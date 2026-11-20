"""Per-session orchestrator state persistence and isolated input file staging."""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from core.config import Settings, get_settings
from core.ingestion.db_reader import read_database_tables, redact_connection_url
from core.ingestion.raw_store import compute_sha256
from core.schemas.orchestrator import PipelineSessionState
from core.schemas.security import DeletionReceipt
from core.security import AuditLogger, SessionManager, validate_upload

_STATE_LOCK_GUARD = threading.Lock()
_STATE_LOCKS: dict[Path, threading.RLock] = {}


def _get_path_lock(path: Path) -> threading.RLock:
    """Return a process-wide reentrant lock for ``path`` to prevent Windows file-replace races."""
    resolved = path.resolve()
    with _STATE_LOCK_GUARD:
        lock = _STATE_LOCKS.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _STATE_LOCKS[resolved] = lock
        return lock


class SessionStateStore:
    """Persist ``PipelineSessionState`` JSON atomically inside each isolated session workspace."""

    _STATE_FILENAME = "orchestrator_state.json"

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize the session state store backed by ``SessionManager``."""
        self._settings = settings or get_settings()
        self._audit = audit_logger or AuditLogger(settings=self._settings)
        self._sessions = SessionManager(self._settings, audit_logger=self._audit)

    @property
    def session_manager(self) -> SessionManager:
        """Return the underlying ``SessionManager``."""
        return self._sessions

    def state_path(self, session_id: str) -> Path:
        """Return the path to the persisted orchestrator state JSON for ``session_id``."""
        ws = self._sessions.create_workspace(session_id)
        return ws.root_dir / self._STATE_FILENAME

    def uploads_dir(self, session_id: str) -> Path:
        """Return the isolated uploads staging directory for ``session_id``."""
        ws = self._sessions.create_workspace(session_id)
        target = ws.root_dir / "uploads"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def load(self, session_id: str) -> PipelineSessionState:
        """Load persisted session state, or return a fresh idle state when none exists."""
        path = self.state_path(session_id)
        with _get_path_lock(path):
            if not path.is_file():
                return PipelineSessionState(session_id=session_id)
            raw = ""
            for attempt in range(6):
                try:
                    raw = path.read_text(encoding="utf-8")
                    break
                except PermissionError:
                    if attempt == 5:
                        raise
                    time.sleep(0.02 * (attempt + 1))
            return PipelineSessionState.model_validate_json(raw)

    def save(self, state: PipelineSessionState) -> Path:
        """Atomically persist ``state`` inside the session's isolated workspace."""
        updated = state.model_copy(update={"updated_at": datetime.now(UTC)})
        path = self.state_path(updated.session_id)
        payload = updated.model_dump_json(indent=2) + "\n"
        with _get_path_lock(path):
            tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
            try:
                tmp_path.write_text(payload, encoding="utf-8")
                for attempt in range(8):
                    try:
                        os.replace(tmp_path, path)
                        break
                    except PermissionError:
                        if attempt == 7:
                            # Final fallback when Windows indexer/antivirus holds target handle.
                            path.write_text(payload, encoding="utf-8")
                            break
                        time.sleep(0.03 * (attempt + 1))
            finally:
                if tmp_path.exists():
                    with contextlib.suppress(OSError):
                        tmp_path.unlink()
        return path

    def save_uploaded_bytes(
        self,
        session_id: str,
        filename: str,
        payload: bytes,
        *,
        actor: str = "streamlit_user",
    ) -> Path:
        """Write uploaded file bytes into the session workspace and enforce upload security."""
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError(f"Invalid upload filename: {filename!r}")
        dest = self.uploads_dir(session_id) / safe_name
        session_cfg = self._sessions.session_settings(session_id)

        if len(payload) > session_cfg.max_upload_bytes:
            from core.security.exceptions import FileSizeSecurityError

            raise FileSizeSecurityError(
                f"Upload '{safe_name}' ({len(payload):,} bytes) exceeds limit "
                f"of {session_cfg.max_upload_mb} MB."
            )

        payload_sha = hashlib.sha256(payload).hexdigest()
        with _get_path_lock(dest):
            # If the exact same upload bytes are already staged (e.g. Streamlit rerun while a
            # background worker is reading the file), never truncate or overwrite in place.
            if dest.is_file() and compute_sha256(dest) == payload_sha:
                return dest

            tmp_dest = dest.parent / f".tmp_{uuid.uuid4().hex[:8]}_{safe_name}"
            try:
                tmp_dest.write_bytes(payload)
                validate_upload(tmp_dest, settings=session_cfg)
                os.replace(tmp_dest, dest)
            finally:
                if tmp_dest.exists():
                    with contextlib.suppress(OSError):
                        tmp_dest.unlink()

        self._audit.record(
            actor=actor,
            session_id=session_id,
            action="ui_upload_staged",
            resource=safe_name,
            status="success",
            metadata={"size_bytes": len(payload)},
        )
        return dest

    def stage_database_tables(
        self,
        session_id: str,
        connection_url: str,
        *,
        tables: Sequence[str] | None = None,
        schema: str | None = None,
        actor: str = "streamlit_user",
    ) -> tuple[list[Path], str]:
        """Extract tables from SQLite/PostgreSQL into isolated session CSV files for ETL.

        Returns ``(staged_csv_paths, redacted_connection_label)``. Credentials are never
        persisted or logged.
        """
        session_cfg = self._sessions.session_settings(session_id)
        redacted = redact_connection_url(connection_url)
        extractions = read_database_tables(
            connection_url,
            tables=tables,
            schema=schema,
            max_rows=session_cfg.max_rows,
        )
        out_dir = self.uploads_dir(session_id) / "db_extract"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for ext in extractions:
            csv_path = out_dir / f"{ext.table_name}.csv"
            ext.dataframe.to_csv(csv_path, index=False)
            paths.append(csv_path)

        self._audit.record(
            actor=actor,
            session_id=session_id,
            action="db_tables_staged",
            resource=redacted,
            status="success",
            metadata={"table_count": len(paths)},
        )
        return paths, redacted

    def delete_session(self, session_id: str, *, actor: str = "user") -> DeletionReceipt:
        """Delete all raw, staging, curated, output, and persisted state for ``session_id``."""
        return self._sessions.delete_session_data(session_id, actor=actor)

