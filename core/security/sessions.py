"""Per-session workspace isolation, retention policy enforcement, and 'delete my data'."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.schemas.security import DeletionReceipt, RetentionReport, SessionWorkspace
from core.security.audit import AuditLogger
from core.security.exceptions import InvalidSessionError

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_SESSION_META_FILENAME = ".session_meta.json"


def validate_session_id(session_id: str) -> str:
    """Validate that a session ID is safe and cannot traverse outside the workspace root."""
    if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
        raise InvalidSessionError(
            f"Invalid session_id {session_id!r}. Must match ^[A-Za-z0-9][A-Za-z0-9_-]{{0,63}}$."
        )
    return session_id


def _count_files(path: Path) -> int:
    """Count regular files recursively under ``path``."""
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _handle_readonly_remove(func: Any, path: str, _exc_info: Any) -> None:
    """Clear read-only attributes (e.g. immutable raw files on Windows) and retry removal."""
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    func(path)


def _force_remove_tree(path: Path) -> int:
    """Remove a directory tree (including read-only raw artifacts) and return files removed."""
    if not path.exists():
        return 0
    file_count = _count_files(path)
    if path.is_file():
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        path.unlink()
    else:
        shutil.rmtree(path, onerror=_handle_readonly_remove)
    return file_count


class SessionManager:
    """Manage isolated per-session directories, retention purges, and user data deletion."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize the session manager backed by ``settings.data_dir`` and ``output_dir``."""
        self._settings = settings or get_settings()
        self._audit = audit_logger or AuditLogger(settings=self._settings)

    @property
    def sessions_data_root(self) -> Path:
        """Return root directory holding per-session data workspaces."""
        return self._settings.data_dir / "sessions"

    @property
    def sessions_output_root(self) -> Path:
        """Return root directory holding per-session generated outputs."""
        return self._settings.output_dir / "sessions"

    def create_workspace(
        self,
        session_id: str,
        *,
        actor: str = "system",
    ) -> SessionWorkspace:
        """Create or retrieve an isolated workspace for ``session_id``."""
        safe_id = validate_session_id(session_id)
        root_dir = self.sessions_data_root / safe_id
        raw_dir = root_dir / "raw"
        staging_dir = root_dir / "staging"
        curated_dir = root_dir / "curated"
        output_dir = self.sessions_output_root / safe_id

        for directory in (raw_dir, staging_dir, curated_dir, output_dir):
            directory.mkdir(parents=True, exist_ok=True)

        meta_path = root_dir / _SESSION_META_FILENAME
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created_at = datetime.fromisoformat(meta["created_at"])
        else:
            created_at = datetime.now(UTC)
            meta_path.write_text(
                json.dumps({"session_id": safe_id, "created_at": created_at.isoformat()}),
                encoding="utf-8",
            )
            self._audit.record(
                actor=actor,
                session_id=safe_id,
                action="session_created",
                resource=safe_id,
                status="success",
            )

        return SessionWorkspace(
            session_id=safe_id,
            root_dir=root_dir,
            raw_dir=raw_dir,
            staging_dir=staging_dir,
            curated_dir=curated_dir,
            output_dir=output_dir,
            created_at=created_at,
        )

    def session_settings(self, session_id: str) -> Settings:
        """Return a ``Settings`` copy scoped strictly to the isolated session workspace."""
        workspace = self.create_workspace(session_id)
        return self._settings.model_copy(
            update={
                "data_dir": workspace.root_dir,
                "output_dir": workspace.output_dir,
            }
        )

    def delete_session_data(
        self,
        session_id: str,
        *,
        actor: str = "user",
    ) -> DeletionReceipt:
        """Immediately delete all raw, staging, curated, and output data for a session."""
        safe_id = validate_session_id(session_id)
        data_root = self.sessions_data_root / safe_id
        out_root = self.sessions_output_root / safe_id

        deleted_paths: list[Path] = []
        files_removed = 0
        for candidate in (data_root, out_root):
            if candidate.exists():
                files_removed += _force_remove_tree(candidate)
                deleted_paths.append(candidate)

        now = datetime.now(UTC)
        self._audit.record(
            actor=actor,
            session_id=safe_id,
            action="delete_my_data",
            resource=safe_id,
            status="success",
            metadata={"files_removed": files_removed, "paths_deleted": len(deleted_paths)},
        )

        return DeletionReceipt(
            session_id=safe_id,
            deleted_paths=deleted_paths,
            files_removed=files_removed,
            timestamp=now,
        )

    def _directory_age_timestamp(self, directory: Path) -> datetime:
        """Determine the creation/last-modified UTC timestamp of a session or batch folder."""
        meta_path = directory / _SESSION_META_FILENAME
        if meta_path.is_file():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                return datetime.fromisoformat(data["created_at"]).astimezone(UTC)
            except (ValueError, KeyError, json.JSONDecodeError):
                pass

        latest_mtime = directory.stat().st_mtime
        for child in directory.rglob("*"):
            try:
                child_mtime = child.stat().st_mtime
                if child_mtime > latest_mtime:
                    latest_mtime = child_mtime
            except OSError:
                continue
        return datetime.fromtimestamp(latest_mtime, tz=UTC)

    def enforce_retention_policy(
        self,
        *,
        now: datetime | None = None,
        retention_hours: int | None = None,
        actor: str = "retention_policy",
    ) -> RetentionReport:
        """Delete raw, staging, curated, and output data older than ``retention_hours``."""
        current_time = (now or datetime.now(UTC)).astimezone(UTC)
        hours = retention_hours if retention_hours is not None else self._settings.retention_hours
        cutoff = current_time - timedelta(hours=hours)

        purged_sessions: list[str] = []
        purged_paths: list[Path] = []
        files_removed = 0

        # 1. Purge expired isolated session workspaces.
        if self.sessions_data_root.is_dir():
            for session_dir in sorted(self.sessions_data_root.iterdir()):
                if not session_dir.is_dir():
                    continue
                ts = self._directory_age_timestamp(session_dir)
                if ts < cutoff:
                    session_id = session_dir.name
                    files_removed += _force_remove_tree(session_dir)
                    purged_paths.append(session_dir)
                    out_dir = self.sessions_output_root / session_id
                    if out_dir.exists():
                        files_removed += _force_remove_tree(out_dir)
                        purged_paths.append(out_dir)
                    purged_sessions.append(session_id)

        # 2. Purge expired non-session batch directories under raw/staging/curated.
        for layer_dir in (
            self._settings.raw_dir,
            self._settings.staging_dir,
            self._settings.curated_dir,
        ):
            if not layer_dir.is_dir():
                continue
            for batch_dir in sorted(layer_dir.iterdir()):
                if not batch_dir.is_dir():
                    continue
                ts = self._directory_age_timestamp(batch_dir)
                if ts < cutoff:
                    files_removed += _force_remove_tree(batch_dir)
                    purged_paths.append(batch_dir)

        self._audit.record(
            actor=actor,
            action="retention_purge",
            status="success",
            metadata={
                "retention_hours": hours,
                "purged_sessions_count": len(purged_sessions),
                "purged_paths_count": len(purged_paths),
                "files_removed": files_removed,
            },
        )

        return RetentionReport(
            retention_hours=hours,
            cutoff_timestamp=cutoff,
            purged_sessions=purged_sessions,
            purged_paths=purged_paths,
            files_removed=files_removed,
            executed_at=current_time,
        )

