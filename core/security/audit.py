"""Append-only security audit log that never stores raw cell data or secrets."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.logging import REDACTED, redact_text, redact_value
from core.schemas.security import AuditEvent, AuditStatus
from core.security.sanitization import strip_control_chars

_FORBIDDEN_AUDIT_KEYS: frozenset[str] = frozenset(
    {
        "raw_data",
        "raw_rows",
        "sample_rows",
        "rows",
        "dataframe",
        "cells",
        "records",
        "api_key",
        "llm_api_key",
        "password",
        "passwd",
        "secret",
        "token",
        "authorization",
        "connection_url",
    }
)


class AuditLogger:
    """Append-only JSON Lines security audit logger."""

    def __init__(
        self,
        log_path: Path | None = None,
        *,
        settings: Settings | None = None,
        extra_secrets: Iterable[str] = (),
    ) -> None:
        """Initialize the audit logger rooted under ``<data_dir>/audit/audit_log.jsonl``."""
        cfg = settings or get_settings()
        self._log_path = log_path or (cfg.data_dir / "audit" / "audit_log.jsonl")
        self._secrets: tuple[str, ...] = tuple(
            s for s in (*cfg.known_secret_values(), *extra_secrets) if s
        )

    @property
    def log_path(self) -> Path:
        """Return the path to the JSONL audit log file."""
        return self._log_path

    def _sanitize_metadata(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, str | int | float | bool]:
        """Filter out forbidden keys and redact secrets from structured audit metadata."""
        if not metadata:
            return {}
        clean: dict[str, str | int | float | bool] = {}
        for raw_key, raw_val in metadata.items():
            key = strip_control_chars(str(raw_key))[:80]
            lowered = key.lower()
            if lowered in _FORBIDDEN_AUDIT_KEYS or any(
                marker in lowered for marker in ("secret", "password", "token", "api_key")
            ):
                clean[key] = REDACTED
                continue
            redacted = redact_value(raw_val, self._secrets)
            if isinstance(redacted, bool | int | float):
                clean[key] = redacted
            else:
                clean[key] = strip_control_chars(
                    redact_text(str(redacted), self._secrets)
                )[:250]
        return clean

    def record(
        self,
        *,
        actor: str,
        action: str,
        status: AuditStatus = "success",
        session_id: str | None = None,
        batch_id: str | None = None,
        resource: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AuditEvent:
        """Append a sanitized, secret-free audit event to the audit log."""
        event = AuditEvent(
            event_id=uuid.uuid4().hex,
            timestamp=datetime.now(UTC),
            actor=strip_control_chars(redact_text(actor, self._secrets))[:120] or "unknown",
            session_id=(
                strip_control_chars(redact_text(session_id, self._secrets))[:120]
                if session_id
                else None
            ),
            batch_id=(
                strip_control_chars(redact_text(batch_id, self._secrets))[:120]
                if batch_id
                else None
            ),
            action=strip_control_chars(redact_text(action, self._secrets))[:120],
            resource=(
                strip_control_chars(redact_text(resource, self._secrets))[:250]
                if resource
                else None
            ),
            status=status,
            metadata=self._sanitize_metadata(metadata),
        )

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        return event

    def read_events(
        self,
        *,
        session_id: str | None = None,
        batch_id: str | None = None,
    ) -> list[AuditEvent]:
        """Read recorded audit events, optionally filtered by session or batch ID."""
        if not self._log_path.exists():
            return []
        events: list[AuditEvent] = []
        with self._log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                event = AuditEvent.model_validate_json(stripped)
                if session_id is not None and event.session_id != session_id:
                    continue
                if batch_id is not None and event.batch_id != batch_id:
                    continue
                events.append(event)
        return events
