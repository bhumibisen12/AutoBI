"""Structured application logging with automatic secret and API-key redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, TextIO

from core.config import Settings, get_settings

REDACTED = "[REDACTED]"
LOGGER_NAMESPACE = "autobi"

_SENSITIVE_FIELD_MARKERS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "authorization",
    "credential",
    "private_key",
)

# Standard LogRecord attributes excluded from structured extra-field serialization.
_STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime"}

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Bearer / Authorization tokens
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    # Common provider key formats (OpenAI, Anthropic, Google AI, Groq, GitHub, AWS-like)
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    # Generic key=value, key: value, or JSON "key": "value" patterns
    re.compile(
        r'(?i)(["\']?(?:llm_api_key|api[_-]?key|secret[_-]?key|access[_-]?token|'
        r'auth[_-]?token|password|passwd|secret)["\']?\s*[:=]\s*["\']?)'
        r'([^\s"\',;}\]]+)'
    ),
)


def _is_sensitive_key(key: str) -> bool:
    """Return True if a structured log field name indicates sensitive content."""
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_FIELD_MARKERS)


def redact_text(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Mask known secrets and API-key patterns within a string.

    Args:
        text: Input string that may contain secret material.
        extra_secrets: Concrete secret values (e.g., from settings) to mask verbatim.

    Returns:
        Sanitized string with secrets replaced by ``[REDACTED]``.
    """
    sanitized = text
    for secret in extra_secrets:
        if secret and len(secret) >= 4:
            sanitized = sanitized.replace(secret, REDACTED)

    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 1:
            sanitized = pattern.sub(rf"\g<1>{REDACTED}", sanitized)
        else:
            sanitized = pattern.sub(REDACTED, sanitized)

    return sanitized


def redact_value(value: Any, extra_secrets: Iterable[str] = ()) -> Any:
    """Recursively redact sensitive strings and dictionary entries."""
    secrets_tuple = tuple(extra_secrets)
    if isinstance(value, str):
        return redact_text(value, secrets_tuple)
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and _is_sensitive_key(key)
                else redact_value(val, secrets_tuple)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, secrets_tuple) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, secrets_tuple) for item in value)
    return value


class SecretRedactionFilter(logging.Filter):
    """Logging filter that strips secrets from messages, arguments, and extra fields."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        """Initialize the filter with an optional iterable of known secret values."""
        super().__init__()
        self._secrets: tuple[str, ...] = tuple(s for s in secrets if s)

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive values in place and always allow the sanitized record through."""
        if record.args:
            redacted_args = redact_value(record.args, self._secrets)
            try:
                rendered = str(record.msg) % redacted_args
            except (TypeError, ValueError):
                rendered = f"{record.msg} {redacted_args}"
            record.msg = redact_text(rendered, self._secrets)
            record.args = None
        elif isinstance(record.msg, str):
            record.msg = redact_text(record.msg, self._secrets)
        else:
            record.msg = redact_value(record.msg, self._secrets)

        if record.exc_text:
            record.exc_text = redact_text(record.exc_text, self._secrets)

        for key in list(record.__dict__.keys()):
            if key in _STANDARD_LOG_RECORD_ATTRS:
                continue
            if _is_sensitive_key(key):
                setattr(record, key, REDACTED)
            else:
                setattr(record, key, redact_value(getattr(record, key), self._secrets))

        return True


class StructuredJsonFormatter(logging.Formatter):
    """Emit structured single-line JSON log records with UTC ISO-8601 timestamps."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        """Initialize the formatter with known secret values for final-pass redaction."""
        super().__init__()
        self._secrets: tuple[str, ...] = tuple(s for s in secrets if s)

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record into a JSON string with redacted message and extras."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage(), self._secrets),
        }

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = (
                REDACTED if _is_sensitive_key(key) else redact_value(value, self._secrets)
            )

        if record.exc_info:
            formatted_exc = self.formatException(record.exc_info)
            payload["exception"] = redact_text(formatted_exc, self._secrets)

        return json.dumps(payload, default=str, ensure_ascii=False)


class RedactingTextFormatter(logging.Formatter):
    """Human-readable formatter with final-pass secret redaction."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        """Initialize the text formatter with known secret values."""
        super().__init__("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        self._secrets: tuple[str, ...] = tuple(s for s in secrets if s)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as plain text with final-pass redaction."""
        return redact_text(super().format(record), self._secrets)


def configure_logging(
    settings: Settings | None = None,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the AutoBI namespace logger with structured output and secret redaction.

    Args:
        settings: Optional `Settings` instance. Defaults to `get_settings()`.
        stream: Optional output stream (defaults to `sys.stderr`).

    Returns:
        The configured ``autobi`` root namespace logger.
    """
    resolved = settings or get_settings()
    secrets = resolved.known_secret_values()

    logger = logging.getLogger(LOGGER_NAMESPACE)
    logger.setLevel(getattr(logging, resolved.log_level, logging.INFO))
    logger.propagate = False

    # Replace existing handlers to keep repeated calls idempotent.
    for existing in list(logger.handlers):
        logger.removeHandler(existing)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(getattr(logging, resolved.log_level, logging.INFO))
    handler.addFilter(SecretRedactionFilter(secrets))
    handler.setFormatter(
        StructuredJsonFormatter(secrets)
        if resolved.log_json
        else RedactingTextFormatter(secrets)
    )
    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced AutoBI logger, configuring defaults if not yet initialized."""
    root_logger = logging.getLogger(LOGGER_NAMESPACE)
    if not root_logger.handlers:
        configure_logging()

    if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")
