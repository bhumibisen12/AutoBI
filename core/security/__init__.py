"""Security utilities: upload validation, sanitisation, PII detection, sessions, and audit."""

from core.security.audit import AuditLogger
from core.security.exceptions import (
    DisallowedFileTypeError,
    FileSizeSecurityError,
    InvalidSessionError,
    MacroOrExecutableFileError,
    MagicByteMismatchError,
    PasswordProtectedFileError,
    RowLimitSecurityError,
    SecurityError,
    UploadSecurityError,
    ZipBombError,
)
from core.security.pii import (
    detect_pii,
    hash_pii_value,
    is_valid_luhn,
    mask_pii_value,
    mask_sample_rows_for_llm,
    redact_inline_pii,
)
from core.security.sanitization import (
    neutralize_dataframe_for_export,
    neutralize_formula_injection,
    sanitize_for_llm,
    sanitize_prompt_payload,
    strip_control_chars,
)
from core.security.sessions import SessionManager, validate_session_id
from core.security.upload_validator import (
    ALLOWED_EXTENSIONS,
    validate_row_count,
    validate_upload,
)

__all__ = [
    "ALLOWED_EXTENSIONS",
    "AuditLogger",
    "DisallowedFileTypeError",
    "FileSizeSecurityError",
    "InvalidSessionError",
    "MacroOrExecutableFileError",
    "MagicByteMismatchError",
    "PasswordProtectedFileError",
    "RowLimitSecurityError",
    "SecurityError",
    "SessionManager",
    "UploadSecurityError",
    "ZipBombError",
    "detect_pii",
    "hash_pii_value",
    "is_valid_luhn",
    "mask_pii_value",
    "mask_sample_rows_for_llm",
    "neutralize_dataframe_for_export",
    "neutralize_formula_injection",
    "redact_inline_pii",
    "sanitize_for_llm",
    "sanitize_prompt_payload",
    "strip_control_chars",
    "validate_row_count",
    "validate_session_id",
    "validate_upload",
]

