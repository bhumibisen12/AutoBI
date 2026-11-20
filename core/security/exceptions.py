"""Custom exception hierarchy for ``core.security``."""

from __future__ import annotations


class SecurityError(Exception):
    """Base exception for all AutoBI security errors."""


class UploadSecurityError(SecurityError):
    """Raised when an uploaded file fails security validation."""


class DisallowedFileTypeError(UploadSecurityError):
    """Raised when an uploaded file extension is not in the allowlist."""


class MacroOrExecutableFileError(UploadSecurityError):
    """Raised when a file contains macros or executable signatures."""


class PasswordProtectedFileError(UploadSecurityError):
    """Raised when an uploaded spreadsheet or archive is encrypted/password-protected."""


class ZipBombError(UploadSecurityError):
    """Raised when an OOXML archive exhibits zip-bomb or zip-slip characteristics."""


class MagicByteMismatchError(UploadSecurityError):
    """Raised when file magic bytes do not match the declared file extension."""


class FileSizeSecurityError(UploadSecurityError):
    """Raised when an upload exceeds the configured file size limit."""


class RowLimitSecurityError(UploadSecurityError):
    """Raised when a dataset exceeds the configured row count limit."""


class InvalidSessionError(SecurityError):
    """Raised when a session identifier is invalid or attempts path traversal."""

