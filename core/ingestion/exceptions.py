"""Custom exception hierarchy for ``core.ingestion``."""

from __future__ import annotations


class IngestionError(Exception):
    """Base exception for all ingestion-layer errors."""


class UploadSizeLimitError(IngestionError):
    """Raised when an uploaded file exceeds ``Settings.max_upload_bytes``."""


class RowLimitExceededError(IngestionError):
    """Raised when an extracted table or batch exceeds ``Settings.max_rows``."""


class UnsupportedSourceError(IngestionError):
    """Raised when a file extension or database dialect is not supported."""


class ImmutableRawError(IngestionError):
    """Raised when an operation attempts to overwrite an immutable raw artifact."""

