"""Extraction-only ingestion layer for CSV, Excel, and SQLAlchemy database sources."""

from core.ingestion.exceptions import (
    ImmutableRawError,
    IngestionError,
    RowLimitExceededError,
    UnsupportedSourceError,
    UploadSizeLimitError,
)
from core.ingestion.extractor import generate_batch_id, ingest_database, ingest_files
from core.ingestion.raw_store import compute_sha256

__all__ = [
    "ImmutableRawError",
    "IngestionError",
    "RowLimitExceededError",
    "UnsupportedSourceError",
    "UploadSizeLimitError",
    "compute_sha256",
    "generate_batch_id",
    "ingest_database",
    "ingest_files",
]

