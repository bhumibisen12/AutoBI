"""Pydantic schemas exchanged by ingestion and storage modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SourceType = Literal["csv", "excel", "database"]
DataLayer = Literal["raw", "staging", "curated"]
BatchStatus = Literal["started", "completed", "failed"]


class RawFileRecord(BaseModel):
    """Immutable raw artifact metadata stored inside ``manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, description="Sanitized filename saved in the raw layer.")
    raw_path: Path = Field(description="Path to the immutable raw copy under data/raw/<batch_id>/.")
    size_bytes: int = Field(ge=0, description="File size in bytes.")
    sha256: str = Field(
        min_length=64,
        max_length=64,
        description="Hex-encoded SHA-256 checksum of the raw artifact.",
    )
    source_type: SourceType = Field(description="Source format category.")
    row_count: int = Field(ge=0, description="Total rows extracted from this source.")
    column_count: int = Field(ge=0, description="Total columns extracted across tables.")
    timestamp: datetime = Field(description="UTC timestamp when the raw artifact was captured.")
    detected_encoding: str | None = Field(
        default=None,
        description="Detected text encoding for delimited text sources.",
    )
    detected_delimiter: str | None = Field(
        default=None,
        description="Detected field delimiter for delimited text sources.",
    )
    tables_extracted: list[str] = Field(
        default_factory=list,
        description="Logical table names extracted from this source.",
    )


class RawManifest(BaseModel):
    """Manifest describing all immutable raw inputs captured for a batch."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, description="Unique batch identifier.")
    created_at: datetime = Field(description="UTC timestamp when the batch manifest was written.")
    files: list[RawFileRecord] = Field(
        default_factory=list,
        description="Immutable raw file records captured in this batch.",
    )


class TableArtifact(BaseModel):
    """Metadata for a single extracted and optionally staged table."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1, description="Normalized logical table identifier.")
    source_file: str = Field(min_length=1, description="Source filename or database label.")
    source_object: str | None = Field(
        default=None,
        description="Origin sheet name (Excel) or table name (database), if applicable.",
    )
    row_count: int = Field(ge=0, description="Number of data rows extracted.")
    column_count: int = Field(ge=0, description="Number of columns extracted.")
    columns: list[str] = Field(default_factory=list, description="Ordered column names.")
    dtypes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of column name to extracted pandas dtype string.",
    )
    header_row_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based row index detected as the header row.",
    )
    staging_parquet_path: Path | None = Field(
        default=None,
        description="Path to the staging Parquet file once written by core/storage.",
    )


class DatasetBundle(BaseModel):
    """Primary Pydantic contract returned by ingestion and enriched by storage."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, description="Unique batch identifier.")
    created_at: datetime = Field(description="UTC timestamp when the bundle was created.")
    tables: list[TableArtifact] = Field(
        default_factory=list,
        description="Metadata for every extracted table in the batch.",
    )
    source_files: list[RawFileRecord] = Field(
        default_factory=list,
        description="Immutable raw file records captured for this batch.",
    )
    total_rows: int = Field(default=0, ge=0, description="Total rows across all tables.")
    total_columns: int = Field(default=0, ge=0, description="Total columns across all tables.")
    load_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings emitted during extraction or staging.",
    )
    raw_dir: Path | None = Field(
        default=None,
        description="Directory containing immutable raw files and manifest.json.",
    )
    manifest_path: Path | None = Field(
        default=None,
        description="Path to manifest.json inside the raw batch directory.",
    )
    staging_dir: Path | None = Field(
        default=None,
        description="Directory containing staging Parquet files and DuckDB database.",
    )
    duckdb_path: Path | None = Field(
        default=None,
        description="Path to the batch DuckDB database registering staged tables.",
    )


class BatchLogEntry(BaseModel):
    """Append-only batch execution log record maintained by ``core/storage``."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1, description="Batch identifier.")
    layer: DataLayer = Field(description="Pipeline data layer recorded by this entry.")
    status: BatchStatus = Field(description="Execution status for the batch layer.")
    timestamp: datetime = Field(description="UTC timestamp of the log event.")
    table_count: int = Field(ge=0, description="Number of tables recorded in this step.")
    total_rows: int = Field(ge=0, description="Total rows recorded in this step.")
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings associated with this batch step.",
    )
    details: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Additional non-secret structured metadata.",
    )

