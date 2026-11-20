"""Pydantic contracts for security validation, PII detection, sessions, and audit logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AllowedFileKind = Literal["csv", "xlsx", "xls"]
PIIType = Literal[
    "email",
    "phone",
    "credit_card",
    "pan",
    "aadhaar",
    "ssn",
    "person_name",
    "address",
    "date_of_birth",
]
PIIDetectionMethod = Literal["regex", "column_name", "both"]
AuditStatus = Literal["success", "denied", "failure"]


class UploadValidationResult(BaseModel):
    """Structured outcome of validating an uploaded file before ingestion."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, description="Original filename evaluated.")
    file_kind: AllowedFileKind = Field(description="Validated file format.")
    size_bytes: int = Field(ge=0, description="File size in bytes.")
    is_valid: bool = Field(default=True, description="True if all security checks passed.")
    checks_passed: list[str] = Field(
        default_factory=list,
        description="Names of security checks passed during validation.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings emitted during validation.",
    )


class PIIFinding(BaseModel):
    """Single column-level PII detection finding."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1, description="Column name flagged for PII.")
    pii_type: PIIType = Field(description="Detected PII category.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence in [0, 1].")
    row_percentage: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of non-null rows matching the PII pattern.",
    )
    matched_rows: int = Field(ge=0, description="Number of rows matching the PII pattern.")
    detection_method: PIIDetectionMethod = Field(
        description="Whether detected by regex, column-name heuristic, or both."
    )


class PIIReport(BaseModel):
    """Summary of PII findings across a single table."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1, description="Logical table name scanned.")
    total_rows: int = Field(ge=0, description="Total rows evaluated in the table.")
    findings: list[PIIFinding] = Field(
        default_factory=list,
        description="List of detected PII findings.",
    )

    @property
    def flagged_columns(self) -> dict[str, PIIType]:
        """Return mapping of flagged column name to highest-confidence PII type."""
        best: dict[str, tuple[float, PIIType]] = {}
        for finding in self.findings:
            existing = best.get(finding.column)
            if existing is None or finding.confidence > existing[0]:
                best[finding.column] = (finding.confidence, finding.pii_type)
        return {col: pii_type for col, (_, pii_type) in best.items()}


class SessionWorkspace(BaseModel):
    """Isolated per-session workspace paths across raw, staging, curated, and output layers."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="Validated session identifier.")
    root_dir: Path = Field(description="Session root directory under data/sessions/.")
    raw_dir: Path = Field(description="Session raw layer directory.")
    staging_dir: Path = Field(description="Session staging layer directory.")
    curated_dir: Path = Field(description="Session curated layer directory.")
    output_dir: Path = Field(description="Session generated output directory.")
    created_at: datetime = Field(description="UTC timestamp when the session was created.")


class DeletionReceipt(BaseModel):
    """Receipt returned by ``delete_session_data`` ('delete my data')."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, description="Deleted session identifier.")
    deleted_paths: list[Path] = Field(
        default_factory=list,
        description="Filesystem paths deleted for the session.",
    )
    files_removed: int = Field(ge=0, description="Total files removed.")
    timestamp: datetime = Field(description="UTC timestamp of deletion.")


class RetentionReport(BaseModel):
    """Summary returned by ``enforce_retention_policy``."""

    model_config = ConfigDict(extra="forbid")

    retention_hours: int = Field(ge=1, description="Configured retention threshold in hours.")
    cutoff_timestamp: datetime = Field(description="UTC cutoff older than which data was purged.")
    purged_sessions: list[str] = Field(
        default_factory=list,
        description="Session IDs purged by retention.",
    )
    purged_paths: list[Path] = Field(
        default_factory=list,
        description="Batch or session directories purged.",
    )
    files_removed: int = Field(ge=0, description="Total files removed across purged paths.")
    executed_at: datetime = Field(description="UTC timestamp when retention ran.")


class AuditEvent(BaseModel):
    """Structured audit-log record (never contains raw data or secrets)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, description="Unique audit event identifier.")
    timestamp: datetime = Field(description="UTC timestamp when the event occurred.")
    actor: str = Field(min_length=1, description="Sanitized actor/user identifier.")
    session_id: str | None = Field(default=None, description="Associated session ID, if any.")
    batch_id: str | None = Field(default=None, description="Associated batch ID, if any.")
    action: str = Field(min_length=1, description="Audited action name.")
    resource: str | None = Field(
        default=None,
        description="Resource identifier (filename or table name; never raw cell data).",
    )
    status: AuditStatus = Field(description="Outcome of the audited action.")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Non-secret, non-raw-data metadata for the event.",
    )

