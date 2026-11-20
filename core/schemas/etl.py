"""Pydantic contracts for ETL orchestration, transformation, validation gates, and batch audit."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.schemas.lineage import LineageEvent, LineageGraph
from core.schemas.profiling import CardinalityHint

ETLBatchStatus = Literal["completed", "failed"]


class TableReconciliation(BaseModel):
    """Per-table row reconciliation proving zero silent drops or fabrications."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    rows_read: int = Field(ge=0)
    rows_staged: int = Field(ge=0)
    rows_curated: int = Field(ge=0)
    rows_rejected: int = Field(ge=0)
    rows_dropped_duplicates: int = Field(ge=0)
    rejection_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="Count of quarantined rows grouped by explicit rejection reason.",
    )
    duplicate_reason: str = Field(default="duplicate_row")

    @property
    def is_reconciled(self) -> bool:
        """Return True iff ``rows_read == rows_staged == curated + rejected + duplicates``."""
        return (
            self.rows_read == self.rows_staged
            and self.rows_staged
            == self.rows_curated + self.rows_rejected + self.rows_dropped_duplicates
        )


class QualityGateConfig(BaseModel):
    """Configurable data-quality gate thresholds enforced before loading curated tables."""

    model_config = ConfigDict(extra="forbid")

    max_null_pct: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Maximum allowed null percentage in any curated column.",
    )
    max_duplicate_pct: float = Field(
        default=25.0,
        ge=0.0,
        le=100.0,
        description="Maximum allowed duplicate-row percentage observed in staged input.",
    )
    max_invalid_type_pct: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Maximum allowed percentage of rows rejected for type/date parse errors.",
    )
    max_outlier_pct: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Maximum allowed outlier percentage (IQR or robust z-score) per numeric col.",
    )
    require_non_empty: bool = Field(
        default=True,
        description="Require every curated source table to retain at least one row.",
    )
    require_reconciliation: bool = Field(
        default=True,
        description="Require exact row-count reconciliation across read/staged/curated/quarantine.",
    )


class QualityGateCheck(BaseModel):
    """Outcome of a single data-quality gate evaluation."""

    model_config = ConfigDict(extra="forbid")

    gate_name: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
    column_name: str | None = None
    passed: bool
    observed_value: float
    threshold_value: float
    message: str = Field(min_length=1)


class ValidationReport(BaseModel):
    """Aggregate data-quality gate report returned by ``core.validation``."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1)
    passed: bool
    checks: list[QualityGateCheck] = Field(default_factory=list)

    @property
    def failed_checks(self) -> list[QualityGateCheck]:
        """Return only the checks that failed."""
        return [chk for chk in self.checks if not chk.passed]


class CuratedRelationship(BaseModel):
    """Validated relationship registered in the curated semantic-ready layer."""

    model_config = ConfigDict(extra="forbid")

    from_table: str = Field(min_length=1)
    from_column: str = Field(min_length=1)
    to_table: str = Field(min_length=1)
    to_column: str = Field(min_length=1)
    cardinality: CardinalityHint
    source: Literal["profile_inferred", "generated_date_dim"]


class CuratedTableInfo(BaseModel):
    """Metadata describing a single table in the curated layer."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    primary_key: str | None = None
    surrogate_key_added: bool = False
    is_generated: bool = False
    curated_parquet_path: Path | None = None


class CuratedBuildSummary(BaseModel):
    """Structured metadata produced by ``core.transformation`` when building curated tables."""

    model_config = ConfigDict(extra="forbid")

    tables: list[CuratedTableInfo] = Field(default_factory=list)
    relationships: list[CuratedRelationship] = Field(default_factory=list)
    date_dimension_created: bool = False
    dropped_duplicates_by_table: dict[str, int] = Field(default_factory=dict)
    lineage_events: list[LineageEvent] = Field(default_factory=list)


class ETLBatchLogRecord(BaseModel):
    """Audit batch-log table entry persisted by ``core.storage`` in DuckDB and JSONL."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    status: ETLBatchStatus
    rows_read: int = Field(ge=0)
    rows_staged: int = Field(ge=0)
    rows_curated: int = Field(ge=0)
    rows_rejected: int = Field(ge=0)
    rows_dropped_duplicates: int = Field(ge=0)
    checksum: str = Field(min_length=64, max_length=64)
    code_version: str = Field(min_length=1)
    failure_reason: str | None = None


class ETLRunResult(BaseModel):
    """Complete Pydantic summary returned by ``core.etl.run_etl_pipeline``."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str = Field(min_length=1)
    idempotent_replay: bool = False
    checksum: str = Field(min_length=64, max_length=64)
    raw_dir: Path
    staging_dir: Path
    curated_dir: Path
    curated_duckdb_path: Path
    staging_duckdb_path: Path
    curated_tables: list[CuratedTableInfo] = Field(default_factory=list)
    quarantine_tables: dict[str, Path] = Field(default_factory=dict)
    relationships: list[CuratedRelationship] = Field(default_factory=list)
    reconciliations: list[TableReconciliation] = Field(default_factory=list)
    validation_report: ValidationReport
    lineage_graph: LineageGraph
    batch_log: ETLBatchLogRecord

