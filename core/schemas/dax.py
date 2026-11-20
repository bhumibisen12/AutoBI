"""Pydantic contracts for metric selection, DAX generation, validation, and SQL verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MetricTemplateType = Literal[
    "sum",
    "average",
    "distinct_count",
    "divide_ratio",
    "yoy_growth",
    "mom_growth",
    "rolling_avg",
    "percent_of_total",
    "rank",
    "top_n",
]

VerificationStatus = Literal["VERIFIED", "UNVERIFIED", "FAILED"]


class MetricParameters(BaseModel):
    """Allowlisted parameters used to instantiate a DAX template (never free-form DAX)."""

    model_config = ConfigDict(extra="forbid")

    column: str | None = Field(default=None, description="Primary measure or ID column.")
    numerator_column: str | None = Field(default=None, description="Numerator column for DIVIDE.")
    denominator_column: str | None = Field(
        default=None, description="Denominator column for DIVIDE."
    )
    subtract_column: str | None = Field(
        default=None,
        description="Optional column subtracted from numerator (e.g. revenue - cost).",
    )
    date_table: str | None = Field(
        default=None, description="Marked date dimension table (e.g. dim_date)."
    )
    date_column: str | None = Field(
        default=None, description="Date column in the date dimension (e.g. date)."
    )
    fact_date_column: str | None = Field(
        default=None,
        description="Fact table date column joined to dim_date for SQL reconciliation.",
    )
    dimension_table: str | None = Field(
        default=None, description="Dimension table for % of total, rank, or top N."
    )
    dimension_column: str | None = Field(
        default=None, description="Dimension column for % of total, rank, or top N."
    )
    window_months: int = Field(
        default=3, ge=2, le=24, description="Rolling average lookback window in months."
    )
    top_n: int = Field(default=3, ge=1, le=100, description="Top N cutoff for TOPN measures.")


class SelectedMetricSpec(BaseModel):
    """Single KPI selected and parameterised strictly from the metrics registry."""

    model_config = ConfigDict(extra="forbid")

    measure_name: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
    description: str = Field(default="")
    format_string: str | None = None
    params: MetricParameters = Field(default_factory=MetricParameters)
    rag_source_ids: list[str] = Field(default_factory=list)


class MetricSelectionPlan(BaseModel):
    """Structured LLM output choosing metrics exclusively from the metrics registry."""

    model_config = ConfigDict(extra="forbid")

    selected_metrics: list[SelectedMetricSpec] = Field(default_factory=list)
    rationale: str = Field(default="")


class DAXMeasureSpec(BaseModel):
    """Rendered DAX measure ready for validation, reconciliation, and TMDL export."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    table_name: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    template_type: str = Field(min_length=1)
    expression: str = Field(min_length=1)
    format_string: str = Field(default="#,##0.00")
    display_folder: str = Field(default="KPIs")
    description: str = Field(default="")
    params: MetricParameters = Field(default_factory=MetricParameters)
    rag_source_ids: list[str] = Field(default_factory=list)


class DAXValidationResult(BaseModel):
    """Syntax, schema, DIVIDE, and implicit-measure validation result for one DAX measure."""

    model_config = ConfigDict(extra="forbid")

    measure_name: str = Field(min_length=1)
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    referenced_tables: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)


class MeasureVerificationResult(BaseModel):
    """Side-by-side DAX, equivalent DuckDB SQL, and reconciled result for a single measure."""

    model_config = ConfigDict(extra="forbid")

    measure_name: str = Field(min_length=1)
    template_id: str = Field(min_length=1)
    dax: str = Field(min_length=1)
    sql: str | None = None
    result: float | None = Field(
        default=None,
        description="Scalar result computed by DuckDB SQL against curated tables.",
    )
    breakdown: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional per-category breakdown rows for rank or % of total measures.",
    )
    status: VerificationStatus
    validation_errors: list[str] = Field(default_factory=list)
    notes: str = Field(default="")


class VerificationReport(BaseModel):
    """Complete verification report pairing every generated DAX measure with DuckDB SQL."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = None
    curated_duckdb_path: Path | None = None
    total_measures: int = Field(ge=0)
    verified_count: int = Field(ge=0)
    unverified_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    results: list[MeasureVerificationResult] = Field(default_factory=list)
