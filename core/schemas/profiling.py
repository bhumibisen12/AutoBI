"""Pydantic contracts for dataset profiling reports."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SemanticType = Literal[
    "id",
    "numeric_measure",
    "categorical_dimension",
    "date_time",
    "boolean",
    "free_text",
    "currency",
    "percent",
]

IssueType = Literal[
    "mixed_types",
    "constant_column",
    "high_nulls",
    "outliers_iqr",
    "outliers_robust_z",
]

IssueSeverity = Literal["info", "warning", "error"]
CardinalityHint = Literal["many_to_one", "one_to_one", "one_to_many", "many_to_many"]


class ValueFrequency(BaseModel):
    """Frequency entry for top values observed in a column."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(description="Stringified value.")
    count: int = Field(ge=0, description="Number of occurrences.")
    percentage: float = Field(ge=0.0, le=100.0, description="Percentage of non-null rows.")


class NumericStats(BaseModel):
    """Descriptive statistics and outlier counts for numeric/currency/percent columns."""

    model_config = ConfigDict(extra="forbid")

    min: float
    max: float
    mean: float
    std: float | None = None
    q25: float
    median: float
    q75: float
    iqr: float
    iqr_outlier_count: int = Field(default=0, ge=0)
    robust_z_outlier_count: int = Field(default=0, ge=0)


class DateRangeStats(BaseModel):
    """Date/time coverage statistics for parsed temporal columns."""

    model_config = ConfigDict(extra="forbid")

    min_date: str = Field(description="Minimum ISO-8601 date/timestamp.")
    max_date: str = Field(description="Maximum ISO-8601 date/timestamp.")
    span_days: int = Field(ge=0, description="Calendar days between min_date and max_date.")


class ColumnIssue(BaseModel):
    """Single data-quality or structural issue identified during profiling."""

    model_config = ConfigDict(extra="forbid")

    issue_type: IssueType
    severity: IssueSeverity
    description: str = Field(min_length=1)
    affected_count: int = Field(default=0, ge=0)
    affected_percentage: float = Field(default=0.0, ge=0.0, le=100.0)


class ColumnProfile(BaseModel):
    """Comprehensive statistical and semantic profile for a single column."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(min_length=1)
    raw_dtype: str = Field(min_length=1)
    semantic_type: SemanticType
    row_count: int = Field(ge=0)
    null_count: int = Field(ge=0)
    null_percentage: float = Field(ge=0.0, le=100.0)
    distinct_count: int = Field(ge=0)
    distinct_percentage: float = Field(ge=0.0, le=100.0)
    numeric_stats: NumericStats | None = None
    top_values: list[ValueFrequency] = Field(default_factory=list)
    date_range: DateRangeStats | None = None
    issues: list[ColumnIssue] = Field(default_factory=list)


class TableProfile(BaseModel):
    """Table-level profile combining row/key diagnostics and per-column profiles."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    duplicate_row_count: int = Field(default=0, ge=0)
    duplicate_row_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    candidate_keys: list[str] = Field(default_factory=list)
    columns: list[ColumnProfile] = Field(default_factory=list)


class CandidateRelationship(BaseModel):
    """Candidate join relationship inferred across two tables."""

    model_config = ConfigDict(extra="forbid")

    left_table: str = Field(min_length=1)
    left_column: str = Field(min_length=1)
    right_table: str = Field(min_length=1)
    right_column: str = Field(min_length=1)
    name_similarity: float = Field(ge=0.0, le=1.0)
    value_overlap: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    cardinality_hint: CardinalityHint


class ProfileReport(BaseModel):
    """Complete dataset profile report across one or more tables."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = None
    tables: list[TableProfile] = Field(default_factory=list)
    candidate_relationships: list[CandidateRelationship] = Field(default_factory=list)
    summary_warnings: list[str] = Field(default_factory=list)

