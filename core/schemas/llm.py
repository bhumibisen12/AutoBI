"""Pydantic contracts for the LLM layer: DatasetUnderstanding, UserIntent, and call metadata."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ColumnRole = Literal["dimension", "measure", "date", "id", "text"]
AggregationType = Literal["sum", "avg", "count", "distinct_count", "min", "max", "none"]
TimeGrain = Literal["day", "week", "month", "quarter", "year", "unspecified"]
FilterOperator = Literal[
    "eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "between", "contains"
]


class ColumnUnderstanding(BaseModel):
    """Business semantic interpretation of a single curated column."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    column_name: str = Field(min_length=1)
    role: ColumnRole
    friendly_name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    suggested_aggregations: list[AggregationType] = Field(default_factory=list)


class CandidateKPI(BaseModel):
    """Proposed business KPI grounded in dataset columns with rationale."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    related_columns: list[str] = Field(default_factory=list)
    suggested_aggregation: AggregationType = "sum"


class AmbiguityQuestion(BaseModel):
    """Clarifying question to ask the user when schema semantics are ambiguous."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    context: str = Field(default="")
    options: list[str] = Field(default_factory=list)


class DatasetUnderstanding(BaseModel):
    """Structured LLM understanding of an uploaded dataset profile."""

    model_config = ConfigDict(extra="forbid")

    business_domain: str = Field(min_length=1)
    summary: str = Field(default="")
    columns: list[ColumnUnderstanding] = Field(default_factory=list)
    candidate_kpis: list[CandidateKPI] = Field(default_factory=list)
    ambiguities: list[AmbiguityQuestion] = Field(default_factory=list)


class IntentFilter(BaseModel):
    """Structured filter requested in the user's natural-language prompt."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    operator: FilterOperator = "eq"
    values: list[str | int | float | bool] = Field(default_factory=list)


class UserIntent(BaseModel):
    """Structured interpretation of the user's natural-language analytics prompt."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    audience: str = Field(default="business stakeholders", min_length=1)
    time_grain: TimeGrain = "unspecified"
    filters: list[IntentFilter] = Field(default_factory=list)
    requested_visuals: list[str] = Field(default_factory=list)
    focus_kpis: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token consumption and estimated USD cost for an LLM call."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


class LLMCallMetadata(BaseModel):
    """Operational telemetry returned alongside a validated structured LLM output."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=64, max_length=64)
    cached: bool = False
    attempts: int = Field(default=1, ge=1)
    latency_ms: float = Field(default=0.0, ge=0.0)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class MaskedTableContext(BaseModel):
    """LLM-safe representation of one table: sanitized stats + masked sample rows only."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    candidate_keys: list[str] = Field(default_factory=list)
    columns: list[dict[str, Any]] = Field(default_factory=list)
    masked_sample_rows: list[dict[str, Any]] = Field(default_factory=list)


class SanitizedDatasetContext(BaseModel):
    """Complete PII-masked, sanitized dataset profile context safe for LLM prompts."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = None
    tables: list[MaskedTableContext] = Field(default_factory=list)
    candidate_relationships: list[dict[str, Any]] = Field(default_factory=list)
    summary_warnings: list[str] = Field(default_factory=list)
