"""Pydantic contracts for column-level data lineage tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LineageEvent(BaseModel):
    """Single atomic transformation event linking source columns to target columns."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    timestamp: datetime
    table_name: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    operation: str = Field(min_length=1)
    source_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    rows_affected: int = Field(default=0, ge=0)
    applied: bool = Field(
        default=True,
        description="True if the operation mutated data; False if idempotently skipped.",
    )
    note: str = Field(default="")


class ColumnLineageChain(BaseModel):
    """Trace of original source column(s) and ordered operations producing a target column."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    origin_columns: list[str] = Field(default_factory=list)
    operations: list[str] = Field(default_factory=list)
    events: list[LineageEvent] = Field(default_factory=list)


class LineageGraph(BaseModel):
    """Complete lineage record for a table or pipeline batch."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = None
    events: list[LineageEvent] = Field(default_factory=list)
    column_chains: list[ColumnLineageChain] = Field(default_factory=list)

