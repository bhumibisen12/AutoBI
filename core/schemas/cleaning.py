"""Declarative Pydantic schemas for allowlisted cleaning plans and execution logs."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.schemas.lineage import LineageEvent

CastTargetType = Literal["int", "float", "str", "bool"]
TrimSide = Literal["both", "left", "right"]
CaseStyle = Literal["lower", "upper", "title"]
FillStrategy = Literal["constant", "median", "mode"]
DuplicateKeep = Literal["first", "last"]
ScalarValue = str | int | float | bool | None


class RenameColumnOp(BaseModel):
    """Rename a single column."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["rename_column"] = "rename_column"
    column: str = Field(min_length=1)
    new_name: str = Field(min_length=1)


class CastTypeOp(BaseModel):
    """Cast a column to a target primitive dtype."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["cast_type"] = "cast_type"
    column: str = Field(min_length=1)
    target_type: CastTargetType


class ParseDateOp(BaseModel):
    """Parse a column into datetime values."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["parse_date"] = "parse_date"
    column: str = Field(min_length=1)
    date_format: str | None = Field(default=None)
    utc: bool = Field(default=False)


class TrimTextOp(BaseModel):
    """Strip leading/trailing whitespace from string values in a column."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["trim_text"] = "trim_text"
    column: str = Field(min_length=1)
    side: TrimSide = "both"


class StandardizeCaseOp(BaseModel):
    """Normalize letter casing for string values in a column."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["standardize_case"] = "standardize_case"
    column: str = Field(min_length=1)
    case: CaseStyle


class FillNullOp(BaseModel):
    """Fill missing values using an allowlisted strategy (constant, median, or mode)."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["fill_null"] = "fill_null"
    column: str = Field(min_length=1)
    strategy: FillStrategy
    constant_value: ScalarValue = None

    @model_validator(mode="after")
    def _validate_constant(self) -> FillNullOp:
        """Require ``constant_value`` when strategy is ``constant``."""
        if self.strategy == "constant" and self.constant_value is None:
            raise ValueError("constant_value must be provided when strategy is 'constant'.")
        return self


class DropDuplicatesOp(BaseModel):
    """Drop duplicate rows across all columns or a specified subset."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["drop_duplicates"] = "drop_duplicates"
    subset: list[str] | None = None
    keep: DuplicateKeep = "first"


class DropColumnOp(BaseModel):
    """Remove an unnecessary column (idempotent if already absent)."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["drop_column"] = "drop_column"
    column: str = Field(min_length=1)


class MapValuesOp(BaseModel):
    """Map exact column values using a declarative dictionary allowlist."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["map_values"] = "map_values"
    column: str = Field(min_length=1)
    mapping: dict[str, ScalarValue] = Field(min_length=1)
    case_insensitive: bool = False


class SplitColumnOp(BaseModel):
    """Split a text column by a fixed delimiter into named target columns."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["split_column"] = "split_column"
    column: str = Field(min_length=1)
    delimiter: str = Field(min_length=1)
    into: list[str] = Field(min_length=2)
    drop_source: bool = False


CleaningOperation = Annotated[
    RenameColumnOp
    | CastTypeOp
    | ParseDateOp
    | TrimTextOp
    | StandardizeCaseOp
    | FillNullOp
    | DropDuplicatesOp
    | DropColumnOp
    | MapValuesOp
    | SplitColumnOp,
    Field(discriminator="op"),
]


class CleaningPlan(BaseModel):
    """Declarative, JSON-serializable cleaning plan restricted to allowlisted operations."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    operations: list[CleaningOperation] = Field(default_factory=list)
    description: str = Field(default="")


class CleaningStepLog(BaseModel):
    """Execution audit entry for a single cleaning operation."""

    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(ge=0)
    operation: str = Field(min_length=1)
    applied: bool = True
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    rows_affected: int = Field(default=0, ge=0)
    failed_row_indices: list[int] = Field(
        default_factory=list,
        description="Row indices where parsing/casting failed (for downstream quarantine).",
    )
    message: str = Field(default="")


class CleaningExecutionResult(BaseModel):
    """Structured output returned alongside the cleaned DataFrame (no file I/O)."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    rows_before: int = Field(ge=0)
    rows_after: int = Field(ge=0)
    columns_before: list[str] = Field(default_factory=list)
    columns_after: list[str] = Field(default_factory=list)
    steps: list[CleaningStepLog] = Field(default_factory=list)
    lineage_events: list[LineageEvent] = Field(default_factory=list)

