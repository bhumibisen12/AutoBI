"""Pydantic contracts for the curated-layer Power BI ``SemanticModel``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.schemas.dax import MetricTemplateType
from core.schemas.llm import ColumnRole

TableKind = Literal["fact", "dimension", "date"]
DataTypeBI = Literal["int64", "double", "string", "dateTime", "boolean"]
SummarizeBy = Literal["none", "sum", "average", "count", "distinctCount", "min", "max"]
RelationshipCardinality = Literal["many_to_one", "one_to_one"]
CrossFilterDirection = Literal["single"]


class SemanticColumn(BaseModel):
    """Column specification inside a ``SemanticTable`` (defaults to no implicit measures)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    friendly_name: str = Field(min_length=1)
    role: ColumnRole
    data_type: DataTypeBI
    format_string: str | None = None
    summarize_by: SummarizeBy = "none"
    is_key: bool = False
    is_hidden: bool = False
    description: str = Field(default="")


class SemanticTable(BaseModel):
    """Fact, dimension, or marked date table in the semantic model."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    kind: TableKind
    is_date_table: bool = False
    date_column: str | None = None
    primary_key: str | None = None
    row_count: int = Field(default=0, ge=0)
    columns: list[SemanticColumn] = Field(default_factory=list)
    description: str = Field(default="")

    def get_column(self, column_name: str) -> SemanticColumn | None:
        """Return the ``SemanticColumn`` matching ``column_name`` if present."""
        for col in self.columns:
            if col.name == column_name:
                return col
        return None


class SemanticRelationship(BaseModel):
    """Single-direction one-to-many (many-to-one) relationship in the semantic model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    from_table: str = Field(min_length=1, description="Many-side (fact/child) table.")
    from_column: str = Field(min_length=1, description="Foreign key column.")
    to_table: str = Field(min_length=1, description="One-side (dimension/parent) table.")
    to_column: str = Field(min_length=1, description="Primary key column.")
    cardinality: RelationshipCardinality = "many_to_one"
    cross_filter_direction: CrossFilterDirection = "single"
    is_active: bool = True


class MetricTemplate(BaseModel):
    """Allowlisted metric formula template registered in ``SemanticModel.metrics_registry``."""

    model_config = ConfigDict(extra="forbid")

    template_id: str = Field(min_length=1)
    template_type: MetricTemplateType
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    formula_template: str = Field(min_length=1)
    format_string: str = Field(min_length=1)
    required_roles: list[str] = Field(min_length=1)


class SemanticModel(BaseModel):
    """Complete semantic model built exclusively from the curated data layer."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    batch_id: str | None = None
    business_domain: str = Field(default="general_analytics")
    curated_dir: Path | None = None
    curated_duckdb_path: Path | None = None
    date_table_name: str | None = None
    tables: list[SemanticTable] = Field(default_factory=list)
    relationships: list[SemanticRelationship] = Field(default_factory=list)
    metrics_registry: list[MetricTemplate] = Field(default_factory=list)

    def get_table(self, table_name: str) -> SemanticTable | None:
        """Return the ``SemanticTable`` matching ``table_name`` if present."""
        for tbl in self.tables:
            if tbl.table_name == table_name:
                return tbl
        return None

    def get_template(self, template_id: str) -> MetricTemplate | None:
        """Return the ``MetricTemplate`` matching ``template_id`` if registered."""
        for tpl in self.metrics_registry:
            if tpl.template_id == template_id:
                return tpl
        return None

