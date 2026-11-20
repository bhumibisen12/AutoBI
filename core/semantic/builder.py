"""Build a Power BI ``SemanticModel`` strictly from the curated data layer."""

from __future__ import annotations

import re

from core.schemas.etl import CuratedRelationship, CuratedTableInfo, ETLRunResult
from core.schemas.llm import ColumnRole, ColumnUnderstanding, DatasetUnderstanding
from core.schemas.semantic import (
    DataTypeBI,
    SemanticColumn,
    SemanticModel,
    SemanticRelationship,
    SemanticTable,
    TableKind,
)
from core.semantic.registry import default_metrics_registry

_CURRENCY_NAME_RE = re.compile(
    r"(?:^|_)(?:price|unit_price|revenue|cost|salary|base_salary|amount|billed|budget|usd|eur)(?:$|_)"
)
_PERCENT_NAME_RE = re.compile(r"(?:^|_)(?:pct|percent|percentage|rate|ratio|margin)(?:$|_)")


class SemanticModelError(Exception):
    """Raised when a semantic model cannot be safely built from the curated layer."""


def _map_pandas_dtype_to_bi(dtype_str: str) -> DataTypeBI:
    """Map a pandas dtype string into a Power BI TMDL data type."""
    lowered = dtype_str.strip().lower()
    if "datetime" in lowered or "date" in lowered:
        return "dateTime"
    if "bool" in lowered:
        return "boolean"
    if lowered.startswith(("int", "uint")):
        return "int64"
    if lowered.startswith("float") or "decimal" in lowered:
        return "double"
    return "string"


def _infer_column_role(
    col_name: str,
    bi_type: DataTypeBI,
    *,
    is_primary_key: bool,
    understanding_col: ColumnUnderstanding | None,
) -> ColumnRole:
    """Resolve column role from ``DatasetUnderstanding`` or deterministic curated metadata."""
    if understanding_col is not None:
        return understanding_col.role
    lowered = col_name.lower()
    if bi_type == "dateTime":
        return "date"
    if is_primary_key or lowered.endswith(("_id", "_key", "_sk")) or lowered == "id":
        return "id"
    if bi_type in {"int64", "double"}:
        return "measure"
    return "dimension"


def _default_format_string(col_name: str, bi_type: DataTypeBI, role: ColumnRole) -> str | None:
    """Return a clean Power BI format string for a column."""
    lowered = col_name.lower()
    if bi_type == "dateTime":
        return "yyyy-MM-dd"
    if role == "measure":
        if _CURRENCY_NAME_RE.search(lowered):
            return "$#,##0.00"
        if _PERCENT_NAME_RE.search(lowered):
            return "0.0%"
        return "#,##0" if bi_type == "int64" else "#,##0.00"
    return None


def _normalize_relationships(
    raw_relationships: list[CuratedRelationship],
    table_lookup: dict[str, CuratedTableInfo],
) -> list[SemanticRelationship]:
    """Normalize curated relationships into single-direction one-to-many (many-to-one) links."""
    normalized: list[SemanticRelationship] = []
    seen_keys: set[tuple[str, str, str, str]] = set()

    for rel in raw_relationships:
        if rel.from_table not in table_lookup or rel.to_table not in table_lookup:
            continue

        if rel.cardinality == "many_to_many":
            # Single-direction one-to-many modeling skips ambiguous many-to-many joins.
            continue

        if rel.cardinality == "one_to_many":
            # Flip so from_table is always the many-side and to_table is always the one-side.
            from_table, from_col = rel.to_table, rel.to_column
            to_table, to_col = rel.from_table, rel.from_column
            cardinality = "many_to_one"
        elif rel.cardinality == "one_to_one":
            from_table, from_col = rel.from_table, rel.from_column
            to_table, to_col = rel.to_table, rel.to_column
            cardinality = "one_to_one"
        else:
            from_table, from_col = rel.from_table, rel.from_column
            to_table, to_col = rel.to_table, rel.to_column
            cardinality = "many_to_one"

        dedup_key = (from_table, from_col, to_table, to_col)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        rel_name = f"{from_table}_{from_col}_to_{to_table}_{to_col}"
        normalized.append(
            SemanticRelationship(
                name=rel_name,
                from_table=from_table,
                from_column=from_col,
                to_table=to_table,
                to_column=to_col,
                cardinality=cardinality,
                cross_filter_direction="single",
                is_active=True,
            )
        )
    return normalized


def build_semantic_model(
    etl_result: ETLRunResult,
    *,
    dataset_understanding: DatasetUnderstanding | None = None,
    model_name: str = "AutoBI Semantic Model",
) -> SemanticModel:
    """Build a ``SemanticModel`` strictly from curated ETL outputs.

    Enforces the curated-layer boundary: reads only curated table metadata,
    relationships, and the generated ``dim_date`` table (never raw or staging).
    """
    if not etl_result.curated_dir.is_dir():
        raise SemanticModelError(
            f"Curated directory does not exist: {etl_result.curated_dir}"
        )
    if not etl_result.curated_duckdb_path.is_file():
        raise SemanticModelError(
            f"Curated DuckDB catalog does not exist: {etl_result.curated_duckdb_path}"
        )

    table_lookup = {info.table_name: info for info in etl_result.curated_tables}
    relationships = _normalize_relationships(etl_result.relationships, table_lookup)

    many_side_tables = {rel.from_table for rel in relationships}
    one_side_tables = {rel.to_table for rel in relationships}

    understanding_cols: dict[tuple[str, str], ColumnUnderstanding] = {}
    if dataset_understanding is not None:
        for cu in dataset_understanding.columns:
            understanding_cols[(cu.table_name, cu.column_name)] = cu

    semantic_tables: list[SemanticTable] = []
    date_table_name: str | None = None

    for info in etl_result.curated_tables:
        is_date = info.table_name == "dim_date" or (
            info.is_generated and "date" in info.columns and "date_key" in info.columns
        )
        if is_date:
            kind: TableKind = "date"
            date_table_name = info.table_name
        elif info.table_name in one_side_tables and info.table_name not in many_side_tables:
            kind = "dimension"
        else:
            kind = "fact"

        sem_cols: list[SemanticColumn] = []
        for col_name in info.columns:
            raw_dtype = info.dtypes.get(col_name, "object")
            bi_type = _map_pandas_dtype_to_bi(raw_dtype)
            is_key = col_name == info.primary_key
            u_col = understanding_cols.get((info.table_name, col_name))
            role: ColumnRole = (
                "date"
                if (is_date and col_name == "date")
                else (
                    "dimension"
                    if (is_date and col_name != "date_key")
                    else _infer_column_role(
                        col_name,
                        bi_type,
                        is_primary_key=is_key,
                        understanding_col=u_col,
                    )
                )
            )
            friendly_name = (
                u_col.friendly_name
                if u_col is not None
                else col_name.replace("_", " ").strip().title()
            )
            description = u_col.description if u_col is not None else ""
            fmt = _default_format_string(col_name, bi_type, role)

            # summarize_by='none' on every column disables implicit Power BI measures.
            sem_cols.append(
                SemanticColumn(
                    name=col_name,
                    friendly_name=friendly_name,
                    role=role,
                    data_type=bi_type,
                    format_string=fmt,
                    summarize_by="none",
                    is_key=is_key,
                    is_hidden=col_name.endswith("_sk"),
                    description=description,
                )
            )

        semantic_tables.append(
            SemanticTable(
                table_name=info.table_name,
                kind=kind,
                is_date_table=is_date,
                date_column="date" if is_date and "date" in info.columns else None,
                primary_key=info.primary_key,
                row_count=info.row_count,
                columns=sem_cols,
                description=(
                    "Generated contiguous calendar date dimension."
                    if is_date
                    else f"Curated {kind} table '{info.table_name}'."
                ),
            )
        )

    business_domain = (
        dataset_understanding.business_domain
        if dataset_understanding is not None
        else "general_analytics"
    )

    return SemanticModel(
        model_name=model_name,
        batch_id=etl_result.batch_id,
        business_domain=business_domain,
        curated_dir=etl_result.curated_dir,
        curated_duckdb_path=etl_result.curated_duckdb_path,
        date_table_name=date_table_name,
        tables=semantic_tables,
        relationships=relationships,
        metrics_registry=default_metrics_registry(),
    )

