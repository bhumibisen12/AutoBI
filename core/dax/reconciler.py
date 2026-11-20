"""Reconcile generated DAX measures against equivalent DuckDB SQL queries on curated tables."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.dax.validator import validate_dax_measure
from core.schemas.dax import (
    DAXMeasureSpec,
    MeasureVerificationResult,
    VerificationReport,
)
from core.schemas.semantic import SemanticModel
from core.storage import query_duckdb

_SAFE_SQL_IDENT_RE = re.compile(r"^[A-Za-z0-9_ \-\./()%$#@+:]+$")


def _qident(identifier: str) -> str:
    """Validate and double-quote a DuckDB SQL identifier safely (supports spaces in CSV headers)."""
    if (
        not identifier
        or not _SAFE_SQL_IDENT_RE.match(identifier)
        or "--" in identifier
        or "/*" in identifier
        or "*/" in identifier
    ):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return f'"{identifier}"'


def _find_join_clause(
    model: SemanticModel,
    fact_table: str,
    dim_table: str,
) -> tuple[str, str] | None:
    """Return ``(join_sql, dim_alias)`` linking ``fact_table`` (alias ``f``) to ``dim_table``."""
    if dim_table == fact_table:
        return "", "f"
    for rel in model.relationships:
        if rel.from_table == fact_table and rel.to_table == dim_table:
            join_sql = (
                f"JOIN {_qident(dim_table)} d "
                f"ON f.{_qident(rel.from_column)} = d.{_qident(rel.to_column)}"
            )
            return join_sql, "d"
        if rel.to_table == fact_table and rel.from_table == dim_table:
            join_sql = (
                f"JOIN {_qident(dim_table)} d "
                f"ON f.{_qident(rel.to_column)} = d.{_qident(rel.from_column)}"
            )
            return join_sql, "d"
    return None


def _build_reconciliation_sql(
    measure: DAXMeasureSpec,
    model: SemanticModel,
) -> tuple[str | None, str]:
    """Generate an equivalent DuckDB SQL query for ``measure`` against curated tables.

    Returns ``(sql_or_none, mode_or_reason)``. Returns ``(None, reason)`` when a measure
    cannot be unambiguously reconciled so the caller flags it ``UNVERIFIED``.
    """
    t_type = measure.template_type
    p = measure.params
    tbl = _qident(measure.table_name)

    if t_type == "sum" and p.column:
        return f"SELECT CAST(SUM({_qident(p.column)}) AS DOUBLE) AS value FROM {tbl}", "scalar"

    if t_type == "average" and p.column:
        return f"SELECT CAST(AVG({_qident(p.column)}) AS DOUBLE) AS value FROM {tbl}", "scalar"

    if t_type == "distinct_count" and p.column:
        return (
            f"SELECT CAST(COUNT(DISTINCT {_qident(p.column)}) AS DOUBLE) AS value FROM {tbl}",
            "scalar",
        )

    if t_type == "divide_ratio" and p.numerator_column and p.denominator_column:
        num = f"CAST(SUM({_qident(p.numerator_column)}) AS DOUBLE)"
        if p.subtract_column:
            num = (
                f"CAST(SUM({_qident(p.numerator_column)}) - "
                f"SUM({_qident(p.subtract_column)}) AS DOUBLE)"
            )
        den = f"CAST(SUM({_qident(p.denominator_column)}) AS DOUBLE)"
        sql = (
            f"SELECT CASE WHEN {den} = 0 OR {den} IS NULL THEN 0.0 "
            f"ELSE ({num}) / ({den}) END AS value FROM {tbl}"
        )
        return sql, "scalar"

    if t_type in {"yoy_growth", "mom_growth", "rolling_avg"} and p.column:
        date_tbl = p.date_table or model.date_table_name
        date_col = p.date_column or "date"
        fact_dt = p.fact_date_column
        if not fact_dt and date_tbl:
            for rel in model.relationships:
                if rel.from_table == measure.table_name and rel.to_table == date_tbl:
                    fact_dt = rel.from_column
                    break
        if not date_tbl or not fact_dt:
            return None, "Missing dim_date relationship needed for time-intelligence SQL."

        d_tbl = _qident(date_tbl)
        d_col = _qident(date_col)
        f_dt = _qident(fact_dt)
        m_col = _qident(p.column)

        if t_type == "yoy_growth":
            sql = (
                "WITH yearly AS ("
                f" SELECT d.year, CAST(SUM(f.{m_col}) AS DOUBLE) AS val"
                f" FROM {tbl} f"
                f" JOIN {d_tbl} d ON CAST(f.{f_dt} AS DATE) = CAST(d.{d_col} AS DATE)"
                " GROUP BY d.year"
                "), yoy AS ("
                " SELECT year, val, LAG(val) OVER (ORDER BY year) AS prev_val FROM yearly"
                ")"
                " SELECT CASE WHEN prev_val IS NULL OR prev_val = 0 THEN NULL"
                " ELSE (val - prev_val) / prev_val END AS value"
                " FROM yoy ORDER BY year DESC LIMIT 1"
            )
            return sql, "scalar"

        if t_type == "mom_growth":
            sql = (
                "WITH monthly AS ("
                f" SELECT d.year_month, CAST(SUM(f.{m_col}) AS DOUBLE) AS val"
                f" FROM {tbl} f"
                f" JOIN {d_tbl} d ON CAST(f.{f_dt} AS DATE) = CAST(d.{d_col} AS DATE)"
                " GROUP BY d.year_month"
                "), mom AS ("
                " SELECT year_month, val, LAG(val) OVER (ORDER BY year_month) AS prev_val"
                " FROM monthly"
                ")"
                " SELECT CASE WHEN prev_val IS NULL OR prev_val = 0 THEN NULL"
                " ELSE (val - prev_val) / prev_val END AS value"
                " FROM mom ORDER BY year_month DESC LIMIT 1"
            )
            return sql, "scalar"

        # rolling_avg over trailing window_months
        preceding = max(0, p.window_months - 1)
        sql = (
            "WITH monthly AS ("
            f" SELECT d.year_month, CAST(SUM(f.{m_col}) AS DOUBLE) AS val"
            f" FROM {tbl} f"
            f" JOIN {d_tbl} d ON CAST(f.{f_dt} AS DATE) = CAST(d.{d_col} AS DATE)"
            " GROUP BY d.year_month"
            "), rolling AS ("
            f" SELECT year_month, AVG(val) OVER (ORDER BY year_month "
            f"ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW) AS rolling_avg"
            " FROM monthly"
            ")"
            " SELECT CAST(rolling_avg AS DOUBLE) AS value"
            " FROM rolling ORDER BY year_month DESC LIMIT 1"
        )
        return sql, "scalar"

    if t_type in {"percent_of_total", "rank", "top_n"} and p.column and p.dimension_column:
        dim_tbl = p.dimension_table or measure.table_name
        join_info = _find_join_clause(model, measure.table_name, dim_tbl)
        if join_info is None:
            return None, f"No curated relationship linking '{measure.table_name}' to '{dim_tbl}'."
        join_sql, dim_alias = join_info
        dim_ref = f"{dim_alias}.{_qident(p.dimension_column)}"
        m_ref = f"f.{_qident(p.column)}"

        if t_type == "percent_of_total":
            sql = (
                "WITH grouped AS ("
                f" SELECT {dim_ref} AS category, CAST(SUM({m_ref}) AS DOUBLE) AS val"
                f" FROM {tbl} f {join_sql} GROUP BY {dim_ref}"
                ")"
                " SELECT category, CASE WHEN SUM(val) OVER () = 0 THEN 0.0"
                " ELSE val / SUM(val) OVER () END AS value"
                " FROM grouped ORDER BY value DESC, category ASC"
            )
            return sql, "breakdown"

        if t_type == "rank":
            sql = (
                "WITH grouped AS ("
                f" SELECT {dim_ref} AS category, CAST(SUM({m_ref}) AS DOUBLE) AS val"
                f" FROM {tbl} f {join_sql} GROUP BY {dim_ref}"
                ")"
                " SELECT category, val,"
                " CAST(DENSE_RANK() OVER (ORDER BY val DESC) AS DOUBLE) AS value"
                " FROM grouped ORDER BY value ASC, category ASC"
            )
            return sql, "breakdown"

        # top_n
        sql = (
            "SELECT CAST(SUM(cat_total) AS DOUBLE) AS value FROM ("
            f" SELECT {dim_ref} AS category, CAST(SUM({m_ref}) AS DOUBLE) AS cat_total"
            f" FROM {tbl} f {join_sql} GROUP BY {dim_ref}"
            f" ORDER BY cat_total DESC, category ASC LIMIT {p.top_n}"
            ") ranked"
        )
        return sql, "scalar"

    return None, f"No verified SQL reconciliation template available for '{t_type}'."


def reconcile_measures(
    measures: Sequence[DAXMeasureSpec],
    model: SemanticModel,
    *,
    curated_duckdb_path: Path | None = None,
) -> VerificationReport:
    """Validate and reconcile each DAX measure against DuckDB SQL on the curated layer."""
    db_path = curated_duckdb_path or model.curated_duckdb_path
    known_names = [m.name for m in measures]
    results: list[MeasureVerificationResult] = []

    for measure in measures:
        validation = validate_dax_measure(measure, model, known_measures=known_names)
        if not validation.is_valid:
            results.append(
                MeasureVerificationResult(
                    measure_name=measure.name,
                    template_id=measure.template_id,
                    dax=measure.expression,
                    sql=None,
                    result=None,
                    status="FAILED",
                    validation_errors=validation.errors,
                    notes="Failed DAX validation prior to SQL reconciliation.",
                )
            )
            continue

        if db_path is None or not db_path.is_file():
            results.append(
                MeasureVerificationResult(
                    measure_name=measure.name,
                    template_id=measure.template_id,
                    dax=measure.expression,
                    sql=None,
                    result=None,
                    status="UNVERIFIED",
                    notes="Curated DuckDB path unavailable for SQL reconciliation.",
                )
            )
            continue

        try:
            sql, mode = _build_reconciliation_sql(measure, model)
        except ValueError as exc:
            results.append(
                MeasureVerificationResult(
                    measure_name=measure.name,
                    template_id=measure.template_id,
                    dax=measure.expression,
                    sql=None,
                    result=None,
                    status="UNVERIFIED",
                    notes=str(exc),
                )
            )
            continue

        if sql is None:
            results.append(
                MeasureVerificationResult(
                    measure_name=measure.name,
                    template_id=measure.template_id,
                    dax=measure.expression,
                    sql=None,
                    result=None,
                    status="UNVERIFIED",
                    notes=mode,
                )
            )
            continue

        try:
            df = query_duckdb(db_path, sql)
        except Exception as exc:
            results.append(
                MeasureVerificationResult(
                    measure_name=measure.name,
                    template_id=measure.template_id,
                    dax=measure.expression,
                    sql=sql,
                    result=None,
                    status="UNVERIFIED",
                    notes=f"DuckDB SQL execution failed: {exc}",
                )
            )
            continue

        scalar_val: float | None = None
        breakdown: list[dict[str, Any]] = []
        notes = "Reconciled against curated DuckDB SQL."

        if mode == "scalar":
            if not df.empty and df.iloc[0]["value"] is not None:
                raw_val = df.iloc[0]["value"]
                if raw_val == raw_val:  # not NaN
                    scalar_val = round(float(raw_val), 6)
            if scalar_val is None and measure.template_type in {"yoy_growth", "mom_growth"}:
                notes = "Verified as BLANK/NULL (no prior comparison period in dataset)."
        else:
            for _, row in df.iterrows():
                item: dict[str, Any] = {}
                for col in df.columns:
                    val = row[col]
                    if hasattr(val, "item"):
                        val = val.item()
                    if isinstance(val, float):
                        val = round(val, 6)
                    item[str(col)] = val
                breakdown.append(item)
            if breakdown and isinstance(breakdown[0].get("value"), int | float):
                scalar_val = round(float(breakdown[0]["value"]), 6)

        results.append(
            MeasureVerificationResult(
                measure_name=measure.name,
                template_id=measure.template_id,
                dax=measure.expression,
                sql=sql,
                result=scalar_val,
                breakdown=breakdown,
                status="VERIFIED",
                notes=notes,
            )
        )

    verified = sum(1 for r in results if r.status == "VERIFIED")
    unverified = sum(1 for r in results if r.status == "UNVERIFIED")
    failed = sum(1 for r in results if r.status == "FAILED")

    return VerificationReport(
        batch_id=model.batch_id,
        curated_duckdb_path=db_path,
        total_measures=len(results),
        verified_count=verified,
        unverified_count=unverified,
        failed_count=failed,
        results=results,
    )
