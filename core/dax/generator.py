"""Template-driven Power BI DAX measure generator (no LLM free-form DAX)."""

from __future__ import annotations

from core.schemas.dax import (
    DAXMeasureSpec,
    MetricParameters,
    MetricSelectionPlan,
    SelectedMetricSpec,
)
from core.schemas.semantic import SemanticModel


class DAXGenerationError(Exception):
    """Raised when a metric selection cannot be rendered from an allowlisted DAX template."""


def _qcol(table: str, column: str) -> str:
    """Format a DAX qualified table-column reference as ``'table'[column]``."""
    escaped_table = table.replace("'", "''")
    escaped_col = column.replace("]", "]]")
    return f"'{escaped_table}'[{escaped_col}]"


def _resolve_fact_date_column(
    model: SemanticModel,
    table_name: str,
    params: MetricParameters,
) -> str | None:
    """Resolve the fact table date column for SQL reconciliation metadata."""
    if params.fact_date_column:
        return params.fact_date_column
    if model.date_table_name:
        for rel in model.relationships:
            if rel.from_table == table_name and rel.to_table == model.date_table_name:
                return rel.from_column
    tbl = model.get_table(table_name)
    if tbl is not None:
        for col in tbl.columns:
            if col.role == "date" or col.data_type == "dateTime":
                return col.name
    return None


def generate_dax_measure(
    spec: SelectedMetricSpec,
    model: SemanticModel,
) -> DAXMeasureSpec:
    """Render a single ``SelectedMetricSpec`` using its allowlisted registry template."""
    template = model.get_template(spec.template_id)
    if template is None:
        raise DAXGenerationError(
            f"Template '{spec.template_id}' is not registered in SemanticModel.metrics_registry."
        )

    table = spec.table_name
    params = spec.params
    t_type = template.template_type
    resolved_params = params

    if t_type == "sum":
        if not params.column:
            raise DAXGenerationError(f"Metric '{spec.measure_name}' requires 'column'.")
        expr = f"SUM( {_qcol(table, params.column)} )"
    elif t_type == "average":
        if not params.column:
            raise DAXGenerationError(f"Metric '{spec.measure_name}' requires 'column'.")
        expr = f"AVERAGE( {_qcol(table, params.column)} )"
    elif t_type == "distinct_count":
        if not params.column:
            raise DAXGenerationError(f"Metric '{spec.measure_name}' requires 'column'.")
        expr = f"DISTINCTCOUNT( {_qcol(table, params.column)} )"
    elif t_type == "divide_ratio":
        if not params.numerator_column or not params.denominator_column:
            raise DAXGenerationError(
                f"Metric '{spec.measure_name}' requires numerator_column and denominator_column."
            )
        num_expr = f"SUM( {_qcol(table, params.numerator_column)} )"
        if params.subtract_column:
            num_expr = f"{num_expr} - SUM( {_qcol(table, params.subtract_column)} )"
        den_expr = f"SUM( {_qcol(table, params.denominator_column)} )"
        expr = f"DIVIDE( {num_expr}, {den_expr}, 0 )"
    elif t_type == "yoy_growth":
        if not params.column:
            raise DAXGenerationError(f"Metric '{spec.measure_name}' requires 'column'.")
        date_tbl = params.date_table or model.date_table_name or "dim_date"
        date_col = params.date_column or "date"
        fact_dt = _resolve_fact_date_column(model, table, params)
        resolved_params = params.model_copy(
            update={"date_table": date_tbl, "date_column": date_col, "fact_date_column": fact_dt}
        )
        base = f"SUM( {_qcol(table, params.column)} )"
        expr = (
            f"VAR CurrentValue = {base}\n"
            f"VAR PriorYearValue = CALCULATE( {base}, "
            f"SAMEPERIODLASTYEAR( {_qcol(date_tbl, date_col)} ) )\n"
            "RETURN\n"
            "DIVIDE( CurrentValue - PriorYearValue, PriorYearValue, BLANK() )"
        )
    elif t_type == "mom_growth":
        if not params.column:
            raise DAXGenerationError(f"Metric '{spec.measure_name}' requires 'column'.")
        date_tbl = params.date_table or model.date_table_name or "dim_date"
        date_col = params.date_column or "date"
        fact_dt = _resolve_fact_date_column(model, table, params)
        resolved_params = params.model_copy(
            update={"date_table": date_tbl, "date_column": date_col, "fact_date_column": fact_dt}
        )
        base = f"SUM( {_qcol(table, params.column)} )"
        expr = (
            f"VAR CurrentValue = {base}\n"
            f"VAR PriorMonthValue = CALCULATE( {base}, "
            f"DATEADD( {_qcol(date_tbl, date_col)}, -1, MONTH ) )\n"
            "RETURN\n"
            "DIVIDE( CurrentValue - PriorMonthValue, PriorMonthValue, BLANK() )"
        )
    elif t_type == "rolling_avg":
        if not params.column:
            raise DAXGenerationError(f"Metric '{spec.measure_name}' requires 'column'.")
        date_tbl = params.date_table or model.date_table_name or "dim_date"
        date_col = params.date_column or "date"
        fact_dt = _resolve_fact_date_column(model, table, params)
        resolved_params = params.model_copy(
            update={"date_table": date_tbl, "date_column": date_col, "fact_date_column": fact_dt}
        )
        base = f"SUM( {_qcol(table, params.column)} )"
        expr = (
            f"VAR LastVisibleDate = MAX( {_qcol(date_tbl, date_col)} )\n"
            f"VAR PeriodWindow = DATESINPERIOD( {_qcol(date_tbl, date_col)}, "
            f"LastVisibleDate, -{params.window_months}, MONTH )\n"
            "RETURN\n"
            f"AVERAGEX( VALUES( {_qcol(date_tbl, 'year_month')} ), "
            f"CALCULATE( {base}, PeriodWindow ) )"
        )
    elif t_type == "percent_of_total":
        if not params.column or not params.dimension_column:
            raise DAXGenerationError(
                f"Metric '{spec.measure_name}' requires 'column' and 'dimension_column'."
            )
        dim_tbl = params.dimension_table or table
        resolved_params = params.model_copy(update={"dimension_table": dim_tbl})
        base = f"SUM( {_qcol(table, params.column)} )"
        expr = (
            f"VAR CurrentValue = {base}\n"
            f"VAR AllSelectedTotal = CALCULATE( {base}, "
            f"ALLSELECTED( {_qcol(dim_tbl, params.dimension_column)} ) )\n"
            "RETURN\n"
            "DIVIDE( CurrentValue, AllSelectedTotal, 0 )"
        )
    elif t_type == "rank":
        if not params.column or not params.dimension_column:
            raise DAXGenerationError(
                f"Metric '{spec.measure_name}' requires 'column' and 'dimension_column'."
            )
        dim_tbl = params.dimension_table or table
        resolved_params = params.model_copy(update={"dimension_table": dim_tbl})
        base = f"CALCULATE( SUM( {_qcol(table, params.column)} ) )"
        dim_ref = _qcol(dim_tbl, params.dimension_column)
        expr = (
            f"IF(\n"
            f"    ISINSCOPE( {dim_ref} ),\n"
            f"    RANKX( ALLSELECTED( {dim_ref} ), {base}, , DESC, DENSE )\n"
            ")"
        )
    elif t_type == "top_n":
        if not params.column or not params.dimension_column:
            raise DAXGenerationError(
                f"Metric '{spec.measure_name}' requires 'column' and 'dimension_column'."
            )
        dim_tbl = params.dimension_table or table
        resolved_params = params.model_copy(update={"dimension_table": dim_tbl})
        base = f"SUM( {_qcol(table, params.column)} )"
        dim_ref = _qcol(dim_tbl, params.dimension_column)
        expr = (
            f"CALCULATE(\n"
            f"    {base},\n"
            f"    KEEPFILTERS( TOPN( {params.top_n}, ALLSELECTED( {dim_ref} ), "
            f"CALCULATE( {base} ), DESC ) )\n"
            ")"
        )
    else:
        raise DAXGenerationError(f"Unsupported metric template type '{t_type}'.")

    return DAXMeasureSpec(
        name=spec.measure_name,
        table_name=table,
        template_id=spec.template_id,
        template_type=t_type,
        expression=expr,
        format_string=spec.format_string or template.format_string,
        description=spec.description or template.description,
        params=resolved_params,
        rag_source_ids=list(spec.rag_source_ids),
    )


def generate_dax_measures(
    plan: MetricSelectionPlan,
    model: SemanticModel,
) -> list[DAXMeasureSpec]:
    """Render every selected metric in ``plan`` into a validated ``DAXMeasureSpec``."""
    return [generate_dax_measure(spec, model) for spec in plan.selected_metrics]
