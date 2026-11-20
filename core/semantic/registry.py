"""Allowlisted Power BI metric formula templates for ``SemanticModel.metrics_registry``."""

from __future__ import annotations

from core.schemas.semantic import MetricTemplate


def default_metrics_registry() -> list[MetricTemplate]:
    """Return the fixed allowlist of parameterised DAX metric templates."""
    return [
        MetricTemplate(
            template_id="tpl.sum",
            template_type="sum",
            name="Total Sum",
            description="Additive sum of a numeric measure column under current filters.",
            formula_template="SUM( '{table}'[{column}] )",
            format_string="#,##0.00",
            required_roles=["measure"],
        ),
        MetricTemplate(
            template_id="tpl.average",
            template_type="average",
            name="Average",
            description="Arithmetic mean of a numeric measure column under current filters.",
            formula_template="AVERAGE( '{table}'[{column}] )",
            format_string="#,##0.00",
            required_roles=["measure"],
        ),
        MetricTemplate(
            template_id="tpl.distinct_count",
            template_type="distinct_count",
            name="Distinct Count",
            description="Count of unique non-blank values in an ID or dimension column.",
            formula_template="DISTINCTCOUNT( '{table}'[{column}] )",
            format_string="#,##0",
            required_roles=["id"],
        ),
        MetricTemplate(
            template_id="tpl.divide_ratio",
            template_type="divide_ratio",
            name="Safe DIVIDE Ratio",
            description=(
                "Safe ratio using DIVIDE(numerator, denominator, 0), optionally subtracting "
                "a cost column from numerator for margin calculations."
            ),
            formula_template=(
                "DIVIDE( SUM( '{table}'[{numerator}] ), SUM( '{table}'[{denominator}] ), 0 )"
            ),
            format_string="0.0%",
            required_roles=["measure:numerator", "measure:denominator"],
        ),
        MetricTemplate(
            template_id="tpl.yoy_growth",
            template_type="yoy_growth",
            name="Year-over-Year (YoY) Growth %",
            description=(
                "Percentage change versus the same period last year using SAMEPERIODLASTYEAR "
                "on the marked date table."
            ),
            formula_template=(
                "VAR CurrentVal = SUM( '{table}'[{column}] ) "
                "VAR PriorVal = CALCULATE( SUM( '{table}'[{column}] ), "
                "SAMEPERIODLASTYEAR( '{date_table}'[{date_column}] ) ) "
                "RETURN DIVIDE( CurrentVal - PriorVal, PriorVal, BLANK() )"
            ),
            format_string="+0.0%;-0.0%;0.0%",
            required_roles=["measure", "date"],
        ),
        MetricTemplate(
            template_id="tpl.mom_growth",
            template_type="mom_growth",
            name="Month-over-Month (MoM) Growth %",
            description=(
                "Percentage change versus the immediately preceding month using DATEADD "
                "on the marked date table."
            ),
            formula_template=(
                "VAR CurrentVal = SUM( '{table}'[{column}] ) "
                "VAR PriorVal = CALCULATE( SUM( '{table}'[{column}] ), "
                "DATEADD( '{date_table}'[{date_column}], -1, MONTH ) ) "
                "RETURN DIVIDE( CurrentVal - PriorVal, PriorVal, BLANK() )"
            ),
            format_string="+0.0%;-0.0%;0.0%",
            required_roles=["measure", "date"],
        ),
        MetricTemplate(
            template_id="tpl.rolling_avg",
            template_type="rolling_avg",
            name="Rolling Window Moving Average",
            description=(
                "Trailing N-month moving average over the marked date table using DATESINPERIOD."
            ),
            formula_template=(
                "VAR LastDate = MAX( '{date_table}'[{date_column}] ) "
                "VAR WindowDates = DATESINPERIOD( '{date_table}'[{date_column}], LastDate, "
                "-{window_months}, MONTH ) "
                "RETURN AVERAGEX( VALUES( '{date_table}'[year_month] ), "
                "CALCULATE( SUM( '{table}'[{column}] ), WindowDates ) )"
            ),
            format_string="#,##0.00",
            required_roles=["measure", "date"],
        ),
        MetricTemplate(
            template_id="tpl.percent_of_total",
            template_type="percent_of_total",
            name="Percent of Total",
            description="Category share of grand total across ALLSELECTED dimension values.",
            formula_template=(
                "VAR CurrentVal = SUM( '{table}'[{column}] ) "
                "VAR AllSelectedVal = CALCULATE( SUM( '{table}'[{column}] ), "
                "ALLSELECTED( '{dim_table}'[{dim_column}] ) ) "
                "RETURN DIVIDE( CurrentVal, AllSelectedVal, 0 )"
            ),
            format_string="0.0%",
            required_roles=["measure", "dimension"],
        ),
        MetricTemplate(
            template_id="tpl.rank",
            template_type="rank",
            name="Category Dense Rank (RANKX)",
            description="1-based descending dense rank per category guarded by ISINSCOPE.",
            formula_template=(
                "IF( ISINSCOPE( '{dim_table}'[{dim_column}] ), "
                "RANKX( ALLSELECTED( '{dim_table}'[{dim_column}] ), "
                "CALCULATE( SUM( '{table}'[{column}] ) ), , DESC, DENSE ) )"
            ),
            format_string="#,##0",
            required_roles=["measure", "dimension"],
        ),
        MetricTemplate(
            template_id="tpl.top_n",
            template_type="top_n",
            name="Top N Contribution",
            description="Total measure value contributed by the Top N categories via TOPN.",
            formula_template=(
                "CALCULATE( SUM( '{table}'[{column}] ), "
                "KEEPFILTERS( TOPN( {top_n}, ALLSELECTED( '{dim_table}'[{dim_column}] ), "
                "CALCULATE( SUM( '{table}'[{column}] ) ), DESC ) ) )"
            ),
            format_string="#,##0.00",
            required_roles=["measure", "dimension"],
        ),
    ]

