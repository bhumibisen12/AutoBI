"""Unit tests for Step 6: SemanticModel, KPI selector, DAX templates, validator, SQL verifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.config import Settings
from core.dax import (
    generate_dax_measures,
    reconcile_measures,
    validate_dax_measure,
    validate_semantic_model_no_implicit_measures,
)
from core.etl import run_etl_pipeline
from core.llm import MockLLM, parse_user_intent, understand_dataset
from core.profiling import profile_dataset
from core.rag import LocalHashEmbedder, build_knowledge_index
from core.schemas import (
    DAXMeasureSpec,
    MetricParameters,
    MetricSelectionPlan,
    SelectedMetricSpec,
)
from core.semantic import (
    InvalidMetricSelectionError,
    build_semantic_model,
    select_kpis,
    suggest_default_metric_selections,
)
from core.storage import read_curated_table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def _make_settings(tmp_path: Path) -> Settings:
    """Create isolated Settings rooted under pytest's temporary directory."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
    )


def test_semantic_model_from_curated_layer_and_kpi_selector(tmp_path: Path) -> None:
    """SemanticModel builds from curated tables with marked dim_date and 1:* single-dir links."""
    settings = _make_settings(tmp_path)
    etl_res = run_etl_pipeline(
        [SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv", SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx"],
        settings=settings,
    )

    curated_sales = read_curated_table(etl_res.curated_dir, "synthetic_sales")
    profile = profile_dataset({"synthetic_sales": curated_sales}, batch_id=etl_res.batch_id)
    mock_llm = MockLLM()
    understanding, _ = understand_dataset(mock_llm, profile, {"synthetic_sales": curated_sales})
    intent, _ = parse_user_intent(
        mock_llm,
        "Executive monthly sales revenue and gross margin dashboard",
        profile,
        {"synthetic_sales": curated_sales},
    )

    model = build_semantic_model(etl_res, dataset_understanding=understanding)

    # Marked date table + single-direction one-to-many relationships + zero implicit measures
    assert model.date_table_name == "dim_date"
    date_tbl = model.get_table("dim_date")
    assert date_tbl is not None
    assert date_tbl.is_date_table is True
    assert date_tbl.date_column == "date"
    assert validate_semantic_model_no_implicit_measures(model) == []
    assert len(model.metrics_registry) == 10
    assert all(rel.cross_filter_direction == "single" for rel in model.relationships)
    assert any(
        rel.from_table == "synthetic_hr__employees"
        and rel.to_table == "synthetic_hr__departments"
        and rel.cardinality == "many_to_one"
        for rel in model.relationships
    )

    # Retrieve RAG chunks and drive KPI selector via MockLLM
    kb_index = build_knowledge_index(KNOWLEDGE_DIR, embedder=LocalHashEmbedder())
    rag_chunks = kb_index.retrieve("sales gross margin revenue month over month", k=4)
    default_plan = suggest_default_metric_selections(model, understanding, intent, rag_chunks)

    kpi_llm = MockLLM(responses_by_model={MetricSelectionPlan: default_plan})
    selected_plan, _ = select_kpis(kpi_llm, model, understanding, intent, rag_chunks)
    assert len(selected_plan.selected_metrics) >= 4
    assert "NEVER write free-form DAX" in kpi_llm.recorded_prompts[0].system_prompt

    # Reject any LLM attempt to pick an unregistered template or nonexistent column
    bad_template_plan = MetricSelectionPlan(
        selected_metrics=[
            SelectedMetricSpec(
                measure_name="Rogue Metric",
                template_id="tpl.free_form_dax",
                table_name="synthetic_sales",
                params=MetricParameters(column="revenue"),
            )
        ]
    )
    rogue_llm = MockLLM(responses_by_model={MetricSelectionPlan: bad_template_plan})
    with pytest.raises(InvalidMetricSelectionError, match="unknown template"):
        select_kpis(rogue_llm, model, understanding, intent, rag_chunks)


def test_all_10_dax_templates_reconciled_against_hand_computed_values(tmp_path: Path) -> None:
    """Every DAX template matches hand-computed values via DuckDB SQL on curated sample datasets."""
    settings = _make_settings(tmp_path)
    etl_res = run_etl_pipeline(
        [SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv", SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx"],
        settings=settings,
    )
    model = build_semantic_model(etl_res)

    sales_df = read_curated_table(etl_res.curated_dir, "synthetic_sales")
    emp_df = read_curated_table(etl_res.curated_dir, "synthetic_hr__employees")
    dept_df = read_curated_table(etl_res.curated_dir, "synthetic_hr__departments")

    # --- Hand-computed ground-truth values ---
    expected_sum = float(sales_df["revenue"].sum())
    expected_avg = float(sales_df["revenue"].mean())
    expected_distinct = float(sales_df["order_id"].nunique())
    expected_margin = float(
        (sales_df["revenue"].sum() - sales_df["cost"].sum()) / sales_df["revenue"].sum()
    )

    # Monthly sales for MoM and Rolling 3M Avg (2025-01..2025-12)
    sales_df["ym"] = pd.to_datetime(sales_df["order_date"]).dt.strftime("%Y-%m")
    monthly_rev = sales_df.groupby("ym")["revenue"].sum().sort_index()
    expected_mom = float((monthly_rev.iloc[-1] - monthly_rev.iloc[-2]) / monthly_rev.iloc[-2])
    expected_rolling_3m = float(monthly_rev.iloc[-3:].mean())

    # Yearly HR salary sum across 2020..2024 for YoY growth %
    emp_df["yr"] = pd.to_datetime(emp_df["hire_date"]).dt.year
    yearly_salary = emp_df.groupby("yr")["base_salary_usd"].sum().sort_index()
    expected_yoy = float((yearly_salary.iloc[-1] - yearly_salary.iloc[-2]) / yearly_salary.iloc[-2])

    # Region % of Total, Rank, and Top 2 Contribution on sales
    region_rev = sales_df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    expected_top_region_share = float(region_rev.iloc[0] / region_rev.sum())
    expected_top_2_region_rev = float(region_rev.iloc[:2].sum())

    # Cross-table join test: HR employees joined to departments.department_name for Top 2 salary
    hr_joined = emp_df.merge(dept_df, on="department_id", how="inner")
    dept_sal = (
        hr_joined.groupby("department_name")["base_salary_usd"]
        .sum()
        .sort_values(ascending=False)
    )
    expected_top_2_dept_salary = float(dept_sal.iloc[:2].sum())

    plan = MetricSelectionPlan(
        selected_metrics=[
            SelectedMetricSpec(
                measure_name="Total Revenue",
                template_id="tpl.sum",
                table_name="synthetic_sales",
                params=MetricParameters(column="revenue"),
            ),
            SelectedMetricSpec(
                measure_name="Average Revenue",
                template_id="tpl.average",
                table_name="synthetic_sales",
                params=MetricParameters(column="revenue"),
            ),
            SelectedMetricSpec(
                measure_name="Unique Orders",
                template_id="tpl.distinct_count",
                table_name="synthetic_sales",
                params=MetricParameters(column="order_id"),
            ),
            SelectedMetricSpec(
                measure_name="Gross Margin %",
                template_id="tpl.divide_ratio",
                table_name="synthetic_sales",
                params=MetricParameters(
                    numerator_column="revenue",
                    subtract_column="cost",
                    denominator_column="revenue",
                ),
            ),
            SelectedMetricSpec(
                measure_name="Salary YoY %",
                template_id="tpl.yoy_growth",
                table_name="synthetic_hr__employees",
                params=MetricParameters(
                    column="base_salary_usd",
                    date_table="dim_date",
                    date_column="date",
                    fact_date_column="hire_date",
                ),
            ),
            SelectedMetricSpec(
                measure_name="Revenue MoM %",
                template_id="tpl.mom_growth",
                table_name="synthetic_sales",
                params=MetricParameters(
                    column="revenue",
                    date_table="dim_date",
                    date_column="date",
                    fact_date_column="order_date",
                ),
            ),
            SelectedMetricSpec(
                measure_name="Revenue Rolling 3M Avg",
                template_id="tpl.rolling_avg",
                table_name="synthetic_sales",
                params=MetricParameters(
                    column="revenue",
                    date_table="dim_date",
                    date_column="date",
                    fact_date_column="order_date",
                    window_months=3,
                ),
            ),
            SelectedMetricSpec(
                measure_name="Revenue % of Total by Region",
                template_id="tpl.percent_of_total",
                table_name="synthetic_sales",
                params=MetricParameters(
                    column="revenue",
                    dimension_table="synthetic_sales",
                    dimension_column="region",
                ),
            ),
            SelectedMetricSpec(
                measure_name="Region Revenue Rank",
                template_id="tpl.rank",
                table_name="synthetic_sales",
                params=MetricParameters(
                    column="revenue",
                    dimension_table="synthetic_sales",
                    dimension_column="region",
                ),
            ),
            SelectedMetricSpec(
                measure_name="Top 2 Regions Revenue",
                template_id="tpl.top_n",
                table_name="synthetic_sales",
                params=MetricParameters(
                    column="revenue",
                    dimension_table="synthetic_sales",
                    dimension_column="region",
                    top_n=2,
                ),
            ),
            SelectedMetricSpec(
                measure_name="Top 2 Departments Salary",
                template_id="tpl.top_n",
                table_name="synthetic_hr__employees",
                params=MetricParameters(
                    column="base_salary_usd",
                    dimension_table="synthetic_hr__departments",
                    dimension_column="department_name",
                    top_n=2,
                ),
            ),
        ]
    )

    measures = generate_dax_measures(plan, model)
    report = reconcile_measures(measures, model)

    assert report.total_measures == 11
    assert report.verified_count == 11
    assert report.unverified_count == 0
    assert report.failed_count == 0

    by_name = {res.measure_name: res for res in report.results}
    for res in report.results:
        assert res.status == "VERIFIED"
        assert res.dax and res.sql

    assert by_name["Total Revenue"].result == pytest.approx(expected_sum, rel=1e-5)
    assert by_name["Average Revenue"].result == pytest.approx(expected_avg, rel=1e-5)
    assert by_name["Unique Orders"].result == pytest.approx(expected_distinct, rel=1e-5)
    assert by_name["Gross Margin %"].result == pytest.approx(expected_margin, rel=1e-5)
    assert by_name["Salary YoY %"].result == pytest.approx(expected_yoy, rel=1e-5)
    assert by_name["Revenue MoM %"].result == pytest.approx(expected_mom, rel=1e-5)
    assert by_name["Revenue Rolling 3M Avg"].result == pytest.approx(expected_rolling_3m, rel=1e-5)
    assert by_name["Revenue % of Total by Region"].result == pytest.approx(
        expected_top_region_share, rel=1e-5
    )
    region_share_sum = sum(
        row["value"] for row in by_name["Revenue % of Total by Region"].breakdown
    )
    assert region_share_sum == pytest.approx(1.0, rel=1e-5)
    assert by_name["Region Revenue Rank"].breakdown[0]["value"] == 1.0
    assert by_name["Top 2 Regions Revenue"].result == pytest.approx(
        expected_top_2_region_rev, rel=1e-5
    )
    assert by_name["Top 2 Departments Salary"].result == pytest.approx(
        expected_top_2_dept_salary, rel=1e-5
    )


def test_dax_validator_rules_and_unverified_flagging(tmp_path: Path) -> None:
    """Validator catches '/', missing columns, implicit measures, syntax; unknown -> UNVERIFIED."""
    settings = _make_settings(tmp_path)
    etl_res = run_etl_pipeline([SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv"], settings=settings)
    model = build_semantic_model(etl_res)

    # 1. Raw '/' division rejected
    slash_measure = DAXMeasureSpec(
        name="Bad Ratio",
        table_name="synthetic_sales",
        template_id="tpl.divide_ratio",
        template_type="divide_ratio",
        expression="SUM( 'synthetic_sales'[revenue] ) / SUM( 'synthetic_sales'[cost] )",
    )
    res_slash = validate_dax_measure(slash_measure, model)
    assert res_slash.is_valid is False
    assert any("DIVIDE" in err for err in res_slash.errors)

    # 2. Missing column rejected
    missing_col = DAXMeasureSpec(
        name="Ghost Metric",
        table_name="synthetic_sales",
        template_id="tpl.sum",
        template_type="sum",
        expression="SUM( 'synthetic_sales'[does_not_exist] )",
    )
    res_missing = validate_dax_measure(missing_col, model)
    assert res_missing.is_valid is False
    assert any("does_not_exist" in err for err in res_missing.errors)

    # 3. Implicit bare column measure rejected
    implicit_measure = DAXMeasureSpec(
        name="Implicit Rev",
        table_name="synthetic_sales",
        template_id="tpl.sum",
        template_type="sum",
        expression="'synthetic_sales'[revenue]",
    )
    res_implicit = validate_dax_measure(implicit_measure, model)
    assert res_implicit.is_valid is False
    assert any("Implicit measure" in err for err in res_implicit.errors)

    # 4. Syntax error (unbalanced parentheses + VAR without RETURN) rejected
    bad_syntax = DAXMeasureSpec(
        name="Broken Syntax",
        table_name="synthetic_sales",
        template_id="tpl.sum",
        template_type="sum",
        expression="VAR x = SUM( 'synthetic_sales'[revenue]",
    )
    res_syntax = validate_dax_measure(bad_syntax, model)
    assert res_syntax.is_valid is False
    assert any("Unclosed parenthesis" in err for err in res_syntax.errors)
    assert any("RETURN" in err for err in res_syntax.errors)

    # 5. Valid DAX whose template has no SQL verifier is explicitly flagged UNVERIFIED
    unverifiable = DAXMeasureSpec(
        name="Custom Median",
        table_name="synthetic_sales",
        template_id="tpl.custom_unknown",
        template_type="custom_unverified",
        expression="SUM( 'synthetic_sales'[revenue] )",
    )
    report = reconcile_measures([slash_measure, unverifiable], model)
    by_name = {r.measure_name: r for r in report.results}
    assert by_name["Bad Ratio"].status == "FAILED"
    assert by_name["Custom Median"].status == "UNVERIFIED"
    assert by_name["Custom Median"].sql is None
    assert report.failed_count == 1
    assert report.unverified_count == 1
