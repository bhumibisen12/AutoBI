"""Unit tests for Step 3: profiling golden files, pure idempotent cleaning, lineage, and EDA."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from core.cleaning import (
    CleaningExecutionError,
    execute_cleaning_plan,
    parse_cleaning_plan_json,
    supported_cleaning_operations,
)
from core.lineage import LineageTracker
from core.profiling import (
    plot_category_breakdown,
    plot_correlation_heatmap,
    plot_distribution,
    plot_time_trend,
    profile_dataset,
)
from core.schemas import CleaningPlan, ProfileReport

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


def _build_golden_fixture_tables() -> dict[str, pd.DataFrame]:
    """Return the exact deterministic multi-table dataset backing the golden profile test."""
    orders = pd.DataFrame(
        {
            "order_id": [
                "ORD-1",
                "ORD-2",
                "ORD-3",
                "ORD-4",
                "ORD-5",
                "ORD-6",
                "ORD-7",
                "ORD-8",
                "ORD-8",
            ],
            "customer_id": ["C1", "C2", "C1", "C3", "C2", "C1", "C2", "C3", "C3"],
            "order_date": [
                "2025-01-05",
                "2025-01-12",
                "2025-02-01",
                "2025-02-15",
                "2025-03-01",
                "2025-03-10",
                "2025-03-20",
                "2025-04-01",
                "2025-04-01",
            ],
            "region": ["North", "South", "North", "East", "South", "North", "East", "West", "West"],
            "is_priority": ["Yes", "No", "Yes", "No", "No", "Yes", "No", "Yes", "Yes"],
            "unit_count": [10, 11, 10, 12, 11, 10, 12, 500, 500],
            "revenue_usd": [
                "$100.00",
                "$120.50",
                "$110.00",
                "$130.00",
                "$115.00",
                "$105.00",
                "$125.00",
                "$140.00",
                "$140.00",
            ],
            "discount_pct": ["5%", "10%", "5%", "0%", "5%", "10%", "0%", "15%", "15%"],
            "mixed_score": ["10", "20", "N/A", "30", "40", "bad", "50", "60", "60"],
            "constant_tag": ["SYNTHETIC"] * 9,
            "review_notes": [
                "Customer reported smooth onboarding and timely shipment of items.",
                "Order arrived ahead of schedule with intact protective packaging.",
                "Followed up with support team regarding invoice copy and terms.",
                "Repeat enterprise customer requested multi-site delivery quote.",
                "Standard fulfillment completed without any warehouse exceptions.",
                "Client appreciated proactive status updates throughout transit.",
                "Account manager confirmed renewal discussion for next quarter.",
                "Expedited freight approved due to tight regional launch window.",
                "Expedited freight approved due to tight regional launch window.",
            ],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "segment": ["Enterprise", "SMB", "Enterprise"],
        }
    )
    return {"orders": orders, "customers": customers}


def test_profiler_matches_golden_file() -> None:
    """Profiler output matches the locked golden JSON report exactly across all metrics."""
    tables = _build_golden_fixture_tables()
    report = profile_dataset(tables, batch_id="golden_batch")

    golden_path = FIXTURES_DIR / "golden_profile_report.json"
    expected_report = ProfileReport.model_validate_json(golden_path.read_text(encoding="utf-8"))

    assert report.model_dump() == expected_report.model_dump()


def test_profiler_discovers_synthetic_hr_relationship() -> None:
    """Profiling the SYNTHETIC HR workbook discovers employees.department_id -> departments."""
    emp_df = pd.read_excel(
        SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx",
        sheet_name="employees",
        header=2,
    )
    dept_df = pd.read_excel(
        SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx",
        sheet_name="departments",
    )

    report = profile_dataset({"employees": emp_df, "departments": dept_df})
    rel_pairs = {
        (rel.left_table, rel.left_column, rel.right_table, rel.right_column)
        for rel in report.candidate_relationships
    }
    assert ("departments", "department_id", "employees", "department_id") in rel_pairs


def test_cleaning_plan_allowlist_rejects_unknown_operations() -> None:
    """Declarative cleaning plan schema enforces the 10-operation allowlist strictly."""
    expected_ops = {
        "rename_column",
        "cast_type",
        "parse_date",
        "trim_text",
        "standardize_case",
        "fill_null",
        "drop_duplicates",
        "drop_column",
        "map_values",
        "split_column",
    }
    assert set(supported_cleaning_operations()) == expected_ops

    # Unknown / code-execution operations must be rejected by Pydantic
    bad_plan_json = json.dumps(
        {
            "table_name": "sales",
            "operations": [{"op": "exec_python", "code": "print('pwned')"}],
        }
    )
    with pytest.raises(ValidationError):
        parse_cleaning_plan_json(bad_plan_json)

    # fill_null constant strategy requires constant_value
    with pytest.raises(ValidationError):
        CleaningPlan(
            table_name="sales",
            operations=[{"op": "fill_null", "column": "region", "strategy": "constant"}],
        )


def test_cleaning_plan_all_operations_idempotent_and_complete_lineage() -> None:
    """Applying a plan using all 10 allowlisted ops twice is idempotent with complete lineage."""
    raw_df = pd.DataFrame(
        {
            "Order ID": ["ORD-1", "ORD-2", "ORD-2", "ORD-3"],
            "Customer Name": ["  alice smith  ", "BOB JONES", "BOB JONES", "carol lee"],
            "Region Code": [" ny ", "ca", "ca", None],
            "Qty": ["10", "20", "20", None],
            "Amount": ["100.5", "200.0", "200.0", "bad_num"],
            "Order Date": ["2025-01-10", "2025-02-15", "2025-02-15", "2025-03-20"],
            "Active": ["Yes", "No", "No", "Yes"],
            "Unused Col": ["x", "y", "y", "z"],
        }
    )
    original_snapshot = raw_df.copy(deep=True)

    plan_json = json.dumps(
        {
            "table_name": "orders",
            "description": "Full allowlisted cleaning pipeline",
            "operations": [
                {"op": "rename_column", "column": "Order ID", "new_name": "order_id"},
                {"op": "rename_column", "column": "Customer Name", "new_name": "customer_name"},
                {"op": "rename_column", "column": "Region Code", "new_name": "region"},
                {"op": "trim_text", "column": "customer_name", "side": "both"},
                {"op": "standardize_case", "column": "customer_name", "case": "title"},
                {"op": "trim_text", "column": "region", "side": "both"},
                {
                    "op": "map_values",
                    "column": "region",
                    "mapping": {"ny": "New York", "ca": "California"},
                    "case_insensitive": True,
                },
                {
                    "op": "fill_null",
                    "column": "region",
                    "strategy": "constant",
                    "constant_value": "Unknown",
                },
                {"op": "standardize_case", "column": "region", "case": "title"},
                {"op": "cast_type", "column": "Qty", "target_type": "int"},
                {"op": "fill_null", "column": "Qty", "strategy": "median"},
                {"op": "cast_type", "column": "Amount", "target_type": "float"},
                {"op": "cast_type", "column": "Active", "target_type": "bool"},
                {"op": "parse_date", "column": "Order Date", "date_format": "%Y-%m-%d"},
                {
                    "op": "split_column",
                    "column": "customer_name",
                    "delimiter": " ",
                    "into": ["first_name", "last_name"],
                    "drop_source": True,
                },
                {"op": "drop_column", "column": "Unused Col"},
                {"op": "drop_duplicates", "subset": ["order_id"], "keep": "first"},
            ],
        }
    )
    plan = parse_cleaning_plan_json(plan_json)

    # First execution
    cleaned_1, res_1 = execute_cleaning_plan(raw_df, plan)

    # Pure function: original DataFrame was never mutated
    pd.testing.assert_frame_equal(raw_df, original_snapshot)

    assert len(cleaned_1) == 3
    assert "Unused Col" not in cleaned_1.columns
    assert "customer_name" not in cleaned_1.columns
    assert cleaned_1["first_name"].tolist() == ["Alice", "Bob", "Carol"]
    assert cleaned_1["last_name"].tolist() == ["Smith", "Jones", "Lee"]
    assert cleaned_1["region"].tolist() == ["New York", "California", "Unknown"]
    assert cleaned_1["Qty"].tolist() == [10, 20, 20]
    assert pd.isna(cleaned_1.loc[2, "Amount"])

    # Cast failure on "bad_num" was logged without silent fabrication
    amount_cast_step = next(
        step for step in res_1.steps if step.operation == "cast_type" and step.failed_row_indices
    )
    assert amount_cast_step.failed_row_indices == [3]

    # Second execution: applying the exact same plan twice MUST be idempotent
    cleaned_2, res_2 = execute_cleaning_plan(cleaned_1, plan)
    pd.testing.assert_frame_equal(cleaned_1, cleaned_2)
    assert len(res_2.lineage_events) == len(plan.operations)

    # Lineage completeness: every operation emitted a LineageEvent and column chains resolve origin
    assert len(res_1.lineage_events) == len(plan.operations)
    tracker = LineageTracker(res_1.lineage_events)
    graph = tracker.build_graph({"orders": list(cleaned_1.columns)}, batch_id="b_clean")

    chain_by_col = {chain.target_column: chain for chain in graph.column_chains}
    assert chain_by_col["first_name"].origin_columns == ["Customer Name"]
    assert "rename_column" in chain_by_col["first_name"].operations
    assert "trim_text" in chain_by_col["first_name"].operations
    assert "standardize_case" in chain_by_col["first_name"].operations
    assert "split_column" in chain_by_col["first_name"].operations

    assert chain_by_col["region"].origin_columns == ["Region Code"]
    assert "map_values" in chain_by_col["region"].operations
    assert "fill_null" in chain_by_col["region"].operations


def test_cleaning_mode_fill_and_missing_column_error() -> None:
    """Mode fill uses deterministic mode and missing columns raise CleaningExecutionError."""
    df = pd.DataFrame({"cat": ["B", "A", "A", None]})
    plan = CleaningPlan(
        table_name="demo",
        operations=[{"op": "fill_null", "column": "cat", "strategy": "mode"}],
    )
    cleaned, _ = execute_cleaning_plan(df, plan)
    assert cleaned["cat"].tolist() == ["B", "A", "A", "A"]

    bad_plan = CleaningPlan(
        table_name="demo",
        operations=[{"op": "trim_text", "column": "nonexistent"}],
    )
    with pytest.raises(CleaningExecutionError):
        execute_cleaning_plan(df, bad_plan)


def test_eda_plotting_helpers_and_notebook_structure(tmp_path: Path) -> None:
    """All four Matplotlib/Seaborn EDA helpers generate figures and save cleanly."""
    sales_df = pd.read_csv(SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv")

    dist_path = tmp_path / "dist.png"
    corr_path = tmp_path / "corr.png"
    trend_path = tmp_path / "trend.png"
    cat_path = tmp_path / "cat.png"

    figs = [
        plot_distribution(sales_df, "revenue", output_path=dist_path),
        plot_correlation_heatmap(sales_df, output_path=corr_path),
        plot_time_trend(sales_df, "order_date", "revenue", output_path=trend_path),
        plot_category_breakdown(sales_df, "product", "revenue", output_path=cat_path),
    ]
    for fig in figs:
        assert len(fig.axes) >= 1
        plt.close(fig)

    for png in (dist_path, corr_path, trend_path, cat_path):
        assert png.is_file() and png.stat().st_size > 1_000

    nb_path = PROJECT_ROOT / "notebooks" / "02_profiling_cleaning_eda.ipynb"
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 4
