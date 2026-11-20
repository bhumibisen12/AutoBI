"""Unit tests for Step 9: incremental PBIP (TMDL + PBIR), schema validation, ZIP, and fallback."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import openpyxl
import pandas as pd

from core.config import Settings
from core.dax import generate_dax_measures, reconcile_measures
from core.etl import run_etl_pipeline
from core.pbip import build_pbip_package, validate_pbip_project
from core.report_spec import finalize_report_spec
from core.schemas import (
    DAXMeasureSpec,
    DraftPageSpec,
    DraftReportSpec,
    DraftVisualSpec,
    FieldBinding,
    MetricParameters,
    MetricSelectionPlan,
    PBIPBuildConfig,
    SelectedMetricSpec,
    SlicerSpec,
)
from core.semantic import build_semantic_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


def _make_settings(tmp_path: Path) -> Settings:
    """Create isolated Settings under pytest's temporary directory."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
    )


def test_increment_1_minimal_pbip_project_parquet_and_csv(tmp_path: Path) -> None:
    """Increment 1: 1 curated table via DataFolderPath M param, 1 measure, 1 page with 1 card."""
    minimal_csv = tmp_path / "mini_sales.csv"
    pd.DataFrame(
        {"order_id": ["O1", "O2", "O3"], "amount": [100.0, 200.0, 300.0]}
    ).to_csv(minimal_csv, index=False)

    etl_res = run_etl_pipeline([minimal_csv], settings=_make_settings(tmp_path))
    model = build_semantic_model(etl_res)
    measures = generate_dax_measures(
        MetricSelectionPlan(
            selected_metrics=[
                SelectedMetricSpec(
                    measure_name="Total Amount",
                    template_id="tpl.sum",
                    table_name="mini_sales",
                    params=MetricParameters(column="amount"),
                )
            ]
        ),
        model,
    )
    verification = reconcile_measures(measures, model)
    report_spec = finalize_report_spec(
        DraftReportSpec(
            report_title="Minimal Report",
            pages=[
                DraftPageSpec(
                    title="Overview",
                    visuals=[
                        DraftVisualSpec(
                            visual_type="kpi_card",
                            title="Total Amount",
                            measures=["Total Amount"],
                        )
                    ],
                )
            ],
        ),
        model,
        measures,
    )

    for fmt, m_fn in (("parquet", "Parquet.Document"), ("csv", "Csv.Document")):
        out_dir = tmp_path / f"out_{fmt}"
        res = build_pbip_package(
            model,
            measures,
            report_spec,
            verification,
            out_dir,
            config=PBIPBuildConfig(project_name=f"Mini_{fmt}", source_format=fmt),
        )
        assert res.mode == "pbip"
        assert res.validation_result is not None
        assert res.validation_result.is_valid is True
        assert res.pbip_file_path is not None and res.pbip_file_path.is_file()

        tmdl_dir = res.output_dir / "pbip" / f"Mini_{fmt}.SemanticModel" / "definition"
        expr_text = (tmdl_dir / "expressions.tmdl").read_text(encoding="utf-8")
        assert 'expression DataFolderPath = "../data"' in expr_text

        tbl_text = (tmdl_dir / "tables" / "mini_sales.tmdl").read_text(encoding="utf-8")
        assert m_fn in tbl_text
        assert "DataFolderPath" in tbl_text
        assert "measure 'Total Amount' = SUM( 'mini_sales'[amount] )" in tbl_text


def test_increments_2_3_4_full_pbip_bundle_all_visuals_theme_and_zip(tmp_path: Path) -> None:
    """Increments 2-4: multi-table TMDL, dim_date, 1:* links, all visual types, theme, full ZIP."""
    settings = _make_settings(tmp_path)
    etl_res = run_etl_pipeline(
        [SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv", SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx"],
        settings=settings,
    )
    model = build_semantic_model(etl_res)

    plan = MetricSelectionPlan(
        selected_metrics=[
            SelectedMetricSpec(
                measure_name="Total Revenue",
                template_id="tpl.sum",
                table_name="synthetic_sales",
                params=MetricParameters(column="revenue"),
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
                measure_name="Unique Employees",
                template_id="tpl.distinct_count",
                table_name="synthetic_hr__employees",
                params=MetricParameters(column="employee_id"),
            ),
        ]
    )
    measures = generate_dax_measures(plan, model)
    # Include one intentionally UNVERIFIED custom measure to verify explicit reporting in bundle
    unverified_measure = DAXMeasureSpec(
        name="Custom Unverified Metric",
        table_name="synthetic_sales",
        template_id="tpl.custom_unverified",
        template_type="custom_unverified",
        expression="SUM( 'synthetic_sales'[revenue] )",
    )
    all_measures = [*measures, unverified_measure]
    verification = reconcile_measures(all_measures, model)
    assert verification.unverified_count == 1

    # Cover every required visual type across two clean pages:
    # Page 1: card, line_chart, bar_chart, table + slicers
    # Page 2: column_chart, matrix + slicer
    draft = DraftReportSpec(
        report_title="AutoBI Executive Suite",
        pages=[
            DraftPageSpec(
                title="Sales Overview",
                subtitle="Revenue trends and category performance",
                slicers=[
                    SlicerSpec(table_name="dim_date", column_name="year", title="Year"),
                    SlicerSpec(table_name="synthetic_sales", column_name="region", title="Region"),
                ],
                visuals=[
                    DraftVisualSpec(
                        visual_type="kpi_card",
                        title="Total Revenue",
                        measures=["Total Revenue"],
                    ),
                    DraftVisualSpec(
                        visual_type="kpi_card",
                        title="Gross Margin %",
                        measures=["Gross Margin %"],
                    ),
                    DraftVisualSpec(
                        visual_type="line_chart",
                        title="Monthly Revenue Trend",
                        category_field=FieldBinding(
                            table_name="dim_date", column_name="year_month"
                        ),
                        measures=["Total Revenue"],
                    ),
                    DraftVisualSpec(
                        visual_type="bar_chart",
                        title="Revenue by Product",
                        category_field=FieldBinding(
                            table_name="synthetic_sales", column_name="product"
                        ),
                        measures=["Total Revenue"],
                    ),
                    DraftVisualSpec(
                        visual_type="table",
                        title="Regional Breakdown",
                        columns=[
                            FieldBinding(table_name="synthetic_sales", column_name="region"),
                            FieldBinding(table_name="synthetic_sales", column_name="channel"),
                        ],
                        measures=["Total Revenue", "Gross Margin %"],
                    ),
                ],
            ),
            DraftPageSpec(
                title="Workforce & Matrix",
                subtitle="Departmental view",
                slicers=[
                    SlicerSpec(
                        table_name="synthetic_hr__departments",
                        column_name="department_name",
                        title="Department",
                    )
                ],
                visuals=[
                    DraftVisualSpec(
                        visual_type="column_chart",
                        title="Headcount by Department",
                        category_field=FieldBinding(
                            table_name="synthetic_hr__departments",
                            column_name="department_name",
                        ),
                        measures=["Unique Employees"],
                    ),
                    DraftVisualSpec(
                        visual_type="matrix",
                        title="Region x Category Matrix",
                        category_field=FieldBinding(
                            table_name="synthetic_sales", column_name="region"
                        ),
                        secondary_field=FieldBinding(
                            table_name="synthetic_sales", column_name="category"
                        ),
                        measures=["Total Revenue"],
                    ),
                ],
            ),
        ],
    )
    report_spec = finalize_report_spec(draft, model, all_measures)

    # Set low large_data_warning_mb (0.001 MB) to verify large-data warning triggers cleanly
    cfg = PBIPBuildConfig(
        project_name="AutoBI_Exec_Suite",
        source_format="parquet",
        large_data_warning_mb=0.001,
    )
    export = build_pbip_package(
        model,
        all_measures,
        report_spec,
        verification,
        tmp_path / "exports",
        config=cfg,
    )

    assert export.mode == "pbip"
    assert export.zip_path.is_file()
    assert export.validation_result is not None
    assert export.validation_result.is_valid is True
    assert export.large_data_warning is not None
    assert export.unverified_measures == ["Custom Unverified Metric"]

    # Verify TMDL marked date table and relationships
    sm_def = export.output_dir / "pbip" / "AutoBI_Exec_Suite.SemanticModel" / "definition"
    dim_date_tmdl = (sm_def / "tables" / "dim_date.tmdl").read_text(encoding="utf-8")
    assert "dataCategory: Time" in dim_date_tmdl
    assert "\tisKey" in dim_date_tmdl

    rels_tmdl = (sm_def / "relationships.tmdl").read_text(encoding="utf-8")
    assert "crossFilteringBehavior: oneDirection" in rels_tmdl
    assert "dim_date.date" in rels_tmdl

    # Verify all visual types emitted in PBIR visual.json files
    rep_pages = export.output_dir / "pbip" / "AutoBI_Exec_Suite.Report" / "definition" / "pages"
    emitted_types = set()
    for vf in rep_pages.rglob("visual.json"):
        vdata = json.loads(vf.read_text(encoding="utf-8"))
        emitted_types.add(vdata["visual"]["visualType"])
    assert {
        "card",
        "lineChart",
        "clusteredBarChart",
        "clusteredColumnChart",
        "tableEx",
        "pivotTable",
        "slicer",
    } <= emitted_types

    # Inspect ZIP archive contents
    with zipfile.ZipFile(export.zip_path, "r") as zf:
        names = set(zf.namelist())
    assert "AutoBI_Exec_Suite/pbip/AutoBI_Exec_Suite.pbip" in names
    assert "AutoBI_Exec_Suite/data/synthetic_sales.parquet" in names
    assert "AutoBI_Exec_Suite/data/dim_date.parquet" in names
    assert "AutoBI_Exec_Suite/measures.dax" in names
    assert "AutoBI_Exec_Suite/data_dictionary.xlsx" in names
    assert "AutoBI_Exec_Suite/verification_report.json" in names
    assert "AutoBI_Exec_Suite/verification_report.md" in names
    assert "AutoBI_Exec_Suite/README.md" in names

    # Verify Excel data dictionary has all 4 sheets
    wb = openpyxl.load_workbook(export.output_dir / "data_dictionary.xlsx", read_only=True)
    try:
        assert wb.sheetnames == ["Tables", "Columns", "Measures", "Relationships"]
    finally:
        wb.close()

    # Verify verification_report.md & README.md highlight UNVERIFIED measure & large-data warning
    ver_md = (export.output_dir / "verification_report.md").read_text(encoding="utf-8")
    assert "Custom Unverified Metric" in ver_md
    readme_text = (export.output_dir / "README.md").read_text(encoding="utf-8")
    assert "LARGE DATA WARNING" in readme_text
    assert "Manual Checklist to Open in Power BI Desktop" in readme_text


def test_validator_catches_broken_pbir_and_fallback_export_works(tmp_path: Path) -> None:
    """Validator catches broken visual bindings; fallback writes guide + DAX + dictionary."""
    minimal_csv = tmp_path / "mini.csv"
    pd.DataFrame({"id": ["1", "2"], "val": [10.0, 20.0]}).to_csv(minimal_csv, index=False)

    etl_res = run_etl_pipeline([minimal_csv], settings=_make_settings(tmp_path))
    model = build_semantic_model(etl_res)
    measures = generate_dax_measures(
        MetricSelectionPlan(
            selected_metrics=[
                SelectedMetricSpec(
                    measure_name="Total Val",
                    template_id="tpl.sum",
                    table_name="mini",
                    params=MetricParameters(column="val"),
                )
            ]
        ),
        model,
    )
    verification = reconcile_measures(measures, model)
    report_spec = finalize_report_spec(
        DraftReportSpec(
            report_title="Mini",
            pages=[
                DraftPageSpec(
                    title="Home",
                    visuals=[
                        DraftVisualSpec(
                            visual_type="kpi_card",
                            title="Total Val",
                            measures=["Total Val"],
                        )
                    ],
                )
            ],
        ),
        model,
        measures,
    )

    # 1. Build valid PBIP, tamper with a visual.json reference, and verify validator catches it
    ok_res = build_pbip_package(
        model,
        measures,
        report_spec,
        verification,
        tmp_path / "tamper_test",
        config=PBIPBuildConfig(project_name="TamperProj"),
    )
    vis_files = list((ok_res.output_dir / "pbip").rglob("visual.json"))
    assert vis_files
    tampered = json.loads(vis_files[0].read_text(encoding="utf-8"))
    tampered["visual"]["query"]["queryState"]["Values"]["projections"][0]["field"]["Measure"][
        "Property"
    ] = "Nonexistent Measure"
    vis_files[0].write_text(json.dumps(tampered, indent=2), encoding="utf-8")

    bad_val = validate_pbip_project(ok_res.output_dir / "pbip", "TamperProj")
    assert bad_val.is_valid is False
    assert any("Nonexistent Measure" in iss.message for iss in bad_val.issues)

    # 2. Fallback export generates portable ZIP with measures.dax, data_dictionary.xlsx, build guide
    fb_res = build_pbip_package(
        model,
        measures,
        report_spec,
        verification,
        tmp_path / "fallback_test",
        config=PBIPBuildConfig(project_name="FallbackProj"),
        force_fallback=True,
    )
    assert fb_res.mode == "fallback"
    assert fb_res.pbip_file_path is None
    assert fb_res.fallback_reason is not None
    assert fb_res.zip_path.is_file()
    with zipfile.ZipFile(fb_res.zip_path, "r") as zf:
        fb_names = set(zf.namelist())
    assert "FallbackProj/measures.dax" in fb_names
    assert "FallbackProj/data_dictionary.xlsx" in fb_names
    assert "FallbackProj/STEP_BY_STEP_BUILD_GUIDE.md" in fb_names
    assert "FallbackProj/verification_report.md" in fb_names
    assert "FallbackProj/data/mini.parquet" in fb_names
