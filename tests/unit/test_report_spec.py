"""Unit tests for Step 8: ReportSpec schema, validation, 12-col layout, overlaps, and previews."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pydantic import ValidationError

from core.config import Settings
from core.dax import generate_dax_measures
from core.etl import run_etl_pipeline
from core.llm import MockLLM
from core.rag import LocalHashEmbedder, build_knowledge_index
from core.report_spec import (
    ReportSpecValidationError,
    build_default_draft_report,
    design_report,
    finalize_report_spec,
    render_report_previews_matplotlib,
    render_visual_plotly,
    validate_no_overlaps,
)
from core.schemas import (
    DraftPageSpec,
    DraftReportSpec,
    DraftVisualSpec,
    FieldBinding,
    GridPosition,
    MetricParameters,
    MetricSelectionPlan,
    PixelBounds,
    SelectedMetricSpec,
    UserIntent,
)
from core.semantic import build_semantic_model

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


def _build_sales_context(tmp_path: Path):
    """Run ETL, build curated SemanticModel, and generate DAX measures on SYNTHETIC_sales."""
    settings = _make_settings(tmp_path)
    etl_res = run_etl_pipeline([SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv"], settings=settings)
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
                measure_name="Total Cost",
                template_id="tpl.sum",
                table_name="synthetic_sales",
                params=MetricParameters(column="cost"),
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
                measure_name="Unique Orders",
                template_id="tpl.distinct_count",
                table_name="synthetic_sales",
                params=MetricParameters(column="order_id"),
            ),
        ]
    )
    measures = generate_dax_measures(plan, model)
    return model, measures


def test_end_to_end_report_design_layout_theme_alt_text_and_previews(tmp_path: Path) -> None:
    """LLM proposes DraftReportSpec; deterministic code lays out, themes, and renders previews."""
    model, measures = _build_sales_context(tmp_path)
    intent = UserIntent(
        goal="Executive sales overview showing revenue trends and regional performance",
        audience="executives",
        time_grain="month",
        requested_visuals=["kpi_card", "line_chart", "bar_chart", "table"],
    )

    kb = build_knowledge_index(KNOWLEDGE_DIR, embedder=LocalHashEmbedder())
    visual_rules = kb.retrieve("visual layout max visuals per page colorblind palette", k=4)

    draft = build_default_draft_report(model, intent, measures)
    mock_llm = MockLLM(responses_by_model={DraftReportSpec: draft})

    report, meta = design_report(mock_llm, model, intent, measures, visual_rules)

    assert meta.cached is False
    assert "NEVER output pixel coordinates" in mock_llm.recorded_prompts[0].system_prompt
    assert report.grid_columns == 12
    assert report.theme.name == "Okabe-Ito Accessible"
    assert len(report.pages) == 1
    assert validate_no_overlaps(report) == []

    page = report.pages[0]
    assert page.title == "Executive Overview"
    for vis in page.visuals:
        assert len(vis.alt_text) > 10
        assert 0 <= vis.grid.col < 12
        assert vis.grid.col + vis.grid.col_span <= 12
        assert vis.pixel_bounds.width > 0 and vis.pixel_bounds.height > 0

    # Render Matplotlib page PNG preview from real curated DuckDB data
    preview_dir = tmp_path / "previews"
    figs = render_report_previews_matplotlib(report, model, measures, output_dir=preview_dir)
    assert page.page_id in figs
    png_path = preview_dir / f"{page.page_id}.png"
    assert png_path.is_file() and png_path.stat().st_size > 5_000
    for fig in figs.values():
        plt.close(fig)

    # Render interactive Plotly preview for each visual
    for vis in page.visuals:
        p_fig = render_visual_plotly(vis, model, measures, report.theme)
        assert p_fig is not None


def test_draft_schema_forbids_llm_pixel_positions() -> None:
    """DraftVisualSpec strictly rejects pixel/grid position keys from the LLM."""
    with pytest.raises(ValidationError):
        DraftVisualSpec(
            visual_type="kpi_card",
            title="Total Revenue",
            measures=["Total Revenue"],
            x=100,  # type: ignore[call-arg]
            y=50,  # type: ignore[call-arg]
        )


def test_invalid_report_specs_rejected_with_clear_errors(tmp_path: Path) -> None:
    """Line chart without date axis, pie with >6 categories, >8 visuals, and bad refs fail."""
    model, measures = _build_sales_context(tmp_path)

    # 1. Line chart bound to non-date categorical column ('region')
    bad_line = DraftReportSpec(
        report_title="Bad Line",
        pages=[
            DraftPageSpec(
                title="Page 1",
                visuals=[
                    DraftVisualSpec(
                        visual_type="line_chart",
                        title="Revenue by Region",
                        category_field=FieldBinding(
                            table_name="synthetic_sales", column_name="region"
                        ),
                        measures=["Total Revenue"],
                    )
                ],
            )
        ],
    )
    with pytest.raises(ReportSpecValidationError, match="requires a date/time axis"):
        finalize_report_spec(bad_line, model, measures)

    # 2. Pie chart with >6 categories (checked automatically against curated DuckDB!)
    many_cats_csv = tmp_path / "many_cities.csv"
    pd.DataFrame(
        {
            "order_id": [f"O-{i}" for i in range(12)],
            "city": [f"City_{i}" for i in range(12)],
            "revenue": [100.0 + i for i in range(12)],
        }
    ).to_csv(many_cats_csv, index=False)
    etl_cities = run_etl_pipeline([many_cats_csv], settings=_make_settings(tmp_path / "c2"))
    city_model = build_semantic_model(etl_cities)
    city_measures = generate_dax_measures(
        MetricSelectionPlan(
            selected_metrics=[
                SelectedMetricSpec(
                    measure_name="Total Revenue",
                    template_id="tpl.sum",
                    table_name="many_cities",
                    params=MetricParameters(column="revenue"),
                )
            ]
        ),
        city_model,
    )
    bad_pie = DraftReportSpec(
        report_title="Bad Pie",
        pages=[
            DraftPageSpec(
                title="Cities",
                visuals=[
                    DraftVisualSpec(
                        visual_type="pie_chart",
                        title="Revenue Share by City",
                        category_field=FieldBinding(table_name="many_cities", column_name="city"),
                        measures=["Total Revenue"],
                    )
                ],
            )
        ],
    )
    with pytest.raises(ReportSpecValidationError, match="maximum allowed is 6"):
        finalize_report_spec(bad_pie, city_model, city_measures)

    # 3. Exceeding max visuals per page (> 8 visuals)
    too_many_visuals = DraftReportSpec(
        report_title="Too Crowded",
        pages=[
            DraftPageSpec(
                title="Crowded Page",
                visuals=[
                    DraftVisualSpec(
                        visual_type="bar_chart",
                        title=f"Chart {i}",
                        category_field=FieldBinding(
                            table_name="synthetic_sales", column_name="region"
                        ),
                        measures=["Total Revenue"],
                    )
                    for i in range(9)
                ],
            )
        ],
    )
    with pytest.raises(ReportSpecValidationError, match="exceeding the maximum of 8"):
        finalize_report_spec(too_many_visuals, model, measures)

    # 4. Unknown measure + scatter with only 1 measure + kpi_card with 2 measures
    bad_bindings = DraftReportSpec(
        report_title="Bad Bindings",
        pages=[
            DraftPageSpec(
                title="Broken",
                visuals=[
                    DraftVisualSpec(
                        visual_type="kpi_card",
                        title="Dual KPI",
                        measures=["Total Revenue", "Total Cost"],
                    ),
                    DraftVisualSpec(
                        visual_type="scatter_chart",
                        title="Single Axis Scatter",
                        measures=["Nonexistent Measure"],
                    ),
                ],
            )
        ],
    )
    with pytest.raises(ReportSpecValidationError) as exc_info:
        finalize_report_spec(bad_bindings, model, measures)
    err_text = " | ".join(exc_info.value.errors)
    assert "kpi_card requires exactly 1 measure" in err_text
    assert "unknown measure 'Nonexistent Measure'" in err_text
    assert "scatter_chart requires at least 2 measures" in err_text


@pytest.mark.parametrize("n_kpis", [1, 2, 3, 4])
@pytest.mark.parametrize("n_charts", [1, 2, 3, 4])
def test_auto_layout_never_overlaps_across_grid_combinations(
    tmp_path: Path,
    n_kpis: int,
    n_charts: int,
) -> None:
    """Auto-layout produces zero grid or pixel overlaps for every valid KPI/chart combination."""
    model, measures = _build_sales_context(tmp_path)
    kpi_names = [m.name for m in measures][:n_kpis]
    visuals = [
        DraftVisualSpec(visual_type="kpi_card", title=f"KPI {i}", measures=[name])
        for i, name in enumerate(kpi_names, start=1)
    ]
    for j in range(n_charts):
        visuals.append(
            DraftVisualSpec(
                visual_type="bar_chart" if j % 2 == 0 else "column_chart",
                title=f"Breakdown {j + 1}",
                category_field=FieldBinding(table_name="synthetic_sales", column_name="region"),
                measures=["Total Revenue"],
            )
        )

    draft = DraftReportSpec(
        report_title="Grid Stress Test",
        pages=[DraftPageSpec(title="Stress Page", visuals=visuals)],
    )
    report = finalize_report_spec(draft, model, measures)
    assert validate_no_overlaps(report) == []


def test_overlap_detector_catches_manually_overlapped_visuals(tmp_path: Path) -> None:
    """Explicit overlap validator detects colliding grid cells and colliding pixel rectangles."""
    model, measures = _build_sales_context(tmp_path)
    draft = build_default_draft_report(
        model,
        UserIntent(goal="Sales summary"),
        measures,
    )
    report = finalize_report_spec(draft, model, measures)

    # Mutate visual 2 to intentionally overlap visual 1
    vis0 = report.pages[0].visuals[0]
    vis1 = report.pages[0].visuals[1].model_copy(
        update={
            "grid": GridPosition(
                col=vis0.grid.col,
                row=vis0.grid.row,
                col_span=vis0.grid.col_span,
                row_span=vis0.grid.row_span,
            ),
            "pixel_bounds": PixelBounds(
                x=vis0.pixel_bounds.x,
                y=vis0.pixel_bounds.y,
                width=vis0.pixel_bounds.width,
                height=vis0.pixel_bounds.height,
            ),
        }
    )
    broken_page = report.pages[0].model_copy(
        update={"visuals": [vis0, vis1, *report.pages[0].visuals[2:]]}
    )
    broken_report = report.model_copy(update={"pages": [broken_page]})

    overlaps = validate_no_overlaps(broken_report)
    assert any("grid overlap" in err for err in overlaps)
    assert any("pixel overlap" in err for err in overlaps)

