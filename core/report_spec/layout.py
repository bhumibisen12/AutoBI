"""Deterministic 12-column auto-layout, accessible theming, and alt-text enrichment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from core.report_spec.validator import (
    ReportSpecValidationError,
    validate_draft_report_spec,
    validate_no_overlaps,
)
from core.schemas.dax import DAXMeasureSpec
from core.schemas.report_spec import (
    DraftReportSpec,
    DraftVisualSpec,
    GridPosition,
    PixelBounds,
    ReportPageSpec,
    ReportSpec,
    ThemeSpec,
    VisualSpec,
)
from core.schemas.semantic import SemanticModel

_GUTTER_PX = 16
_HEADER_HEIGHT_PX = 84
_SLICER_BAR_HEIGHT_PX = 44


def build_alt_text(visual: DraftVisualSpec) -> str:
    """Generate concise, WCAG-ready screen-reader alt text for a visual."""
    if visual.alt_text and visual.alt_text.strip():
        return visual.alt_text.strip()
    kind_label = visual.visual_type.replace("_", " ").title()
    measure_text = ", ".join(visual.measures) if visual.measures else "selected fields"
    if visual.category_field is not None:
        axis_text = f"{visual.category_field.table_name}.{visual.category_field.column_name}"
        return f"{kind_label} titled '{visual.title}' showing {measure_text} by {axis_text}."
    if visual.columns:
        cols_text = ", ".join(c.column_name for c in visual.columns[:4])
        return f"{kind_label} titled '{visual.title}' displaying {cols_text} and {measure_text}."
    return f"{kind_label} titled '{visual.title}' displaying {measure_text}."


def _compute_grid_positions(visuals: Sequence[DraftVisualSpec]) -> list[tuple[int, GridPosition]]:
    """Assign non-overlapping 12-column grid positions: KPI strip on top, charts below."""
    kpi_items = [(idx, v) for idx, v in enumerate(visuals) if v.visual_type == "kpi_card"]
    chart_items = [(idx, v) for idx, v in enumerate(visuals) if v.visual_type != "kpi_card"]

    placed: list[tuple[int, GridPosition]] = []
    current_row = 0

    # 1. Place KPI cards evenly across row 0 (row_span=2)
    if kpi_items:
        n_kpis = len(kpi_items)
        base_span = 12 // n_kpis
        remainder = 12 % n_kpis
        cursor_col = 0
        for pos, (orig_idx, _) in enumerate(kpi_items):
            span = base_span + (1 if pos < remainder else 0)
            placed.append(
                (orig_idx, GridPosition(col=cursor_col, row=0, col_span=span, row_span=2))
            )
            cursor_col += span
        current_row = 2

    # 2. Place analytical charts/tables below KPI strip in clean rows
    idx_ptr = 0
    total_charts = len(chart_items)
    while idx_ptr < total_charts:
        orig_idx, vis = chart_items[idx_ptr]
        remaining = total_charts - idx_ptr
        is_wide = vis.visual_type in {"table", "matrix"} or remaining == 1

        if is_wide:
            placed.append(
                (orig_idx, GridPosition(col=0, row=current_row, col_span=12, row_span=4))
            )
            current_row += 4
            idx_ptr += 1
        else:
            # Pair two side-by-side visuals (6 columns each) unless the next visual is a wide table.
            next_orig_idx, next_vis = chart_items[idx_ptr + 1]
            if next_vis.visual_type in {"table", "matrix"}:
                placed.append(
                    (orig_idx, GridPosition(col=0, row=current_row, col_span=12, row_span=4))
                )
                current_row += 4
                idx_ptr += 1
            else:
                placed.append(
                    (orig_idx, GridPosition(col=0, row=current_row, col_span=6, row_span=4))
                )
                placed.append(
                    (next_orig_idx, GridPosition(col=6, row=current_row, col_span=6, row_span=4))
                )
                current_row += 4
                idx_ptr += 2

    # Preserve original visual ordering
    placed.sort(key=lambda item: item[0])
    return placed


def _grid_to_pixels(
    grid: GridPosition,
    *,
    total_rows: int,
    has_slicers: bool,
    canvas_width: int,
    canvas_height: int,
) -> PixelBounds:
    """Convert a 12-column ``GridPosition`` into non-overlapping 16:9 canvas pixel bounds."""
    top_offset = _HEADER_HEIGHT_PX + (_SLICER_BAR_HEIGHT_PX if has_slicers else 0)
    usable_width = canvas_width - 2 * _GUTTER_PX
    usable_height = canvas_height - top_offset - _GUTTER_PX

    col_unit = usable_width / 12.0
    row_unit = usable_height / max(total_rows, 1)

    x1 = round(_GUTTER_PX + grid.col * col_unit + (_GUTTER_PX / 2 if grid.col > 0 else 0))
    x2 = round(
        _GUTTER_PX
        + (grid.col + grid.col_span) * col_unit
        - (_GUTTER_PX / 2 if (grid.col + grid.col_span) < 12 else 0)
    )
    y1 = round(top_offset + grid.row * row_unit + (_GUTTER_PX / 2 if grid.row > 0 else 0))
    y2 = round(
        top_offset
        + (grid.row + grid.row_span) * row_unit
        - (_GUTTER_PX / 2 if (grid.row + grid.row_span) < total_rows else 0)
    )

    return PixelBounds(
        x=max(0, x1),
        y=max(0, y1),
        width=max(1, x2 - x1),
        height=max(1, y2 - y1),
    )


def finalize_report_spec(
    draft: DraftReportSpec,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    *,
    theme: ThemeSpec | None = None,
    category_cardinalities: Mapping[str, int] | None = None,
    curated_duckdb_path: Path | None = None,
    rag_source_ids: Sequence[str] = (),
    max_visuals_per_page: int = 8,
    max_pie_categories: int = 6,
    canvas_width: int = 1280,
    canvas_height: int = 720,
) -> ReportSpec:
    """Validate a ``DraftReportSpec`` and apply deterministic 12-col layout, theme, and alt-text."""
    validate_draft_report_spec(
        draft,
        model,
        measures,
        category_cardinalities=category_cardinalities,
        curated_duckdb_path=curated_duckdb_path,
        max_visuals_per_page=max_visuals_per_page,
        max_pie_categories=max_pie_categories,
    )

    resolved_theme = theme or ThemeSpec()
    final_pages: list[ReportPageSpec] = []

    for page_idx, draft_page in enumerate(draft.pages, start=1):
        page_id = f"page_{page_idx:02d}"
        grid_assignments = _compute_grid_positions(draft_page.visuals)
        total_rows = max((gp.row + gp.row_span for _, gp in grid_assignments), default=1)
        has_slicers = len(draft_page.slicers) > 0

        final_visuals: list[VisualSpec] = []
        for vis_idx, (orig_idx, grid_pos) in enumerate(grid_assignments, start=1):
            dv = draft_page.visuals[orig_idx]
            bounds = _grid_to_pixels(
                grid_pos,
                total_rows=total_rows,
                has_slicers=has_slicers,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            final_visuals.append(
                VisualSpec(
                    visual_id=f"{page_id}_vis_{vis_idx:02d}",
                    visual_type=dv.visual_type,
                    title=dv.title.strip(),
                    category_field=dv.category_field,
                    secondary_field=dv.secondary_field,
                    measures=list(dv.measures),
                    columns=list(dv.columns),
                    format=dv.format,
                    grid=grid_pos,
                    pixel_bounds=bounds,
                    alt_text=build_alt_text(dv),
                )
            )

        final_pages.append(
            ReportPageSpec(
                page_id=page_id,
                title=draft_page.title.strip(),
                subtitle=draft_page.subtitle.strip(),
                slicers=list(draft_page.slicers),
                visuals=final_visuals,
            )
        )

    report = ReportSpec(
        report_title=draft.report_title.strip(),
        business_domain=model.business_domain,
        batch_id=model.batch_id,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        grid_columns=12,
        theme=resolved_theme,
        pages=final_pages,
        rag_source_ids=list(rag_source_ids),
    )

    overlap_errors = validate_no_overlaps(report)
    if overlap_errors:
        raise ReportSpecValidationError(overlap_errors)
    return report

