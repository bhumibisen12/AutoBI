"""Deterministic ReportSpec validation: chart-to-role rules, cardinality caps, and zero overlaps."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from core.schemas.dax import DAXMeasureSpec
from core.schemas.report_spec import (
    DraftReportSpec,
    DraftVisualSpec,
    FieldBinding,
    GridPosition,
    PixelBounds,
    ReportSpec,
)
from core.schemas.semantic import SemanticColumn, SemanticModel
from core.storage import query_duckdb

_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DATE_DIM_AXIS_COLS: frozenset[str] = frozenset(
    {"date", "year", "quarter", "quarter_label", "month", "month_name", "year_month"}
)


class ReportSpecValidationError(Exception):
    """Raised when a proposed or finalized ReportSpec violates visual or layout rules."""

    def __init__(self, errors: Sequence[str]) -> None:
        """Store all validation errors with a concise summary message."""
        self.errors = list(errors)
        joined = "; ".join(self.errors[:5])
        super().__init__(f"ReportSpec validation failed ({len(self.errors)} error(s)): {joined}")


def _lookup_column(model: SemanticModel, binding: FieldBinding) -> SemanticColumn | None:
    """Resolve a ``FieldBinding`` against ``SemanticModel``."""
    tbl = model.get_table(binding.table_name)
    if tbl is None:
        return None
    return tbl.get_column(binding.column_name)


def _is_date_axis_field(model: SemanticModel, binding: FieldBinding) -> bool:
    """Return True if ``binding`` is a valid temporal axis (date role, dateTime, or dim_date)."""
    tbl = model.get_table(binding.table_name)
    if tbl is None:
        return False
    col = tbl.get_column(binding.column_name)
    if col is None:
        return False
    if col.role == "date" or col.data_type == "dateTime":
        return True
    return tbl.is_date_table and col.name in _DATE_DIM_AXIS_COLS


def _resolve_cardinality(
    binding: FieldBinding,
    *,
    category_cardinalities: Mapping[str, int] | None,
    curated_duckdb_path: Path | None,
) -> int | None:
    """Look up distinct category count for ``binding`` from provided map or curated DuckDB."""
    key = f"{binding.table_name}.{binding.column_name}"
    if category_cardinalities is not None:
        if key in category_cardinalities:
            return int(category_cardinalities[key])
        if binding.column_name in category_cardinalities:
            return int(category_cardinalities[binding.column_name])

    if (
        curated_duckdb_path is not None
        and curated_duckdb_path.is_file()
        and _SAFE_IDENT_RE.match(binding.table_name)
        and _SAFE_IDENT_RE.match(binding.column_name)
    ):
        sql = (
            f'SELECT COUNT(DISTINCT "{binding.column_name}") AS cnt '
            f'FROM "{binding.table_name}"'
        )
        df = query_duckdb(curated_duckdb_path, sql)
        if not df.empty:
            return int(df.iloc[0]["cnt"])
    return None


def _validate_visual(
    visual: DraftVisualSpec,
    page_title: str,
    model: SemanticModel,
    measure_names: set[str],
    *,
    category_cardinalities: Mapping[str, int] | None,
    curated_duckdb_path: Path | None,
    max_pie_categories: int,
) -> list[str]:
    """Validate a single visual's field bindings, measure references, and chart-type rules."""
    errors: list[str] = []
    ctx = f"Page '{page_title}', visual '{visual.title}' ({visual.visual_type})"

    # 1. Validate every measure exists
    for m_name in visual.measures:
        if m_name not in measure_names:
            errors.append(f"{ctx}: unknown measure '{m_name}'.")

    # 2. Validate field bindings exist in SemanticModel
    for label, binding in (
        ("category_field", visual.category_field),
        ("secondary_field", visual.secondary_field),
    ):
        if binding is not None and _lookup_column(model, binding) is None:
            errors.append(
                f"{ctx}: {label} '{binding.table_name}.{binding.column_name}' does not exist."
            )

    for col_bind in visual.columns:
        if _lookup_column(model, col_bind) is None:
            errors.append(
                f"{ctx}: table column '{col_bind.table_name}.{col_bind.column_name}' "
                "does not exist."
            )

    # 3. Chart type vs field role rules
    vtype = visual.visual_type
    if vtype == "kpi_card":
        if len(visual.measures) != 1:
            errors.append(f"{ctx}: kpi_card requires exactly 1 measure.")
        if visual.category_field is not None:
            errors.append(f"{ctx}: kpi_card must not define a category_field axis.")

    elif vtype in {"line_chart", "area_chart"}:
        if visual.category_field is None:
            errors.append(f"{ctx}: {vtype} requires a date/time category_field axis.")
        elif _lookup_column(model, visual.category_field) is not None and not _is_date_axis_field(
            model, visual.category_field
        ):
            errors.append(
                f"{ctx}: {vtype} requires a date/time axis field, got "
                f"'{visual.category_field.table_name}.{visual.category_field.column_name}'."
            )
        if not visual.measures:
            errors.append(f"{ctx}: {vtype} requires at least 1 measure.")

    elif vtype in {"pie_chart", "donut_chart"}:
        if visual.category_field is None:
            errors.append(f"{ctx}: {vtype} requires a categorical category_field.")
        else:
            col = _lookup_column(model, visual.category_field)
            if col is not None:
                if col.role not in {"dimension", "id"}:
                    errors.append(
                        f"{ctx}: {vtype} requires a categorical dimension field, got role "
                        f"'{col.role}'."
                    )
                card = _resolve_cardinality(
                    visual.category_field,
                    category_cardinalities=category_cardinalities,
                    curated_duckdb_path=curated_duckdb_path,
                )
                if card is not None and card > max_pie_categories:
                    errors.append(
                        f"{ctx}: {vtype} rejected because "
                        f"'{visual.category_field.table_name}.{visual.category_field.column_name}' "
                        f"has {card} distinct categories (maximum allowed is {max_pie_categories})."
                    )
        if len(visual.measures) != 1:
            errors.append(f"{ctx}: {vtype} requires exactly 1 measure.")

    elif vtype in {"bar_chart", "column_chart"}:
        if visual.category_field is None:
            errors.append(f"{ctx}: {vtype} requires a category_field.")
        if not visual.measures:
            errors.append(f"{ctx}: {vtype} requires at least 1 measure.")

    elif vtype == "scatter_chart":
        if len(visual.measures) < 2:
            errors.append(f"{ctx}: scatter_chart requires at least 2 measures (X and Y axes).")

    elif vtype in {"table", "matrix"}:
        if not visual.columns and not visual.measures and visual.category_field is None:
            errors.append(f"{ctx}: {vtype} requires at least one column or measure.")

    return errors


def validate_draft_report_spec(
    draft: DraftReportSpec,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    *,
    category_cardinalities: Mapping[str, int] | None = None,
    curated_duckdb_path: Path | None = None,
    max_visuals_per_page: int = 8,
    max_kpi_cards_per_page: int = 4,
    max_slicers_per_page: int = 4,
    max_pie_categories: int = 6,
) -> None:
    """Validate a ``DraftReportSpec`` against semantic schema and visual design rules.

    Raises ``ReportSpecValidationError`` with all violations if any check fails.
    """
    errors: list[str] = []
    measure_names = {m.name for m in measures}
    db_path = curated_duckdb_path or model.curated_duckdb_path

    if not draft.pages:
        errors.append("ReportSpec must contain at least one page.")

    for page in draft.pages:
        if not page.title.strip():
            errors.append("Every report page must have a non-empty title.")
        if not page.visuals:
            errors.append(f"Page '{page.title}' must contain at least one visual.")
        if len(page.visuals) > max_visuals_per_page:
            errors.append(
                f"Page '{page.title}' has {len(page.visuals)} visuals, exceeding the maximum "
                f"of {max_visuals_per_page} visuals per page."
            )

        kpi_count = sum(1 for v in page.visuals if v.visual_type == "kpi_card")
        if kpi_count > max_kpi_cards_per_page:
            errors.append(
                f"Page '{page.title}' has {kpi_count} KPI cards, exceeding the maximum "
                f"of {max_kpi_cards_per_page} per page."
            )

        if len(page.slicers) > max_slicers_per_page:
            errors.append(
                f"Page '{page.title}' has {len(page.slicers)} slicers, exceeding the maximum "
                f"of {max_slicers_per_page} slicers per page."
            )

        for slicer in page.slicers:
            if _lookup_column(
                model, FieldBinding(table_name=slicer.table_name, column_name=slicer.column_name)
            ) is None:
                errors.append(
                    f"Page '{page.title}': slicer column "
                    f"'{slicer.table_name}.{slicer.column_name}' does not exist in SemanticModel."
                )

        for visual in page.visuals:
            errors.extend(
                _validate_visual(
                    visual,
                    page.title,
                    model,
                    measure_names,
                    category_cardinalities=category_cardinalities,
                    curated_duckdb_path=db_path,
                    max_pie_categories=max_pie_categories,
                )
            )

    if errors:
        raise ReportSpecValidationError(errors)


def _grids_overlap(a: GridPosition, b: GridPosition) -> bool:
    """Return True if two 12-column grid rectangles intersect."""
    a_x2, a_y2 = a.col + a.col_span, a.row + a.row_span
    b_x2, b_y2 = b.col + b.col_span, b.row + b.row_span
    return not (a_x2 <= b.col or b_x2 <= a.col or a_y2 <= b.row or b_y2 <= a.row)


def _pixels_overlap(a: PixelBounds, b: PixelBounds) -> bool:
    """Return True if two pixel rectangles intersect."""
    a_x2, a_y2 = a.x + a.width, a.y + a.height
    b_x2, b_y2 = b.x + b.width, b.y + b.height
    return not (a_x2 <= b.x or b_x2 <= a.x or a_y2 <= b.y or b_y2 <= a.y)


def validate_no_overlaps(report: ReportSpec) -> list[str]:
    """Verify every visual stays within canvas bounds and no two visuals overlap on any page."""
    errors: list[str] = []
    for page in report.pages:
        for vis in page.visuals:
            if vis.grid.col + vis.grid.col_span > report.grid_columns:
                errors.append(
                    f"Page '{page.title}', visual '{vis.visual_id}' overflows 12-column grid."
                )
            if (
                vis.pixel_bounds.x + vis.pixel_bounds.width > report.canvas_width
                or vis.pixel_bounds.y + vis.pixel_bounds.height > report.canvas_height
            ):
                errors.append(
                    f"Page '{page.title}', visual '{vis.visual_id}' overflows canvas bounds."
                )

        for i, vis_a in enumerate(page.visuals):
            for vis_b in page.visuals[i + 1 :]:
                if _grids_overlap(vis_a.grid, vis_b.grid):
                    errors.append(
                        f"Page '{page.title}': grid overlap between '{vis_a.visual_id}' "
                        f"and '{vis_b.visual_id}'."
                    )
                if _pixels_overlap(vis_a.pixel_bounds, vis_b.pixel_bounds):
                    errors.append(
                        f"Page '{page.title}': pixel overlap between '{vis_a.visual_id}' "
                        f"and '{vis_b.visual_id}'."
                    )
    return errors

