"""Preview renderer (Matplotlib + Plotly) drawing ReportSpec pages from real curated data."""

from __future__ import annotations

import html
import io
import re
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from core.schemas.dax import DAXMeasureSpec
from core.schemas.report_spec import ReportPageSpec, ReportSpec, ThemeSpec, VisualSpec
from core.schemas.semantic import SemanticModel
from core.storage import query_duckdb

matplotlib.use("Agg")

_SAFE_IDENT_RE = re.compile(r"^[A-Za-z0-9_ \-\./()%$#@+:]+$")


def _qident(name: str) -> str:
    """Validate and double-quote a DuckDB SQL identifier safely (supports spaces in CSV headers)."""
    if (
        not name
        or not _SAFE_IDENT_RE.match(name)
        or "--" in name
        or "/*" in name
        or "*/" in name
    ):
        raise ValueError(f"Unsafe SQL identifier in preview query: {name!r}")
    return f'"{name}"'


def _measure_sql_expr(measure: DAXMeasureSpec, table_alias: str = "f") -> str:
    """Translate a template-backed ``DAXMeasureSpec`` into a DuckDB SQL aggregation snippet."""
    p = measure.params
    t = measure.template_type
    if t == "average" and p.column:
        return f"CAST(AVG({table_alias}.{_qident(p.column)}) AS DOUBLE)"
    if t == "distinct_count" and p.column:
        return f"CAST(COUNT(DISTINCT {table_alias}.{_qident(p.column)}) AS DOUBLE)"
    if t == "divide_ratio" and p.numerator_column and p.denominator_column:
        num = f"CAST(SUM({table_alias}.{_qident(p.numerator_column)}) AS DOUBLE)"
        if p.subtract_column:
            num = (
                f"CAST(SUM({table_alias}.{_qident(p.numerator_column)}) - "
                f"SUM({table_alias}.{_qident(p.subtract_column)}) AS DOUBLE)"
            )
        den = f"CAST(SUM({table_alias}.{_qident(p.denominator_column)}) AS DOUBLE)"
        return f"CASE WHEN {den} = 0 OR {den} IS NULL THEN 0.0 ELSE ({num}) / ({den}) END"
    if p.column:
        return f"CAST(SUM({table_alias}.{_qident(p.column)}) AS DOUBLE)"
    return "CAST(COUNT(*) AS DOUBLE)"


def _format_scalar(val: float, fmt: str | None) -> str:
    """Format a scalar KPI value cleanly using a Power BI format string hint."""
    if fmt and "%" in fmt:
        return f"{val * 100.0:.1f}%"
    if fmt and "$" in fmt:
        return f"${val:,.2f}"
    if fmt and ".00" not in fmt and "." not in fmt:
        return f"{val:,.0f}"
    return f"{val:,.2f}"


def _build_join_for_dim(
    model: SemanticModel,
    fact_table: str,
    dim_table: str,
) -> tuple[str, str]:
    """Return ``(join_sql, dim_alias)`` joining ``fact_table`` (alias ``f``) to ``dim_table``."""
    if dim_table == fact_table:
        return "", "f"
    for rel in model.relationships:
        if rel.from_table == fact_table and rel.to_table == dim_table:
            if dim_table == model.date_table_name:
                return (
                    f"JOIN {_qident(dim_table)} d ON "
                    f"CAST(f.{_qident(rel.from_column)} AS DATE) = "
                    f"CAST(d.{_qident(rel.to_column)} AS DATE)",
                    "d",
                )
            return (
                f"JOIN {_qident(dim_table)} d ON "
                f"f.{_qident(rel.from_column)} = d.{_qident(rel.to_column)}",
                "d",
            )
    return "", "f"


def _query_visual_data(
    visual: VisualSpec,
    model: SemanticModel,
    measure_map: dict[str, DAXMeasureSpec],
) -> pd.DataFrame:
    """Query curated DuckDB to materialize the aggregated data backing ``visual``."""
    db_path = model.curated_duckdb_path
    if db_path is None or not db_path.is_file():
        raise ValueError("SemanticModel.curated_duckdb_path is required for preview rendering.")

    primary_measure = measure_map[visual.measures[0]] if visual.measures else None
    fact_table = (
        primary_measure.table_name
        if primary_measure is not None
        else (
            visual.category_field.table_name
            if visual.category_field is not None
            else visual.columns[0].table_name
        )
    )
    f_tbl = _qident(fact_table)

    if visual.visual_type == "kpi_card" and primary_measure is not None:
        expr = _measure_sql_expr(primary_measure, "f")
        return query_duckdb(db_path, f"SELECT {expr} AS value FROM {f_tbl} f")

    if visual.visual_type in {"table", "matrix"}:
        group_cols: list[str] = []
        select_parts: list[str] = []
        for col_bind in visual.columns:
            ref = f"f.{_qident(col_bind.column_name)}"
            group_cols.append(ref)
            select_parts.append(f"{ref} AS {_qident(col_bind.column_name)}")
        for m_name in visual.measures:
            m_spec = measure_map[m_name]
            if m_spec.table_name == fact_table:
                alias_ident = _qident(
                    re.sub(r"[^A-Za-z0-9_]+", "_", m_spec.name.replace("%", "pct")).strip("_")
                )
                select_parts.append(f"{_measure_sql_expr(m_spec, 'f')} AS {alias_ident}")
        group_clause = f"GROUP BY {', '.join(group_cols)}" if group_cols else ""
        sql = f"SELECT {', '.join(select_parts)} FROM {f_tbl} f {group_clause} LIMIT 12"
        return query_duckdb(db_path, sql)

    if visual.visual_type == "scatter_chart" and len(visual.measures) >= 2:
        m_x = measure_map[visual.measures[0]]
        m_y = measure_map[visual.measures[1]]
        if visual.category_field is not None:
            join_sql, alias = _build_join_for_dim(
                model, fact_table, visual.category_field.table_name
            )
            cat_ref = f"{alias}.{_qident(visual.category_field.column_name)}"
            sql = (
                f"SELECT {cat_ref} AS category, "
                f"{_measure_sql_expr(m_x, 'f')} AS x_val, "
                f"{_measure_sql_expr(m_y, 'f')} AS y_val "
                f"FROM {f_tbl} f {join_sql} GROUP BY {cat_ref}"
            )
        else:
            sql = (
                f"SELECT {_measure_sql_expr(m_x, 'f')} AS x_val, "
                f"{_measure_sql_expr(m_y, 'f')} AS y_val FROM {f_tbl} f"
            )
        return query_duckdb(db_path, sql)

    if visual.category_field is not None and primary_measure is not None:
        join_sql, alias = _build_join_for_dim(model, fact_table, visual.category_field.table_name)
        cat_ref = f"{alias}.{_qident(visual.category_field.column_name)}"
        m_expr = _measure_sql_expr(primary_measure, "f")
        is_time = visual.visual_type in {"line_chart", "area_chart"}
        if is_time:
            order_clause = "ORDER BY category ASC"
        elif visual.format.sort_descending:
            order_clause = "ORDER BY value DESC, category ASC"
        else:
            order_clause = "ORDER BY value ASC"
        sql = (
            f"SELECT {cat_ref} AS category, {m_expr} AS value "
            f"FROM {f_tbl} f {join_sql} GROUP BY {cat_ref} {order_clause} LIMIT 25"
        )
        return query_duckdb(db_path, sql)

    return pd.DataFrame()


def _place_matplotlib_title(
    ax: Axes,
    title: str,
    *,
    title_loc: str,
    title_pos: str,
    title_size: float,
    color: str,
    pad: float = 8.0,
) -> None:
    """Place a Matplotlib chart title either above or just below the chart."""
    if title_pos == "bottom":
        tx = 0.50 if title_loc == "center" else (0.98 if title_loc == "right" else 0.02)
        ax.text(
            tx,
            -0.26,
            title,
            transform=ax.transAxes,
            fontsize=title_size,
            color=color,
            fontweight="bold",
            ha=title_loc,
            va="top",
        )
    else:
        ax.set_title(
            title,
            fontsize=title_size,
            color=color,
            fontweight="bold",
            loc=title_loc,
            pad=pad,
        )


def _draw_matplotlib_visual(
    ax: Axes,
    visual: VisualSpec,
    data: pd.DataFrame,
    measure_map: dict[str, DAXMeasureSpec],
    theme: ThemeSpec,
) -> None:
    """Render a single visual with executive styling into its Matplotlib subplot axes."""
    fmt_spec = visual.format
    palette = fmt_spec.custom_colors or theme.palette
    surface_bg = fmt_spec.background_color or theme.surface_color
    ax.set_facecolor(surface_bg)
    primary_color = palette[0]
    secondary_color = palette[1] if len(palette) > 1 else primary_color
    title_loc = fmt_spec.title_align
    title_pos = fmt_spec.title_position
    title_size = float(fmt_spec.title_font_size or 10.5)

    if visual.visual_type == "kpi_card":
        ax.set_xticks([])
        ax.set_yticks([])
        for spine_name, spine in ax.spines.items():
            if spine_name == "left":
                spine.set_edgecolor(primary_color)
                spine.set_linewidth(4.5)
            else:
                spine.set_edgecolor(theme.border_color)
                spine.set_linewidth(1.0)
        has_val = not data.empty and pd.notna(data.iloc[0]["value"])
        val = float(data.iloc[0]["value"]) if has_val else 0.0
        fmt = fmt_spec.number_format or (
            measure_map[visual.measures[0]].format_string if visual.measures else None
        )
        tx = 0.50 if title_loc == "center" else (0.92 if title_loc == "right" else 0.09)
        title_y, val_y = (0.20, 0.58) if title_pos == "bottom" else (0.70, 0.24)
        ax.text(
            tx,
            title_y,
            visual.title.upper(),
            fontsize=8.5,
            color=theme.muted_text_color,
            transform=ax.transAxes,
            fontweight="bold",
            ha=title_loc,
        )
        ax.text(
            tx,
            val_y,
            _format_scalar(val, fmt),
            fontsize=20,
            color=primary_color,
            transform=ax.transAxes,
            fontweight="bold",
            ha=title_loc,
        )
        return

    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_edgecolor(theme.border_color)

    if visual.visual_type in {"line_chart", "area_chart"} and not data.empty:
        x_labels = data["category"].astype(str).tolist()
        y_vals = data["value"].astype(float).tolist()
        ax.plot(
            x_labels,
            y_vals,
            marker="o",
            markersize=5,
            color=primary_color,
            linewidth=2.4,
        )
        ax.fill_between(range(len(x_labels)), y_vals, color=primary_color, alpha=0.14)
        _place_matplotlib_title(
            ax,
            visual.title,
            title_loc=title_loc,
            title_pos=title_pos,
            title_size=title_size,
            color=theme.text_color,
        )
        ax.tick_params(axis="x", rotation=30, labelsize=8, colors=theme.muted_text_color)
        ax.tick_params(axis="y", labelsize=8, colors=theme.muted_text_color)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35, color=theme.border_color)
        return

    if visual.visual_type == "bar_chart" and not data.empty:
        cats = data["category"].astype(str).tolist()[::-1]
        vals = data["value"].astype(float).tolist()[::-1]
        bar_colors = (
            [palette[i % len(palette)] for i in range(len(cats))]
            if fmt_spec.custom_colors and len(palette) > 1
            else primary_color
        )
        bars = ax.barh(cats, vals, color=bar_colors, height=0.62, alpha=0.92)
        max_val = max((abs(v) for v in vals), default=1.0) or 1.0
        if fmt_spec.show_data_labels:
            for bar, val in zip(bars, vals, strict=False):
                ax.text(
                    val + max_val * 0.02,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{val:,.1f}",
                    va="center",
                    fontsize=7.5,
                    color=theme.text_color,
                    fontweight="bold",
                )
        ax.set_xlim(0, max_val * 1.18)
        _place_matplotlib_title(
            ax,
            visual.title,
            title_loc=title_loc,
            title_pos=title_pos,
            title_size=title_size,
            color=theme.text_color,
        )
        ax.tick_params(labelsize=8, colors=theme.muted_text_color)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35, color=theme.border_color)
        return

    if visual.visual_type == "column_chart" and not data.empty:
        cats = data["category"].astype(str).tolist()
        vals = data["value"].astype(float).tolist()
        bar_colors = [palette[i % len(palette)] for i in range(len(cats))]
        bars = ax.bar(cats, vals, color=bar_colors, width=0.58, alpha=0.92)
        max_val = max((abs(v) for v in vals), default=1.0) or 1.0
        if fmt_spec.show_data_labels:
            for bar, val in zip(bars, vals, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    val + max_val * 0.02,
                    f"{val:,.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=theme.text_color,
                    fontweight="bold",
                )
        ax.set_ylim(0, max_val * 1.18)
        _place_matplotlib_title(
            ax,
            visual.title,
            title_loc=title_loc,
            title_pos=title_pos,
            title_size=title_size,
            color=theme.text_color,
        )
        ax.tick_params(axis="x", rotation=20, labelsize=8, colors=theme.muted_text_color)
        ax.tick_params(axis="y", labelsize=8, colors=theme.muted_text_color)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35, color=theme.border_color)
        return

    if visual.visual_type in {"pie_chart", "donut_chart"} and not data.empty:
        cats = data["category"].astype(str).tolist()
        vals = [max(0.0, float(v)) for v in data["value"].tolist()]
        colors = [palette[i % len(palette)] for i in range(len(cats))]
        has_white = any(c.upper() in {"#FFFFFF", "#FFF", "#F8FAFC", "#F3F4F6"} for c in colors)
        edge_col = "#111827" if has_white else theme.background_color
        wedge_props = (
            {"width": 0.45, "edgecolor": edge_col, "linewidth": 1.6}
            if visual.visual_type == "donut_chart"
            else {"edgecolor": edge_col, "linewidth": 1.4}
        )
        ax.pie(
            vals,
            labels=cats if fmt_spec.show_legend else None,
            colors=colors,
            autopct="%1.1f%%" if fmt_spec.show_data_labels else None,
            wedgeprops=wedge_props,
            textprops={"fontsize": 8, "color": theme.text_color},
        )
        _place_matplotlib_title(
            ax,
            visual.title,
            title_loc=title_loc,
            title_pos=title_pos,
            title_size=title_size,
            color=theme.text_color,
        )
        return

    if visual.visual_type == "scatter_chart" and not data.empty:
        ax.scatter(
            data["x_val"],
            data["y_val"],
            color=secondary_color,
            edgecolors=primary_color,
            linewidths=1.2,
            alpha=0.88,
            s=75,
        )
        if fmt_spec.show_data_labels and "category" in data.columns and len(data) <= 12:
            for _, row in data.iterrows():
                ax.annotate(
                    str(row["category"])[:14],
                    (float(row["x_val"]), float(row["y_val"])),
                    fontsize=7,
                    color=theme.muted_text_color,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        _place_matplotlib_title(
            ax,
            visual.title,
            title_loc=title_loc,
            title_pos=title_pos,
            title_size=title_size,
            color=theme.text_color,
        )
        ax.tick_params(labelsize=8, colors=theme.muted_text_color)
        ax.grid(True, linestyle="--", alpha=0.35, color=theme.border_color)
        return

    if visual.visual_type in {"table", "matrix"}:
        ax.axis("off")
        _place_matplotlib_title(
            ax,
            visual.title,
            title_loc=title_loc,
            title_pos=title_pos,
            title_size=title_size,
            color=theme.text_color,
            pad=6.0,
        )
        if not data.empty:
            preview_df = data.head(7).copy()
            for col in preview_df.select_dtypes(include="number").columns:
                preview_df[col] = preview_df[col].map(lambda v: f"{float(v):,.2f}")
            tbl = ax.table(
                cellText=preview_df.values,
                colLabels=list(preview_df.columns),
                loc="center",
                cellLoc="left",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1.0, 1.4)
            for (row_idx, _col_idx), cell in tbl.get_celld().items():
                cell.set_edgecolor(theme.border_color)
                if row_idx == 0:
                    cell.set_facecolor(primary_color)
                    cell.set_text_props(color="#FFFFFF", fontweight="bold")
                elif row_idx % 2 == 0:
                    cell.set_facecolor(surface_bg)
                    cell.set_text_props(color=theme.text_color)
                else:
                    cell.set_facecolor(theme.background_color)
                    cell.set_text_props(color=theme.text_color)
        return

    ax.text(0.5, 0.5, "No data", ha="center", va="center", color=theme.muted_text_color)
    ax.set_title(visual.title, fontsize=title_size, color=theme.text_color, loc=title_loc)


def _render_single_page_matplotlib(
    page: ReportPageSpec,
    report: ReportSpec,
    model: SemanticModel,
    measure_map: dict[str, DAXMeasureSpec],
) -> Figure:
    """Render one complete ``ReportPageSpec`` onto an executive 12-column Matplotlib canvas."""
    total_rows = max((v.grid.row + v.grid.row_span for v in page.visuals), default=4)
    fig = plt.figure(figsize=(13.6, 7.8), facecolor=report.theme.background_color)
    gs = GridSpec(total_rows, 12, figure=fig, hspace=0.62, wspace=0.45, top=0.82, bottom=0.06)

    # Top executive header accent stripe
    header_ax = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    header_ax.set_facecolor(report.theme.palette[0])
    header_ax.axis("off")

    slicer_text = (
        "Filters: " + "  |  ".join(s.title for s in page.slicers)
        if page.slicers
        else "All Records"
    )
    header_ax.text(
        0.03,
        0.58,
        f"{report.report_title}  —  {page.title}",
        fontsize=15,
        fontweight="bold",
        color="#FFFFFF",
        va="center",
    )
    sub_line = f"{page.subtitle}   •   {slicer_text}" if page.subtitle else slicer_text
    header_ax.text(
        0.03,
        0.22,
        sub_line[:135],
        fontsize=9,
        color="#E2E8F0",
        va="center",
    )

    for visual in page.visuals:
        g = visual.grid
        ax = fig.add_subplot(gs[g.row : g.row + g.row_span, g.col : g.col + g.col_span])
        data = _query_visual_data(visual, model, measure_map)
        _draw_matplotlib_visual(ax, visual, data, measure_map, report.theme)

    return fig


def render_report_previews_matplotlib(
    report: ReportSpec,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    *,
    output_dir: Path | None = None,
) -> dict[str, Figure]:
    """Render every page of ``ReportSpec`` from curated DuckDB data as Matplotlib figures."""
    measure_map = {m.name: m for m in measures}
    figures: dict[str, Figure] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for page in report.pages:
        fig = _render_single_page_matplotlib(page, report, model, measure_map)
        figures[page.page_id] = fig
        if output_dir is not None:
            fig.savefig(output_dir / f"{page.page_id}.png", dpi=130, bbox_inches="tight")
    return figures


def render_visual_plotly(
    visual: VisualSpec,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    theme: ThemeSpec | None = None,
) -> go.Figure:
    """Render an executive interactive Plotly visual from curated DuckDB data."""
    resolved_theme = theme or ThemeSpec()
    fmt_spec = visual.format
    palette = fmt_spec.custom_colors or resolved_theme.palette
    surface_bg = fmt_spec.background_color or resolved_theme.surface_color
    paper_bg = fmt_spec.background_color or resolved_theme.background_color
    measure_map = {m.name: m for m in measures}
    data = _query_visual_data(visual, model, measure_map)
    primary = palette[0]
    secondary = palette[1] if len(palette) > 1 else primary

    fmt = fmt_spec.number_format or (
        measure_map[visual.measures[0]].format_string
        if visual.measures and visual.measures[0] in measure_map
        else None
    )
    title_size = fmt_spec.title_font_size or 15
    title_x = (
        0.5
        if fmt_spec.title_align == "center"
        else (0.98 if fmt_spec.title_align == "right" else 0.02)
    )

    if visual.visual_type == "kpi_card":
        has_val = not data.empty and pd.notna(data.iloc[0]["value"])
        val = float(data.iloc[0]["value"]) if has_val else 0.0
        num_prefix = "$" if fmt and "$" in fmt else ""
        num_suffix = "%" if fmt and "%" in fmt else ""
        display_val = val * 100.0 if num_suffix == "%" else val
        indicator_title = (
            None
            if fmt_spec.title_position == "bottom"
            else {
                "text": f"<b>{visual.title}</b>",
                "align": fmt_spec.title_align,
                "font": {
                    "size": fmt_spec.title_font_size or 14,
                    "color": resolved_theme.muted_text_color,
                },
            }
        )
        fig = go.Figure(
            go.Indicator(
                mode="number",
                value=display_val,
                number={
                    "prefix": num_prefix,
                    "suffix": num_suffix,
                    "valueformat": ",.1f" if num_suffix == "%" else ",.2f",
                    "font": {"size": 36, "color": primary},
                },
                title=indicator_title,
            )
        )
        if fmt_spec.title_position == "bottom":
            fig.update_layout(
                height=165,
                margin={"l": 16, "r": 16, "t": 18, "b": 44},
                title={
                    "text": f"<b>{visual.title}</b>",
                    "x": title_x,
                    "xanchor": fmt_spec.title_align,
                    "y": 0.08,
                    "yanchor": "bottom",
                    "yref": "container",
                    "font": {
                        "size": fmt_spec.title_font_size or 13,
                        "color": resolved_theme.muted_text_color,
                    },
                },
            )
        else:
            fig.update_layout(height=165, margin={"l": 16, "r": 16, "t": 44, "b": 12})
    elif visual.visual_type in {"line_chart", "area_chart"} and not data.empty:
        fig = go.Figure(
            go.Scatter(
                x=data["category"].astype(str),
                y=data["value"],
                mode="lines+markers",
                fill="tozeroy",
                line={"color": primary, "width": 3, "shape": "spline"},
                marker={"size": 7, "color": primary},
            )
        )
        fig.update_layout(hovermode="x unified", height=360, showlegend=fmt_spec.show_legend)
    elif visual.visual_type == "bar_chart" and not data.empty:
        ordered = data.iloc[::-1]
        bar_colors = (
            [palette[i % len(palette)] for i in range(len(ordered))]
            if fmt_spec.custom_colors and len(palette) > 1
            else primary
        )
        fig = go.Figure(
            go.Bar(
                y=ordered["category"].astype(str),
                x=ordered["value"],
                orientation="h",
                marker={"color": bar_colors, "line": {"width": 0}},
                text=(
                    ordered["value"].map(lambda v: _format_scalar(float(v), fmt))
                    if fmt_spec.show_data_labels
                    else None
                ),
                textposition="outside" if fmt_spec.show_data_labels else None,
                cliponaxis=False,
            )
        )
        fig.update_layout(height=360, yaxis={"showgrid": False}, showlegend=fmt_spec.show_legend)
    elif visual.visual_type == "column_chart" and not data.empty:
        colors = [palette[i % len(palette)] for i in range(len(data))]
        fig = go.Figure(
            go.Bar(
                x=data["category"].astype(str),
                y=data["value"],
                marker={"color": colors},
                text=(
                    data["value"].map(lambda v: _format_scalar(float(v), fmt))
                    if fmt_spec.show_data_labels
                    else None
                ),
                textposition="outside" if fmt_spec.show_data_labels else None,
                cliponaxis=False,
            )
        )
        fig.update_layout(height=360, xaxis={"showgrid": False}, showlegend=fmt_spec.show_legend)
    elif visual.visual_type in {"pie_chart", "donut_chart"} and not data.empty:
        hole = 0.50 if visual.visual_type == "donut_chart" else 0.0
        has_white = any(c.upper() in {"#FFFFFF", "#FFF", "#F8FAFC", "#F3F4F6"} for c in palette)
        border_col = "#111827" if has_white else resolved_theme.background_color
        fig = go.Figure(
            go.Pie(
                labels=data["category"].astype(str),
                values=data["value"],
                hole=hole,
                textinfo="label+percent" if fmt_spec.show_data_labels else "none",
                marker={
                    "colors": palette,
                    "line": {"color": border_col, "width": 2},
                },
            )
        )
        fig.update_layout(height=360, showlegend=fmt_spec.show_legend)
    elif visual.visual_type == "scatter_chart" and not data.empty:
        fig = go.Figure(
            go.Scatter(
                x=data["x_val"],
                y=data["y_val"],
                text=data["category"].astype(str) if "category" in data.columns else None,
                mode=(
                    "markers+text"
                    if fmt_spec.show_data_labels and len(data) <= 10
                    else "markers"
                ),
                textposition="top center",
                marker={
                    "color": secondary,
                    "size": 12,
                    "line": {"color": primary, "width": 1.5},
                },
            )
        )
        fig.update_layout(
            height=360,
            showlegend=fmt_spec.show_legend,
            xaxis_title=fmt_spec.x_axis_title or "X",
            yaxis_title=fmt_spec.y_axis_title or "Y",
        )
    elif visual.visual_type in {"table", "matrix"} and not data.empty:
        formatted_cols = []
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                formatted_cols.append(
                    [f"{float(v):,.2f}" if pd.notna(v) else "" for v in data[col]]
                )
            else:
                formatted_cols.append(data[col].astype(str).tolist())
        row_fills = [
            surface_bg if idx % 2 == 0 else paper_bg
            for idx in range(len(data))
        ]
        fig = go.Figure(
            go.Table(
                header={
                    "values": [f"<b>{c}</b>" for c in data.columns],
                    "fill_color": primary,
                    "font": {"color": "#FFFFFF", "size": 12},
                    "align": "left",
                    "height": 32,
                },
                cells={
                    "values": formatted_cols,
                    "fill_color": [row_fills],
                    "font": {"color": resolved_theme.text_color, "size": 11},
                    "align": "left",
                    "height": 28,
                },
            )
        )
        fig.update_layout(height=340, margin={"l": 8, "r": 8, "t": 44, "b": 8})
    else:
        fig = go.Figure()

    layout_kwargs: dict[str, object] = {
        "paper_bgcolor": paper_bg,
        "plot_bgcolor": surface_bg,
        "font": {"family": resolved_theme.font_family, "color": resolved_theme.text_color},
    }
    if visual.visual_type != "kpi_card":
        is_bottom = fmt_spec.title_position == "bottom"
        layout_kwargs["title"] = {
            "text": f"<b>{visual.title}</b>",
            "x": title_x,
            "xanchor": fmt_spec.title_align,
            "y": 0.03 if is_bottom else 0.96,
            "yanchor": "bottom" if is_bottom else "top",
            "yref": "container",
            "font": {"size": title_size, "color": resolved_theme.text_color},
        }
        layout_kwargs["margin"] = (
            {"l": 24, "r": 24, "t": 22, "b": 72}
            if is_bottom
            else {"l": 24, "r": 24, "t": 54, "b": 36}
        )
    fig.update_layout(**layout_kwargs)
    return fig


def export_interactive_html_report(
    report: ReportSpec,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
) -> str:
    """Export the exact interactive Plotly executive dashboard as a standalone HTML file."""
    theme = report.theme
    primary = theme.palette[0]
    safe_title = html.escape(report.report_title)
    safe_theme = html.escape(theme.name)

    page_sections: list[str] = []
    plotly_js_included = False

    for p_idx, page in enumerate(report.pages, start=1):
        kpi_visuals = [v for v in page.visuals if v.visual_type == "kpi_card"]
        table_visuals = [v for v in page.visuals if v.visual_type in {"table", "matrix"}]
        chart_visuals = [
            v for v in page.visuals if v.visual_type not in {"kpi_card", "table", "matrix"}
        ]

        def _fig_html(visual: VisualSpec) -> str:
            nonlocal plotly_js_included
            fig = render_visual_plotly(visual, model, measures, theme=theme)
            include_js: str | bool = "cdn" if not plotly_js_included else False
            plotly_js_included = True
            return fig.to_html(full_html=False, include_plotlyjs=include_js)

        kpi_html = ""
        if kpi_visuals:
            cards = "".join(
                f'<div class="card">{_fig_html(v)}</div>' for v in kpi_visuals
            )
            kpi_html = f'<div class="kpi-grid">{cards}</div>'

        charts_html = ""
        if chart_visuals:
            cards = "".join(
                f'<div class="card">{_fig_html(v)}</div>' for v in chart_visuals
            )
            charts_html = f'<div class="chart-grid">{cards}</div>'

        tables_html = "".join(
            f'<div class="card full-width">{_fig_html(v)}</div>' for v in table_visuals
        )

        slicers_html = (
            '<div class="slicer-bar">Filters: '
            + " &nbsp;&bull;&nbsp; ".join(html.escape(s.title) for s in page.slicers)
            + "</div>"
            if page.slicers
            else ""
        )
        sub_html = (
            f'<p class="page-sub">{html.escape(page.subtitle)}</p>' if page.subtitle else ""
        )

        page_sections.append(
            f"""
            <section class="page-section">
              <div class="page-heading">
                <h2>{p_idx}. {html.escape(page.title)}</h2>
                {sub_html}
                {slicers_html}
              </div>
              {kpi_html}
              {charts_html}
              {tables_html}
            </section>
            """
        )

    body_html = "\n".join(page_sections)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px 32px;
      background: {theme.background_color};
      color: {theme.text_color};
      font-family: "{theme.font_family}", "Segoe UI", Arial, sans-serif;
    }}
    .banner {{
      background: {primary};
      color: #FFFFFF;
      padding: 22px 28px;
      border-radius: 12px;
      margin-bottom: 24px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }}
    .banner h1 {{ margin: 0; font-size: 26px; font-weight: 700; }}
    .banner .meta {{ margin-top: 6px; font-size: 13px; color: #E2E8F0; }}
    .page-section {{ margin-bottom: 36px; }}
    .page-heading h2 {{ margin: 0 0 4px 0; font-size: 20px; color: {theme.text_color}; }}
    .page-sub {{ margin: 0 0 8px 0; font-size: 13px; color: {theme.muted_text_color}; }}
    .slicer-bar {{
      display: inline-block;
      background: {theme.surface_color};
      border: 1px solid {theme.border_color};
      border-radius: 6px;
      padding: 4px 12px;
      font-size: 12px;
      color: {theme.muted_text_color};
      margin-bottom: 14px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 18px;
      margin-bottom: 18px;
    }}
    .card {{
      background: {theme.surface_color};
      border: 1px solid {theme.border_color};
      border-radius: 10px;
      padding: 10px;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }}
    .full-width {{ margin-bottom: 18px; }}
    @media print {{
      .page-section {{ page-break-after: always; }}
    }}
  </style>
</head>
<body>
  <header class="banner">
    <h1>{safe_title}</h1>
    <div class="meta">
      Theme: <b>{safe_theme}</b> &nbsp;|&nbsp;
      Pages: <b>{len(report.pages)}</b> &nbsp;|&nbsp;
      Verified Measures: <b>{len(measures)}</b>
    </div>
  </header>
  {body_html}
</body>
</html>
"""


def export_report_pdf_bytes(
    report: ReportSpec,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
) -> bytes:
    """Export all executive report pages as a multi-page printable PDF (bytes)."""
    measure_map = {m.name: m for m in measures}
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for page in report.pages:
            fig = _render_single_page_matplotlib(page, report, model, measure_map)
            try:
                pdf.savefig(fig, bbox_inches="tight")
            finally:
                plt.close(fig)
    return buf.getvalue()

