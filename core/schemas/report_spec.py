"""Pydantic contracts for LLM-proposed draft specs and finalized 12-column ``ReportSpec``."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VisualType = Literal[
    "kpi_card",
    "line_chart",
    "area_chart",
    "bar_chart",
    "column_chart",
    "donut_chart",
    "pie_chart",
    "scatter_chart",
    "table",
    "matrix",
]

SlicerStyle = Literal["dropdown", "between", "list"]


class FieldBinding(BaseModel):
    """Reference to a specific column in a curated semantic table."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    column_name: str = Field(min_length=1)


class VisualFormatSpec(BaseModel):
    """Axis labels, number formatting, styling overrides, and display toggles for a visual."""

    model_config = ConfigDict(extra="forbid")

    x_axis_title: str | None = None
    y_axis_title: str | None = None
    number_format: str | None = None
    show_data_labels: bool = True
    show_legend: bool = True
    sort_descending: bool = True
    title_align: Literal["left", "center", "right"] = "left"
    title_position: Literal["top", "bottom"] = "top"
    title_font_size: int | None = Field(default=None, ge=8, le=32)
    custom_colors: list[str] | None = None
    background_color: str | None = None


class SlicerSpec(BaseModel):
    """Page-level slicer / filter control bound to a curated column."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = Field(min_length=1)
    column_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    slicer_style: SlicerStyle = "dropdown"


class DraftVisualSpec(BaseModel):
    """Visual proposed by the LLM (strictly NO pixel or grid positions)."""

    model_config = ConfigDict(extra="forbid")

    visual_type: VisualType
    title: str = Field(min_length=1)
    category_field: FieldBinding | None = None
    secondary_field: FieldBinding | None = None
    measures: list[str] = Field(default_factory=list)
    columns: list[FieldBinding] = Field(
        default_factory=list,
        description="Explicit columns for table/matrix visuals.",
    )
    format: VisualFormatSpec = Field(default_factory=VisualFormatSpec)
    alt_text: str | None = None


class DraftPageSpec(BaseModel):
    """Report page proposed by the LLM before deterministic validation and auto-layout."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    subtitle: str = Field(default="")
    slicers: list[SlicerSpec] = Field(default_factory=list)
    visuals: list[DraftVisualSpec] = Field(default_factory=list)


class DraftReportSpec(BaseModel):
    """Unpositioned report design proposed by the LLM (no pixel coordinates allowed)."""

    model_config = ConfigDict(extra="forbid")

    report_title: str = Field(min_length=1)
    pages: list[DraftPageSpec] = Field(min_length=1)
    rationale: str = Field(default="")


class GridPosition(BaseModel):
    """Deterministic 12-column layout grid coordinates."""

    model_config = ConfigDict(extra="forbid")

    col: int = Field(ge=0, le=11)
    row: int = Field(ge=0)
    col_span: int = Field(ge=1, le=12)
    row_span: int = Field(ge=1, le=12)


class PixelBounds(BaseModel):
    """Canvas pixel rectangle computed deterministically from ``GridPosition``."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ThemeSpec(BaseModel):
    """Accessible Power BI theme and colorblind-safe palette."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="Okabe-Ito Accessible")
    palette: list[str] = Field(
        default_factory=lambda: [
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#56B4E9",
            "#D55E00",
            "#CC79A7",
        ]
    )
    background_color: str = Field(default="#FFFFFF")
    surface_color: str = Field(default="#F8FAFC")
    border_color: str = Field(default="#E2E8F0")
    text_color: str = Field(default="#0F172A")
    muted_text_color: str = Field(default="#475569")
    font_family: str = Field(default="Segoe UI")


class VisualSpec(BaseModel):
    """Validated, auto-laid-out visual ready for preview rendering and PBIR export."""

    model_config = ConfigDict(extra="forbid")

    visual_id: str = Field(min_length=1)
    visual_type: VisualType
    title: str = Field(min_length=1)
    category_field: FieldBinding | None = None
    secondary_field: FieldBinding | None = None
    measures: list[str] = Field(default_factory=list)
    columns: list[FieldBinding] = Field(default_factory=list)
    format: VisualFormatSpec = Field(default_factory=VisualFormatSpec)
    grid: GridPosition
    pixel_bounds: PixelBounds
    alt_text: str = Field(min_length=1)


class ReportPageSpec(BaseModel):
    """Validated, auto-laid-out report page."""

    model_config = ConfigDict(extra="forbid")

    page_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    subtitle: str = Field(default="")
    slicers: list[SlicerSpec] = Field(default_factory=list)
    visuals: list[VisualSpec] = Field(default_factory=list)


class ReportSpec(BaseModel):
    """Final validated, themed, and auto-laid-out Power BI report specification."""

    model_config = ConfigDict(extra="forbid")

    report_title: str = Field(min_length=1)
    business_domain: str = Field(default="general_analytics")
    batch_id: str | None = None
    canvas_width: int = Field(default=1280, ge=640)
    canvas_height: int = Field(default=720, ge=360)
    grid_columns: int = Field(default=12, ge=12, le=12)
    theme: ThemeSpec = Field(default_factory=ThemeSpec)
    pages: list[ReportPageSpec] = Field(min_length=1)
    rag_source_ids: list[str] = Field(default_factory=list)

