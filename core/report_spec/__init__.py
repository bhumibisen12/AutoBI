"""ReportSpec design, validation, deterministic 12-column layout, and curated-data previews."""

from core.report_spec.designer import (
    build_default_draft_report,
    build_report_design_prompt,
    design_report,
    propose_report_spec,
)
from core.report_spec.layout import build_alt_text, finalize_report_spec
from core.report_spec.preview import (
    export_interactive_html_report,
    export_report_pdf_bytes,
    render_report_previews_matplotlib,
    render_visual_plotly,
)
from core.report_spec.validator import (
    ReportSpecValidationError,
    validate_draft_report_spec,
    validate_no_overlaps,
)

__all__ = [
    "ReportSpecValidationError",
    "build_alt_text",
    "build_default_draft_report",
    "build_report_design_prompt",
    "design_report",
    "export_interactive_html_report",
    "export_report_pdf_bytes",
    "finalize_report_spec",
    "propose_report_spec",
    "render_report_previews_matplotlib",
    "render_visual_plotly",
    "validate_draft_report_spec",
    "validate_no_overlaps",
]

