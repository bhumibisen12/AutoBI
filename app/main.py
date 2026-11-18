"""Streamlit entry point for AutoBI.

Business logic stays in ``core/``; this module only renders the UI shell.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logging import configure_logging, get_logger  # noqa: E402


def main() -> None:
    """Render the placeholder AutoBI Streamlit landing page."""
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger("app.main")
    logger.info("Rendering AutoBI placeholder landing page")

    st.set_page_config(
        page_title="AutoBI",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.title("AutoBI")
    st.caption(
        "Dataset + natural language to verified Power BI PBIP projects "
        "(TMDL semantic model + PBIR report)."
    )

    st.info(
        "Project scaffold initialized. Upload, profiling, prompt builder, and report "
        "pages will be wired in upcoming steps."
    )

    st.subheader("Active Configuration Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM Provider", settings.llm_provider)
    col2.metric("LLM Model", settings.llm_model)
    col3.metric("Max Upload (MB)", settings.max_upload_mb)
    col4.metric("DL Module Enabled", "Yes" if settings.enable_dl else "No")


if __name__ == "__main__":
    main()
