"""Page 1: Upload files or connect a database, enter the analytics prompt, and inspect PII."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    LOGO_PATH,
    init_session,
    load_session_state,
    render_job_progress,
    render_pii_warnings,
    render_sidebar_shell,
    render_top_workspace_bar,
)
from core.schemas.etl import QualityGateConfig  # noqa: E402

SAMPLE_DATA_DIR = _PROJECT_ROOT / "sample_data"
SYNTHETIC_SAMPLES = {
    "Sales (SYNTHETIC_sales.csv)": SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv",
    "HR (SYNTHETIC_hr.xlsx)": SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx",
    "Healthcare (SYNTHETIC_healthcare.csv)": SAMPLE_DATA_DIR / "SYNTHETIC_healthcare.csv",
}


def main() -> None:
    """Render Page 1 — Upload, Optional Database Connection, Prompt, and PII Detection."""
    st.set_page_config(
        page_title="1. Upload | AutoBI Studio",
        page_icon=str(LOGO_PATH),
        layout="wide",
    )
    session_id, settings, orchestrator, worker = init_session()
    state = load_session_state(orchestrator, session_id)
    render_sidebar_shell(session_id, settings, orchestrator, state, worker)

    render_top_workspace_bar(
        "Datasets — Upload & Prompt",
        f"Upload CSV/Excel or connect a database · Limit: {settings.max_upload_mb} MB / "
        f"{settings.max_rows:,} rows · Isolated immutable session workspace.",
        state,
    )

    is_running = render_job_progress(worker, session_id)
    if state.error_message:
        st.error(f"Pipeline stopped at `{state.failed_node}`: {state.error_message}")

    staged_paths: list[Path] = []
    db_label: str | None = None

    tab_upload, tab_sample, tab_db = st.tabs(
        ["Upload Files", "SYNTHETIC Sample Datasets", "Database Connection (Optional)"]
    )

    with tab_upload:
        uploaded_files = st.file_uploader(
            "Select one or more CSV / Excel files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            disabled=is_running,
        )
        if uploaded_files:
            for uf in uploaded_files:
                raw_bytes = uf.getvalue()
                if len(raw_bytes) > settings.max_upload_bytes:
                    st.error(
                        f"'{uf.name}' exceeds the configured {settings.max_upload_mb} MB limit."
                    )
                    continue
                if is_running:
                    st.caption(f"Processing `{uf.name}` ({len(raw_bytes):,} bytes)...")
                    continue
                try:
                    saved = orchestrator.state_store.save_uploaded_bytes(
                        session_id,
                        uf.name,
                        raw_bytes,
                    )
                    staged_paths.append(saved)
                    st.success(f"Validated and staged `{uf.name}` ({len(raw_bytes):,} bytes)")
                except Exception as exc:
                    st.error(f"Rejected `{uf.name}`: {exc}")

    with tab_sample:
        selected_samples = st.multiselect(
            "Load bundled SYNTHETIC demo dataset(s)",
            options=list(SYNTHETIC_SAMPLES.keys()),
            default=[],
            disabled=is_running,
        )
        for label in selected_samples:
            sample_path = SYNTHETIC_SAMPLES[label]
            if sample_path.is_file():
                staged_paths.append(sample_path)
                st.caption(f"Ready: `{sample_path.name}`")

    with tab_db:
        st.caption(
            "Connect read-only to SQLite (`sqlite:///path/to/db.sqlite`) or PostgreSQL "
            "(`postgresql+psycopg2://user:pass@host:5432/db`). Passwords are never logged."
        )
        db_url = st.text_input(
            "SQLAlchemy Connection URL",
            type="password",
            key="upload_db_url",
            disabled=is_running,
        )
        db_tables_raw = st.text_input(
            "Table names (comma-separated, leave blank for all tables)",
            key="upload_db_tables",
            disabled=is_running,
        )
        if st.button("Extract Database Tables", disabled=is_running or not db_url.strip()):
            requested_tables = [t.strip() for t in db_tables_raw.split(",") if t.strip()] or None
            try:
                extracted_paths, redacted = orchestrator.state_store.stage_database_tables(
                    session_id,
                    db_url.strip(),
                    tables=requested_tables,
                )
                st.session_state["autobi_db_staged_paths"] = [str(p) for p in extracted_paths]
                st.session_state["autobi_db_label"] = redacted
                st.success(f"Extracted {len(extracted_paths)} table(s) from `{redacted}`.")
            except Exception as exc:
                st.error(f"Database extraction failed: {exc}")

        for p_str in st.session_state.get("autobi_db_staged_paths", []):
            p_obj = Path(p_str)
            if p_obj.is_file():
                staged_paths.append(p_obj)
        db_label = st.session_state.get("autobi_db_label")

    st.divider()
    user_prompt = st.text_area(
        "What would you like to analyze?",
        value=state.user_prompt
        or "Build an executive dashboard showing key KPIs, trends, and category breakdowns.",
        height=100,
        disabled=is_running,
    )

    with st.expander("Advanced Quality Gate Thresholds (Optional)", expanded=False):
        st.caption(
            "Relax Max Null % up to 100% if your dataset has intentionally sparse columns "
            "(e.g. optional phone numbers or notes)."
        )
        g1, g2, g3, g4 = st.columns(4)
        max_null_pct = g1.slider("Max Null %", 0, 100, 50, disabled=is_running)
        max_dup_pct = g2.slider("Max Duplicate %", 0, 100, 25, disabled=is_running)
        max_inv_pct = g3.slider("Max Invalid Type %", 0, 100, 20, disabled=is_running)
        max_out_pct = g4.slider("Max Outlier %", 0, 100, 30, disabled=is_running)

    col_run, col_next = st.columns([2, 1])
    if col_run.button(
        "Run ETL, Profiling & Understanding",
        type="primary",
        disabled=is_running or not staged_paths,
        use_container_width=True,
    ):
        gate_cfg = QualityGateConfig(
            max_null_pct=float(max_null_pct),
            max_duplicate_pct=float(max_dup_pct),
            max_invalid_type_pct=float(max_inv_pct),
            max_outlier_pct=float(max_out_pct),
        )
        worker.submit(
            session_id,
            "ETL + Profiling + Understanding",
            orchestrator.start_ingestion,
            session_id,
            staged_paths,
            user_prompt,
            db_source_label=db_label,
            gate_config=gate_cfg,
        )
        st.rerun()

    if state.status == "awaiting_human_confirm":
        col_next.page_link(
            "pages/2_data_profile.py",
            label="Continue to 2. Data Profile ->",
            icon="🔍",
        )

    if state.pii_reports:
        st.divider()
        st.subheader("PII Detection Results")
        render_pii_warnings(state.pii_reports)


if __name__ == "__main__":
    main()

