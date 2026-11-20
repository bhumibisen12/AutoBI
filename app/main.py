"""Streamlit entry point for AutoBI — Clean SaaS Analytics Workspace shell.

Business logic stays in ``core/``; this module only renders the SaaS workspace UI.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    LOGO_PATH,
    format_compact_number,
    init_session,
    load_session_state,
    render_node_execution_log,
    render_sidebar_shell,
    render_top_workspace_bar,
)
from core.logging import get_logger  # noqa: E402
from core.report_spec import render_visual_plotly  # noqa: E402
from core.schemas.orchestrator import HumanReviewPayload  # noqa: E402
from core.storage import read_curated_table  # noqa: E402

_ROLE_ICONS: dict[str, str] = {
    "id": "#",
    "dimension": "Aa",
    "date": "📅",
    "measure": "$",
    "text": "T",
}


def main() -> None:
    """Render the InsightIQ-style SaaS Analytics Workspace landing dashboard."""
    st.set_page_config(
        page_title="AutoBI Studio — Verified BI & PBIP Platform",
        page_icon=str(LOGO_PATH),
        layout="wide",
    )

    session_id, settings, orchestrator, worker = init_session()
    state = load_session_state(orchestrator, session_id)
    render_sidebar_shell(session_id, settings, orchestrator, state, worker)

    logger = get_logger("app.main")
    logger.info("Rendering AutoBI SaaS workspace for session %s", session_id)

    render_top_workspace_bar(
        "Workspace",
        "Upload datasets, inspect quality & lineage, preview live dashboards, and export PBIP.",
        state,
    )

    if state.error_message:
        st.error(f"Pipeline stopped at `{state.failed_node}`: {state.error_message}")

    main_col, ai_col = st.columns([2.35, 1.0], gap="large")

    # ================= LEFT / MAIN ANALYTICS CANVAS =================
    with main_col:
        # 1. Quick Action / Upload Banner Card
        with st.container(border=True):
            st.markdown(
                """
                <div style="text-align:center;padding:12px 8px 6px 8px;">
                  <div style="font-size:26px;">☁️</div>
                  <div style="font-weight:700;font-size:16px;color:#0F172A;margin-top:4px;">
                    Upload Excel, CSV or connect a database
                  </div>
                  <div style="font-size:13px;color:#64748B;margin-top:2px;">
                    Audited ETL &rarr; DuckDB-Verified DAX &rarr; Interactive Report &amp; PBIP
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            nav1, nav2, nav3, nav4 = st.columns(4)
            nav1.page_link("pages/1_upload.py", label="1. Upload Data", icon="📤")
            nav2.page_link("pages/2_data_profile.py", label="2. Data Profile", icon="🔍")
            nav3.page_link("pages/3_prompt_builder.py", label="3. KPI Builder", icon="🎯")
            nav4.page_link("pages/4_reports.py", label="4. Full Report", icon="📊")

        # 2. Dataset Summary + Detected KPIs Row
        sum_col, kpi_col = st.columns(2, gap="medium")

        with sum_col, st.container(border=True):
            st.markdown("#### Dataset Summary")
            if state.profile_report is not None and state.etl_result is not None:
                primary_tbl = state.profile_report.tables[0]
                chips_html = ""
                u_roles = (
                    {
                        (c.table_name, c.column_name): c.role
                        for c in state.dataset_understanding.columns
                    }
                    if state.dataset_understanding
                    else {}
                )
                for col in primary_tbl.columns[:8]:
                    role = u_roles.get((primary_tbl.table_name, col.column_name), "dimension")
                    icon = _ROLE_ICONS.get(role, "•")
                    chips_html += (
                        f'<span class="autobi-col-chip">'
                        f"<b>{icon}</b> {html.escape(col.column_name)} "
                        f'<span class="autobi-col-chip-type">{html.escape(role)}</span>'
                        f"</span>"
                    )
                st.markdown(
                    f'<div class="autobi-chip-grid">{chips_html}</div>',
                    unsafe_allow_html=True,
                )

                total_rows = state.etl_result.batch_log.rows_curated
                total_read = max(1, state.etl_result.batch_log.rows_read)
                rejected = state.etl_result.batch_log.rows_rejected
                quality_pct = max(0.0, (1.0 - (rejected / total_read)) * 100.0)
                m1, m2 = st.columns(2)
                m1.metric("Curated Rows", f"{total_rows:,}")
                m2.metric("Data Quality", f"{quality_pct:.1f}%")
            else:
                st.caption("No dataset loaded yet. Upload a CSV/Excel or pick a sample on Page 1.")
                m1, m2 = st.columns(2)
                m1.metric("LLM Provider", settings.llm_provider)
                m2.metric("Max Upload", f"{settings.max_upload_mb} MB")

        with kpi_col, st.container(border=True):
            st.markdown("#### Detected KPIs")
            if state.verification_report is not None and state.verification_report.results:
                measure_fmts = {m.name: m.format_string for m in state.dax_measures}
                tiles_html = ""
                for item in state.verification_report.results[:4]:
                    val_txt = format_compact_number(
                        item.result,
                        measure_fmts.get(item.measure_name),
                    )
                    tiles_html += (
                        f'<div class="autobi-kpi-tile">'
                        f'<div class="autobi-kpi-label">{html.escape(item.measure_name)}</div>'
                        f'<div class="autobi-kpi-val">{html.escape(val_txt)}</div>'
                        f'<span class="autobi-kpi-badge">{html.escape(item.status)}</span>'
                        f"</div>"
                    )
                st.markdown(
                    f'<div class="autobi-kpi-grid">{tiles_html}</div>',
                    unsafe_allow_html=True,
                )
            elif state.proposed_metrics and state.proposed_metrics.selected_metrics:
                tiles_html = "".join(
                    f'<div class="autobi-kpi-tile">'
                    f'<div class="autobi-kpi-label">{html.escape(m.measure_name)}</div>'
                    f'<div style="font-size:12px;color:#64748B;margin-top:4px;">'
                    f"{html.escape(m.template_id)}</div></div>"
                    for m in state.proposed_metrics.selected_metrics[:4]
                )
                st.markdown(
                    f'<div class="autobi-kpi-grid">{tiles_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Suggested & DuckDB-reconciled KPIs will appear here after ETL.")

        # 3. Live Dashboard Preview Card
        with st.container(border=True):
            st.markdown("#### Dashboard Preview")
            if state.report_spec is not None and state.semantic_model is not None:
                first_page = state.report_spec.pages[0]
                preview_charts = [
                    v
                    for v in first_page.visuals
                    if v.visual_type not in {"kpi_card", "table", "matrix"}
                ][:2]
                if preview_charts:
                    c_slots = st.columns(len(preview_charts))
                    for slot, vis in zip(c_slots, preview_charts, strict=False):
                        with slot:
                            try:
                                fig = render_visual_plotly(
                                    vis,
                                    state.semantic_model,
                                    state.dax_measures,
                                    theme=state.report_spec.theme,
                                )
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"home_prev_{vis.visual_id}_{len(state.chat_history)}",
                                )
                            except Exception as exc:
                                st.caption(f"Preview unavailable for {vis.title}: {exc}")
                else:
                    st.caption("Open Page 4 to inspect full report visuals.")
            else:
                st.info(
                    "Start on **1. Upload Data** to profile your dataset and generate an "
                    "executive dashboard preview."
                )

        # 4. Sample Data (Top 10) Card
        if state.etl_result is not None:
            non_gen = [t for t in state.etl_result.curated_tables if not t.is_generated]
            if non_gen and non_gen[0].curated_parquet_path is not None:
                with st.container(border=True):
                    st.markdown(f"#### Sample Data (Top 10) — `{non_gen[0].table_name}`")
                    sample_df = read_curated_table(
                        state.etl_result.curated_dir,
                        non_gen[0].table_name,
                    ).head(10)
                    st.dataframe(sample_df, use_container_width=True, hide_index=True)

    # ================= RIGHT PANEL: ASK YOUR DATA ✨ =================
    with ai_col, st.container(border=True):
        st.markdown("#### Ask your data ✨")
        if state.user_prompt:
            st.markdown(
                f'<div style="background:#F1F5F9;border-radius:10px;padding:10px 12px;'
                f'font-size:13px;color:#0F172A;margin-bottom:10px;">'
                f"{html.escape(state.user_prompt)}</div>",
                unsafe_allow_html=True,
            )

        if state.dataset_understanding is not None:
            st.caption(
                f"**AutoBI AI · Domain:** `{state.dataset_understanding.business_domain}`"
            )
            st.write(state.dataset_understanding.summary)
        else:
            st.caption(
                "Upload a dataset on Page 1 to see AI insights, live DuckDB SQL previews, "
                "and one-click report refinements here."
            )

        # SQL / DAX Preview Box matching reference right panel
        if state.verification_report is not None and state.verification_report.results:
            first_vr = next(
                (r for r in state.verification_report.results if r.sql),
                state.verification_report.results[0],
            )
            st.markdown("**SQL Preview (DuckDB Verified)**")
            st.code(first_vr.sql or first_vr.dax, language="sql")

        if state.report_spec is not None:
            act1, act2 = st.columns(2)
            act1.page_link("pages/4_reports.py", label="Open Full Report", icon="📊")
            if act2.button("Generate Power BI", type="primary", use_container_width=True):
                worker.submit(
                    session_id,
                    "PBIP (TMDL + PBIR) Generation & Packaging",
                    orchestrator.approve_and_generate_pbip,
                    session_id,
                    HumanReviewPayload(approved=True),
                )
                st.rerun()

            quick_msg = st.chat_input("Ask to refine your report...")
            if quick_msg:
                try:
                    _, patch = orchestrator.apply_chat_refinement(session_id, quick_msg)
                    st.success(patch.summary)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not apply patch: {exc}")

        with st.expander("Pipeline Activity Log", expanded=False):
            render_node_execution_log(state.node_logs)


if __name__ == "__main__":
    main()
