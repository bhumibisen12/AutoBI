"""Page 4: SaaS Executive Report Studio — Live Dashboard + Ask Your Data + Exports."""

from __future__ import annotations

import html
import json
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
    render_sidebar_shell,
    render_top_workspace_bar,
    render_under_the_hood,
)
from core.orchestrator import THEME_CATALOG  # noqa: E402
from core.report_spec import (  # noqa: E402
    export_interactive_html_report,
    export_report_pdf_bytes,
    render_visual_plotly,
)
from core.schemas.orchestrator import HumanReviewPayload  # noqa: E402


def main() -> None:
    """Render Page 4 — Side-by-Side SaaS Report Studio, Ask Your Data Chat, and Exports."""
    st.set_page_config(
        page_title="4. Reports & Export | AutoBI Studio",
        page_icon=str(LOGO_PATH),
        layout="wide",
    )
    session_id, settings, orchestrator, worker = init_session()
    state = load_session_state(orchestrator, session_id)
    render_sidebar_shell(session_id, settings, orchestrator, state, worker)

    render_top_workspace_bar(
        "Executive Report Studio",
        "Inspect live visuals, refine via natural chat, and export HTML, PDF, or Power BI PBIP.",
        state,
    )

    is_running = render_job_progress(worker, session_id)
    if state.error_message:
        st.error(f"Pipeline stopped at `{state.failed_node}`: {state.error_message}")

    if state.report_spec is None or state.semantic_model is None:
        st.info("Confirm columns and KPIs on **3. Prompt & KPI Builder** first.")
        st.page_link("pages/3_prompt_builder.py", label="Go to 3. Prompt Builder", icon="🎯")
        return

    report_spec = state.report_spec
    model = state.semantic_model
    theme = report_spec.theme
    safe_slug = (
        "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in report_spec.report_title
        ).strip("_")
        or "AutoBI_Report"
    )

    # Split into Main Dashboard Canvas (left) + Ask Your Data ✨ Copilot Panel (right)
    dash_col, chat_col = st.columns([2.5, 1.0], gap="large")

    # ================= LEFT: LIVE EXECUTIVE REPORT STUDIO =================
    with dash_col:
        hdr_left, hdr_right = st.columns([3.2, 1.1])
        with hdr_left:
            st.markdown(
                f"""
                <div style="background:{theme.palette[0]};padding:16px 22px;border-radius:14px;
                            margin-bottom:14px;box-shadow:0 3px 10px rgba(15,23,42,0.08);">
                  <div style="color:#FFFFFF;font-size:22px;font-weight:700;">
                    {html.escape(report_spec.report_title)}
                  </div>
                  <div style="color:#E2E8F0;font-size:12.5px;margin-top:4px;">
                    Theme: <b>{html.escape(theme.name)}</b> &nbsp;&bull;&nbsp;
                    Pages: <b>{len(report_spec.pages)}</b> &nbsp;&bull;&nbsp;
                    Verified Measures: <b>{len(state.dax_measures)}</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hdr_right:
            theme_names = list(THEME_CATALOG.keys())
            cur_idx = theme_names.index(theme.name) if theme.name in theme_names else 0
            chosen_theme = st.selectbox(
                "Quick Theme",
                options=theme_names,
                index=cur_idx,
                disabled=is_running,
                key="reports_quick_theme",
            )
            if chosen_theme != theme.name and st.button(
                "Apply Theme", use_container_width=True, disabled=is_running
            ):
                orchestrator.apply_chat_refinement(session_id, f"switch theme to {chosen_theme}")
                st.rerun()

        tab_interactive, tab_layout = st.tabs(
            ["Interactive Executive Dashboard", "Full-Page Print Preview"]
        )

        with tab_interactive:
            page_tabs = st.tabs(
                [f"{i}. {p.title}" for i, p in enumerate(report_spec.pages, start=1)]
            )
            render_ver = f"{len(state.chat_history)}_{theme.name}"
            for page_tab, page in zip(page_tabs, report_spec.pages, strict=False):
                with page_tab:
                    if page.subtitle:
                        st.caption(page.subtitle)
                    if page.slicers:
                        slicer_labels = "  •  ".join(s.title for s in page.slicers)
                        st.caption(f"Configured Power BI Slicers: {slicer_labels}")

                    kpi_visuals = [v for v in page.visuals if v.visual_type == "kpi_card"]
                    table_visuals = [
                        v for v in page.visuals if v.visual_type in {"table", "matrix"}
                    ]
                    chart_visuals = [
                        v
                        for v in page.visuals
                        if v.visual_type not in {"kpi_card", "table", "matrix"}
                    ]

                    if kpi_visuals:
                        kpi_cols = st.columns(len(kpi_visuals))
                        for col_slot, visual in zip(kpi_cols, kpi_visuals, strict=False):
                            with col_slot, st.container(border=True):
                                try:
                                    fig = render_visual_plotly(
                                        visual, model, state.dax_measures, theme=theme
                                    )
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"kpi_{page.page_id}_{visual.visual_id}_{render_ver}",
                                    )
                                except Exception as exc:
                                    st.warning(f"Could not render `{visual.title}`: {exc}")

                    if chart_visuals:
                        cols = st.columns(2)
                        for idx, visual in enumerate(chart_visuals):
                            with cols[idx % 2], st.container(border=True):
                                try:
                                    fig = render_visual_plotly(
                                        visual, model, state.dax_measures, theme=theme
                                    )
                                    st.plotly_chart(
                                        fig,
                                        use_container_width=True,
                                        key=f"ch_{page.page_id}_{visual.visual_id}_{render_ver}",
                                    )
                                except Exception as exc:
                                    st.warning(f"Could not render `{visual.title}`: {exc}")

                    for visual in table_visuals:
                        with st.container(border=True):
                            try:
                                fig = render_visual_plotly(
                                    visual, model, state.dax_measures, theme=theme
                                )
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"tbl_{page.page_id}_{visual.visual_id}_{render_ver}",
                                )
                            except Exception as exc:
                                st.warning(f"Could not render `{visual.title}`: {exc}")

        with tab_layout:
            shown = 0
            for img_path_str in state.preview_image_paths:
                img_path = Path(img_path_str)
                if img_path.is_file():
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
                    shown += 1
            if shown == 0:
                st.caption("No rendered page preview PNGs found.")

        st.divider()
        render_under_the_hood(state)

    # ================= RIGHT: ASK YOUR DATA ✨ + SQL PREVIEW + EXPORTS =================
    with chat_col:
        with st.container(border=True):
            st.markdown("#### Ask your data ✨")
            st.caption(
                "Refine visuals live: `fix title below centre for all visuals`, "
                "`only change pie chart color to black and white`, `use pink theme`, "
                "`add pie chart`, `all charts in one page`."
            )

            for msg in state.chat_history[-6:]:
                with st.chat_message(msg.get("role", "assistant")):
                    st.write(msg.get("content", ""))

            chat_prompt = st.chat_input(
                "Ask a refinement for this report...",
                disabled=is_running,
            )
            if chat_prompt:
                try:
                    _, patch = orchestrator.apply_chat_refinement(session_id, chat_prompt)
                    st.success(patch.summary)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Refinement patch rejected: {exc}")

        with st.container(border=True):
            st.markdown("#### SQL Preview")
            if state.verification_report is not None and state.verification_report.results:
                first_vr = next(
                    (r for r in state.verification_report.results if r.sql),
                    state.verification_report.results[0],
                )
                st.code(first_vr.sql or first_vr.dax, language="sql")
            else:
                st.caption("No SQL reconciliation queries recorded.")

        with st.container(border=True):
            st.markdown("#### Export & Download")
            html_bytes = export_interactive_html_report(
                report_spec, model, state.dax_measures
            ).encode("utf-8")
            st.download_button(
                "Interactive Report (.html)",
                data=html_bytes,
                file_name=f"{safe_slug}.html",
                mime="text/html",
                use_container_width=True,
            )

            pdf_bytes = export_report_pdf_bytes(report_spec, model, state.dax_measures)
            st.download_button(
                "Executive Report (.pdf)",
                data=pdf_bytes,
                file_name=f"{safe_slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.divider()
            if st.button(
                "Generate Power BI Report (.pbip)",
                type="primary",
                disabled=is_running,
                use_container_width=True,
            ):
                worker.submit(
                    session_id,
                    "PBIP (TMDL + PBIR) Generation & Packaging",
                    orchestrator.approve_and_generate_pbip,
                    session_id,
                    HumanReviewPayload(approved=True),
                )
                st.rerun()

            if state.pbip_export is not None:
                exp = state.pbip_export
                if exp.mode == "pbip":
                    st.success("PBIP validated offline (open checklist inside README).")
                else:
                    st.warning(f"Fallback export generated ({exp.fallback_reason}).")

                if exp.large_data_warning:
                    st.warning(exp.large_data_warning)

                if exp.zip_path.is_file():
                    st.download_button(
                        "PBIP Bundle (.zip)",
                        data=exp.zip_path.read_bytes(),
                        file_name=exp.zip_path.name,
                        mime="application/zip",
                        use_container_width=True,
                    )

                measures_dax_path = exp.output_dir / "measures.dax"
                data_dict_path = exp.output_dir / "data_dictionary.xlsx"
                verification_md_path = exp.output_dir / "verification_report.md"

                if measures_dax_path.is_file():
                    st.download_button(
                        "measures.dax",
                        data=measures_dax_path.read_bytes(),
                        file_name="measures.dax",
                        mime="text/plain",
                        use_container_width=True,
                    )
                if data_dict_path.is_file():
                    st.download_button(
                        "Data Dictionary (.xlsx)",
                        data=data_dict_path.read_bytes(),
                        file_name="data_dictionary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                if verification_md_path.is_file():
                    st.download_button(
                        "Verification Report (.md)",
                        data=verification_md_path.read_bytes(),
                        file_name="verification_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                if state.insights_summary is not None:
                    insights_json = json.dumps(
                        state.insights_summary.model_dump(mode="json"),
                        indent=2,
                    ).encode("utf-8")
                    st.download_button(
                        "Insights (.json)",
                        data=insights_json,
                        file_name="insights_summary.json",
                        mime="application/json",
                        use_container_width=True,
                    )


if __name__ == "__main__":
    main()
