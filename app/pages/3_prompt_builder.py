"""Page 3: Parsed user intent, suggested KPIs with rationale, design options, and confirmation."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    LOGO_PATH,
    init_session,
    load_session_state,
    render_job_progress,
    render_sidebar_shell,
    render_top_workspace_bar,
)
from core.orchestrator import THEME_CATALOG  # noqa: E402
from core.schemas.dax import SelectedMetricSpec  # noqa: E402
from core.schemas.insights import InsightsRequest  # noqa: E402
from core.schemas.orchestrator import (  # noqa: E402
    ColumnConfirmationEdit,
    DesignOptions,
    HumanConfirmPayload,
)


def main() -> None:
    """Render Page 3 — Parsed Intent, Suggested KPIs (Editable), Design Options, Gate 1 Submit."""
    st.set_page_config(
        page_title="3. KPI Builder | AutoBI Studio",
        page_icon=str(LOGO_PATH),
        layout="wide",
    )
    session_id, settings, orchestrator, worker = init_session()
    state = load_session_state(orchestrator, session_id)
    render_sidebar_shell(session_id, settings, orchestrator, state, worker)

    render_top_workspace_bar(
        "KPI Selection & Design Studio",
        "Review parsed intent, select registry-backed DAX metrics, and pick theme & ML options.",
        state,
    )

    is_running = render_job_progress(worker, session_id)
    if state.error_message:
        st.error(f"Pipeline stopped at `{state.failed_node}`: {state.error_message}")

    if state.dataset_understanding is None or state.user_intent is None:
        st.info("Complete **1. Upload** and **2. Data Profile** first.")
        st.page_link("pages/1_upload.py", label="Go to 1. Upload", icon="📤")
        return

    intent = state.user_intent
    understanding = state.dataset_understanding

    st.subheader("Parsed User Intent")
    c1, c2, c3 = st.columns(3)
    c1.metric("Business Domain", understanding.business_domain.replace("_", " ").title())
    c2.metric("Time Grain", intent.time_grain.title())
    c3.metric("Audience", intent.audience.title())
    st.markdown(f"**Goal:** {intent.goal}")

    if understanding.ambiguities:
        with st.expander("Clarifying Questions Flagged by Dataset Understanding", expanded=False):
            for amb in understanding.ambiguities:
                st.markdown(f"- **{amb.question}** — _{amb.context}_")

    st.divider()
    st.subheader("Suggested KPIs (Registry Templates Only)")
    st.caption(
        "All measures are instantiated from allowlisted DAX templates and reconciled "
        "against DuckDB SQL. Select or rename KPIs below."
    )

    proposed_specs: list[SelectedMetricSpec] = (
        list(state.proposed_metrics.selected_metrics) if state.proposed_metrics else []
    )
    if state.proposed_metrics and state.proposed_metrics.rationale:
        st.caption(f"Selection rationale: {state.proposed_metrics.rationale}")

    kpi_rows = [
        {
            "include": True,
            "measure_name": spec.measure_name,
            "template_id": spec.template_id,
            "table_name": spec.table_name,
            "description": spec.description,
        }
        for spec in proposed_specs
    ]
    edited_kpis = st.data_editor(
        pd.DataFrame(kpi_rows),
        use_container_width=True,
        hide_index=True,
        disabled=["template_id", "table_name"],
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "measure_name": st.column_config.TextColumn("Measure Name"),
            "description": st.column_config.TextColumn("Description / Rationale"),
        },
        key="kpi_selection_editor",
    )

    st.divider()
    st.subheader("Design & Analytics Options")
    opt_left, opt_right = st.columns(2)
    with opt_left:
        project_name = st.text_input(
            "Power BI Project Name",
            value=f"AutoBI_{understanding.business_domain.title()}",
            disabled=is_running,
        )
        theme_name = st.selectbox(
            "Report Theme Palette",
            options=["Auto (Best for Domain)", *THEME_CATALOG.keys()],
            index=0,
            disabled=is_running,
        )
        audience = st.selectbox(
            "Target Audience",
            options=["executives", "business analysts", "operations managers", "finance team"],
            index=0 if intent.audience.lower() == "executives" else 1,
            disabled=is_running,
        )
        source_format = st.radio(
            "PBIP Curated Data Connector",
            options=["parquet", "csv"],
            horizontal=True,
            disabled=is_running,
        )

    with opt_right:
        st.markdown("**Optional ML Insights (Curated Layer Only)**")
        enable_anomaly = st.checkbox("Run Time-Series Anomaly Detection", value=False)
        enable_forecast = st.checkbox(
            "Run Time-Series Forecasting (vs Seasonal Naive)", value=False
        )
        enable_cluster = st.checkbox("Run Dimension Clustering (KMeans)", value=False)
        run_any_insights = enable_anomaly or enable_forecast or enable_cluster

    col_confirm, col_next = st.columns([2, 1])
    if col_confirm.button(
        "Confirm & Build Semantic Model, DAX & Report Preview",
        type="primary",
        disabled=is_running,
        use_container_width=True,
    ):
        selected_specs: list[SelectedMetricSpec] = []
        for idx, row in edited_kpis.iterrows():
            if bool(row["include"]):
                base = proposed_specs[int(idx)]
                selected_specs.append(
                    base.model_copy(
                        update={
                            "measure_name": str(row["measure_name"]).strip() or base.measure_name,
                            "description": str(row["description"] or base.description),
                        }
                    )
                )

        if not selected_specs:
            st.error("Select at least one KPI before building the semantic model.")
            return

        saved_edits_raw = st.session_state.get("confirmed_column_edits", [])
        col_edits = [ColumnConfirmationEdit.model_validate(item) for item in saved_edits_raw]

        insights_req = (
            InsightsRequest(
                run_anomaly=enable_anomaly,
                run_forecasting=enable_forecast,
                run_clustering=enable_cluster,
            )
            if run_any_insights
            else None
        )
        payload = HumanConfirmPayload(
            confirmed=True,
            column_edits=col_edits,
            selected_metrics=selected_specs,
            design_options=DesignOptions(
                project_name=project_name.strip() or "AutoBI_Report",
                theme_name=theme_name,
                audience=audience,
                source_format=source_format,  # type: ignore[arg-type]
                run_insights=run_any_insights,
                insights_request=insights_req,
            ),
        )
        worker.submit(
            session_id,
            "Semantic Model + DAX Reconciliation + Insights + Report Spec",
            orchestrator.submit_human_confirmation,
            session_id,
            payload,
        )
        st.rerun()

    if state.status in {"awaiting_human_review", "completed"}:
        col_next.page_link(
            "pages/4_reports.py",
            label="Continue to 4. Report Preview & Export ->",
            icon="📊",
        )


if __name__ == "__main__":
    main()
