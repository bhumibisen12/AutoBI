"""Unified SaaS light-mode stylesheet, top bar, KPI formatters, and reusable UI widgets."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from core.orchestrator import BackgroundWorker
from core.schemas.orchestrator import NodeExecutionRecord, PipelineSessionState
from core.schemas.security import PIIReport

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.svg"
LOGO_SVG_SMALL = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
    'width="36" height="36" fill="none">'
    '<defs><linearGradient id="abg" x1="6" y1="6" x2="58" y2="58" gradientUnits="userSpaceOnUse">'
    '<stop offset="0%" stop-color="#2563EB"/><stop offset="55%" stop-color="#4F46E5"/>'
    '<stop offset="100%" stop-color="#0EA5E9"/></linearGradient></defs>'
    '<rect x="4" y="4" width="56" height="56" rx="15" fill="url(#abg)"/>'
    '<rect x="14" y="34" width="8" height="16" rx="3" fill="#FFFFFF" fill-opacity="0.78"/>'
    '<rect x="26" y="25" width="8" height="25" rx="3" fill="#FFFFFF" fill-opacity="0.92"/>'
    '<rect x="38" y="16" width="8" height="34" rx="3" fill="#FFFFFF"/>'
    '<path d="M49 10L50.7 14.3L55 16L50.7 17.7L49 22L47.3 17.7L43 16L47.3 14.3L49 10Z"'
    ' fill="#FDE047"/></svg>'
)

_SAAS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background-color: #F5F7FB !important;
    color: #0F172A !important;
}

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1480px !important;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E9F2 !important;
}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #334155;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E9F2 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04) !important;
    padding: 0.35rem 0.5rem !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    transition: all 0.15s ease-in-out !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
    transform: translateY(-1px);
}
button[kind="secondary"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #D8E0EC !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
}
button[kind="secondary"]:hover {
    background-color: #F8FAFC !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background-color: #F8FAFF !important;
    border: 1.5px dashed #93C5FD !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
}

div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
}
div[data-testid="stMetric"] label {
    color: #64748B !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-weight: 700 !important;
    font-size: 1.55rem !important;
}

button[data-baseweb="tab"] {
    font-weight: 600 !important;
    color: #64748B !important;
    border-radius: 8px 8px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #E5E9F2;
    border-radius: 10px;
    overflow: hidden;
}

details[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E9F2 !important;
    border-radius: 12px !important;
}

.autobi-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 14px;
    padding: 14px 20px;
    margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
}
.autobi-topbar-title {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
}
.autobi-topbar-sub {
    font-size: 13px;
    color: #64748B;
    margin-top: 2px;
}
.autobi-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
}
.autobi-pill-green { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.autobi-pill-amber { background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
.autobi-pill-red { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
.autobi-chip-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 14px 0; }
.autobi-col-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 10px;
    border-radius: 8px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    font-size: 12px;
    color: #1E293B;
}
.autobi-col-chip-type {
    font-size: 10px;
    color: #64748B;
    text-transform: uppercase;
    font-weight: 600;
}
.autobi-kpi-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
    margin-top: 8px;
}
.autobi-kpi-tile {
    background: #F8FAFC;
    border: 1px solid #E5E9F2;
    border-radius: 10px;
    padding: 12px 14px;
}
.autobi-kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}
.autobi-kpi-val { font-size: 20px; font-weight: 700; color: #0F172A; margin-top: 4px; }
.autobi-kpi-badge {
    display: inline-block;
    margin-top: 4px;
    font-size: 11px;
    font-weight: 600;
    color: #059669;
    background: #ECFDF5;
    padding: 2px 7px;
    border-radius: 999px;
}
.autobi-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 2px 14px 2px;
    border-bottom: 1px solid #E5E9F2;
    margin-bottom: 14px;
}
.autobi-brand-icon {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: linear-gradient(135deg, #2563EB, #3B82F6);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 15px;
}
</style>
"""


def inject_saas_theme() -> None:
    """Inject the unified clean SaaS stylesheet across every AutoBI Streamlit page."""
    st.markdown(_SAAS_CSS, unsafe_allow_html=True)


def render_top_workspace_bar(
    title: str,
    subtitle: str,
    state: PipelineSessionState,
) -> None:
    """Render the clean SaaS top workspace bar with live pipeline status pill."""
    pill_map = {
        "idle": ("Idle — Ready for data", "autobi-pill"),
        "running": ("Running pipeline...", "autobi-pill autobi-pill-amber"),
        "awaiting_human_confirm": ("Step 2/3 — Confirm Profile & KPIs", "autobi-pill"),
        "awaiting_human_review": ("Step 4 — Review & Export Report", "autobi-pill"),
        "completed": ("PBIP Ready to Download", "autobi-pill autobi-pill-green"),
        "failed": (f"Stopped at {state.failed_node or 'error'}", "autobi-pill autobi-pill-red"),
    }
    label, cls = pill_map.get(state.status, (state.status, "autobi-pill"))
    st.markdown(
        f"""
        <div class="autobi-topbar">
          <div>
            <div class="autobi-topbar-title">{html.escape(title)}</div>
            <div class="autobi-topbar-sub">{html.escape(subtitle)}</div>
          </div>
          <span class="{cls}">&bull;&nbsp;{html.escape(label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_compact_number(val: float | None, fmt: str | None = None) -> str:
    """Format KPI numbers cleanly (e.g. 14.2M, 24.6K, $576, 18.2%) for SaaS KPI tiles."""
    if val is None:
        return "—"
    if fmt and "%" in fmt:
        return f"{val * 100.0:.1f}%"
    prefix = "$" if fmt and "$" in fmt else ""
    abs_v = abs(val)
    if abs_v >= 1_000_000:
        return f"{prefix}{val / 1_000_000:.2f}M"
    if abs_v >= 10_000:
        return f"{prefix}{val / 1_000:.1f}K"
    if fmt and "." not in fmt:
        return f"{prefix}{val:,.0f}"
    return f"{prefix}{val:,.2f}"


def render_pii_warnings(pii_reports: dict[str, PIIReport]) -> None:
    """Display a prominent warning banner and table whenever PII is detected."""
    rows: list[dict[str, object]] = []
    for table_name, report in pii_reports.items():
        for finding in report.findings:
            rows.append(
                {
                    "Table": table_name,
                    "Column": finding.column,
                    "PII Type": finding.pii_type,
                    "Confidence": f"{finding.confidence:.0%}",
                    "% Rows Matched": f"{finding.row_percentage:.1f}%",
                    "Method": finding.detection_method,
                }
            )
    if not rows:
        st.success("No PII patterns detected across uploaded tables.")
        return

    st.warning(
        f"**PII Warning:** Detected {len(rows)} potential PII column(s). "
        "All sample values sent to the LLM are automatically masked, and full data "
        "remains inside your isolated session workspace."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_job_progress(worker: BackgroundWorker, session_id: str) -> bool:
    """Render live progress for an active background job. Returns ``True`` while still running."""
    job = worker.latest_for_session(session_id)
    if job is None:
        return False

    if job.status in {"queued", "running"}:
        st.info(f"**Background worker running:** {job.message}")
        st.progress(job.progress)
        col1, _ = st.columns([1, 3])
        if col1.button("Refresh status", key=f"refresh_{job.job_id}"):
            st.rerun()
        return True

    if job.status == "failed" and job.error:
        st.error(f"Background task `{job.stage}` failed: {job.error}")
    return False


def render_node_execution_log(node_logs: list[NodeExecutionRecord]) -> None:
    """Render a compact table of executed orchestrator nodes with durations and outcomes."""
    if not node_logs:
        st.caption("No orchestrator nodes have executed in this session yet.")
        return
    rows = [
        {
            "Node": rec.node,
            "Status": rec.status,
            "Duration (ms)": round(rec.duration_ms, 2),
            "Message": rec.message,
        }
        for rec in node_logs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_under_the_hood(state: PipelineSessionState) -> None:
    """Render the 'Under the hood' expander with DAX, reconciled SQL, verification, and insights."""
    with st.expander("Under the hood — DAX, DuckDB SQL, Verification & Insights", expanded=False):
        if state.verification_report is not None:
            vr = state.verification_report
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Measures", vr.total_measures)
            c2.metric("VERIFIED", vr.verified_count)
            c3.metric("UNVERIFIED", vr.unverified_count)
            c4.metric("FAILED", vr.failed_count)

            if vr.unverified_count > 0:
                st.warning(
                    "Some measures are flagged UNVERIFIED and should be checked manually "
                    "in Power BI Desktop."
                )

            rows = [
                {
                    "Measure": item.measure_name,
                    "Template": item.template_id,
                    "Status": item.status,
                    "Reconciled Value": (
                        f"{item.result:,.4f}" if item.result is not None else "N/A"
                    ),
                    "Notes": item.notes,
                }
                for item in vr.results
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("#### Side-by-Side DAX & DuckDB SQL")
            for item in vr.results:
                with st.container(border=True):
                    st.markdown(f"**{item.measure_name}** (`{item.status}`)")
                    left, right = st.columns(2)
                    left.code(item.dax, language="dax")
                    right.code(item.sql or "-- Not applicable / unverified", language="sql")
        else:
            st.caption("DAX verification report has not been generated yet.")

        if state.insights_summary is not None and state.insights_summary.modules_executed:
            st.divider()
            st.markdown("#### Grounded ML Insights & Evidence")
            st.caption(
                f"Modules executed: {', '.join(state.insights_summary.modules_executed)}"
            )
            if state.insights_summary.narrative is not None:
                st.markdown(f"**{state.insights_summary.narrative.headline}**")
                for bullet in state.insights_summary.narrative.bullet_points:
                    st.markdown(f"- {bullet}")
                for caveat in state.insights_summary.narrative.caveats:
                    st.caption(caveat)
            if state.insights_summary.grounding_check is not None:
                gc = state.insights_summary.grounding_check
                st.caption(
                    f"Number grounding check: "
                    f"{'PASSED' if gc.all_numbers_grounded else 'FAILED'} "
                    f"({len(gc.narrative_numbers)} checked number(s))"
                )
            st.json(state.insights_summary.evidence)

        st.divider()
        st.markdown("#### Orchestrator Node Execution Log")
        render_node_execution_log(state.node_logs)

