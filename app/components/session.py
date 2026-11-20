"""Streamlit session state management, caching, isolated workspaces, and 'delete my data'."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app.components.ui import LOGO_SVG_SMALL, inject_saas_theme  # noqa: E402
from core.config import Settings, get_settings  # noqa: E402
from core.logging import configure_logging  # noqa: E402
from core.orchestrator import AutoBIOrchestrator, BackgroundWorker  # noqa: E402
from core.rag import KnowledgeIndex, build_knowledge_index  # noqa: E402
from core.schemas.orchestrator import PipelineSessionState  # noqa: E402
from core.security import SessionManager  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_background_worker() -> BackgroundWorker:
    """Return a shared thread-pool ``BackgroundWorker`` cached across Streamlit reruns."""
    return BackgroundWorker(max_workers=2)


@st.cache_resource(show_spinner=False)
def get_cached_rag_index() -> KnowledgeIndex:
    """Build and cache the local RAG knowledge index once per server process."""
    return build_knowledge_index()


@st.cache_resource(show_spinner=False)
def _run_startup_retention_purge() -> int:
    """Enforce configured retention policy once on app startup."""
    settings = get_settings()
    report = SessionManager(settings).enforce_retention_policy()
    return len(report.purged_sessions)


def init_session() -> tuple[str, Settings, AutoBIOrchestrator, BackgroundWorker]:
    """Initialize isolated per-session workspace, orchestrator, and background worker."""
    inject_saas_theme()
    settings = get_settings()
    configure_logging(settings)
    _run_startup_retention_purge()

    if "autobi_session_id" not in st.session_state:
        st.session_state["autobi_session_id"] = f"sess_{uuid.uuid4().hex[:12]}"

    session_id = str(st.session_state["autobi_session_id"])
    orchestrator = AutoBIOrchestrator(
        settings=settings,
        knowledge_index=get_cached_rag_index(),
    )
    orchestrator.state_store.session_manager.create_workspace(
        session_id,
        actor="streamlit_user",
    )
    worker = get_background_worker()
    return session_id, settings, orchestrator, worker


def load_session_state(
    orchestrator: AutoBIOrchestrator,
    session_id: str,
) -> PipelineSessionState:
    """Load persisted ``PipelineSessionState`` and mirror a snapshot in ``st.session_state``."""
    state = orchestrator.get_state(session_id)
    st.session_state["autobi_pipeline_status"] = state.status
    return state


def render_sidebar_shell(
    session_id: str,
    settings: Settings,
    orchestrator: AutoBIOrchestrator,
    state: PipelineSessionState,
    worker: BackgroundWorker,
) -> None:
    """Render clean SaaS sidebar brand, workflow stepper, limits, job status, and privacy."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="autobi-brand">
              <div style="flex-shrink:0;display:flex;align-items:center;">{LOGO_SVG_SMALL}</div>
              <div>
                <div style="font-weight:800;font-size:17px;color:#0F172A;
                            letter-spacing:-0.02em;line-height:1.15;">
                  AutoBI <span style="color:#2563EB;">Studio</span>
                </div>
                <div style="font-size:11px;color:#64748B;">
                  Verified BI &amp; PBIP Platform
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Workflow**")
        steps = [
            ("1. Upload & Prompt", state.etl_result is not None),
            ("2. Data Profile", state.profile_report is not None),
            ("3. KPI & Design", state.semantic_model is not None),
            ("4. Reports & PBIP", state.report_spec is not None),
        ]
        for label, done in steps:
            icon = "✅" if done else "○"
            st.caption(f"{icon}  {label}")

        latest_job = worker.latest_for_session(session_id)
        if latest_job is not None and latest_job.status in {"queued", "running"}:
            st.warning(f"Running: {latest_job.stage}")
            st.progress(latest_job.progress)

        st.divider()
        st.markdown("**Workspace & Guardrails**")
        st.caption(f"Session: `{session_id}`")
        st.caption(
            f"LLM: `{settings.llm_provider}` · Limit: **{settings.max_upload_mb} MB** · "
            f"Retention: **{settings.retention_hours}h**"
        )

        with st.expander("Privacy & Data Deletion", expanded=False):
            confirm_delete = st.checkbox(
                "Confirm permanent deletion",
                key="sidebar_confirm_delete",
            )
            if st.button(
                "Delete my data",
                type="secondary",
                disabled=not confirm_delete,
                use_container_width=True,
            ):
                receipt = orchestrator.delete_session_data(session_id, actor="streamlit_user")
                for key in list(st.session_state.keys()):
                    if key.startswith(("autobi_", "confirmed_", "kpi_", "sidebar_")):
                        del st.session_state[key]
                st.session_state["autobi_session_id"] = f"sess_{uuid.uuid4().hex[:12]}"
                st.success(f"Deleted {receipt.files_removed} file(s).")
                st.rerun()
