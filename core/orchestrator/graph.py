"""LangGraph state-machine orchestrator coordinating AutoBI from ETL to PBIP export."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
from langgraph.graph import END, START, StateGraph

from core.config import Settings, get_settings
from core.dax import generate_dax_measures, reconcile_measures
from core.etl import run_etl_pipeline
from core.insights import insights_request_from_intent, run_insights
from core.llm import (
    BaseLLMClient,
    LLMRequestError,
    MockLLMClient,
    create_llm_client,
    parse_user_intent,
    understand_dataset,
)
from core.logging import get_logger
from core.orchestrator.patcher import apply_report_patch, get_theme_by_name, propose_report_patch
from core.orchestrator.state_store import SessionStateStore
from core.pbip import build_pbip_package
from core.profiling import profile_dataset
from core.rag import KnowledgeIndex, build_knowledge_index
from core.report_spec import (
    build_default_draft_report,
    design_report,
    finalize_report_spec,
    render_report_previews_matplotlib,
)
from core.schemas.dax import MetricSelectionPlan
from core.schemas.etl import QualityGateConfig
from core.schemas.insights import InsightsRequest
from core.schemas.llm import ColumnUnderstanding, DatasetUnderstanding, UserIntent
from core.schemas.orchestrator import (
    HumanConfirmPayload,
    HumanReviewPayload,
    NodeExecutionRecord,
    PipelineSessionState,
    ReportSpecPatch,
    WorkflowNodeName,
)
from core.schemas.pbip import PBIPBuildConfig
from core.schemas.rag import RetrievedChunk
from core.schemas.security import DeletionReceipt, PIIReport
from core.schemas.semantic import SemanticModel
from core.security import AuditLogger, detect_pii
from core.semantic import (
    build_semantic_model,
    select_kpis,
    suggest_default_metric_selections,
    validate_metric_selection_plan,
)
from core.storage import read_curated_table
from core.validation import QualityGateFailedError

logger = get_logger("orchestrator")


class _GraphEnvelope(TypedDict):
    """LangGraph carrier holding the single validated ``PipelineSessionState``."""

    state: PipelineSessionState


def _load_curated_dataframes(state: PipelineSessionState) -> dict[str, pd.DataFrame]:
    """Load curated tables from Parquet for profiling, masked samples, and PII checks."""
    if state.etl_result is None:
        return {}
    curated_dir = state.etl_result.curated_dir
    frames: dict[str, pd.DataFrame] = {}
    for tbl in state.etl_result.curated_tables:
        if tbl.curated_parquet_path is not None and tbl.curated_parquet_path.is_file():
            frames[tbl.table_name] = read_curated_table(curated_dir, tbl.table_name)
    return frames


def _apply_column_edits(
    understanding: DatasetUnderstanding,
    confirm: HumanConfirmPayload,
) -> DatasetUnderstanding:
    """Apply user-confirmed column roles, friendly names, and descriptions."""
    if not confirm.column_edits:
        return understanding

    edit_map = {(e.table_name, e.column_name): e for e in confirm.column_edits}
    updated_cols: list[ColumnUnderstanding] = []
    seen: set[tuple[str, str]] = set()
    for col in understanding.columns:
        key = (col.table_name, col.column_name)
        seen.add(key)
        if key in edit_map:
            edit = edit_map[key]
            updated_cols.append(
                col.model_copy(
                    update={
                        "role": edit.role,
                        "friendly_name": edit.friendly_name,
                        "description": edit.description or col.description,
                    }
                )
            )
        else:
            updated_cols.append(col)

    for key, edit in edit_map.items():
        if key not in seen:
            updated_cols.append(
                ColumnUnderstanding(
                    table_name=edit.table_name,
                    column_name=edit.column_name,
                    role=edit.role,
                    friendly_name=edit.friendly_name,
                    description=edit.description or f"User-confirmed {edit.role} column.",
                    suggested_aggregations=(
                        ["sum", "avg"] if edit.role == "measure" else ["distinct_count"]
                    ),
                )
            )

    return understanding.model_copy(update={"columns": updated_cols})


def _pick_clusterable_dimension(
    curated_dir: Path | None,
    table_name: str,
    candidate_cols: Sequence[str],
) -> str | None:
    """Pick the first dimension column that has at least 3 distinct values for KMeans."""
    if not candidate_cols:
        return None
    if curated_dir is None:
        return candidate_cols[0]
    try:
        df = read_curated_table(curated_dir, table_name)
    except Exception:
        return candidate_cols[0]
    for col in candidate_cols:
        if col in df.columns and int(df[col].dropna().nunique()) >= 3:
            return col
    return None


def _resolve_insights_request(
    model: SemanticModel,
    intent: UserIntent,
    confirm: HumanConfirmPayload | None,
    *,
    curated_dir: Path | None = None,
) -> tuple[str, InsightsRequest] | None:
    """Resolve primary fact table and opt-in ``InsightsRequest``, or ``None`` if disabled."""
    fact_tables = [t for t in model.tables if t.kind == "fact"] or [
        t for t in model.tables if not t.is_date_table
    ]
    if not fact_tables:
        return None
    fact = fact_tables[0]

    date_col = next((c.name for c in fact.columns if c.role == "date"), None)
    measure_col = next((c.name for c in fact.columns if c.role == "measure"), None)
    dim_candidates = [c.name for c in fact.columns if c.role == "dimension"]
    id_candidates = [c.name for c in fact.columns if c.role == "id" and not c.is_hidden]
    dim_col = _pick_clusterable_dimension(
        curated_dir, fact.table_name, [*dim_candidates, *id_candidates]
    )

    explicit_req = confirm.design_options.insights_request if confirm else None
    wants_insights = bool(confirm and confirm.design_options.run_insights)

    if explicit_req is not None:
        req = explicit_req.model_copy(
            update={
                "date_column": explicit_req.date_column or date_col,
                "measure_column": explicit_req.measure_column or measure_col,
                "dimension_column": explicit_req.dimension_column or dim_col,
            }
        )
    elif wants_insights:
        req = InsightsRequest(
            run_anomaly=bool(date_col and measure_col),
            run_forecasting=bool(date_col and measure_col),
            run_clustering=bool(dim_col and measure_col),
            date_column=date_col,
            measure_column=measure_col,
            dimension_column=dim_col,
        )
    else:
        req = insights_request_from_intent(
            intent,
            date_column=date_col,
            measure_column=measure_col,
            dimension_column=dim_col,
        )

    # Guard against unavailable columns so optional ML never crashes on non-time-series tables.
    if req.run_anomaly and (not req.date_column or not req.measure_column):
        req = req.model_copy(update={"run_anomaly": False})
    if req.run_forecasting and (not req.date_column or not req.measure_column):
        req = req.model_copy(update={"run_forecasting": False})
    if req.run_clustering and not req.dimension_column:
        req = req.model_copy(update={"run_clustering": False})
    if req.run_supervised and not req.target_column:
        req = req.model_copy(update={"run_supervised": False})

    any_enabled = any(
        (
            req.run_anomaly,
            req.run_forecasting,
            req.run_supervised,
            req.run_clustering,
            req.run_dl_forecast,
        )
    )
    return (fact.table_name, req) if any_enabled else None


class AutoBIOrchestrator:
    """LangGraph-powered end-to-end AutoBI pipeline orchestrator with human-in-the-loop gates."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm_client: BaseLLMClient | None = None,
        knowledge_index: KnowledgeIndex | None = None,
        gate_config: QualityGateConfig | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize the orchestrator, session store, LLM client, RAG index, and LangGraph."""
        self._settings = settings or get_settings()
        self._audit = audit_logger or AuditLogger(settings=self._settings)
        self._store = SessionStateStore(self._settings, audit_logger=self._audit)
        self._llm_client = llm_client or create_llm_client(self._settings)
        self._knowledge_index = knowledge_index
        self._gate_config = gate_config or QualityGateConfig()
        self._graph = self._compile_graph()

    @property
    def state_store(self) -> SessionStateStore:
        """Expose the underlying per-session state store."""
        return self._store

    def _get_rag_index(self) -> KnowledgeIndex:
        """Lazily build and cache the local RAG knowledge index."""
        if self._knowledge_index is None:
            self._knowledge_index = build_knowledge_index()
        return self._knowledge_index

    def _retrieve_rag(
        self,
        query: str,
        *,
        category: str | None = None,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant knowledge chunks safely from the local RAG index."""
        try:
            index = self._get_rag_index()
            return index.retrieve(query, k=k, category=category)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("RAG retrieval unavailable (%s); continuing with registry defaults", exc)
            return []

    def _run_node(
        self,
        node_name: WorkflowNodeName,
        state: PipelineSessionState,
        fn: Callable[[PipelineSessionState], tuple[PipelineSessionState, str, str]],
    ) -> _GraphEnvelope:
        """Execute ``fn`` with timing, structured logging, error containment, and persistence."""
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()
        logger.info(
            "Orchestrator node started: %s (session=%s)",
            node_name,
            state.session_id,
        )
        try:
            updated_state, node_status, message = fn(state)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            finished_at = datetime.now(UTC)
            err_msg = str(exc)
            if isinstance(exc, QualityGateFailedError):
                failed_checks = [c.message for c in exc.report.checks if not c.passed]
                err_msg = (
                    "Quality gate failed — pipeline stopped before curated load. "
                    + ("; ".join(failed_checks) if failed_checks else str(exc))
                )
            record = NodeExecutionRecord(
                node=node_name,
                status="failed",
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                message=err_msg,
            )
            failed_state = state.model_copy(
                update={
                    "status": "failed",
                    "current_node": node_name,
                    "failed_node": node_name,
                    "error_message": err_msg,
                    "node_logs": [*state.node_logs, record],
                }
            )
            self._store.save(failed_state)
            logger.error(
                "Orchestrator node failed: %s (session=%s, duration_ms=%.2f): %s",
                node_name,
                state.session_id,
                duration_ms,
                err_msg,
            )
            return {"state": failed_state}

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        finished_at = datetime.now(UTC)
        record = NodeExecutionRecord(
            node=node_name,
            status=node_status,  # type: ignore[arg-type]
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            message=message,
        )
        final_state = updated_state.model_copy(
            update={
                "current_node": node_name,
                "node_logs": [*updated_state.node_logs, record],
            }
        )
        self._store.save(final_state)
        logger.info(
            "Orchestrator node finished: %s (session=%s, status=%s, duration_ms=%.2f)",
            node_name,
            state.session_id,
            node_status,
            duration_ms,
        )
        return {"state": final_state}

    # --- Node implementations -------------------------------------------------

    def _node_etl(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if not state.input_files:
                raise ValueError("No input files provided for ETL ingestion.")
            paths = [Path(p) for p in state.input_files]
            etl_res = run_etl_pipeline(
                paths,
                gate_config=self._gate_config,
                settings=self._settings,
                session_id=state.session_id,
                actor="orchestrator",
                audit_logger=self._audit,
            )

            # Run PII detection on every curated non-date table so the UI warns the user early.
            pii_reports: dict[str, PIIReport] = {}
            for tbl in etl_res.curated_tables:
                if (
                    tbl.is_generated
                    or tbl.curated_parquet_path is None
                    or not tbl.curated_parquet_path.is_file()
                ):
                    continue
                df = read_curated_table(etl_res.curated_dir, tbl.table_name)
                pii_reports[tbl.table_name] = detect_pii(df, table_name=tbl.table_name)

            updated = state.model_copy(
                update={
                    "status": "running",
                    "error_message": None,
                    "failed_node": None,
                    "etl_result": etl_res,
                    "pii_reports": pii_reports,
                }
            )
            msg = (
                f"Batch {etl_res.batch_id}: {len(etl_res.curated_tables)} curated table(s), "
                f"{etl_res.batch_log.rows_curated} curated rows."
            )
            return updated, "completed", msg

        return self._run_node("etl_pipeline", envelope["state"], _exec)

    def _node_profile(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if state.etl_result is None:
                raise ValueError("ETL result is required before profiling.")
            tables = {
                name: df
                for name, df in _load_curated_dataframes(state).items()
                if name != "dim_date"
            }
            report = profile_dataset(tables, batch_id=state.etl_result.batch_id)
            updated = state.model_copy(update={"profile_report": report})
            return (
                updated,
                "completed",
                f"Profiled {len(report.tables)} table(s), "
                f"{len(report.candidate_relationships)} candidate relationship(s).",
            )

        return self._run_node("profile", envelope["state"], _exec)

    def _node_understand(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if state.profile_report is None or state.etl_result is None:
                raise ValueError("ProfileReport and ETL result are required for LLM understanding.")
            sample_tables = {
                name: df
                for name, df in _load_curated_dataframes(state).items()
                if name != "dim_date"
            }
            understanding, _ = understand_dataset(
                self._llm_client,
                state.profile_report,
                sample_tables=sample_tables,
                settings=self._settings,
            )
            prompt_text = (
                state.user_prompt.strip()
                or "Create an executive BI dashboard with key KPIs, trends, and category breakdown."
            )
            intent, _ = parse_user_intent(
                self._llm_client,
                prompt_text,
                state.profile_report,
                sample_tables=sample_tables,
                settings=self._settings,
            )

            # Build preview semantic model to propose registry-backed candidate metrics for Page 3.
            preview_model = build_semantic_model(
                state.etl_result,
                dataset_understanding=understanding,
            )
            proposed = suggest_default_metric_selections(preview_model, understanding, intent)

            updated = state.model_copy(
                update={
                    "dataset_understanding": understanding,
                    "user_intent": intent,
                    "proposed_metrics": proposed,
                }
            )
            return (
                updated,
                "completed",
                f"Domain='{understanding.business_domain}', "
                f"{len(understanding.columns)} columns, "
                f"{len(proposed.selected_metrics)} suggested KPIs.",
            )

        return self._run_node("understand", envelope["state"], _exec)

    def _node_human_confirm(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if state.human_confirm is None or not state.human_confirm.confirmed:
                paused = state.model_copy(update={"status": "awaiting_human_confirm"})
                return (
                    paused,
                    "paused",
                    "Waiting for user confirmation of column roles, KPIs, and design options.",
                )
            understanding = state.dataset_understanding
            if understanding is not None:
                understanding = _apply_column_edits(understanding, state.human_confirm)
            intent = state.user_intent
            if intent is not None and state.human_confirm.design_options.audience:
                intent = intent.model_copy(
                    update={"audience": state.human_confirm.design_options.audience}
                )
            updated = state.model_copy(
                update={
                    "status": "running",
                    "dataset_understanding": understanding,
                    "user_intent": intent,
                }
            )
            return (
                updated,
                "completed",
                f"Confirmed {len(state.human_confirm.column_edits)} column edit(s).",
            )

        return self._run_node("human_confirm", envelope["state"], _exec)

    def _node_semantic_model(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if state.etl_result is None:
                raise ValueError("ETL result is required to build the SemanticModel.")
            model = build_semantic_model(
                state.etl_result,
                dataset_understanding=state.dataset_understanding,
            )
            updated = state.model_copy(update={"semantic_model": model})
            return (
                updated,
                "completed",
                f"Built SemanticModel with {len(model.tables)} table(s), "
                f"{len(model.relationships)} relationship(s), "
                f"date_table={model.date_table_name}.",
            )

        return self._run_node("semantic_model", envelope["state"], _exec)

    def _node_dax_reconciliation(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            model = state.semantic_model
            if model is None:
                raise ValueError("SemanticModel is required before DAX generation.")

            plan: MetricSelectionPlan | None = None
            if (
                state.human_confirm is not None
                and state.human_confirm.selected_metrics is not None
                and len(state.human_confirm.selected_metrics) > 0
            ):
                plan = MetricSelectionPlan(
                    selected_metrics=list(state.human_confirm.selected_metrics),
                    rationale="User-confirmed KPI selection.",
                )
                validate_metric_selection_plan(plan, model)
            else:
                rag_chunks = self._retrieve_rag(
                    f"{model.business_domain} KPI metrics DAX formulas",
                    k=5,
                )
                if (
                    state.dataset_understanding is not None
                    and state.user_intent is not None
                    and not isinstance(self._llm_client, MockLLMClient)
                ):
                    plan, _ = select_kpis(
                        self._llm_client,
                        model,
                        state.dataset_understanding,
                        state.user_intent,
                        rag_chunks,
                    )
                else:
                    try:
                        if state.dataset_understanding and state.user_intent:
                            plan, _ = select_kpis(
                                self._llm_client,
                                model,
                                state.dataset_understanding,
                                state.user_intent,
                                rag_chunks,
                            )
                    except LLMRequestError:
                        plan = None
                    if (
                        plan is None
                        and state.dataset_understanding is not None
                        and state.user_intent is not None
                    ):
                        plan = suggest_default_metric_selections(
                            model,
                            state.dataset_understanding,
                            state.user_intent,
                        )
                    elif plan is None:
                        raise ValueError("Cannot suggest default metrics without understanding.")

            measures = generate_dax_measures(plan, model)
            verification = reconcile_measures(measures, model)
            if verification.failed_count > 0:
                failed_names = [
                    r.measure_name for r in verification.results if r.status == "FAILED"
                ]
                raise ValueError(
                    f"DAX validation failed for measure(s): {', '.join(failed_names)}."
                )

            updated = state.model_copy(
                update={
                    "proposed_metrics": plan,
                    "dax_measures": measures,
                    "verification_report": verification,
                }
            )
            return (
                updated,
                "completed",
                f"Generated {len(measures)} measure(s): "
                f"{verification.verified_count} VERIFIED, "
                f"{verification.unverified_count} UNVERIFIED.",
            )

        return self._run_node("dax_reconciliation", envelope["state"], _exec)

    def _node_insights(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if (
                state.etl_result is None
                or state.semantic_model is None
                or state.user_intent is None
            ):
                return state, "skipped", "Skipped insights (missing prerequisites)."

            resolved = _resolve_insights_request(
                state.semantic_model,
                state.user_intent,
                state.human_confirm,
                curated_dir=state.etl_result.curated_dir,
            )
            if resolved is None:
                return (
                    state,
                    "skipped",
                    "Optional ML insights skipped (not requested or insufficient data shape).",
                )

            table_name, req = resolved
            llm_for_narrator = (
                None if isinstance(self._llm_client, MockLLMClient) else self._llm_client
            )
            try:
                summary = run_insights(
                    state.etl_result.curated_dir,
                    table_name,
                    req,
                    llm_client=llm_for_narrator,
                    settings=self._settings,
                    batch_id=state.etl_result.batch_id,
                )
            except ValueError as exc:
                logger.warning("Skipping optional ML insights due to data constraint: %s", exc)
                return state, "skipped", f"Skipped optional insights: {exc}"
            updated = state.model_copy(update={"insights_summary": summary})
            return (
                updated,
                "completed",
                f"Ran insight module(s): {', '.join(summary.modules_executed) or 'none'}.",
            )

        return self._run_node("insights", envelope["state"], _exec)

    def _node_report_spec(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            model = state.semantic_model
            intent = state.user_intent
            if model is None or intent is None or not state.dax_measures:
                raise ValueError("SemanticModel, UserIntent, and DAX measures are required.")

            theme_name = (
                state.human_confirm.design_options.theme_name
                if state.human_confirm is not None
                else "Auto (Best for Domain)"
            )
            theme = get_theme_by_name(theme_name, business_domain=model.business_domain)
            rag_chunks = self._retrieve_rag(
                f"Power BI visual design layout rules {intent.audience}",
                category="visual_rules",
                k=4,
            )

            report_spec = None
            if not isinstance(self._llm_client, MockLLMClient):
                report_spec, _ = design_report(
                    self._llm_client,
                    model,
                    intent,
                    state.dax_measures,
                    rag_chunks,
                    theme=theme,
                    curated_duckdb_path=model.curated_duckdb_path,
                )
            else:
                try:
                    report_spec, _ = design_report(
                        self._llm_client,
                        model,
                        intent,
                        state.dax_measures,
                        rag_chunks,
                        theme=theme,
                        curated_duckdb_path=model.curated_duckdb_path,
                    )
                except LLMRequestError:
                    report_spec = None

            if report_spec is None:
                draft = build_default_draft_report(
                    model,
                    intent,
                    state.dax_measures,
                    multi_page=True,
                )
                report_spec = finalize_report_spec(
                    draft,
                    model,
                    state.dax_measures,
                    theme=theme,
                    curated_duckdb_path=model.curated_duckdb_path,
                    rag_source_ids=[ch.source_id for ch in rag_chunks],
                )

            ws = self._store.session_manager.create_workspace(state.session_id)
            preview_dir = ws.output_dir / "previews"
            figs = render_report_previews_matplotlib(
                report_spec,
                model,
                state.dax_measures,
                output_dir=preview_dir,
            )
            for fig in figs.values():
                plt.close(fig)
            preview_paths = [preview_dir / f"{page.page_id}.png" for page in report_spec.pages]
            updated = state.model_copy(
                update={
                    "report_spec": report_spec,
                    "preview_image_paths": [str(p) for p in preview_paths if p.is_file()],
                }
            )
            return (
                updated,
                "completed",
                f"Designed {len(report_spec.pages)} page(s) and rendered "
                f"{len(preview_paths)} preview image(s).",
            )

        return self._run_node("report_spec", envelope["state"], _exec)

    def _node_human_review(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if state.human_review is None or not state.human_review.approved:
                paused = state.model_copy(update={"status": "awaiting_human_review"})
                return (
                    paused,
                    "paused",
                    "Waiting for user review and approval of the ReportSpec preview.",
                )
            updated = state.model_copy(update={"status": "running"})
            return updated, "completed", "User approved ReportSpec for PBIP generation."

        return self._run_node("human_review", envelope["state"], _exec)

    def _node_pbip_generation(self, envelope: _GraphEnvelope) -> _GraphEnvelope:
        def _exec(state: PipelineSessionState) -> tuple[PipelineSessionState, str, str]:
            if (
                state.semantic_model is None
                or state.report_spec is None
                or state.verification_report is None
            ):
                raise ValueError("SemanticModel, ReportSpec, and VerificationReport are required.")

            ws = self._store.session_manager.create_workspace(state.session_id)
            opts = state.human_confirm.design_options if state.human_confirm else None
            cfg = PBIPBuildConfig(
                project_name=opts.project_name if opts else "AutoBI_Report",
                source_format=opts.source_format if opts else "parquet",
            )
            export_res = build_pbip_package(
                state.semantic_model,
                state.dax_measures,
                state.report_spec,
                state.verification_report,
                ws.output_dir / "pbip_bundle",
                config=cfg,
            )
            updated = state.model_copy(
                update={
                    "status": "completed",
                    "pbip_export": export_res,
                }
            )
            return (
                updated,
                "completed",
                f"Exported {export_res.mode.upper()} bundle to {export_res.zip_path.name}.",
            )

        return self._run_node("pbip_generation", envelope["state"], _exec)

    # --- LangGraph routing & compilation --------------------------------------

    @staticmethod
    def _route_entry(envelope: _GraphEnvelope) -> str:
        """Resume cleanly at the exact waiting human gate when re-invoked."""
        st = envelope["state"]
        if st.status == "failed":
            return END
        if st.report_spec is not None and st.human_review is not None and st.human_review.approved:
            return "human_review"
        if (
            st.dataset_understanding is not None
            and st.human_confirm is not None
            and st.human_confirm.confirmed
            and st.report_spec is None
        ):
            return "human_confirm"
        return "etl_pipeline"

    @staticmethod
    def _continue_or_stop(next_node: str) -> Callable[[_GraphEnvelope], str]:
        def _router(envelope: _GraphEnvelope) -> str:
            st = envelope["state"]
            if st.status in {"failed", "awaiting_human_confirm", "awaiting_human_review"}:
                return END
            return next_node

        return _router

    def _compile_graph(self) -> Any:
        """Build and compile the 10-node LangGraph ``StateGraph``."""
        builder: StateGraph = StateGraph(_GraphEnvelope)
        builder.add_node("etl_pipeline", self._node_etl)
        builder.add_node("profile", self._node_profile)
        builder.add_node("understand", self._node_understand)
        builder.add_node("human_confirm", self._node_human_confirm)
        builder.add_node("semantic_model", self._node_semantic_model)
        builder.add_node("dax_reconciliation", self._node_dax_reconciliation)
        builder.add_node("insights", self._node_insights)
        builder.add_node("report_spec", self._node_report_spec)
        builder.add_node("human_review", self._node_human_review)
        builder.add_node("pbip_generation", self._node_pbip_generation)

        builder.add_conditional_edges(
            START,
            self._route_entry,
            {
                "etl_pipeline": "etl_pipeline",
                "human_confirm": "human_confirm",
                "human_review": "human_review",
                END: END,
            },
        )
        builder.add_conditional_edges(
            "etl_pipeline",
            self._continue_or_stop("profile"),
            {"profile": "profile", END: END},
        )
        builder.add_conditional_edges(
            "profile",
            self._continue_or_stop("understand"),
            {"understand": "understand", END: END},
        )
        builder.add_conditional_edges(
            "understand",
            self._continue_or_stop("human_confirm"),
            {"human_confirm": "human_confirm", END: END},
        )
        builder.add_conditional_edges(
            "human_confirm",
            self._continue_or_stop("semantic_model"),
            {"semantic_model": "semantic_model", END: END},
        )
        builder.add_conditional_edges(
            "semantic_model",
            self._continue_or_stop("dax_reconciliation"),
            {"dax_reconciliation": "dax_reconciliation", END: END},
        )
        builder.add_conditional_edges(
            "dax_reconciliation",
            self._continue_or_stop("insights"),
            {"insights": "insights", END: END},
        )
        builder.add_conditional_edges(
            "insights",
            self._continue_or_stop("report_spec"),
            {"report_spec": "report_spec", END: END},
        )
        builder.add_conditional_edges(
            "report_spec",
            self._continue_or_stop("human_review"),
            {"human_review": "human_review", END: END},
        )
        builder.add_conditional_edges(
            "human_review",
            self._continue_or_stop("pbip_generation"),
            {"pbip_generation": "pbip_generation", END: END},
        )
        builder.add_edge("pbip_generation", END)
        return builder.compile()

    # --- Public Session Workflow API ------------------------------------------

    def get_state(self, session_id: str) -> PipelineSessionState:
        """Load the latest persisted ``PipelineSessionState`` for ``session_id``."""
        return self._store.load(session_id)

    def start_ingestion(
        self,
        session_id: str,
        file_paths: Sequence[Path | str],
        user_prompt: str,
        *,
        db_source_label: str | None = None,
        gate_config: QualityGateConfig | None = None,
    ) -> PipelineSessionState:
        """Run Stage 1 (``etl_pipeline -> profile -> understand -> human_confirm [pause]``)."""
        if gate_config is not None:
            self._gate_config = gate_config
        fresh_state = PipelineSessionState(
            session_id=session_id,
            status="running",
            input_files=[str(Path(p)) for p in file_paths],
            db_source_label=db_source_label,
            user_prompt=user_prompt,
        )
        self._store.save(fresh_state)
        result = self._graph.invoke({"state": fresh_state})
        return result["state"]

    def submit_human_confirmation(
        self,
        session_id: str,
        confirm_payload: HumanConfirmPayload,
    ) -> PipelineSessionState:
        """Submit Gate 1 confirmation and run through ``report_spec -> human_review [pause]``."""
        state = self._store.load(session_id)
        if state.etl_result is None or state.dataset_understanding is None:
            raise ValueError("Cannot confirm columns/KPIs before ingestion and understanding run.")
        updated = state.model_copy(
            update={
                "status": "running",
                "human_confirm": confirm_payload,
                "report_spec": None,
                "human_review": None,
                "pbip_export": None,
                "error_message": None,
                "failed_node": None,
            }
        )
        self._store.save(updated)
        result = self._graph.invoke({"state": updated})
        return result["state"]

    def apply_chat_refinement(
        self,
        session_id: str,
        user_message: str,
    ) -> tuple[PipelineSessionState, ReportSpecPatch]:
        """Translate a chat message into a validated ``ReportSpecPatch``, apply, and re-preview."""
        state = self._store.load(session_id)
        if state.report_spec is None or state.semantic_model is None:
            raise ValueError("ReportSpec must be generated before applying chat refinements.")

        patch = propose_report_patch(
            user_message,
            state.report_spec,
            state.semantic_model,
            measures=state.dax_measures,
            llm_client=self._llm_client,
        )
        new_spec = apply_report_patch(
            state.report_spec,
            patch,
            state.semantic_model,
            state.dax_measures,
        )
        ws = self._store.session_manager.create_workspace(session_id)
        preview_dir = ws.output_dir / "previews"
        figs = render_report_previews_matplotlib(
            new_spec,
            state.semantic_model,
            state.dax_measures,
            output_dir=preview_dir,
        )
        for fig in figs.values():
            plt.close(fig)
        preview_paths = [preview_dir / f"{page.page_id}.png" for page in new_spec.pages]
        chat_entries = [
            *state.chat_history,
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": patch.summary},
        ]
        updated = state.model_copy(
            update={
                "report_spec": new_spec,
                "preview_image_paths": [str(p) for p in preview_paths if p.is_file()],
                "chat_history": chat_entries,
            }
        )
        self._store.save(updated)
        return updated, patch

    def approve_and_generate_pbip(
        self,
        session_id: str,
        review_payload: HumanReviewPayload | None = None,
    ) -> PipelineSessionState:
        """Submit Gate 2 approval and execute ``human_review -> pbip_generation``."""
        state = self._store.load(session_id)
        if state.report_spec is None:
            raise ValueError("ReportSpec must be built before PBIP generation.")
        review = review_payload or HumanReviewPayload(approved=True)
        updated = state.model_copy(
            update={
                "status": "running",
                "human_review": review,
                "error_message": None,
                "failed_node": None,
            }
        )
        self._store.save(updated)
        result = self._graph.invoke({"state": updated})
        return result["state"]

    def delete_session_data(self, session_id: str, *, actor: str = "user") -> DeletionReceipt:
        """Delete all files, curated tables, outputs, and state for ``session_id``."""
        return self._store.delete_session(session_id, actor=actor)
