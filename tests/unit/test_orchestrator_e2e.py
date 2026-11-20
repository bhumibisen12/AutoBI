"""End-to-end tests for Step 10: LangGraph orchestrator with MockLLM on all sample datasets."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from core.config import Settings
from core.llm import MockLLM
from core.orchestrator import AutoBIOrchestrator, BackgroundWorker
from core.rag import build_knowledge_index
from core.schemas import (
    ColumnConfirmationEdit,
    DesignOptions,
    HumanConfirmPayload,
    HumanReviewPayload,
    QualityGateConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

ALL_SAMPLE_DATASETS = [
    (
        "sales",
        SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv",
        "Build an executive sales dashboard with revenue trends and category breakdown.",
    ),
    (
        "hr",
        SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx",
        "Analyze headcount, salaries, and attrition across departments.",
    ),
    (
        "healthcare",
        SAMPLE_DATA_DIR / "SYNTHETIC_healthcare.csv",
        "Track patient visits, billed amounts, and length of stay over time.",
    ),
]


def _make_settings(tmp_path: Path) -> Settings:
    """Create isolated Settings rooted under pytest's temporary directory."""
    return Settings(
        _env_file=None,
        llm_provider="mock",
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
    )


@pytest.mark.slow
@pytest.mark.parametrize(("dataset_label", "dataset_path", "prompt"), ALL_SAMPLE_DATASETS)
def test_orchestrator_end_to_end_on_all_sample_datasets(
    tmp_path: Path,
    dataset_label: str,
    dataset_path: Path,
    prompt: str,
) -> None:
    """Full 10-node LangGraph run with MockLLM, human gates, chat patch, PBIP zip, and delete."""
    settings = _make_settings(tmp_path)
    rag_index = build_knowledge_index()
    orch = AutoBIOrchestrator(
        settings=settings,
        llm_client=MockLLM(),
        knowledge_index=rag_index,
    )
    session_id = f"sess_e2e_{dataset_label}"

    # Stage 1: etl_pipeline -> profile -> understand -> human_confirm (pause)
    s1 = orch.start_ingestion(session_id, [dataset_path], prompt)
    assert s1.status == "awaiting_human_confirm", s1.error_message
    assert s1.etl_result is not None
    assert s1.profile_report is not None
    assert s1.dataset_understanding is not None
    assert s1.user_intent is not None
    assert s1.proposed_metrics is not None
    assert len(s1.proposed_metrics.selected_metrics) >= 1
    assert orch.state_store.state_path(session_id).is_file()

    # Gate 1: edit first column friendly name & submit confirmation
    first_col = s1.dataset_understanding.columns[0]
    confirm_payload = HumanConfirmPayload(
        confirmed=True,
        column_edits=[
            ColumnConfirmationEdit(
                table_name=first_col.table_name,
                column_name=first_col.column_name,
                role=first_col.role,
                friendly_name=f"{first_col.friendly_name} (Confirmed)",
                description="Confirmed by user in Step 2.",
            )
        ],
        selected_metrics=list(s1.proposed_metrics.selected_metrics),
        design_options=DesignOptions(
            project_name=f"AutoBI_{dataset_label}",
            theme_name="AutoBI Default",
            audience="executives",
            source_format="parquet",
            run_insights=(dataset_label == "sales"),
        ),
    )

    # Stage 2: human_confirm -> semantic_model -> dax_reconciliation -> insights -> report_spec
    # -> human_review (pause)
    s2 = orch.submit_human_confirmation(session_id, confirm_payload)
    assert s2.status == "awaiting_human_review", s2.error_message
    assert s2.semantic_model is not None
    assert len(s2.dax_measures) >= 1
    assert s2.verification_report is not None
    assert s2.verification_report.failed_count == 0
    assert s2.report_spec is not None
    assert len(s2.preview_image_paths) >= 1
    for p_str in s2.preview_image_paths:
        assert Path(p_str).is_file()
    if dataset_label == "sales":
        assert s2.insights_summary is not None
        assert len(s2.insights_summary.modules_executed) >= 1

    # Verify confirmed column friendly name carried into SemanticModel
    confirmed_tbl = s2.semantic_model.get_table(first_col.table_name)
    assert confirmed_tbl is not None
    confirmed_col = confirmed_tbl.get_column(first_col.column_name)
    assert confirmed_col is not None
    assert confirmed_col.friendly_name.endswith("(Confirmed)")

    # Chat refinement patch before approval
    s2_patched, patch = orch.apply_chat_refinement(
        session_id,
        "Change title to Board Ready View and switch theme to Executive Navy",
    )
    assert len(patch.operations) == 2
    assert s2_patched.report_spec is not None
    assert s2_patched.report_spec.report_title == "Board Ready View"
    assert s2_patched.report_spec.theme.name == "Executive Navy"

    # Stage 3: human_review -> pbip_generation -> completed
    s3 = orch.approve_and_generate_pbip(session_id, HumanReviewPayload(approved=True))
    assert s3.status == "completed", s3.error_message
    assert s3.pbip_export is not None
    assert s3.pbip_export.mode == "pbip"
    assert s3.pbip_export.zip_path.is_file()

    with zipfile.ZipFile(s3.pbip_export.zip_path, "r") as zf:
        names = set(zf.namelist())
    prefix = f"AutoBI_{dataset_label}"
    assert f"{prefix}/pbip/{prefix}.pbip" in names
    assert f"{prefix}/measures.dax" in names
    assert f"{prefix}/data_dictionary.xlsx" in names
    assert f"{prefix}/verification_report.md" in names
    assert f"{prefix}/README.md" in names

    # Verify every node logged with duration_ms >= 0
    logged_nodes = {rec.node for rec in s3.node_logs}
    expected_nodes = {
        "etl_pipeline",
        "profile",
        "understand",
        "human_confirm",
        "semantic_model",
        "dax_reconciliation",
        "insights",
        "report_spec",
        "human_review",
        "pbip_generation",
    }
    assert expected_nodes <= logged_nodes
    assert all(rec.duration_ms >= 0.0 for rec in s3.node_logs)

    # Verify "Delete my data" cleans up the isolated session workspace completely
    receipt = orch.delete_session_data(session_id)
    assert receipt.files_removed > 0
    assert not (settings.data_dir / "sessions" / session_id).exists()
    assert not (settings.output_dir / "sessions" / session_id).exists()


def test_orchestrator_stops_cleanly_on_quality_gate_failure(tmp_path: Path) -> None:
    """A failed ETL quality gate stops the graph immediately with a clear error message."""
    dirty_csv = tmp_path / "dirty_nulls.csv"
    pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3", "O4"],
            "amount": [10.0, None, None, None],
        }
    ).to_csv(dirty_csv, index=False)

    orch = AutoBIOrchestrator(
        settings=_make_settings(tmp_path),
        llm_client=MockLLM(),
        gate_config=QualityGateConfig(max_null_pct=0.10),
    )
    state = orch.start_ingestion("sess_gate_fail", [dirty_csv], "Show total amount")
    assert state.status == "failed"
    assert state.failed_node == "etl_pipeline"
    assert state.error_message is not None
    assert "Quality gate failed" in state.error_message
    assert state.profile_report is None
    assert [r.node for r in state.node_logs] == ["etl_pipeline"]


def test_background_worker_and_sqlite_staging(tmp_path: Path) -> None:
    """BackgroundWorker executes Stage 1 off-thread and SQLite staging redacts connection info."""
    db_path = tmp_path / "demo.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE orders (order_id TEXT, region TEXT, revenue REAL)")
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [("O1", "North", 120.0), ("O2", "South", 250.0), ("O3", "North", 180.0)],
        )
        conn.commit()

    settings = _make_settings(tmp_path)
    orch = AutoBIOrchestrator(
        settings=settings,
        llm_client=MockLLM(),
        knowledge_index=build_knowledge_index(),
    )
    session_id = "sess_bg_sqlite"
    staged_paths, redacted = orch.state_store.stage_database_tables(
        session_id,
        f"sqlite:///{db_path.as_posix()}",
        tables=["orders"],
    )
    assert len(staged_paths) == 1
    assert "sqlite:///" in redacted

    worker = BackgroundWorker(max_workers=1)
    try:
        job = worker.submit(
            session_id,
            "ETL + Understanding",
            orch.start_ingestion,
            session_id,
            staged_paths,
            "Analyze regional revenue",
            db_source_label=redacted,
        )
        final_job = worker.wait(job.job_id, timeout=30.0)
        assert final_job.status == "completed"
        assert final_job.progress == 1.0
        persisted = orch.get_state(session_id)
        assert persisted.status == "awaiting_human_confirm"
        assert persisted.db_source_label == redacted
    finally:
        worker.shutdown(wait=True)


def test_session_state_store_concurrent_read_write_is_windows_safe(tmp_path: Path) -> None:
    """Simultaneous load/save across UI and worker threads never raises WinError 5."""
    from concurrent.futures import ThreadPoolExecutor

    from core.orchestrator import SessionStateStore
    from core.schemas import PipelineSessionState

    store = SessionStateStore(_make_settings(tmp_path))
    sid = "sess_concurrent"
    store.save(PipelineSessionState(session_id=sid, status="running"))

    def _writer(idx: int) -> None:
        for step in range(15):
            store.save(
                PipelineSessionState(
                    session_id=sid,
                    status="running",
                    user_prompt=f"writer-{idx}-step-{step}",
                )
            )

    def _reader() -> None:
        for _ in range(30):
            loaded = store.load(sid)
            assert loaded.session_id == sid

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_writer, 1), pool.submit(_writer, 2), pool.submit(_reader)]
        for fut in futs:
            fut.result(timeout=15.0)


def test_column_names_with_spaces_work_end_to_end_and_block_sql_injection(
    tmp_path: Path,
) -> None:
    """Real CSV headers with spaces (e.g. 'Heart Disease') work across DAX, preview & PBIP."""
    from core.report_spec.preview import _qident

    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        _qident('Heart"; DROP TABLE x--')

    csv_path = tmp_path / "heart.csv"
    pd.DataFrame(
        {
            "Patient ID": ["P1", "P2", "P3", "P4"],
            "Heart Disease": ["Presence", "Absence", "Presence", "Absence"],
            "Max HR": [150, 172, 138, 165],
            "Cholesterol (mg/dL)": [240.0, 198.0, 265.0, 210.0],
        }
    ).to_csv(csv_path, index=False)

    orch = AutoBIOrchestrator(
        settings=_make_settings(tmp_path),
        llm_client=MockLLM(),
        knowledge_index=build_knowledge_index(),
    )
    sid = "sess_heart_spaces"
    s1 = orch.start_ingestion(sid, [csv_path], "Analyze Max HR and Cholesterol by Heart Disease")
    assert s1.status == "awaiting_human_confirm", s1.error_message
    s2 = orch.submit_human_confirmation(
        sid,
        HumanConfirmPayload(
            confirmed=True,
            selected_metrics=(
                list(s1.proposed_metrics.selected_metrics) if s1.proposed_metrics else None
            ),
            design_options=DesignOptions(project_name="Heart_Analysis"),
        ),
    )
    assert s2.status == "awaiting_human_review", s2.error_message
    assert s2.verification_report is not None
    assert s2.verification_report.verified_count >= 1
    assert s2.report_spec is not None
    # AutoBI automatically acts like a senior analyst: detects healthcare domain,
    # picks clinical Emerald Slate palette, uses AVERAGE on clinical metrics, and auto-adds
    # a donut share chart for low-cardinality 'Heart Disease' + scatter deep dive.
    assert s2.semantic_model is not None
    assert s2.semantic_model.business_domain == "healthcare"
    assert s2.report_spec.theme.name == "Emerald Slate"
    auto_vtypes = {v.visual_type for p in s2.report_spec.pages for v in p.visuals}
    assert "donut_chart" in auto_vtypes
    assert "scatter_chart" in auto_vtypes
    measure_names = [m.name for m in s2.dax_measures]
    assert any(name.startswith("Average ") for name in measure_names)
    assert len(s2.report_spec.pages) == 2
    assert len(s2.preview_image_paths) == 2

    # Natural chat refinements: pink theme + add pie chart both work cleanly offline.
    s2_pink, pink_patch = orch.apply_chat_refinement(
        sid,
        "make a professional report using pink color",
    )
    assert len(pink_patch.operations) >= 1
    assert s2_pink.report_spec is not None
    assert s2_pink.report_spec.theme.name == "Rose Pink"

    s2_pie, pie_patch = orch.apply_chat_refinement(sid, "add pie chart")
    assert any(op.op == "add_visual" for op in pie_patch.operations)
    assert s2_pie.report_spec is not None
    all_vtypes = {v.visual_type for p in s2_pie.report_spec.pages for v in p.visuals}
    assert "pie_chart" in all_vtypes

    s2_one, one_patch = orch.apply_chat_refinement(sid, "i want all things or charts in ine page")
    assert any(op.op == "merge_to_single_page" for op in one_patch.operations)
    assert s2_one.report_spec is not None
    assert len(s2_one.report_spec.pages) == 1
    assert len(s2_one.report_spec.pages[0].visuals) <= 8

    # Per-visual & all-chart styling commands requested via natural chat:
    s2_ctr, ctr_patch = orch.apply_chat_refinement(
        sid, "fix the chart title in the just below centre of the chart this is for all visuals"
    )
    assert any(op.op == "update_visual_style" for op in ctr_patch.operations)
    assert s2_ctr.report_spec is not None
    assert all(
        v.format.title_align == "center" and v.format.title_position == "bottom"
        for p in s2_ctr.report_spec.pages
        for v in p.visuals
    )
    from core.report_spec import (
        export_interactive_html_report,
        export_report_pdf_bytes,
        render_visual_plotly,
    )

    non_kpi = next(v for v in s2_ctr.report_spec.pages[0].visuals if v.visual_type != "kpi_card")
    pfig = render_visual_plotly(non_kpi, s2_ctr.semantic_model, s2_ctr.dax_measures)  # type: ignore[arg-type]
    assert pfig.layout.title.x == 0.5
    assert pfig.layout.title.y == 0.03
    assert pfig.layout.title.yanchor == "bottom"

    html_doc = export_interactive_html_report(
        s2_ctr.report_spec, s2_ctr.semantic_model, s2_ctr.dax_measures  # type: ignore[arg-type]
    )
    import html

    assert "<!DOCTYPE html>" in html_doc
    assert html.escape(s2_ctr.report_spec.report_title) in html_doc
    pdf_bytes = export_report_pdf_bytes(
        s2_ctr.report_spec, s2_ctr.semantic_model, s2_ctr.dax_measures  # type: ignore[arg-type]
    )
    assert pdf_bytes.startswith(b"%PDF-")

    prev_theme = s2_ctr.report_spec.theme.name
    s2_bw, bw_patch = orch.apply_chat_refinement(
        sid, "only change the color of pie chart to black and white"
    )
    assert any(op.op == "update_visual_style" for op in bw_patch.operations)
    assert s2_bw.report_spec is not None
    assert s2_bw.report_spec.theme.name == prev_theme
    pie_visuals = [
        v for p in s2_bw.report_spec.pages for v in p.visuals if v.visual_type == "pie_chart"
    ]
    styled_targets = pie_visuals or [
        v for p in s2_bw.report_spec.pages for v in p.visuals if v.visual_type == "donut_chart"
    ]
    assert len(styled_targets) >= 1
    for v in styled_targets:
        assert v.format.custom_colors is not None
        assert v.format.custom_colors[:2] == ["#111827", "#FFFFFF"]
    target_ids = {v.visual_id for v in styled_targets}
    for p in s2_bw.report_spec.pages:
        for v in p.visuals:
            if v.visual_id not in target_ids:
                assert v.format.custom_colors is None

    s3 = orch.approve_and_generate_pbip(sid, HumanReviewPayload(approved=True))
    assert s3.status == "completed", s3.error_message
    assert s3.pbip_export is not None and s3.pbip_export.mode == "pbip"


@pytest.mark.slow
def test_streamlit_multipage_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All Streamlit pages render with zero exceptions on both empty and populated sessions."""
    from streamlit.testing.v1 import AppTest

    from core.config import clear_settings_cache

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    clear_settings_cache()
    try:
        # Pre-populate a full completed session (including insights + PBIP) so every page
        # renders its populated UI branches (editors, Plotly charts, Under the hood, downloads).
        settings = _make_settings(tmp_path)
        orch = AutoBIOrchestrator(
            settings=settings,
            llm_client=MockLLM(),
            knowledge_index=build_knowledge_index(),
        )
        sid = "sess_ui_populated"
        s1 = orch.start_ingestion(
            sid,
            [SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv"],
            "Executive sales overview",
        )
        assert s1.status == "awaiting_human_confirm"
        s2 = orch.submit_human_confirmation(
            sid,
            HumanConfirmPayload(
                confirmed=True,
                selected_metrics=(
                    list(s1.proposed_metrics.selected_metrics) if s1.proposed_metrics else None
                ),
                design_options=DesignOptions(project_name="UI_Demo", run_insights=True),
            ),
        )
        assert s2.status == "awaiting_human_review"
        s3 = orch.approve_and_generate_pbip(sid, HumanReviewPayload(approved=True))
        assert s3.status == "completed"

        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"))
        at.session_state["autobi_session_id"] = sid
        at = at.run(timeout=20)
        assert len(at.exception) == 0, f"Exception on main.py: {at.exception}"

        for page_path in (
            "pages/1_upload.py",
            "pages/2_data_profile.py",
            "pages/3_prompt_builder.py",
            "pages/4_reports.py",
        ):
            at = at.switch_page(page_path).run(timeout=20)
            assert len(at.exception) == 0, f"Exception on {page_path}: {at.exception}"
    finally:
        clear_settings_cache()

