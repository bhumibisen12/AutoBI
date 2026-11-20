"""AutoBI 6-dataset pipeline evaluation and 200k-row scale benchmark.

Run from repository root:
    python evals/pipeline_eval.py
    python evals/pipeline_eval.py --mandi-rows 200000 --output-dir output/evals
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.config import Settings  # noqa: E402
from core.llm import MockLLM  # noqa: E402
from core.orchestrator import AutoBIOrchestrator  # noqa: E402
from core.rag import build_knowledge_index  # noqa: E402
from core.schemas import (  # noqa: E402
    DesignOptions,
    HumanConfirmPayload,
    HumanReviewPayload,
    PipelineSessionState,
)
from evals.datasets import EvalDatasetSpec, prepare_eval_datasets  # noqa: E402


@dataclass
class DatasetEvalResult:
    """Measured accuracy, reconciliation, validation, and stage timing for one dataset."""

    dataset: str
    domain: str
    rows_read: int
    status: str
    schema_matched: int
    schema_total: int
    schema_accuracy_pct: float
    measures_total: int
    measures_verified: int
    measures_unverified: int
    measures_reconciled_pct: float
    report_spec_valid: bool
    pbip_valid: bool
    stage_timings_ms: dict[str, float]
    total_runtime_ms: float
    error: str | None = None


def _compute_schema_accuracy(
    spec: EvalDatasetSpec,
    state: PipelineSessionState,
) -> tuple[int, int]:
    """Compare inferred SemanticModel column roles against ground-truth expected roles."""
    matched = 0
    total = 0
    model = state.semantic_model
    if model is None:
        total = sum(len(cols) for cols in spec.expected_roles.values())
        return 0, total

    for table_name, expected_cols in spec.expected_roles.items():
        sem_tbl = model.get_table(table_name)
        for col_name, expected_role in expected_cols.items():
            total += 1
            if sem_tbl is None:
                continue
            sem_col = sem_tbl.get_column(col_name)
            if sem_col is not None and sem_col.role == expected_role:
                matched += 1
    return matched, total


def evaluate_single_dataset(
    spec: EvalDatasetSpec,
    *,
    work_dir: Path,
    rag_index: object,
) -> DatasetEvalResult:
    """Run the full 10-node orchestrator end-to-end on ``spec`` and collect eval metrics."""
    settings = Settings(
        _env_file=None,
        llm_provider="mock",
        max_upload_mb=200,
        max_rows=500_000,
        data_dir=work_dir / "data" / spec.name,
        output_dir=work_dir / "output" / spec.name,
    )
    orch = AutoBIOrchestrator(
        settings=settings,
        llm_client=MockLLM(),
        knowledge_index=rag_index,  # type: ignore[arg-type]
    )
    session_id = f"eval_{spec.name}"
    t0 = time.perf_counter()

    s1 = orch.start_ingestion(session_id, [spec.path], spec.prompt)
    if s1.status != "awaiting_human_confirm":
        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return DatasetEvalResult(
            dataset=spec.name,
            domain=spec.domain,
            rows_read=s1.etl_result.batch_log.rows_read if s1.etl_result else 0,
            status=s1.status,
            schema_matched=0,
            schema_total=sum(len(c) for c in spec.expected_roles.values()),
            schema_accuracy_pct=0.0,
            measures_total=0,
            measures_verified=0,
            measures_unverified=0,
            measures_reconciled_pct=0.0,
            report_spec_valid=False,
            pbip_valid=False,
            stage_timings_ms={r.node: r.duration_ms for r in s1.node_logs},
            total_runtime_ms=total_ms,
            error=s1.error_message,
        )

    confirm = HumanConfirmPayload(
        confirmed=True,
        selected_metrics=(
            list(s1.proposed_metrics.selected_metrics) if s1.proposed_metrics else None
        ),
        design_options=DesignOptions(
            project_name=f"Eval_{spec.name}",
            theme_name="AutoBI Default",
            audience="executives",
            source_format="parquet",
            run_insights=False,
        ),
    )
    s2 = orch.submit_human_confirmation(session_id, confirm)
    final_state = (
        orch.approve_and_generate_pbip(session_id, HumanReviewPayload(approved=True))
        if s2.status == "awaiting_human_review"
        else s2
    )
    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    matched, total_cols = _compute_schema_accuracy(spec, final_state)
    schema_acc = round((matched / total_cols) * 100.0, 2) if total_cols else 0.0

    vr = final_state.verification_report
    m_total = vr.total_measures if vr else 0
    m_verified = vr.verified_count if vr else 0
    m_unverified = vr.unverified_count if vr else 0
    reconciled_pct = round((m_verified / m_total) * 100.0, 2) if m_total else 0.0

    report_ok = final_state.report_spec is not None and len(final_state.report_spec.pages) > 0
    pbip_ok = bool(
        final_state.pbip_export is not None
        and final_state.pbip_export.mode == "pbip"
        and final_state.pbip_export.validation_result is not None
        and final_state.pbip_export.validation_result.is_valid
    )

    stage_timings: dict[str, float] = {}
    for rec in final_state.node_logs:
        stage_timings[rec.node] = round(stage_timings.get(rec.node, 0.0) + rec.duration_ms, 2)

    return DatasetEvalResult(
        dataset=spec.name,
        domain=spec.domain,
        rows_read=final_state.etl_result.batch_log.rows_read if final_state.etl_result else 0,
        status=final_state.status,
        schema_matched=matched,
        schema_total=total_cols,
        schema_accuracy_pct=schema_acc,
        measures_total=m_total,
        measures_verified=m_verified,
        measures_unverified=m_unverified,
        measures_reconciled_pct=reconciled_pct,
        report_spec_valid=report_ok,
        pbip_valid=pbip_ok,
        stage_timings_ms=stage_timings,
        total_runtime_ms=total_ms,
        error=final_state.error_message,
    )


def run_evaluation_suite(
    *,
    mandi_rows: int = 200_000,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Execute the full 6-dataset evaluation suite and optionally save JSON + Markdown reports."""
    rag_index = build_knowledge_index()
    with tempfile.TemporaryDirectory(prefix="autobi_eval_") as tmp:
        work_dir = Path(tmp)
        specs = prepare_eval_datasets(work_dir, mandi_rows=mandi_rows)
        results = [
            evaluate_single_dataset(spec, work_dir=work_dir, rag_index=rag_index)
            for spec in specs
        ]

    total_matched = sum(r.schema_matched for r in results)
    total_cols = sum(r.schema_total for r in results)
    total_verified = sum(r.measures_verified for r in results)
    total_measures = sum(r.measures_total for r in results)
    specs_passed = sum(1 for r in results if r.report_spec_valid and r.pbip_valid)

    summary = {
        "datasets_evaluated": len(results),
        "overall_schema_accuracy_pct": (
            round((total_matched / total_cols) * 100.0, 2) if total_cols else 0.0
        ),
        "overall_measures_reconciled_pct": (
            round((total_verified / total_measures) * 100.0, 2) if total_measures else 0.0
        ),
        "specs_passing_validation_pct": (
            round((specs_passed / len(results)) * 100.0, 2) if results else 0.0
        ),
        "results": [asdict(r) for r in results],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pipeline_eval_report.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        md_lines = [
            "# AutoBI 6-Dataset Pipeline Evaluation Report",
            "",
            f"- **Datasets Evaluated**: {summary['datasets_evaluated']}",
            f"- **Overall Schema-Detection Accuracy**: {summary['overall_schema_accuracy_pct']}%",
            f"- **Overall DuckDB Measure Reconciliation**: "
            f"{summary['overall_measures_reconciled_pct']}%",
            f"- **Specs Passing Validation (ReportSpec + PBIP)**: "
            f"{summary['specs_passing_validation_pct']}%",
            "",
            "| Dataset | Rows | Schema Acc | Reconciled DAX | Specs Valid | "
            "ETL (ms) | Profile (ms) | DAX (ms) | Report (ms) | PBIP (ms) | Total (ms) |",
            "| --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in results:
            t = r.stage_timings_ms
            valid_badge = "PASS" if (r.report_spec_valid and r.pbip_valid) else "FAIL"
            md_lines.append(
                f"| `{r.dataset}` | {r.rows_read:,} | {r.schema_accuracy_pct:.1f}% "
                f"({r.schema_matched}/{r.schema_total}) | {r.measures_reconciled_pct:.1f}% "
                f"({r.measures_verified}/{r.measures_total}) | {valid_badge} | "
                f"{t.get('etl_pipeline', 0.0):.1f} | {t.get('profile', 0.0):.1f} | "
                f"{t.get('dax_reconciliation', 0.0):.1f} | {t.get('report_spec', 0.0):.1f} | "
                f"{t.get('pbip_generation', 0.0):.1f} | {r.total_runtime_ms:.1f} |"
            )
        (output_dir / "pipeline_eval_report.md").write_text(
            "\n".join(md_lines) + "\n",
            encoding="utf-8",
        )

    return summary


def main() -> int:
    """CLI entrypoint for the AutoBI 6-dataset evaluation suite."""
    parser = argparse.ArgumentParser(description="Run AutoBI 6-dataset evaluation & scale suite.")
    parser.add_argument(
        "--mandi-rows",
        type=int,
        default=200_000,
        help="Number of rows to generate for the mandi-price scale dataset (default: 200000).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "output" / "evals",
        help="Directory to write pipeline_eval_report.json and .md.",
    )
    args = parser.parse_args()

    summary = run_evaluation_suite(mandi_rows=args.mandi_rows, output_dir=args.output_dir)
    print(f"Datasets evaluated: {summary['datasets_evaluated']}")
    print(f"Overall schema-detection accuracy: {summary['overall_schema_accuracy_pct']}%")
    print(f"Overall DAX DuckDB reconciliation: {summary['overall_measures_reconciled_pct']}%")
    print(f"Specs passing validation: {summary['specs_passing_validation_pct']}%")
    for item in summary["results"]:  # type: ignore[union-attr]
        t = item["stage_timings_ms"]
        print(
            f"  - {item['dataset']:18s} | rows={item['rows_read']:7,d} | "
            f"schema={item['schema_accuracy_pct']:5.1f}% | "
            f"dax={item['measures_reconciled_pct']:5.1f}% | "
            f"valid={item['report_spec_valid'] and item['pbip_valid']} | "
            f"etl={t.get('etl_pipeline', 0.0):7.1f}ms | "
            f"total={item['total_runtime_ms']:7.1f}ms"
        )
    print(f"Saved reports to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

