"""Thin ETL orchestrator coordinating ingestion, profiling, cleaning, transform, gates, storage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from core.cleaning import execute_cleaning_plan
from core.config import Settings, get_settings
from core.ingestion import ImmutableRawError, compute_sha256, ingest_files
from core.lineage import LineageTracker
from core.logging import get_logger
from core.profiling import profile_dataset
from core.schemas.cleaning import CleaningPlan, RenameColumnOp
from core.schemas.etl import (
    ETLBatchLogRecord,
    ETLRunResult,
    QualityGateConfig,
    TableReconciliation,
)
from core.schemas.lineage import LineageEvent
from core.security import AuditLogger, SessionManager, validate_upload
from core.storage import (
    load_existing_etl_result,
    upsert_etl_batch_log,
    write_curated_metadata,
    write_curated_tables,
    write_quarantine_tables,
    write_staging_tables,
)
from core.transformation import (
    build_curated_layer,
    quarantine_cleaning_failures,
    split_cleaning_plan_around_dedup,
    stage_and_quarantine_table,
)
from core.validation import QualityGateFailedError, evaluate_quality_gates

logger = get_logger("etl")


def _resolve_code_version() -> str:
    """Return installed AutoBI package version or fallback scaffold version."""
    try:
        return importlib.metadata.version("autobi")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"


def _compute_run_checksum(
    file_paths: Sequence[Path],
    cleaning_plans: Mapping[str, CleaningPlan],
    gate_config: QualityGateConfig,
) -> str:
    """Compute a deterministic SHA-256 fingerprint across input files, plans, and gate config."""
    digest = hashlib.sha256()
    file_fingerprints = sorted((path.name, compute_sha256(path)) for path in file_paths)
    payload = {
        "files": file_fingerprints,
        "cleaning_plans": {
            name: cleaning_plans[name].model_dump(mode="json")
            for name in sorted(cleaning_plans.keys())
        },
        "gate_config": gate_config.model_dump(mode="json"),
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def run_etl_pipeline(
    file_paths: Sequence[Path | str],
    *,
    cleaning_plans: Mapping[str, CleaningPlan] | None = None,
    gate_config: QualityGateConfig | None = None,
    settings: Settings | None = None,
    batch_id: str | None = None,
    session_id: str | None = None,
    actor: str = "system",
    audit_logger: AuditLogger | None = None,
) -> ETLRunResult:
    """Orchestrate Extract -> Stage/Quarantine -> Clean -> Transform -> Validate -> Load.

    Contains orchestration logic only; all domain logic lives in ``core/ingestion``,
    ``core/profiling``, ``core/cleaning``, ``core/transformation``, ``core/validation``,
    ``core/storage``, ``core/lineage``, and ``core/security``.
    """
    started_at = datetime.now(UTC)
    base_settings = settings or get_settings()
    audit = audit_logger or AuditLogger(settings=base_settings)
    cfg = (
        SessionManager(base_settings, audit_logger=audit).session_settings(session_id)
        if session_id is not None
        else base_settings
    )
    plans = dict(cleaning_plans or {})
    gates = gate_config or QualityGateConfig()
    resolved_paths = [Path(p) for p in file_paths]

    # Pre-validate uploads before hashing so malicious/oversize inputs fail fast.
    for path in resolved_paths:
        validate_upload(path, settings=cfg)

    checksum = _compute_run_checksum(resolved_paths, plans, gates)
    resolved_batch_id = batch_id or f"batch_{checksum[:16]}"
    source_label = ",".join(path.name for path in resolved_paths)
    code_version = _resolve_code_version()

    # Idempotency check: return existing completed batch verbatim when checksum matches.
    existing = load_existing_etl_result(resolved_batch_id, settings=cfg)
    if existing is not None:
        if existing.checksum != checksum:
            raise ImmutableRawError(
                f"Batch '{resolved_batch_id}' already exists with a different input checksum."
            )
        logger.info("Idempotent replay hit for batch %s", resolved_batch_id)
        return existing.model_copy(update={"idempotent_replay": True})

    # 1. EXTRACT: immutable raw copy + manifest, then raw staging Parquet + DuckDB.
    bundle, raw_frames = ingest_files(
        resolved_paths,
        settings=cfg,
        batch_id=resolved_batch_id,
        actor=actor,
        session_id=session_id,
        audit_logger=audit,
    )
    staged_bundle = write_staging_tables(bundle, raw_frames, settings=cfg)
    assert staged_bundle.raw_dir is not None
    assert staged_bundle.staging_dir is not None
    assert staged_bundle.duckdb_path is not None

    # 2. TRANSFORM: profile -> stage typing & quarantine -> cleaning -> curated layer.
    raw_profile = profile_dataset(raw_frames, batch_id=resolved_batch_id)
    profile_by_table = {tp.table_name: tp for tp in raw_profile.tables}

    cleaned_frames: dict[str, pd.DataFrame] = {}
    quarantine_frames: dict[str, pd.DataFrame] = {}
    rejection_reasons_by_table: dict[str, dict[str, int]] = {}
    cleaning_dups_by_table: dict[str, int] = {}
    rename_maps: dict[str, dict[str, str]] = {}
    all_lineage_events: list[LineageEvent] = []

    for table_name in sorted(raw_frames.keys()):
        raw_df = raw_frames[table_name]
        plan = plans.get(table_name)
        valid_df, stage_q_df, stage_reasons, stage_lineage = stage_and_quarantine_table(
            raw_df,
            profile_by_table[table_name],
            batch_id=resolved_batch_id,
            explicit_cleaning_plan=plan,
        )
        all_lineage_events.extend(stage_lineage)

        q_parts: list[pd.DataFrame] = [stage_q_df] if not stage_q_df.empty else []
        combined_reasons: Counter[str] = Counter(stage_reasons)
        cleaning_dups = 0

        if plan is not None:
            for op in plan.operations:
                if isinstance(op, RenameColumnOp):
                    rename_maps.setdefault(table_name, {})[op.column] = op.new_name

            non_dedup_plan, dedup_plan = split_cleaning_plan_around_dedup(plan)
            if non_dedup_plan.operations:
                pre_clean_df = valid_df.reset_index(drop=True)
                cleaned_df, non_dedup_res = execute_cleaning_plan(pre_clean_df, non_dedup_plan)
                all_lineage_events.extend(non_dedup_res.lineage_events)
                valid_df, clean_q_df, clean_reasons = quarantine_cleaning_failures(
                    pre_clean_df,
                    cleaned_df,
                    non_dedup_res,
                    batch_id=resolved_batch_id,
                )
                if not clean_q_df.empty:
                    q_parts.append(clean_q_df)
                    combined_reasons.update(clean_reasons)

            if dedup_plan.operations:
                rows_before_dedup = len(valid_df)
                valid_df, dedup_res = execute_cleaning_plan(valid_df, dedup_plan)
                all_lineage_events.extend(dedup_res.lineage_events)
                cleaning_dups = rows_before_dedup - len(valid_df)

        cleaned_frames[table_name] = valid_df
        quarantine_frames[table_name] = (
            pd.concat(q_parts, ignore_index=True) if q_parts else pd.DataFrame()
        )
        rejection_reasons_by_table[table_name] = dict(combined_reasons)
        cleaning_dups_by_table[table_name] = cleaning_dups

    written_quarantine = write_quarantine_tables(
        resolved_batch_id,
        quarantine_frames,
        settings=cfg,
    )

    curated_frames, curated_summary = build_curated_layer(
        cleaned_frames,
        profile_report=raw_profile,
        rename_map_by_table=rename_maps,
    )
    all_lineage_events.extend(curated_summary.lineage_events)

    # 3. RECONCILIATION: rows read == staged == curated + rejected + dropped duplicates.
    reconciliations: list[TableReconciliation] = []
    for table_name in sorted(raw_frames.keys()):
        rows_read = len(raw_frames[table_name])
        rows_staged = rows_read
        rows_curated = len(curated_frames[table_name])
        rows_rejected = len(quarantine_frames[table_name])
        rows_dropped_dups = (
            cleaning_dups_by_table[table_name]
            + curated_summary.dropped_duplicates_by_table.get(table_name, 0)
        )
        reconciliations.append(
            TableReconciliation(
                table_name=table_name,
                rows_read=rows_read,
                rows_staged=rows_staged,
                rows_curated=rows_curated,
                rows_rejected=rows_rejected,
                rows_dropped_duplicates=rows_dropped_dups,
                rejection_reasons=rejection_reasons_by_table[table_name],
            )
        )

    total_read = sum(r.rows_read for r in reconciliations)
    total_staged = sum(r.rows_staged for r in reconciliations)
    total_curated = sum(r.rows_curated for r in reconciliations)
    total_rejected = sum(r.rows_rejected for r in reconciliations)
    total_dups = sum(r.rows_dropped_duplicates for r in reconciliations)

    # 4. VALIDATE: quality gates halt pipeline before writing curated layer on failure.
    curated_profile = profile_dataset(curated_frames, batch_id=resolved_batch_id)
    validation_report = evaluate_quality_gates(
        batch_id=resolved_batch_id,
        curated_profile=curated_profile,
        reconciliations=reconciliations,
        config=gates,
    )

    if not validation_report.passed:
        failure_msg = "; ".join(chk.message for chk in validation_report.failed_checks[:5])
        failed_log = ETLBatchLogRecord(
            batch_id=resolved_batch_id,
            source=source_label,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="failed",
            rows_read=total_read,
            rows_staged=total_staged,
            rows_curated=total_curated,
            rows_rejected=total_rejected,
            rows_dropped_duplicates=total_dups,
            checksum=checksum,
            code_version=code_version,
            failure_reason=failure_msg,
        )
        upsert_etl_batch_log(failed_log, settings=cfg)
        audit.record(
            actor=actor,
            session_id=session_id,
            batch_id=resolved_batch_id,
            action="etl_quality_gate_failed",
            resource=source_label,
            status="failure",
            metadata={"failed_checks": len(validation_report.failed_checks)},
        )
        raise QualityGateFailedError(validation_report)

    # 5. LOAD: persist curated Parquet + DuckDB, lineage, validation report, and batch log.
    curated_dir, curated_duckdb_path, updated_infos = write_curated_tables(
        resolved_batch_id,
        curated_frames,
        curated_summary.tables,
        settings=cfg,
    )

    tracker = LineageTracker(all_lineage_events)
    lineage_graph = tracker.build_graph(
        {name: [str(c) for c in df.columns] for name, df in curated_frames.items()},
        batch_id=resolved_batch_id,
    )

    batch_log_record = ETLBatchLogRecord(
        batch_id=resolved_batch_id,
        source=source_label,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        status="completed",
        rows_read=total_read,
        rows_staged=total_staged,
        rows_curated=total_curated,
        rows_rejected=total_rejected,
        rows_dropped_duplicates=total_dups,
        checksum=checksum,
        code_version=code_version,
    )
    upsert_etl_batch_log(batch_log_record, settings=cfg)

    result = ETLRunResult(
        batch_id=resolved_batch_id,
        idempotent_replay=False,
        checksum=checksum,
        raw_dir=staged_bundle.raw_dir,
        staging_dir=staged_bundle.staging_dir,
        curated_dir=curated_dir,
        curated_duckdb_path=curated_duckdb_path,
        staging_duckdb_path=staged_bundle.duckdb_path,
        curated_tables=updated_infos,
        quarantine_tables=written_quarantine,
        relationships=curated_summary.relationships,
        reconciliations=reconciliations,
        validation_report=validation_report,
        lineage_graph=lineage_graph,
        batch_log=batch_log_record,
    )
    write_curated_metadata(curated_dir, result)
    audit.record(
        actor=actor,
        session_id=session_id,
        batch_id=resolved_batch_id,
        action="etl_completed",
        resource=source_label,
        status="success",
        metadata={
            "rows_read": total_read,
            "rows_curated": total_curated,
            "rows_rejected": total_rejected,
            "rows_dropped_duplicates": total_dups,
        },
    )
    return result

