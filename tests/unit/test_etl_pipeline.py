"""Unit tests for Step 3B: ETL orchestration, quarantine, reconciliation, gates, and idempotency."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.config import Settings
from core.etl import run_etl_pipeline
from core.schemas import CleaningPlan, QualityGateConfig
from core.security import SessionManager
from core.storage import (
    list_duckdb_tables,
    query_duckdb,
    read_curated_table,
    read_etl_batch_log,
)
from core.validation import QualityGateFailedError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


def _make_settings(tmp_path: Path, *, retention_hours: int = 24) -> Settings:
    """Create isolated Settings under pytest's temporary directory."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        retention_hours=retention_hours,
    )


def test_bad_rows_quarantined_with_reason_and_exact_reconciliation(tmp_path: Path) -> None:
    """Rows failing date/numeric parsing go to quarantine with reasons and exact reconciliation."""
    bad_csv = tmp_path / "dirty_orders.csv"
    bad_csv.write_text(
        "\n".join(
            [
                "order_id,order_date,amount,region",
                "O-1,2025-01-05,100.50,North",
                "O-2,2025-01-06,200.00,South",
                "O-2,2025-01-06,200.00,South",
                "O-3,2025-01-07,300.00,East",
                "O-4,2025-01-08,150.00,West",
                "O-5,2025-01-09,175.00,North",
                "O-6,2025-01-10,225.00,South",
                "O-7,2025-01-11,250.00,East",
                "O-8,not_a_date,180.00,West",
                "O-9,2025-01-13,not_a_number,North",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = _make_settings(tmp_path)
    gates = QualityGateConfig(max_invalid_type_pct=25.0, max_duplicate_pct=25.0)

    result = run_etl_pipeline(
        [bad_csv],
        gate_config=gates,
        settings=settings,
        actor="tester",
    )

    assert result.validation_report.passed is True
    assert len(result.reconciliations) == 1
    recon = result.reconciliations[0]

    # 10 rows read: 7 unique clean rows + 2 quarantined bad rows + 1 duplicate row
    assert recon.rows_read == 10
    assert recon.rows_staged == 10
    assert recon.rows_curated == 7
    assert recon.rows_rejected == 2
    assert recon.rows_dropped_duplicates == 1
    assert recon.is_reconciled is True
    assert len(recon.rejection_reasons) >= 1

    # Verify quarantine table exists in staging Parquet + DuckDB with original values and reasons
    assert "dirty_orders__quarantine" in result.quarantine_tables
    q_df = query_duckdb(result.staging_duckdb_path, 'SELECT * FROM "dirty_orders__quarantine"')
    assert len(q_df) == 2
    assert set(q_df["order_id"].tolist()) == {"O-8", "O-9"}
    reasons_text = " | ".join(q_df["_quarantine_reason"].tolist())
    assert "invalid_date:order_date" in reasons_text
    assert "invalid_currency:amount" in reasons_text or "invalid_numeric" in reasons_text

    # Verify curated table has only the 7 clean deduplicated rows and dim_date was generated
    curated_tables = set(list_duckdb_tables(result.curated_duckdb_path))
    assert {"dirty_orders", "dim_date"} <= curated_tables
    curated_orders = read_curated_table(result.curated_dir, "dirty_orders")
    assert len(curated_orders) == 7
    expected_ids = {"O-1", "O-2", "O-3", "O-4", "O-5", "O-6", "O-7"}
    assert set(curated_orders["order_id"].tolist()) == expected_ids

    # Verify dim_date spans 2025-01-05..2025-01-11 (7 calendar days) and relationship was registered
    dim_date = read_curated_table(result.curated_dir, "dim_date")
    assert len(dim_date) == 7
    assert any(
        rel.from_table == "dirty_orders" and rel.to_table == "dim_date"
        for rel in result.relationships
    )


def test_etl_rerun_is_idempotent_with_no_duplicates(tmp_path: Path) -> None:
    """Running ETL twice on the same input yields identical data and a single batch log row."""
    settings = _make_settings(tmp_path)
    sample_files = [
        SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv",
        SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx",
    ]
    plans = {
        "synthetic_sales": CleaningPlan(
            table_name="synthetic_sales",
            operations=[
                {"op": "trim_text", "column": "region"},
                {"op": "standardize_case", "column": "region", "case": "title"},
                {"op": "drop_column", "column": "data_origin"},
            ],
        )
    }

    run_1 = run_etl_pipeline(sample_files, cleaning_plans=plans, settings=settings, actor="analyst")
    run_2 = run_etl_pipeline(sample_files, cleaning_plans=plans, settings=settings, actor="analyst")

    assert run_1.idempotent_replay is False
    assert run_2.idempotent_replay is True
    assert run_1.batch_id == run_2.batch_id
    assert run_1.checksum == run_2.checksum
    assert run_1.reconciliations == run_2.reconciliations

    # Batch log table must have exactly ONE row (no duplicate audit records)
    log_df = read_etl_batch_log(settings=settings)
    assert len(log_df) == 1
    assert log_df.loc[0, "batch_id"] == run_1.batch_id
    assert log_df.loc[0, "status"] == "completed"
    assert int(log_df.loc[0, "rows_read"]) == 66  # 36 sales + 25 employees + 5 departments
    assert int(log_df.loc[0, "rows_curated"]) == 66

    # Relationships from Step 3 (departments <-> employees) + generated dim_date links preserved
    rel_tuples = {(r.from_table, r.to_table, r.source) for r in run_1.relationships}
    hr_rel = ("synthetic_hr__departments", "synthetic_hr__employees", "profile_inferred")
    assert hr_rel in rel_tuples
    assert ("synthetic_sales", "dim_date", "generated_date_dim") in rel_tuples
    assert ("synthetic_hr__employees", "dim_date", "generated_date_dim") in rel_tuples

    # Lineage file persisted under curated/<batch_id>/lineage.json
    assert (run_1.curated_dir / "lineage.json").is_file()


def test_failed_quality_gate_stops_pipeline_and_logs_failure(tmp_path: Path) -> None:
    """When a quality gate fails, pipeline halts before curated load and logs failed status."""
    noisy_csv = tmp_path / "noisy_nulls.csv"
    noisy_csv.write_text(
        "\n".join(
            [
                "id,category,amount",
                "1,,100.0",
                "2,,200.0",
                "3,,300.0",
                "4,A,400.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = _make_settings(tmp_path)
    strict_gates = QualityGateConfig(max_null_pct=20.0)

    with pytest.raises(QualityGateFailedError) as exc_info:
        run_etl_pipeline(
            [noisy_csv],
            gate_config=strict_gates,
            settings=settings,
            batch_id="batch_gate_fail",
        )

    report = exc_info.value.report
    assert report.passed is False
    assert any(
        chk.gate_name == "max_null_pct" and chk.column_name == "category"
        for chk in report.failed_checks
    )

    # Curated directory must NOT have been created/loaded when gate failed
    assert not (settings.curated_dir / "batch_gate_fail").exists()

    # Batch log table records the failed batch with failure_reason
    log_df = read_etl_batch_log(settings=settings)
    assert len(log_df) == 1
    assert log_df.loc[0, "batch_id"] == "batch_gate_fail"
    assert log_df.loc[0, "status"] == "failed"
    assert "category" in str(log_df.loc[0, "failure_reason"])


def test_surrogate_key_generation_and_session_retention_across_all_layers(tmp_path: Path) -> None:
    """Tables without natural keys get surrogate keys; retention purges raw/staging/curated."""
    no_key_csv = tmp_path / "events.csv"
    no_key_csv.write_text(
        "\n".join(
            [
                "region,channel,metric_val",
                "North,Online,10.5",
                "North,Partner,20.5",
                "South,Online,10.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    settings = _make_settings(tmp_path, retention_hours=24)
    result = run_etl_pipeline(
        [no_key_csv],
        settings=settings,
        session_id="sess_etl_01",
        actor="privacy_user",
    )

    events_info = next(t for t in result.curated_tables if t.table_name == "events")
    assert events_info.surrogate_key_added is True
    assert events_info.primary_key == "events_sk"

    curated_events = read_curated_table(result.curated_dir, "events")
    assert curated_events["events_sk"].tolist() == [1, 2, 3]

    # Verify all three layers live inside session workspace and retention purges all of them
    assert result.raw_dir.exists()
    assert result.staging_dir.exists()
    assert result.curated_dir.exists()

    manager = SessionManager(settings)
    ws_root = manager.sessions_data_root / "sess_etl_01"
    expired_ts = datetime.now(UTC) - timedelta(hours=72)
    (ws_root / ".session_meta.json").write_text(
        json.dumps({"session_id": "sess_etl_01", "created_at": expired_ts.isoformat()}),
        encoding="utf-8",
    )

    retention = manager.enforce_retention_policy(now=datetime.now(UTC), retention_hours=24)
    assert "sess_etl_01" in retention.purged_sessions
    assert not result.raw_dir.exists()
    assert not result.staging_dir.exists()
    assert not result.curated_dir.exists()


def test_step3b_notebook_is_valid_nbformat() -> None:
    """Step 3B ETL demo notebook exists and is valid nbformat v4."""
    nb_path = PROJECT_ROOT / "notebooks" / "03_etl_pipeline_demo.ipynb"
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 4

