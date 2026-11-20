"""Curated-layer Parquet/DuckDB persistence, quarantine tables, and ETL batch log table."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import duckdb
import pandas as pd

from core.config import Settings, get_settings
from core.logging import get_logger
from core.schemas.etl import (
    CuratedTableInfo,
    ETLBatchLogRecord,
    ETLRunResult,
)
from core.storage.staging import _prepare_dataframe_for_parquet, _validate_identifier

logger = get_logger("storage.curated")

_BATCH_LOG_DDL = """
CREATE TABLE IF NOT EXISTS etl_batch_log (
    batch_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    status VARCHAR NOT NULL,
    rows_read BIGINT NOT NULL,
    rows_staged BIGINT NOT NULL,
    rows_curated BIGINT NOT NULL,
    rows_rejected BIGINT NOT NULL,
    rows_dropped_duplicates BIGINT NOT NULL,
    checksum VARCHAR NOT NULL,
    code_version VARCHAR NOT NULL,
    failure_reason VARCHAR
)
"""


def write_quarantine_tables(
    batch_id: str,
    quarantine_tables: Mapping[str, pd.DataFrame],
    *,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Write non-empty quarantine tables to staging Parquet and register them in staging DuckDB."""
    cfg = settings or get_settings()
    staging_batch_dir = cfg.staging_dir / batch_id
    staging_batch_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = staging_batch_dir / "staging.duckdb"

    written: dict[str, Path] = {}
    non_empty = {name: df for name, df in quarantine_tables.items() if not df.empty}
    if not non_empty:
        return written

    conn = duckdb.connect(str(duckdb_path))
    try:
        for base_table, q_df in sorted(non_empty.items()):
            q_name = _validate_identifier(f"{base_table}__quarantine")
            prepared, _ = _prepare_dataframe_for_parquet(q_df, q_name)
            parquet_path = staging_batch_dir / f"{q_name}.parquet"
            prepared.to_parquet(parquet_path, engine="pyarrow", index=False)
            conn.execute(
                f'CREATE OR REPLACE TABLE "{q_name}" AS SELECT * FROM read_parquet(?)',
                [parquet_path.as_posix()],
            )
            written[q_name] = parquet_path
    finally:
        conn.close()

    logger.info(
        "Wrote %d quarantine table(s) for batch %s",
        len(written),
        batch_id,
    )
    return written


def write_curated_tables(
    batch_id: str,
    curated_tables: Mapping[str, pd.DataFrame],
    curated_infos: list[CuratedTableInfo],
    *,
    settings: Settings | None = None,
) -> tuple[Path, Path, list[CuratedTableInfo]]:
    """Write curated tables to Parquet under ``data/curated/<batch_id>/`` and register in DuckDB."""
    cfg = settings or get_settings()
    curated_batch_dir = cfg.curated_dir / batch_id
    curated_batch_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = curated_batch_dir / "curated.duckdb"

    updated_infos: list[CuratedTableInfo] = []
    conn = duckdb.connect(str(duckdb_path))
    try:
        for info in curated_infos:
            table_name = _validate_identifier(info.table_name)
            df = curated_tables[table_name]
            prepared, _ = _prepare_dataframe_for_parquet(df, table_name)
            parquet_path = curated_batch_dir / f"{table_name}.parquet"
            prepared.to_parquet(parquet_path, engine="pyarrow", index=False)
            conn.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM read_parquet(?)',
                [parquet_path.as_posix()],
            )
            updated_infos.append(
                info.model_copy(update={"curated_parquet_path": parquet_path})
            )
    finally:
        conn.close()

    logger.info(
        "Loaded batch %s into curated layer: %d table(s) in %s",
        batch_id,
        len(updated_infos),
        duckdb_path.name,
    )
    return curated_batch_dir, duckdb_path, updated_infos


def write_curated_metadata(curated_batch_dir: Path, result: ETLRunResult) -> Path:
    """Persist lineage graph, validation report, and ETL run summary inside curated batch dir."""
    curated_batch_dir.mkdir(parents=True, exist_ok=True)
    (curated_batch_dir / "lineage.json").write_text(
        result.lineage_graph.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (curated_batch_dir / "validation_report.json").write_text(
        result.validation_report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    summary_path = curated_batch_dir / "etl_result.json"
    summary_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return summary_path


def load_existing_etl_result(
    batch_id: str,
    *,
    settings: Settings | None = None,
) -> ETLRunResult | None:
    """Load an existing completed ``ETLRunResult`` for ``batch_id`` if present."""
    cfg = settings or get_settings()
    summary_path = cfg.curated_dir / batch_id / "etl_result.json"
    if not summary_path.is_file():
        return None
    return ETLRunResult.model_validate_json(summary_path.read_text(encoding="utf-8"))


def get_batch_log_db_path(*, settings: Settings | None = None) -> Path:
    """Return path to the DuckDB database storing the ``etl_batch_log`` audit table."""
    cfg = settings or get_settings()
    return cfg.curated_dir / "batch_log.duckdb"


def upsert_etl_batch_log(
    record: ETLBatchLogRecord,
    *,
    settings: Settings | None = None,
) -> Path:
    """Insert or replace an ``ETLBatchLogRecord`` in DuckDB ``etl_batch_log`` (idempotent)."""
    db_path = get_batch_log_db_path(settings=settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(_BATCH_LOG_DDL)
        conn.execute(
            """
            INSERT OR REPLACE INTO etl_batch_log (
                batch_id, source, started_at, finished_at, status,
                rows_read, rows_staged, rows_curated, rows_rejected,
                rows_dropped_duplicates, checksum, code_version, failure_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.batch_id,
                record.source,
                record.started_at,
                record.finished_at,
                record.status,
                record.rows_read,
                record.rows_staged,
                record.rows_curated,
                record.rows_rejected,
                record.rows_dropped_duplicates,
                record.checksum,
                record.code_version,
                record.failure_reason,
            ],
        )
    finally:
        conn.close()
    return db_path


def read_etl_batch_log(*, settings: Settings | None = None) -> pd.DataFrame:
    """Return the ``etl_batch_log`` table as a DataFrame ordered by ``started_at``."""
    db_path = get_batch_log_db_path(settings=settings)
    if not db_path.is_file():
        return pd.DataFrame(
            columns=[
                "batch_id",
                "source",
                "started_at",
                "finished_at",
                "status",
                "rows_read",
                "rows_staged",
                "rows_curated",
                "rows_rejected",
                "rows_dropped_duplicates",
                "checksum",
                "code_version",
                "failure_reason",
            ]
        )
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return conn.execute("SELECT * FROM etl_batch_log ORDER BY started_at").df()
    finally:
        conn.close()


def read_curated_table(curated_dir: Path, table_name: str) -> pd.DataFrame:
    """Read a curated Parquet table from ``curated_dir`` into a DataFrame."""
    safe_name = _validate_identifier(table_name)
    parquet_path = curated_dir / f"{safe_name}.parquet"
    return pd.read_parquet(parquet_path, engine="pyarrow")

