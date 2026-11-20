"""Parquet and DuckDB staging storage helpers plus append-only batch logging."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from core.config import Settings, get_settings
from core.logging import get_logger
from core.schemas.ingestion import BatchLogEntry, DatasetBundle, TableArtifact

_SAFE_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

logger = get_logger("storage")


class StorageError(Exception):
    """Raised when reading, writing, or registering storage artifacts fails."""


def _validate_identifier(identifier: str) -> str:
    """Ensure a table name is a safe SQL identifier before quoting in DuckDB DDL."""
    if not _SAFE_SQL_IDENT_RE.match(identifier):
        raise StorageError(f"Unsafe table identifier for storage registration: {identifier!r}")
    return identifier


def _prepare_dataframe_for_parquet(
    df: pd.DataFrame,
    table_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare raw DataFrames for PyArrow Parquet serialization without cleaning values.

    Only coerces mixed-type ``object`` columns (e.g., numbers and text mixed in a raw
    Excel column) to strings so PyArrow can write them losslessly.
    """
    warnings: list[str] = []
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        observed_types = {type(val) for val in non_null}
        if len(observed_types) > 1:
            out[col] = out[col].map(lambda val: None if pd.isna(val) else str(val))
            warnings.append(
                f"Mixed Python types in raw column '{table_name}.{col}' serialized as text "
                "for staging Parquet compatibility."
            )
    return out, warnings


def append_batch_log(entry: BatchLogEntry, log_path: Path) -> Path:
    """Append a single ``BatchLogEntry`` record to a JSON Lines batch log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json() + "\n")
    return log_path


def read_batch_log(log_path: Path) -> list[BatchLogEntry]:
    """Read all ``BatchLogEntry`` records from a JSON Lines batch log file."""
    if not log_path.exists():
        return []
    entries: list[BatchLogEntry] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                entries.append(BatchLogEntry.model_validate_json(stripped))
    return entries


def write_staging_tables(
    bundle: DatasetBundle,
    tables: Mapping[str, pd.DataFrame],
    *,
    settings: Settings | None = None,
) -> DatasetBundle:
    """Persist extracted tables to Parquet in ``data/staging/<batch_id>/`` and register in DuckDB.

    Args:
        bundle: `DatasetBundle` produced by `core.ingestion`.
        tables: Mapping of ``table_name -> DataFrame`` matching ``bundle.tables``.
        settings: Optional runtime settings override.

    Returns:
        Updated `DatasetBundle` populated with ``staging_dir``, ``duckdb_path``,
        and per-table ``staging_parquet_path``.
    """
    cfg = settings or get_settings()
    staging_batch_dir = cfg.staging_dir / bundle.batch_id
    staging_batch_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = staging_batch_dir / "staging.duckdb"
    batch_log_path = cfg.staging_dir / "batch_log.jsonl"

    updated_tables: list[TableArtifact] = []
    storage_warnings: list[str] = []

    conn = duckdb.connect(str(duckdb_path))
    try:
        for artifact in bundle.tables:
            table_name = _validate_identifier(artifact.table_name)
            if table_name not in tables:
                raise StorageError(
                    f"Table '{table_name}' listed in DatasetBundle is missing from input tables."
                )

            raw_df = tables[table_name]
            prepared_df, prep_warnings = _prepare_dataframe_for_parquet(raw_df, table_name)
            storage_warnings.extend(prep_warnings)

            parquet_path = staging_batch_dir / f"{table_name}.parquet"
            prepared_df.to_parquet(parquet_path, engine="pyarrow", index=False)

            conn.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM read_parquet(?)',
                [parquet_path.as_posix()],
            )

            updated_tables.append(
                artifact.model_copy(update={"staging_parquet_path": parquet_path})
            )
    finally:
        conn.close()

    updated_bundle = bundle.model_copy(
        update={
            "tables": updated_tables,
            "load_warnings": [*bundle.load_warnings, *storage_warnings],
            "staging_dir": staging_batch_dir,
            "duckdb_path": duckdb_path,
        }
    )

    append_batch_log(
        BatchLogEntry(
            batch_id=bundle.batch_id,
            layer="staging",
            status="completed",
            timestamp=datetime.now(UTC),
            table_count=len(updated_tables),
            total_rows=updated_bundle.total_rows,
            warnings=storage_warnings,
            details={"duckdb_path": duckdb_path.as_posix()},
        ),
        batch_log_path,
    )

    logger.info(
        "Staged batch %s: wrote %d Parquet table(s) and registered in %s",
        bundle.batch_id,
        len(updated_tables),
        duckdb_path.name,
    )
    return updated_bundle


def read_staging_table(bundle: DatasetBundle, table_name: str) -> pd.DataFrame:
    """Read a staged Parquet table referenced by ``bundle`` into a DataFrame."""
    for artifact in bundle.tables:
        if artifact.table_name == table_name:
            if artifact.staging_parquet_path is None:
                raise StorageError(f"Table '{table_name}' has not been written to staging yet.")
            return pd.read_parquet(artifact.staging_parquet_path, engine="pyarrow")
    raise StorageError(f"Table '{table_name}' not found in DatasetBundle '{bundle.batch_id}'.")


def list_duckdb_tables(duckdb_path: Path) -> list[str]:
    """Return sorted table names currently registered in a DuckDB database file."""
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        conn.close()


def query_duckdb(duckdb_path: Path, sql: str) -> pd.DataFrame:
    """Execute a read-only SQL query against a DuckDB database and return a DataFrame."""
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        return conn.execute(sql).df()
    finally:
        conn.close()

