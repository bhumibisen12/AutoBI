"""High-level ingestion entry points: raw capture, manifest writing, and extraction."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from core.config import Settings, get_settings
from core.ingestion.csv_reader import read_csv_file
from core.ingestion.db_reader import read_database_tables, redact_connection_url
from core.ingestion.excel_reader import read_excel_file
from core.ingestion.exceptions import (
    ImmutableRawError,
    IngestionError,
    RowLimitExceededError,
)
from core.ingestion.raw_store import (
    compute_sha256,
    make_read_only,
    save_immutable_raw_copy,
    write_manifest,
)
from core.logging import get_logger
from core.schemas.ingestion import (
    DatasetBundle,
    RawFileRecord,
    RawManifest,
    TableArtifact,
)
from core.security import AuditLogger, UploadSecurityError, validate_upload

_CSV_EXTENSIONS: frozenset[str] = frozenset({".csv"})
_EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls"})
_SAFE_IDENT_RE = re.compile(r"[^a-z0-9_]+")

logger = get_logger("ingestion")


def generate_batch_id() -> str:
    """Return a sortable, unique, filesystem-safe batch identifier."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def sanitize_table_name(raw_name: str, existing: set[str]) -> str:
    """Normalize a logical table name into a unique lowercase identifier."""
    normalized = _SAFE_IDENT_RE.sub("_", raw_name.strip().lower()).strip("_")
    if not normalized:
        normalized = "table"
    if normalized[0].isdigit():
        normalized = f"t_{normalized}"

    candidate = normalized
    suffix = 2
    while candidate in existing:
        candidate = f"{normalized}_{suffix}"
        suffix += 1
    existing.add(candidate)
    return candidate


def _dtypes_map(df: pd.DataFrame) -> dict[str, str]:
    """Return a stringified mapping of column names to pandas dtype strings."""
    return {str(col): str(dtype) for col, dtype in df.dtypes.items()}


def ingest_files(
    file_paths: Sequence[Path | str],
    *,
    settings: Settings | None = None,
    batch_id: str | None = None,
    actor: str = "system",
    session_id: str | None = None,
    audit_logger: AuditLogger | None = None,
) -> tuple[DatasetBundle, dict[str, pd.DataFrame]]:
    """Validate uploads, capture immutable raw copies, and extract raw tables.

    Performs security validation (extension allowlist, magic bytes, size limit,
    macro/executable/encryption/zip-bomb checks) and extraction only (no cleaning).
    Before any parsing, each file is copied into ``data/raw/<batch_id>/``,
    checksummed with SHA-256, and marked read-only. ``manifest.json`` is then written
    into the same raw batch directory.

    Args:
        file_paths: One or more CSV/Excel file paths to ingest.
        settings: Optional runtime settings override.
        batch_id: Optional explicit batch identifier.
        actor: Sanitized actor/user identifier recorded in the audit log.
        session_id: Optional session identifier recorded in the audit log.
        audit_logger: Optional ``AuditLogger`` instance (defaults to settings-scoped logger).

    Returns:
        Tuple of ``(DatasetBundle, extracted_tables_by_name)``.
    """
    if not file_paths:
        raise IngestionError("At least one input file path is required for ingestion.")

    cfg = settings or get_settings()
    resolved_batch_id = batch_id or generate_batch_id()
    raw_batch_dir = cfg.raw_dir / resolved_batch_id
    audit = audit_logger or AuditLogger(settings=cfg)
    now = datetime.now(UTC)

    table_artifacts: list[TableArtifact] = []
    source_records: list[RawFileRecord] = []
    extracted_frames: dict[str, pd.DataFrame] = {}
    used_table_names: set[str] = set()
    load_warnings: list[str] = []
    cumulative_rows = 0

    for raw_input in file_paths:
        source_path = Path(raw_input)
        try:
            validation = validate_upload(source_path, settings=cfg)
        except UploadSecurityError as exc:
            audit.record(
                actor=actor,
                session_id=session_id,
                batch_id=resolved_batch_id,
                action="upload_rejected",
                resource=source_path.name,
                status="denied",
                metadata={"reason": type(exc).__name__},
            )
            raise

        audit.record(
            actor=actor,
            session_id=session_id,
            batch_id=resolved_batch_id,
            action="upload_validated",
            resource=source_path.name,
            status="success",
            metadata={
                "file_kind": validation.file_kind,
                "size_bytes": validation.size_bytes,
            },
        )
        load_warnings.extend(validation.warnings)

        # Save immutable copy BEFORE parsing anything.
        raw_copy_path, size_bytes, sha256_hex = save_immutable_raw_copy(
            source_path,
            raw_batch_dir,
            max_upload_bytes=cfg.max_upload_bytes,
        )
        file_timestamp = datetime.now(UTC)
        suffix = raw_copy_path.suffix.lower()

        if suffix in _CSV_EXTENSIONS:
            csv_res = read_csv_file(raw_copy_path, max_rows=cfg.max_rows)
            df = csv_res.dataframe
            row_count = len(df)
            col_count = len(df.columns)
            cumulative_rows += row_count
            if cumulative_rows > cfg.max_rows:
                raise RowLimitExceededError(
                    f"Batch '{resolved_batch_id}' exceeds the configured limit of "
                    f"{cfg.max_rows:,} total rows."
                )

            table_name = sanitize_table_name(raw_copy_path.stem, used_table_names)
            extracted_frames[table_name] = df
            load_warnings.extend(csv_res.warnings)

            table_artifacts.append(
                TableArtifact(
                    table_name=table_name,
                    source_file=raw_copy_path.name,
                    source_object=None,
                    row_count=row_count,
                    column_count=col_count,
                    columns=[str(c) for c in df.columns],
                    dtypes=_dtypes_map(df),
                    header_row_index=csv_res.header_row_index,
                )
            )
            source_records.append(
                RawFileRecord(
                    filename=raw_copy_path.name,
                    raw_path=raw_copy_path,
                    size_bytes=size_bytes,
                    sha256=sha256_hex,
                    source_type="csv",
                    row_count=row_count,
                    column_count=col_count,
                    timestamp=file_timestamp,
                    detected_encoding=csv_res.encoding,
                    detected_delimiter=csv_res.delimiter,
                    tables_extracted=[table_name],
                )
            )
        else:
            sheets = read_excel_file(raw_copy_path, max_rows=cfg.max_rows)
            file_rows = 0
            file_cols = 0
            file_tables: list[str] = []

            for sheet_res in sheets:
                load_warnings.extend(sheet_res.warnings)
                df = sheet_res.dataframe
                if df.empty and len(df.columns) == 0:
                    continue

                row_count = len(df)
                col_count = len(df.columns)
                cumulative_rows += row_count
                if cumulative_rows > cfg.max_rows:
                    raise RowLimitExceededError(
                        f"Batch '{resolved_batch_id}' exceeds the configured limit of "
                        f"{cfg.max_rows:,} total rows."
                    )

                raw_logical_name = f"{raw_copy_path.stem}__{sheet_res.sheet_name}"
                table_name = sanitize_table_name(raw_logical_name, used_table_names)
                extracted_frames[table_name] = df
                file_tables.append(table_name)
                file_rows += row_count
                file_cols += col_count

                table_artifacts.append(
                    TableArtifact(
                        table_name=table_name,
                        source_file=raw_copy_path.name,
                        source_object=sheet_res.sheet_name,
                        row_count=row_count,
                        column_count=col_count,
                        columns=[str(c) for c in df.columns],
                        dtypes=_dtypes_map(df),
                        header_row_index=sheet_res.header_row_index,
                    )
                )

            source_records.append(
                RawFileRecord(
                    filename=raw_copy_path.name,
                    raw_path=raw_copy_path,
                    size_bytes=size_bytes,
                    sha256=sha256_hex,
                    source_type="excel",
                    row_count=file_rows,
                    column_count=file_cols,
                    timestamp=file_timestamp,
                    tables_extracted=file_tables,
                )
            )

    manifest = RawManifest(
        batch_id=resolved_batch_id,
        created_at=now,
        files=source_records,
    )
    manifest_path = write_manifest(manifest, raw_batch_dir)

    bundle = DatasetBundle(
        batch_id=resolved_batch_id,
        created_at=now,
        tables=table_artifacts,
        source_files=source_records,
        total_rows=sum(t.row_count for t in table_artifacts),
        total_columns=sum(t.column_count for t in table_artifacts),
        load_warnings=load_warnings,
        raw_dir=raw_batch_dir,
        manifest_path=manifest_path,
    )

    logger.info(
        "Ingested batch %s: %d file(s), %d table(s), %d total row(s)",
        resolved_batch_id,
        len(source_records),
        len(table_artifacts),
        bundle.total_rows,
    )
    audit.record(
        actor=actor,
        session_id=session_id,
        batch_id=resolved_batch_id,
        action="ingest_completed",
        resource=",".join(rec.filename for rec in source_records),
        status="success",
        metadata={
            "file_count": len(source_records),
            "table_count": len(table_artifacts),
            "total_rows": bundle.total_rows,
        },
    )
    return bundle, extracted_frames


def ingest_database(
    connection_url: str,
    *,
    tables: Sequence[str] | None = None,
    schema: str | None = None,
    settings: Settings | None = None,
    batch_id: str | None = None,
) -> tuple[DatasetBundle, dict[str, pd.DataFrame]]:
    """Extract tables from SQLite or PostgreSQL and persist immutable raw snapshots.

    Each extracted table is saved as an immutable Parquet snapshot under
    ``data/raw/<batch_id>/`` with SHA-256 checksum recorded in ``manifest.json``.
    Connection passwords are never logged or persisted in metadata.
    """
    cfg = settings or get_settings()
    resolved_batch_id = batch_id or generate_batch_id()
    raw_batch_dir = cfg.raw_dir / resolved_batch_id
    raw_batch_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)

    safe_url = redact_connection_url(connection_url)
    db_extractions = read_database_tables(
        connection_url,
        tables=tables,
        schema=schema,
        max_rows=cfg.max_rows,
    )

    table_artifacts: list[TableArtifact] = []
    source_records: list[RawFileRecord] = []
    extracted_frames: dict[str, pd.DataFrame] = {}
    used_table_names: set[str] = set()
    load_warnings: list[str] = []
    cumulative_rows = 0

    for db_res in db_extractions:
        df = db_res.dataframe
        row_count = len(df)
        col_count = len(df.columns)
        cumulative_rows += row_count
        if cumulative_rows > cfg.max_rows:
            raise RowLimitExceededError(
                f"Database batch '{resolved_batch_id}' exceeds the configured limit "
                f"of {cfg.max_rows:,} total rows."
            )

        table_name = sanitize_table_name(db_res.table_name, used_table_names)
        snapshot_path = raw_batch_dir / f"{table_name}.parquet"
        if snapshot_path.exists():
            raise ImmutableRawError(f"Immutable raw snapshot already exists: {snapshot_path}")

        df.to_parquet(snapshot_path, engine="pyarrow", index=False)
        size_bytes = snapshot_path.stat().st_size
        sha256_hex = compute_sha256(snapshot_path)
        make_read_only(snapshot_path)

        extracted_frames[table_name] = df
        load_warnings.extend(db_res.warnings)

        table_artifacts.append(
            TableArtifact(
                table_name=table_name,
                source_file=safe_url,
                source_object=db_res.table_name,
                row_count=row_count,
                column_count=col_count,
                columns=[str(c) for c in df.columns],
                dtypes=_dtypes_map(df),
                header_row_index=0,
            )
        )
        source_records.append(
            RawFileRecord(
                filename=snapshot_path.name,
                raw_path=snapshot_path,
                size_bytes=size_bytes,
                sha256=sha256_hex,
                source_type="database",
                row_count=row_count,
                column_count=col_count,
                timestamp=datetime.now(UTC),
                tables_extracted=[table_name],
            )
        )

    manifest = RawManifest(
        batch_id=resolved_batch_id,
        created_at=now,
        files=source_records,
    )
    manifest_path = write_manifest(manifest, raw_batch_dir)

    bundle = DatasetBundle(
        batch_id=resolved_batch_id,
        created_at=now,
        tables=table_artifacts,
        source_files=source_records,
        total_rows=sum(t.row_count for t in table_artifacts),
        total_columns=sum(t.column_count for t in table_artifacts),
        load_warnings=load_warnings,
        raw_dir=raw_batch_dir,
        manifest_path=manifest_path,
    )

    logger.info(
        "Ingested database batch %s from %s: %d table(s), %d total row(s)",
        resolved_batch_id,
        safe_url,
        len(table_artifacts),
        bundle.total_rows,
    )
    return bundle, extracted_frames

