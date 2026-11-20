"""Unit tests for Step 2: ingestion (CSV/Excel/DB), raw manifests, and staging storage."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from core.config import Settings
from core.ingestion import (
    ImmutableRawError,
    RowLimitExceededError,
    UnsupportedSourceError,
    compute_sha256,
    ingest_database,
    ingest_files,
)
from core.ingestion.db_reader import redact_connection_url
from core.schemas import DatasetBundle, RawManifest
from core.security import DisallowedFileTypeError, FileSizeSecurityError
from core.storage import (
    list_duckdb_tables,
    query_duckdb,
    read_batch_log,
    read_staging_table,
    write_staging_tables,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


def _make_settings(
    tmp_path: Path,
    *,
    max_upload_mb: int = 50,
    max_rows: int = 100_000,
) -> Settings:
    """Create isolated Settings rooted under pytest's temporary directory."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        max_upload_mb=max_upload_mb,
        max_rows=max_rows,
    )


def test_messy_csv_encoding_delimiter_and_extra_headers(tmp_path: Path) -> None:
    """Messy cp1252 semicolon CSV with extra preamble rows extracts accurately without cleaning."""
    fixture_path = FIXTURES_DIR / "messy_sales_cp1252.csv"
    settings = _make_settings(tmp_path)

    bundle, frames = ingest_files([fixture_path], settings=settings, batch_id="batch_messy_csv")

    assert isinstance(bundle, DatasetBundle)
    assert len(bundle.tables) == 1
    assert len(bundle.source_files) == 1

    table_meta = bundle.tables[0]
    assert table_meta.row_count == 4
    assert table_meta.column_count == 5
    assert table_meta.header_row_index == 3
    assert table_meta.columns == ["order_id", "customer_name", "city", "amount_eur", "notes"]

    raw_rec = bundle.source_files[0]
    assert raw_rec.detected_delimiter == ";"
    assert raw_rec.detected_encoding is not None
    assert raw_rec.detected_encoding.lower() != "utf-8"

    df = frames[table_meta.table_name]
    # Verify non-UTF-8 characters decoded accurately and NO silent trimming/cleaning occurred
    assert df.loc[0, "customer_name"] == "Café Lumière"
    assert df.loc[1, "customer_name"] == "Müller & Söhne"
    assert df.loc[2, "customer_name"] == "Señor Tacos"
    assert df.loc[2, "notes"] == "  untrimmed note  "
    assert any("preamble row" in warning.lower() for warning in bundle.load_warnings)


def test_multisheet_excel_extraction_and_header_detection(tmp_path: Path) -> None:
    """Multi-sheet Excel reads every non-empty sheet, detects header rows, and warns on empty."""
    fixture_path = FIXTURES_DIR / "multi_sheet_workbook.xlsx"
    settings = _make_settings(tmp_path)

    bundle, frames = ingest_files([fixture_path], settings=settings, batch_id="batch_excel")

    assert len(bundle.tables) == 2
    by_sheet = {tbl.source_object: tbl for tbl in bundle.tables}
    assert set(by_sheet.keys()) == {"Summary_Orders", "Products"}

    orders_meta = by_sheet["Summary_Orders"]
    assert orders_meta.header_row_index == 3
    assert orders_meta.row_count == 3
    assert orders_meta.columns == ["order_id", "product_id", "qty", "unit_price"]
    assert len(frames[orders_meta.table_name]) == 3

    products_meta = by_sheet["Products"]
    assert products_meta.header_row_index == 0
    assert products_meta.row_count == 2
    assert products_meta.columns == ["product_id", "product_name", "category"]

    assert any("Empty_Notes" in warning for warning in bundle.load_warnings)
    assert any("preamble row" in warning.lower() for warning in bundle.load_warnings)


def test_raw_copy_checksum_manifest_and_immutability(tmp_path: Path) -> None:
    """Raw copy matches source SHA-256, writes valid manifest.json, and blocks overwrites."""
    fixture_path = FIXTURES_DIR / "messy_sales_cp1252.csv"
    settings = _make_settings(tmp_path)

    bundle, _ = ingest_files([fixture_path], settings=settings, batch_id="batch_immutable")

    assert bundle.raw_dir == settings.raw_dir / "batch_immutable"
    assert bundle.manifest_path == bundle.raw_dir / "manifest.json"
    assert bundle.manifest_path.is_file()

    manifest = RawManifest.model_validate_json(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest.batch_id == "batch_immutable"
    assert len(manifest.files) == 1

    record = manifest.files[0]
    expected_sha256 = compute_sha256(fixture_path)
    assert record.sha256 == expected_sha256
    assert compute_sha256(record.raw_path) == expected_sha256
    assert record.size_bytes == fixture_path.stat().st_size
    assert record.row_count == 4
    assert record.column_count == 5
    assert record.filename == fixture_path.name
    assert record.timestamp is not None

    # Re-ingesting into the same batch_id must refuse to overwrite immutable raw copy
    with pytest.raises(ImmutableRawError):
        ingest_files([fixture_path], settings=settings, batch_id="batch_immutable")


def test_oversize_file_rejected_before_raw_copy(tmp_path: Path) -> None:
    """Files exceeding MAX_UPLOAD_MB are rejected before copying to the raw layer."""
    oversize_file = tmp_path / "too_large.csv"
    # Create a >1 MiB CSV file while configuring max_upload_mb=1
    header = "a,b,c\n"
    row = "1234567890,abcdefghij,9876543210\n"
    oversize_file.write_text(header + row * 40_000, encoding="utf-8")
    assert oversize_file.stat().st_size > 1 * 1024 * 1024

    settings = _make_settings(tmp_path, max_upload_mb=1)

    with pytest.raises(FileSizeSecurityError):
        ingest_files([oversize_file], settings=settings, batch_id="batch_oversize")

    assert not (settings.raw_dir / "batch_oversize" / "too_large.csv").exists()


def test_row_limit_enforced_for_csv_and_excel(tmp_path: Path) -> None:
    """Tables exceeding MAX_ROWS raise RowLimitExceededError."""
    settings = _make_settings(tmp_path, max_rows=2)

    with pytest.raises(RowLimitExceededError):
        ingest_files([FIXTURES_DIR / "messy_sales_cp1252.csv"], settings=settings, batch_id="b_csv")

    with pytest.raises(RowLimitExceededError):
        ingest_files(
            [FIXTURES_DIR / "multi_sheet_workbook.xlsx"],
            settings=settings,
            batch_id="b_xlsx",
        )


def test_unsupported_file_extension_rejected(tmp_path: Path) -> None:
    """Unsupported file formats raise DisallowedFileTypeError via upload validation."""
    bad_file = tmp_path / "payload.json"
    bad_file.write_text('{"a": 1}', encoding="utf-8")
    settings = _make_settings(tmp_path)

    with pytest.raises(DisallowedFileTypeError):
        ingest_files([bad_file], settings=settings, batch_id="batch_bad_ext")


def test_sqlite_database_ingestion_and_password_redaction(tmp_path: Path) -> None:
    """SQLAlchemy reader extracts SQLite tables, snapshots raw Parquet, and masks credentials."""
    db_file = tmp_path / "demo.sqlite"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT, balance REAL)")
        conn.executemany(
            "INSERT INTO accounts (id, name, balance) VALUES (?, ?, ?)",
            [(1, "Alpha", 100.5), (2, "Beta", 250.0)],
        )
        conn.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, event TEXT)")
        conn.execute("INSERT INTO logs (id, event) VALUES (1, 'login')")
        conn.commit()

    settings = _make_settings(tmp_path)
    db_url = f"sqlite:///{db_file.as_posix()}"

    bundle, frames = ingest_database(
        db_url,
        tables=["accounts"],
        settings=settings,
        batch_id="batch_sqlite",
    )

    assert len(bundle.tables) == 1
    assert bundle.tables[0].table_name == "accounts"
    assert bundle.tables[0].row_count == 2
    assert len(frames["accounts"]) == 2
    assert bundle.manifest_path is not None and bundle.manifest_path.is_file()
    assert compute_sha256(bundle.source_files[0].raw_path) == bundle.source_files[0].sha256

    # Password redaction helper must mask DB credentials
    masked = redact_connection_url("postgresql+psycopg2://analyst:SuperSecret99@db.local:5432/bi")
    assert "SuperSecret99" not in masked
    assert "analyst" in masked

    with pytest.raises(UnsupportedSourceError):
        ingest_database("mysql://user:pass@localhost/db", settings=settings)


def test_storage_parquet_duckdb_registration_and_batch_log(tmp_path: Path) -> None:
    """Staging storage writes Parquet files, registers tables in DuckDB, and writes batch_log."""
    settings = _make_settings(tmp_path)
    bundle, frames = ingest_files(
        [FIXTURES_DIR / "messy_sales_cp1252.csv", FIXTURES_DIR / "multi_sheet_workbook.xlsx"],
        settings=settings,
        batch_id="batch_stage_01",
    )

    staged_bundle = write_staging_tables(bundle, frames, settings=settings)

    assert staged_bundle.staging_dir == settings.staging_dir / "batch_stage_01"
    assert staged_bundle.duckdb_path == staged_bundle.staging_dir / "staging.duckdb"
    assert staged_bundle.duckdb_path.is_file()

    expected_tables = sorted(t.table_name for t in staged_bundle.tables)
    assert list_duckdb_tables(staged_bundle.duckdb_path) == expected_tables

    for artifact in staged_bundle.tables:
        assert artifact.staging_parquet_path is not None
        assert artifact.staging_parquet_path.is_file()
        loaded_df = read_staging_table(staged_bundle, artifact.table_name)
        assert len(loaded_df) == artifact.row_count

        duck_count = query_duckdb(
            staged_bundle.duckdb_path,
            f'SELECT COUNT(*) AS cnt FROM "{artifact.table_name}"',
        )
        assert int(duck_count.loc[0, "cnt"]) == artifact.row_count

    log_entries = read_batch_log(settings.staging_dir / "batch_log.jsonl")
    assert len(log_entries) == 1
    assert log_entries[0].batch_id == "batch_stage_01"
    assert log_entries[0].layer == "staging"
    assert log_entries[0].status == "completed"
    assert log_entries[0].table_count == len(expected_tables)


def test_synthetic_sample_datasets_and_notebook_structure(tmp_path: Path) -> None:
    """All three SYNTHETIC sample datasets ingest and stage cleanly and the notebook is valid."""
    sample_files = [
        SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv",
        SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx",
        SAMPLE_DATA_DIR / "SYNTHETIC_healthcare.csv",
    ]
    for path in sample_files:
        assert path.is_file(), f"Missing sample dataset: {path}"
        assert "SYNTHETIC" in path.name

    settings = _make_settings(tmp_path)
    bundle, frames = ingest_files(sample_files, settings=settings, batch_id="batch_samples")
    staged = write_staging_tables(bundle, frames, settings=settings)

    assert len(staged.tables) == 4  # sales + hr(employees, departments) + healthcare
    assert staged.total_rows == 36 + 25 + 5 + 30
    assert staged.duckdb_path is not None
    assert set(list_duckdb_tables(staged.duckdb_path)) == {
        "synthetic_sales",
        "synthetic_hr__employees",
        "synthetic_hr__departments",
        "synthetic_healthcare",
    }

    notebook_path = PROJECT_ROOT / "notebooks" / "01_sample_data_ingestion.ipynb"
    nb_data = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert nb_data["nbformat"] == 4
    assert len(nb_data["cells"]) >= 4

