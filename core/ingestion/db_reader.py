"""Optional SQLite/PostgreSQL extraction via SQLAlchemy (read-only allowlisted tables)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine.url import make_url

from core.ingestion.exceptions import (
    IngestionError,
    RowLimitExceededError,
    UnsupportedSourceError,
)

_ALLOWED_DIALECTS: frozenset[str] = frozenset({"sqlite", "postgresql"})


@dataclass(frozen=True)
class DatabaseTableExtraction:
    """Structured result for a single extracted database table."""

    table_name: str
    dataframe: pd.DataFrame
    warnings: list[str]


def redact_connection_url(connection_url: str) -> str:
    """Render a SQLAlchemy URL with any password masked for safe logging and manifests."""
    parsed = make_url(connection_url)
    return parsed.render_as_string(hide_password=True)


def read_database_tables(
    connection_url: str,
    *,
    tables: Sequence[str] | None = None,
    schema: str | None = None,
    max_rows: int,
) -> list[DatabaseTableExtraction]:
    """Extract tables from a SQLite or PostgreSQL database via SQLAlchemy.

    Validates the URL dialect and reflects table metadata so only existing tables are
    queried without dynamic SQL string interpolation.
    """
    parsed_url = make_url(connection_url)
    backend = parsed_url.get_backend_name()
    if backend not in _ALLOWED_DIALECTS:
        raise UnsupportedSourceError(
            f"Unsupported database dialect '{backend}'. "
            f"Allowed dialects: {sorted(_ALLOWED_DIALECTS)}."
        )

    engine = create_engine(connection_url)
    extractions: list[DatabaseTableExtraction] = []
    try:
        inspector = inspect(engine)
        available_tables = inspector.get_table_names(schema=schema)
        available_set = set(available_tables)

        if tables is None:
            target_tables = list(available_tables)
        else:
            missing = [name for name in tables if name not in available_set]
            if missing:
                raise IngestionError(
                    f"Requested database table(s) not found: {', '.join(missing)}"
                )
            target_tables = list(tables)

        metadata = MetaData()
        with engine.connect() as conn:
            for table_name in target_tables:
                reflected = Table(table_name, metadata, autoload_with=engine, schema=schema)
                stmt = select(reflected).limit(max_rows + 1)
                result = conn.execute(stmt)
                rows = result.fetchall()
                if len(rows) > max_rows:
                    raise RowLimitExceededError(
                        f"Database table '{table_name}' exceeds the configured row "
                        f"limit of {max_rows:,} rows."
                    )
                columns = list(map(str, result.keys()))
                df = pd.DataFrame(rows, columns=columns)
                extractions.append(
                    DatabaseTableExtraction(
                        table_name=table_name,
                        dataframe=df,
                        warnings=[],
                    )
                )
    finally:
        engine.dispose()

    return extractions
