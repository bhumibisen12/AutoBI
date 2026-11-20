"""Curated-layer builder: deduplication, date dimension, surrogate keys, and relationships."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from core.lineage import build_lineage_event
from core.schemas.etl import CuratedBuildSummary, CuratedRelationship, CuratedTableInfo
from core.schemas.lineage import LineageEvent
from core.schemas.profiling import ProfileReport


def generate_date_dimension(
    tables: Mapping[str, pd.DataFrame],
    *,
    dim_table_name: str = "dim_date",
) -> tuple[pd.DataFrame | None, list[tuple[str, str]]]:
    """Generate a contiguous Power BI-ready ``dim_date`` spanning all datetime columns.

    Returns:
        Tuple of ``(dim_date_df_or_none, list_of_(table_name, date_column)_links)``.
    """
    date_links: list[tuple[str, str]] = []
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None

    for table_name in sorted(tables.keys()):
        df = tables[table_name]
        for col in df.columns:
            series = df[col]
            if not pd.api.types.is_datetime64_any_dtype(series):
                continue
            valid = series.dropna()
            if valid.empty:
                continue
            # Normalize timezone to naive midnight for clean calendar dimension alignment.
            if getattr(valid.dt, "tz", None) is not None:
                valid = valid.dt.tz_convert(None)
            col_min = pd.Timestamp(valid.min()).normalize()
            col_max = pd.Timestamp(valid.max()).normalize()
            min_ts = col_min if min_ts is None else min(min_ts, col_min)
            max_ts = col_max if max_ts is None else max(max_ts, col_max)
            date_links.append((table_name, str(col)))

    if min_ts is None or max_ts is None:
        return None, []

    dates = pd.date_range(start=min_ts, end=max_ts, freq="D")
    dim_date = pd.DataFrame(
        {
            "date_key": dates.strftime("%Y%m%d").astype("int64"),
            "date": dates,
            "year": dates.year.astype("int64"),
            "quarter": dates.quarter.astype("int64"),
            "quarter_label": [f"Q{q}" for q in dates.quarter],
            "month": dates.month.astype("int64"),
            "month_name": dates.strftime("%B"),
            "year_month": dates.strftime("%Y-%m"),
            "day": dates.day.astype("int64"),
            "day_of_week": (dates.dayofweek + 1).astype("int64"),
            "day_name": dates.strftime("%A"),
            "is_weekend": (dates.dayofweek >= 5),
        }
    )
    return dim_date, date_links


def _find_natural_primary_key(
    df: pd.DataFrame,
    preferred_keys: list[str],
) -> str | None:
    """Return the first valid unique non-null non-float natural key in ``df``, if any."""
    row_count = len(df)
    if row_count == 0:
        return None

    candidates = [col for col in preferred_keys if col in df.columns]
    for col in df.columns:
        col_str = str(col)
        if col_str not in candidates:
            candidates.append(col_str)

    for col in candidates:
        series = df[col]
        if pd.api.types.is_float_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        if int(series.isna().sum()) == 0 and int(series.nunique(dropna=True)) == row_count:
            return col
    return None


def build_curated_layer(
    cleaned_tables: Mapping[str, pd.DataFrame],
    *,
    profile_report: ProfileReport,
    rename_map_by_table: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[dict[str, pd.DataFrame], CuratedBuildSummary]:
    """Build curated tables: deduplicate, add surrogate keys where needed, build ``dim_date``.

    Pure function (no file I/O, does not mutate input DataFrames).
    """
    profile_by_table = {tp.table_name: tp for tp in profile_report.tables}
    curated_frames: dict[str, pd.DataFrame] = {}
    table_infos: list[CuratedTableInfo] = []
    dropped_duplicates_by_table: dict[str, int] = {}
    lineage_events: list[LineageEvent] = []
    step_counter = 0

    for table_name in sorted(cleaned_tables.keys()):
        raw_df = cleaned_tables[table_name].copy()
        rows_before = len(raw_df)
        dedup_df = raw_df.drop_duplicates().reset_index(drop=True)
        dropped_dups = rows_before - len(dedup_df)
        dropped_duplicates_by_table[table_name] = dropped_dups

        if dropped_dups > 0:
            cols = [str(c) for c in dedup_df.columns]
            lineage_events.append(
                build_lineage_event(
                    table_name=table_name,
                    step_index=step_counter,
                    operation="curated_deduplicate",
                    source_columns=cols,
                    target_columns=cols,
                    rows_before=rows_before,
                    rows_after=len(dedup_df),
                    rows_affected=dropped_dups,
                    applied=True,
                    note=f"Removed {dropped_dups} duplicate row(s) while building curated table.",
                )
            )
            step_counter += 1

        # Determine preferred candidate keys (translated through cleaning renames if any).
        rename_map = (rename_map_by_table or {}).get(table_name, {})
        preferred: list[str] = []
        if table_name in profile_by_table:
            for key_col in profile_by_table[table_name].candidate_keys:
                preferred.append(rename_map.get(key_col, key_col))

        primary_key = _find_natural_primary_key(dedup_df, preferred)
        surrogate_added = False
        if primary_key is None and len(dedup_df) > 0:
            sk_col = f"{table_name}_sk"
            if sk_col not in dedup_df.columns:
                dedup_df.insert(0, sk_col, pd.Series(range(1, len(dedup_df) + 1), dtype="int64"))
                lineage_events.append(
                    build_lineage_event(
                        table_name=table_name,
                        step_index=step_counter,
                        operation="generate_surrogate_key",
                        source_columns=[],
                        target_columns=[sk_col],
                        rows_before=len(dedup_df),
                        rows_after=len(dedup_df),
                        rows_affected=len(dedup_df),
                        applied=True,
                        note=f"Generated deterministic surrogate key '{sk_col}'.",
                    )
                )
                step_counter += 1
            primary_key = sk_col
            surrogate_added = True

        curated_frames[table_name] = dedup_df
        table_infos.append(
            CuratedTableInfo(
                table_name=table_name,
                row_count=len(dedup_df),
                column_count=len(dedup_df.columns),
                columns=[str(c) for c in dedup_df.columns],
                dtypes={str(c): str(dt) for c, dt in dedup_df.dtypes.items()},
                primary_key=primary_key,
                surrogate_key_added=surrogate_added,
                is_generated=False,
            )
        )

    # Carry forward validated relationships from Step 3 profile report.
    relationships: list[CuratedRelationship] = []
    for rel in profile_report.candidate_relationships:
        left_rename = (rename_map_by_table or {}).get(rel.left_table, {})
        right_rename = (rename_map_by_table or {}).get(rel.right_table, {})
        l_col = left_rename.get(rel.left_column, rel.left_column)
        r_col = right_rename.get(rel.right_column, rel.right_column)
        if (
            rel.left_table in curated_frames
            and rel.right_table in curated_frames
            and l_col in curated_frames[rel.left_table].columns
            and r_col in curated_frames[rel.right_table].columns
        ):
            relationships.append(
                CuratedRelationship(
                    from_table=rel.left_table,
                    from_column=l_col,
                    to_table=rel.right_table,
                    to_column=r_col,
                    cardinality=rel.cardinality_hint,
                    source="profile_inferred",
                )
            )

    # Generate date dimension if any datetime columns exist across curated tables.
    dim_date_df, date_links = generate_date_dimension(curated_frames)
    date_dim_created = dim_date_df is not None
    if dim_date_df is not None:
        curated_frames["dim_date"] = dim_date_df
        table_infos.append(
            CuratedTableInfo(
                table_name="dim_date",
                row_count=len(dim_date_df),
                column_count=len(dim_date_df.columns),
                columns=[str(c) for c in dim_date_df.columns],
                dtypes={str(c): str(dt) for c, dt in dim_date_df.dtypes.items()},
                primary_key="date_key",
                surrogate_key_added=False,
                is_generated=True,
            )
        )
        source_date_cols = [f"{tbl}.{col}" for tbl, col in date_links]
        lineage_events.append(
            build_lineage_event(
                table_name="dim_date",
                step_index=step_counter,
                operation="generate_date_dimension",
                source_columns=source_date_cols,
                target_columns=[str(c) for c in dim_date_df.columns],
                rows_before=0,
                rows_after=len(dim_date_df),
                rows_affected=len(dim_date_df),
                applied=True,
                note=f"Generated calendar dim_date covering {len(dim_date_df)} day(s).",
            )
        )
        for tbl_name, date_col in date_links:
            relationships.append(
                CuratedRelationship(
                    from_table=tbl_name,
                    from_column=date_col,
                    to_table="dim_date",
                    to_column="date",
                    cardinality="many_to_one",
                    source="generated_date_dim",
                )
            )

    summary = CuratedBuildSummary(
        tables=table_infos,
        relationships=relationships,
        date_dimension_created=date_dim_created,
        dropped_duplicates_by_table=dropped_duplicates_by_table,
        lineage_events=lineage_events,
    )
    return curated_frames, summary

