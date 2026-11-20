"""Deterministic statistical and semantic dataset profiler producing ``ProfileReport``."""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from core.schemas.profiling import (
    CandidateRelationship,
    CardinalityHint,
    ColumnIssue,
    ColumnProfile,
    DateRangeStats,
    NumericStats,
    ProfileReport,
    SemanticType,
    TableProfile,
    ValueFrequency,
)

_BOOL_STRINGS: frozenset[str] = frozenset(
    {"true", "false", "yes", "no", "y", "n", "t", "f"}
)
_BOOL_NAME_RE = re.compile(
    r"(?:^|_)(?:is|has|flag|active|enabled|readmitted|attrition|churn|verified)(?:$|_)"
)
_ID_NAME_RE = re.compile(r"(?:^|_)(?:id|key|code|uuid|guid|sku)$")
_CURRENCY_VALUE_RE = re.compile(
    r"^\s*(?:[$€£¥₹]|USD|EUR|GBP|INR)\s*-?\d[\d,]*(?:\.\d+)?\s*$"
    r"|^\s*-?\d[\d,]*(?:\.\d+)?\s*(?:[$€£¥₹]|USD|EUR|GBP|INR)\s*$",
    re.IGNORECASE,
)
_CURRENCY_NAME_RE = re.compile(
    r"(?:^|_)(?:price|unit_price|revenue|cost|salary|base_salary|amount|billed|"
    r"sales|profit|margin_usd|budget|spend|usd|eur|gbp|inr|currency)(?:$|_)"
)
_PERCENT_VALUE_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*%\s*$")
_PERCENT_NAME_RE = re.compile(r"(?:^|_)(?:pct|percent|percentage|rate|ratio|margin)(?:$|_)")
_NUMERIC_STR_RE = re.compile(r"^\s*[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*$")
_DATE_HINT_RE = re.compile(
    r"^\s*(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
    r"(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)


def _normalize_col_name(name: str) -> str:
    """Normalize a column name into lowercase snake tokens for matching heuristics."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _strip_currency(val: str) -> str:
    """Strip currency symbols, ISO codes, and thousands separators for numeric parsing."""
    cleaned = re.sub(r"(?i)(?:USD|EUR|GBP|INR|[$€£¥₹,\s])", "", val)
    return cleaned


def _strip_percent(val: str) -> str:
    """Strip trailing percent sign and whitespace for numeric parsing."""
    return val.replace("%", "").strip()


def _parse_numeric_series(series: pd.Series, *, mode: str = "plain") -> pd.Series:
    """Convert non-null values to numeric floats according to ``mode`` without mutating input."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    non_null = series.dropna().astype(str)
    if mode == "currency":
        cleaned = non_null.map(_strip_currency)
    elif mode == "percent":
        cleaned = non_null.map(_strip_percent)
    else:
        cleaned = non_null.str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _compute_numeric_stats(numeric_series: pd.Series) -> NumericStats | None:
    """Compute descriptive statistics plus IQR and Modified Z-score outlier counts."""
    clean = numeric_series.dropna().to_numpy(dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return None

    q25 = float(np.percentile(clean, 25))
    median = float(np.percentile(clean, 50))
    q75 = float(np.percentile(clean, 75))
    iqr = q75 - q25
    std_val = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0

    iqr_outliers = 0
    robust_z_outliers = 0
    if clean.size >= 4:
        if iqr > 0:
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            iqr_outliers = int(np.sum((clean < lower) | (clean > upper)))

        mad = float(np.median(np.abs(clean - median)))
        if mad > 0:
            modified_z = 0.6745 * np.abs(clean - median) / mad
            robust_z_outliers = int(np.sum(modified_z > 3.5))

    return NumericStats(
        min=round(float(np.min(clean)), 4),
        max=round(float(np.max(clean)), 4),
        mean=round(float(np.mean(clean)), 4),
        std=round(std_val, 4),
        q25=round(q25, 4),
        median=round(median, 4),
        q75=round(q75, 4),
        iqr=round(iqr, 4),
        iqr_outlier_count=iqr_outliers,
        robust_z_outlier_count=robust_z_outliers,
    )


def _detect_mixed_types(series: pd.Series) -> tuple[bool, int, str]:
    """Detect heterogeneous primitive types inside an ``object``/string column."""
    if series.dtype != "object" and not pd.api.types.is_string_dtype(series):
        return False, 0, ""

    non_null = series.dropna()
    if non_null.empty:
        return False, 0, ""

    kinds: dict[str, int] = {}
    for val in non_null.tolist():
        if isinstance(val, bool):
            kind = "boolean"
        elif isinstance(val, int | float | np.number):
            kind = "numeric"
        else:
            text = str(val).strip()
            lowered = text.lower()
            if lowered in _BOOL_STRINGS:
                kind = "boolean"
            elif _NUMERIC_STR_RE.match(text) or _CURRENCY_VALUE_RE.match(text):
                kind = "numeric"
            elif _DATE_HINT_RE.match(text):
                kind = "date"
            else:
                kind = "text"
        kinds[kind] = kinds.get(kind, 0) + 1

    if len(kinds) <= 1:
        return False, 0, ""

    dominant_kind = max(kinds.items(), key=lambda item: (item[1], item[0]))[0]
    minority_count = sum(cnt for k, cnt in kinds.items() if k != dominant_kind)
    breakdown = ", ".join(f"{k}={cnt}" for k, cnt in sorted(kinds.items()))
    return True, minority_count, f"Mixed value types observed ({breakdown})."


def _try_parse_dates(series: pd.Series) -> pd.Series | None:
    """Parse datetime or date-like string series cleanly without turning integers into epochs."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None

    non_null = series.dropna().astype(str).str.strip()
    if non_null.empty:
        return None

    # Require most values to look date-like before invoking pandas date parsing.
    date_like_ratio = float(non_null.map(lambda s: bool(_DATE_HINT_RE.match(s))).mean())
    if date_like_ratio < 0.8:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")

    valid_ratio = float(parsed.notna().mean())
    if valid_ratio >= 0.85:
        return parsed
    return None


def _infer_semantic_type(
    series: pd.Series,
    col_name: str,
    *,
    row_count: int,
    null_count: int,
    distinct_count: int,
) -> tuple[SemanticType, NumericStats | None, DateRangeStats | None]:
    """Infer a column's BI semantic type alongside numeric or temporal summary stats."""
    norm_name = _normalize_col_name(col_name)
    non_null = series.dropna()

    # 1. Boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean", None, None
    if not non_null.empty:
        if series.dtype == "object" or pd.api.types.is_string_dtype(series):
            lower_vals = {str(v).strip().lower() for v in non_null.tolist()}
            if lower_vals and lower_vals <= _BOOL_STRINGS:
                return "boolean", None, None
        elif pd.api.types.is_integer_dtype(series):
            uniq_ints = set(non_null.unique().tolist())
            if uniq_ints <= {0, 1} and _BOOL_NAME_RE.search(norm_name):
                return "boolean", None, None

    # 2. Explicit string currency or percent patterns
    if not non_null.empty and (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
        str_vals = non_null.astype(str)
        if float(str_vals.map(lambda s: bool(_CURRENCY_VALUE_RE.match(s))).mean()) >= 0.8:
            stats = _compute_numeric_stats(_parse_numeric_series(series, mode="currency"))
            return "currency", stats, None
        if float(str_vals.map(lambda s: bool(_PERCENT_VALUE_RE.match(s))).mean()) >= 0.8:
            stats = _compute_numeric_stats(_parse_numeric_series(series, mode="percent"))
            return "percent", stats, None

    # 3. Date / Time
    parsed_dates = _try_parse_dates(series)
    if parsed_dates is not None:
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            min_dt = valid_dates.min()
            max_dt = valid_dates.max()
            span_days = int((max_dt - min_dt).days)
            date_range = DateRangeStats(
                min_date=min_dt.isoformat(),
                max_date=max_dt.isoformat(),
                span_days=max(span_days, 0),
            )
            return "date_time", None, date_range
        return "date_time", None, None

    # 4. ID (primary or foreign key identifier)
    distinct_ratio = (distinct_count / row_count) if row_count > 0 else 0.0
    if _ID_NAME_RE.search(norm_name) and not pd.api.types.is_float_dtype(series):
        return "id", None, None
    if (
        row_count >= 5
        and null_count == 0
        and distinct_ratio == 1.0
        and (series.dtype == "object" or pd.api.types.is_string_dtype(series))
    ):
        avg_len = float(non_null.astype(str).str.len().mean())
        has_spaces = bool(non_null.astype(str).str.contains(r"\s", regex=True).any())
        if avg_len <= 32 and not has_spaces:
            return "id", None, None

    # 5. Numeric (including currency / percent by column name)
    if pd.api.types.is_numeric_dtype(series):
        stats = _compute_numeric_stats(_parse_numeric_series(series))
        if _CURRENCY_NAME_RE.search(norm_name):
            return "currency", stats, None
        if _PERCENT_NAME_RE.search(norm_name):
            return "percent", stats, None
        return "numeric_measure", stats, None

    # Numeric-parseable string column (e.g. raw CSV column with numbers stored as text)
    if not non_null.empty:
        parsed_num = _parse_numeric_series(series, mode="plain")
        if float(parsed_num.notna().mean()) >= 0.9:
            stats = _compute_numeric_stats(parsed_num)
            if _CURRENCY_NAME_RE.search(norm_name):
                return "currency", stats, None
            if _PERCENT_NAME_RE.search(norm_name):
                return "percent", stats, None
            return "numeric_measure", stats, None

    # 6. Free text vs Categorical dimension
    if not non_null.empty:
        str_series = non_null.astype(str).str.strip()
        avg_len = float(str_series.str.len().mean())
        avg_words = float(str_series.str.split().str.len().mean())
        if avg_len >= 40.0 or (distinct_ratio > 0.7 and avg_words >= 4.0 and row_count >= 5):
            return "free_text", None, None

    return "categorical_dimension", None, None


def _compute_top_values(series: pd.Series, *, top_k: int = 5) -> list[ValueFrequency]:
    """Compute top-K most frequent non-null values with deterministic tie-breaking."""
    non_null = series.dropna()
    non_null_count = len(non_null)
    if non_null_count == 0:
        return []

    counts = non_null.astype(str).value_counts()
    sorted_items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:top_k]
    return [
        ValueFrequency(
            value=str(val),
            count=int(cnt),
            percentage=round((int(cnt) / non_null_count) * 100.0, 2),
        )
        for val, cnt in sorted_items
    ]


def profile_table(
    df: pd.DataFrame,
    table_name: str,
    *,
    top_k: int = 5,
) -> TableProfile:
    """Profile a single pandas DataFrame and return a deterministic ``TableProfile``."""
    row_count = len(df)
    col_count = len(df.columns)
    duplicate_rows = int(df.duplicated().sum()) if row_count > 0 else 0
    duplicate_pct = round((duplicate_rows / row_count) * 100.0, 2) if row_count > 0 else 0.0

    column_profiles: list[ColumnProfile] = []
    candidate_keys: list[str] = []

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        null_count = int(series.isna().sum())
        null_pct = round((null_count / row_count) * 100.0, 2) if row_count > 0 else 0.0
        distinct_count = int(series.nunique(dropna=True))
        non_null_count = row_count - null_count
        distinct_pct = (
            round((distinct_count / non_null_count) * 100.0, 2) if non_null_count > 0 else 0.0
        )

        semantic_type, numeric_stats, date_range = _infer_semantic_type(
            series,
            col_name,
            row_count=row_count,
            null_count=null_count,
            distinct_count=distinct_count,
        )

        issues: list[ColumnIssue] = []

        # Constant column check
        if row_count > 0 and distinct_count <= 1:
            issues.append(
                ColumnIssue(
                    issue_type="constant_column",
                    severity="info",
                    description=(
                        f"Column '{col_name}' has {distinct_count} distinct non-null value."
                    ),
                    affected_count=row_count,
                    affected_percentage=100.0,
                )
            )

        # High null rate check
        if null_pct >= 20.0:
            issues.append(
                ColumnIssue(
                    issue_type="high_nulls",
                    severity="error" if null_pct >= 50.0 else "warning",
                    description=f"Column '{col_name}' has {null_pct:.2f}% null values.",
                    affected_count=null_count,
                    affected_percentage=null_pct,
                )
            )

        # Mixed types check
        is_mixed, mixed_count, mixed_desc = _detect_mixed_types(series)
        if is_mixed:
            mixed_pct = (
                round((mixed_count / non_null_count) * 100.0, 2) if non_null_count > 0 else 0.0
            )
            issues.append(
                ColumnIssue(
                    issue_type="mixed_types",
                    severity="warning",
                    description=mixed_desc,
                    affected_count=mixed_count,
                    affected_percentage=mixed_pct,
                )
            )

        # Outlier checks
        if numeric_stats is not None and non_null_count > 0:
            if numeric_stats.iqr_outlier_count > 0:
                iqr_pct = round((numeric_stats.iqr_outlier_count / non_null_count) * 100.0, 2)
                issues.append(
                    ColumnIssue(
                        issue_type="outliers_iqr",
                        severity="warning",
                        description=(
                            f"{numeric_stats.iqr_outlier_count} IQR outlier(s) detected "
                            f"in '{col_name}'."
                        ),
                        affected_count=numeric_stats.iqr_outlier_count,
                        affected_percentage=iqr_pct,
                    )
                )
            if numeric_stats.robust_z_outlier_count > 0:
                rz_pct = round((numeric_stats.robust_z_outlier_count / non_null_count) * 100.0, 2)
                issues.append(
                    ColumnIssue(
                        issue_type="outliers_robust_z",
                        severity="warning",
                        description=(
                            f"{numeric_stats.robust_z_outlier_count} robust z-score outlier(s) "
                            f"detected in '{col_name}'."
                        ),
                        affected_count=numeric_stats.robust_z_outlier_count,
                        affected_percentage=rz_pct,
                    )
                )

        # Candidate primary key check
        if (
            row_count > 0
            and null_count == 0
            and distinct_count == row_count
            and semantic_type not in {"free_text", "boolean", "currency", "percent"}
            and not pd.api.types.is_float_dtype(series)
        ):
            candidate_keys.append(col_name)

        column_profiles.append(
            ColumnProfile(
                column_name=col_name,
                raw_dtype=str(series.dtype),
                semantic_type=semantic_type,
                row_count=row_count,
                null_count=null_count,
                null_percentage=null_pct,
                distinct_count=distinct_count,
                distinct_percentage=distinct_pct,
                numeric_stats=numeric_stats,
                top_values=_compute_top_values(series, top_k=top_k),
                date_range=date_range,
                issues=issues,
            )
        )

    return TableProfile(
        table_name=table_name,
        row_count=row_count,
        column_count=col_count,
        duplicate_row_count=duplicate_rows,
        duplicate_row_percentage=duplicate_pct,
        candidate_keys=candidate_keys,
        columns=column_profiles,
    )


def _column_name_similarity(
    left_table: str,
    left_col: str,
    right_table: str,
    right_col: str,
) -> float:
    """Compute column-name similarity with boost for ``<table>_id`` <-> ``id`` FK conventions."""
    l_norm = _normalize_col_name(left_col)
    r_norm = _normalize_col_name(right_col)
    if l_norm == r_norm:
        return 1.0

    # Check common foreign-key pattern: orders.customer_id <-> customers.id
    l_table_stem = _normalize_col_name(left_table).rstrip("s")
    r_table_stem = _normalize_col_name(right_table).rstrip("s")
    if (l_norm == "id" and r_norm in {f"{l_table_stem}_id", f"{l_table_stem}_key"}) or (
        r_norm == "id" and l_norm in {f"{r_table_stem}_id", f"{r_table_stem}_key"}
    ):
        return 0.92

    return round(SequenceMatcher(None, l_norm, r_norm).ratio(), 4)


def _infer_candidate_relationships(
    tables: Mapping[str, pd.DataFrame],
    profiles: Mapping[str, TableProfile],
) -> list[CandidateRelationship]:
    """Infer candidate cross-table join relationships using name similarity and value overlap."""
    table_names = sorted(tables.keys())
    relationships: list[CandidateRelationship] = []
    excluded_types: frozenset[SemanticType] = frozenset(
        {"boolean", "date_time", "free_text", "currency", "percent"}
    )

    for idx_a, left_name in enumerate(table_names):
        left_df = tables[left_name]
        left_prof = profiles[left_name]
        left_keys = set(left_prof.candidate_keys)

        for right_name in table_names[idx_a + 1 :]:
            right_df = tables[right_name]
            right_prof = profiles[right_name]
            right_keys = set(right_prof.candidate_keys)

            for l_col in left_prof.columns:
                if l_col.semantic_type in excluded_types or l_col.distinct_count == 0:
                    continue
                if l_col.distinct_count <= 1 and left_prof.row_count > 1:
                    continue

                l_vals = set(left_df[l_col.column_name].dropna().astype(str).tolist())
                if not l_vals:
                    continue

                for r_col in right_prof.columns:
                    if r_col.semantic_type in excluded_types or r_col.distinct_count == 0:
                        continue
                    if r_col.distinct_count <= 1 and right_prof.row_count > 1:
                        continue

                    name_sim = _column_name_similarity(
                        left_name,
                        l_col.column_name,
                        right_name,
                        r_col.column_name,
                    )
                    if name_sim < 0.55:
                        continue

                    r_vals = set(right_df[r_col.column_name].dropna().astype(str).tolist())
                    if not r_vals:
                        continue

                    overlap_count = len(l_vals & r_vals)
                    if overlap_count == 0:
                        continue

                    min_card = min(len(l_vals), len(r_vals))
                    value_overlap = round(overlap_count / min_card, 4)
                    if value_overlap < 0.5:
                        continue

                    l_is_key = l_col.column_name in left_keys
                    r_is_key = r_col.column_name in right_keys
                    if l_is_key and r_is_key:
                        cardinality: CardinalityHint = "one_to_one"
                    elif r_is_key and not l_is_key:
                        cardinality = "many_to_one"
                    elif l_is_key and not r_is_key:
                        cardinality = "one_to_many"
                    else:
                        cardinality = "many_to_many"

                    key_bonus = 0.10 if (l_is_key or r_is_key) else 0.0
                    raw_conf = 0.45 * name_sim + 0.45 * value_overlap + key_bonus
                    confidence = round(min(1.0, raw_conf), 4)
                    if confidence < 0.60:
                        continue

                    relationships.append(
                        CandidateRelationship(
                            left_table=left_name,
                            left_column=l_col.column_name,
                            right_table=right_name,
                            right_column=r_col.column_name,
                            name_similarity=name_sim,
                            value_overlap=value_overlap,
                            confidence=confidence,
                            cardinality_hint=cardinality,
                        )
                    )

    relationships.sort(
        key=lambda rel: (
            -rel.confidence,
            rel.left_table,
            rel.left_column,
            rel.right_table,
            rel.right_column,
        )
    )
    return relationships


def profile_dataset(
    tables: Mapping[str, pd.DataFrame],
    *,
    batch_id: str | None = None,
    top_k: int = 5,
) -> ProfileReport:
    """Profile all tables in a dataset and infer cross-table candidate relationships."""
    ordered_names = sorted(tables.keys())
    table_profiles: list[TableProfile] = []
    profile_map: dict[str, TableProfile] = {}
    summary_warnings: list[str] = []

    for name in ordered_names:
        prof = profile_table(tables[name], table_name=name, top_k=top_k)
        table_profiles.append(prof)
        profile_map[name] = prof
        if prof.duplicate_row_count > 0:
            summary_warnings.append(
                f"Table '{name}' contains {prof.duplicate_row_count} duplicate row(s)."
            )

    relationships = _infer_candidate_relationships(tables, profile_map)
    return ProfileReport(
        batch_id=batch_id,
        tables=table_profiles,
        candidate_relationships=relationships,
        summary_warnings=summary_warnings,
    )
