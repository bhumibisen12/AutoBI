"""Staging column typing and row-level quarantine separation (pure functions, no file I/O)."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Any

import pandas as pd

from core.lineage import build_lineage_event
from core.schemas.cleaning import CleaningExecutionResult, CleaningPlan
from core.schemas.lineage import LineageEvent
from core.schemas.profiling import TableProfile

_TRUE_TOKENS: frozenset[str] = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS: frozenset[str] = frozenset({"false", "0", "no", "n", "f"})
_CURRENCY_STRIP_RE = re.compile(r"(?i)(?:USD|EUR|GBP|INR|[$€£¥₹,\s])")


def _parse_typed_series(
    series: pd.Series,
    semantic_type: str,
) -> tuple[pd.Series, list[int], str]:
    """Attempt strict typing for a single column; return ``(typed_series, failed_idx, reason)``."""
    was_non_null = series.notna()

    if semantic_type == "date_time":
        if pd.api.types.is_datetime64_any_dtype(series):
            return series, [], ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        failed_mask = was_non_null & parsed.isna()
        failed_idx = [int(i) for i in series.index[failed_mask].tolist()]
        return parsed, failed_idx, "invalid_date"

    if semantic_type in {"numeric_measure", "currency", "percent"}:
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return series, [], ""
        cleaned = series.astype(str)
        if semantic_type == "currency":
            cleaned = cleaned.map(lambda s: _CURRENCY_STRIP_RE.sub("", s))
        elif semantic_type == "percent":
            cleaned = cleaned.str.replace("%", "", regex=False).str.strip()
        else:
            cleaned = cleaned.str.replace(",", "", regex=False).str.strip()
        cleaned = cleaned.where(was_non_null, other=pd.NA)
        parsed_num = pd.to_numeric(cleaned, errors="coerce").astype("float64")
        failed_mask = was_non_null & parsed_num.isna()
        failed_idx = [int(i) for i in series.index[failed_mask].tolist()]
        return parsed_num, failed_idx, f"invalid_{semantic_type}"

    if semantic_type == "boolean":
        if pd.api.types.is_bool_dtype(series):
            return series, [], ""
        parsed_vals: list[Any] = []
        failed_idx = []
        for idx, val in zip(series.index, series.tolist(), strict=True):
            if pd.isna(val):
                parsed_vals.append(pd.NA)
            elif isinstance(val, bool):
                parsed_vals.append(val)
            else:
                tok = str(val).strip().lower()
                if tok in _TRUE_TOKENS:
                    parsed_vals.append(True)
                elif tok in _FALSE_TOKENS:
                    parsed_vals.append(False)
                else:
                    parsed_vals.append(pd.NA)
                    failed_idx.append(int(idx))
        typed_bool = pd.Series(parsed_vals, index=series.index, dtype="boolean")
        return typed_bool, failed_idx, "invalid_boolean"

    return series, [], ""


def stage_and_quarantine_table(
    raw_df: pd.DataFrame,
    table_profile: TableProfile,
    *,
    batch_id: str,
    explicit_cleaning_plan: CleaningPlan | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int], list[LineageEvent]]:
    """Type staged columns and route rows failing parsing to a quarantine DataFrame.

    Never silently drops or fabricates bad values. Rows that fail parsing retain
    their original raw values inside the returned quarantine DataFrame alongside
    ``_quarantine_batch_id``, ``_quarantine_source_row_index``, and
    ``_quarantine_reason``.
    """
    source_df = raw_df.reset_index(drop=True).copy()
    typed_df = source_df.copy()

    # Columns explicitly handled by cast_type/parse_date/drop_column in a user cleaning plan
    # are left for the cleaning plan step to avoid double-parsing conflicts.
    plan_typed_cols: set[str] = set()
    if explicit_cleaning_plan is not None:
        for op in explicit_cleaning_plan.operations:
            if op.op in {"cast_type", "parse_date", "drop_column"}:
                plan_typed_cols.add(op.column)

    row_reasons: dict[int, list[str]] = {}
    lineage_events: list[LineageEvent] = []
    step_idx = 0

    for col_prof in table_profile.columns:
        col_name = col_prof.column_name
        if col_name not in typed_df.columns or col_name in plan_typed_cols:
            continue

        # Determine if this column should undergo strict staging typing.
        target_semantic = col_prof.semantic_type
        has_mixed = any(issue.issue_type == "mixed_types" for issue in col_prof.issues)
        typed_targets = {"date_time", "numeric_measure", "currency", "percent", "boolean"}
        if target_semantic not in typed_targets:
            if not has_mixed:
                continue
            # If mixed types are predominantly numeric, parse numeric and quarantine bad rows.
            series_non_null = (
                typed_df[col_name].dropna().astype(str).str.replace(",", "", regex=False)
            )
            num_ratio = (
                float(pd.to_numeric(series_non_null, errors="coerce").notna().mean())
                if len(series_non_null)
                else 0.0
            )
            if num_ratio >= 0.6:
                target_semantic = "numeric_measure"
            else:
                continue

        parsed_series, failed_idx, reason_code = _parse_typed_series(
            typed_df[col_name],
            target_semantic,
        )
        typed_df[col_name] = parsed_series
        for r_idx in failed_idx:
            row_reasons.setdefault(r_idx, []).append(f"{reason_code}:{col_name}")

        lineage_events.append(
            build_lineage_event(
                table_name=table_profile.table_name,
                step_index=step_idx,
                operation=f"stage_type_{target_semantic}",
                source_columns=[col_name],
                target_columns=[col_name],
                parameters={"semantic_type": target_semantic},
                rows_before=len(source_df),
                rows_after=len(source_df),
                rows_affected=len(source_df) - len(failed_idx),
                applied=True,
                note=(
                    f"Typed '{col_name}' as {target_semantic}; "
                    f"{len(failed_idx)} row(s) flagged for quarantine."
                ),
            )
        )
        step_idx += 1

    if not row_reasons:
        return typed_df, pd.DataFrame(), {}, lineage_events

    bad_indices = sorted(row_reasons.keys())
    reason_strings = ["; ".join(row_reasons[i]) for i in bad_indices]
    quarantine_df = source_df.loc[bad_indices].copy()
    quarantine_df.insert(0, "_quarantine_reason", reason_strings)
    quarantine_df.insert(0, "_quarantine_source_row_index", bad_indices)
    quarantine_df.insert(0, "_quarantine_batch_id", batch_id)
    quarantine_df = quarantine_df.reset_index(drop=True)

    valid_df = typed_df.drop(index=bad_indices).reset_index(drop=True)
    reason_counts = dict(Counter(reason_strings))
    return valid_df, quarantine_df, reason_counts, lineage_events


def split_cleaning_plan_around_dedup(
    plan: CleaningPlan,
) -> tuple[CleaningPlan, CleaningPlan]:
    """Separate non-dedup operations from ``drop_duplicates`` so bad rows quarantine first."""
    non_dedup_ops = [op for op in plan.operations if op.op != "drop_duplicates"]
    dedup_ops = [op for op in plan.operations if op.op == "drop_duplicates"]
    return (
        CleaningPlan(
            table_name=plan.table_name,
            operations=non_dedup_ops,
            description=plan.description,
        ),
        CleaningPlan(
            table_name=plan.table_name,
            operations=dedup_ops,
            description=f"{plan.description} (dedup phase)".strip(),
        ),
    )


def quarantine_cleaning_failures(
    pre_clean_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    cleaning_result: CleaningExecutionResult,
    *,
    batch_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Extract rows whose cleaning cast/parse failed into a quarantine DataFrame."""
    row_reasons: dict[int, list[str]] = {}
    for step in cleaning_result.steps:
        if not step.failed_row_indices:
            continue
        reason = f"{step.operation}_failed(step={step.step_index})"
        for r_idx in step.failed_row_indices:
            row_reasons.setdefault(r_idx, []).append(reason)

    if not row_reasons:
        return cleaned_df, pd.DataFrame(), {}

    bad_indices = sorted(idx for idx in row_reasons if idx in cleaned_df.index)
    if not bad_indices:
        return cleaned_df, pd.DataFrame(), {}

    reason_strings = ["; ".join(row_reasons[i]) for i in bad_indices]
    quarantine_df = pre_clean_df.loc[bad_indices].copy()
    quarantine_df.insert(0, "_quarantine_reason", reason_strings)
    quarantine_df.insert(0, "_quarantine_source_row_index", bad_indices)
    quarantine_df.insert(0, "_quarantine_batch_id", batch_id)
    quarantine_df = quarantine_df.reset_index(drop=True)

    valid_df = cleaned_df.drop(index=bad_indices).reset_index(drop=True)
    return valid_df, quarantine_df, dict(Counter(reason_strings))

