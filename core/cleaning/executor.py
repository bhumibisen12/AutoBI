"""Pure, idempotent cleaning-plan executor with strict allowlist and lineage emission."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd

from core.lineage import build_lineage_event
from core.schemas.cleaning import (
    CastTypeOp,
    CleaningExecutionResult,
    CleaningOperation,
    CleaningPlan,
    CleaningStepLog,
    DropColumnOp,
    DropDuplicatesOp,
    FillNullOp,
    MapValuesOp,
    ParseDateOp,
    RenameColumnOp,
    SplitColumnOp,
    StandardizeCaseOp,
    TrimTextOp,
)
from core.schemas.lineage import LineageEvent

_TRUE_TOKENS: frozenset[str] = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS: frozenset[str] = frozenset({"false", "0", "no", "n", "f"})


class CleaningExecutionError(Exception):
    """Raised when a cleaning operation cannot be applied to a DataFrame."""


def parse_cleaning_plan_json(raw_json: str) -> CleaningPlan:
    """Validate a declarative JSON cleaning plan against the allowlisted Pydantic schema."""
    return CleaningPlan.model_validate_json(raw_json)


def _require_column(df: pd.DataFrame, column: str, op_name: str) -> None:
    """Raise ``CleaningExecutionError`` if ``column`` is missing from ``df``."""
    if column not in df.columns:
        raise CleaningExecutionError(
            f"Operation '{op_name}' failed: column '{column}' does not exist."
        )


def _apply_rename_column(
    df: pd.DataFrame,
    op: RenameColumnOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    if op.column in df.columns:
        if op.new_name in df.columns and op.new_name != op.column:
            raise CleaningExecutionError(
                f"Cannot rename '{op.column}' to '{op.new_name}': target already exists."
            )
        out = df.rename(columns={op.column: op.new_name})
        return (
            out,
            [op.column],
            [op.new_name],
            len(out),
            [],
            True,
            f"Renamed column '{op.column}' -> '{op.new_name}'.",
        )
    if op.new_name in df.columns:
        return (
            df,
            [op.column],
            [op.new_name],
            0,
            [],
            False,
            f"Idempotent skip: '{op.new_name}' already exists.",
        )
    raise CleaningExecutionError(
        f"Operation 'rename_column' failed: neither '{op.column}' nor '{op.new_name}' exists."
    )


def _apply_cast_type(
    df: pd.DataFrame,
    op: CastTypeOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    _require_column(df, op.column, op.op)
    out = df.copy()
    series = out[op.column]
    was_non_null = series.notna()
    failed_indices: list[int] = []

    if op.target_type == "int":
        numeric = pd.to_numeric(series, errors="coerce")
        is_whole = numeric.isna() | (numeric % 1 == 0)
        failed_mask = was_non_null & (numeric.isna() | ~is_whole)
        failed_indices = [int(i) for i in out.index[failed_mask].tolist()]
        safe_numeric = numeric.where(is_whole, other=pd.NA)
        out[op.column] = safe_numeric.astype("Int64")
    elif op.target_type == "float":
        numeric = pd.to_numeric(series, errors="coerce")
        failed_mask = was_non_null & numeric.isna()
        failed_indices = [int(i) for i in out.index[failed_mask].tolist()]
        out[op.column] = numeric.astype("float64")
    elif op.target_type == "str":
        converted = series.map(lambda v: pd.NA if pd.isna(v) else str(v))
        out[op.column] = converted.astype("string")
    elif op.target_type == "bool":
        parsed_vals: list[bool | Any] = []
        for idx, val in zip(out.index, series.tolist(), strict=True):
            if pd.isna(val):
                parsed_vals.append(pd.NA)
            elif isinstance(val, bool):
                parsed_vals.append(val)
            else:
                token = str(val).strip().lower()
                if token in _TRUE_TOKENS:
                    parsed_vals.append(True)
                elif token in _FALSE_TOKENS:
                    parsed_vals.append(False)
                else:
                    parsed_vals.append(pd.NA)
                    failed_indices.append(int(idx))
        out[op.column] = pd.Series(parsed_vals, index=out.index, dtype="boolean")

    affected = int(was_non_null.sum())
    note = f"Cast '{op.column}' to '{op.target_type}'."
    if failed_indices:
        note += f" {len(failed_indices)} value(s) could not be cast and became null."
    return out, [op.column], [op.column], affected, failed_indices, True, note


def _apply_parse_date(
    df: pd.DataFrame,
    op: ParseDateOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    _require_column(df, op.column, op.op)
    out = df.copy()
    series = out[op.column]
    was_non_null = series.notna()

    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, utc=op.utc, errors="coerce")
        out[op.column] = parsed
        return out, [op.column], [op.column], 0, [], False, f"'{op.column}' already datetime."

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if op.date_format:
            parsed = pd.to_datetime(series, format=op.date_format, utc=op.utc, errors="coerce")
        else:
            parsed = pd.to_datetime(series, format="mixed", utc=op.utc, errors="coerce")

    failed_mask = was_non_null & parsed.isna()
    failed_indices = [int(i) for i in out.index[failed_mask].tolist()]
    out[op.column] = parsed
    affected = int((was_non_null & parsed.notna()).sum())
    note = f"Parsed dates in '{op.column}'."
    if failed_indices:
        note += f" {len(failed_indices)} value(s) failed date parsing."
    return out, [op.column], [op.column], affected, failed_indices, True, note


def _apply_trim_text(
    df: pd.DataFrame,
    op: TrimTextOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    _require_column(df, op.column, op.op)
    out = df.copy()
    series = out[op.column]

    def _trim(val: Any) -> Any:
        if pd.isna(val) or not isinstance(val, str):
            return val
        if op.side == "left":
            return val.lstrip()
        if op.side == "right":
            return val.rstrip()
        return val.strip()

    trimmed = series.map(_trim)
    changed = int(
        sum(
            1
            for old, new in zip(series.tolist(), trimmed.tolist(), strict=True)
            if not (pd.isna(old) and pd.isna(new)) and old != new
        )
    )
    out[op.column] = trimmed
    return out, [op.column], [op.column], changed, [], True, f"Trimmed whitespace ({op.side})."


def _apply_standardize_case(
    df: pd.DataFrame,
    op: StandardizeCaseOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    _require_column(df, op.column, op.op)
    out = df.copy()
    series = out[op.column]

    def _convert(val: Any) -> Any:
        if pd.isna(val) or not isinstance(val, str):
            return val
        if op.case == "lower":
            return val.lower()
        if op.case == "upper":
            return val.upper()
        return val.title()

    converted = series.map(_convert)
    changed = int(
        sum(
            1
            for old, new in zip(series.tolist(), converted.tolist(), strict=True)
            if not (pd.isna(old) and pd.isna(new)) and old != new
        )
    )
    out[op.column] = converted
    return out, [op.column], [op.column], changed, [], True, f"Standardized case to '{op.case}'."


def _apply_fill_null(
    df: pd.DataFrame,
    op: FillNullOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    _require_column(df, op.column, op.op)
    out = df.copy()
    series = out[op.column]
    null_count = int(series.isna().sum())
    if null_count == 0:
        return out, [op.column], [op.column], 0, [], False, f"No nulls to fill in '{op.column}'."

    fill_value: Any
    if op.strategy == "constant":
        fill_value = op.constant_value
    elif op.strategy == "median":
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            raise CleaningExecutionError(
                f"Cannot fill '{op.column}' with median: column has no numeric values."
            )
        fill_value = float(numeric.median())
        if pd.api.types.is_integer_dtype(series):
            fill_value = round(fill_value)
    else:
        non_null = series.dropna()
        if non_null.empty:
            raise CleaningExecutionError(
                f"Cannot fill '{op.column}' with mode: column is entirely null."
            )
        modes = non_null.mode(dropna=True).tolist()
        fill_value = sorted(modes, key=lambda v: str(v))[0]

    out[op.column] = series.fillna(fill_value)
    return (
        out,
        [op.column],
        [op.column],
        null_count,
        [],
        True,
        f"Filled {null_count} null(s) in '{op.column}' using strategy='{op.strategy}'.",
    )


def _apply_drop_duplicates(
    df: pd.DataFrame,
    op: DropDuplicatesOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    if op.subset:
        for col in op.subset:
            _require_column(df, col, op.op)
    cols = list(op.subset) if op.subset else [str(c) for c in df.columns]
    out = df.drop_duplicates(subset=op.subset, keep=op.keep).reset_index(drop=True)
    removed = len(df) - len(out)
    return (
        out,
        cols,
        cols,
        removed,
        [],
        removed > 0,
        f"Removed {removed} duplicate row(s).",
    )


def _apply_drop_column(
    df: pd.DataFrame,
    op: DropColumnOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    if op.column not in df.columns:
        return (
            df,
            [op.column],
            [],
            0,
            [],
            False,
            f"Idempotent skip: column '{op.column}' already absent.",
        )
    out = df.drop(columns=[op.column])
    return out, [op.column], [], len(out), [], True, f"Dropped column '{op.column}'."


def _apply_map_values(
    df: pd.DataFrame,
    op: MapValuesOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    _require_column(df, op.column, op.op)
    out = df.copy()
    series = out[op.column]

    lookup = (
        {k.lower(): v for k, v in op.mapping.items()}
        if op.case_insensitive
        else dict(op.mapping)
    )

    def _map(val: Any) -> Any:
        if pd.isna(val):
            return val
        key = str(val).lower() if op.case_insensitive else str(val)
        return lookup.get(key, val)

    mapped = series.map(_map)
    changed = int(
        sum(
            1
            for old, new in zip(series.tolist(), mapped.tolist(), strict=True)
            if not (pd.isna(old) and pd.isna(new)) and old != new
        )
    )
    out[op.column] = mapped
    return out, [op.column], [op.column], changed, [], True, f"Mapped {changed} cell value(s)."


def _apply_split_column(
    df: pd.DataFrame,
    op: SplitColumnOp,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str]:
    if op.column not in df.columns:
        if op.drop_source and all(target in df.columns for target in op.into):
            return (
                df,
                [op.column],
                list(op.into),
                0,
                [],
                False,
                f"Idempotent skip: '{op.column}' already split into {op.into}.",
            )
        raise CleaningExecutionError(
            f"Operation 'split_column' failed: column '{op.column}' does not exist."
        )

    out = df.copy()
    max_splits = len(op.into) - 1
    split_rows: list[list[Any]] = []
    for val in out[op.column].tolist():
        if pd.isna(val):
            split_rows.append([None] * len(op.into))
        else:
            parts = str(val).split(op.delimiter, max_splits)
            if len(parts) < len(op.into):
                parts.extend([None] * (len(op.into) - len(parts)))
            split_rows.append(parts)

    for col_idx, target_col in enumerate(op.into):
        out[target_col] = [row[col_idx] for row in split_rows]

    if op.drop_source:
        out = out.drop(columns=[op.column])

    return (
        out,
        [op.column],
        list(op.into),
        len(out),
        [],
        True,
        f"Split '{op.column}' on {op.delimiter!r} into {op.into}.",
    )


_OpHandler = Callable[
    [pd.DataFrame, Any],
    tuple[pd.DataFrame, list[str], list[str], int, list[int], bool, str],
]

_OPERATION_ALLOWLIST: dict[str, _OpHandler] = {
    "rename_column": _apply_rename_column,
    "cast_type": _apply_cast_type,
    "parse_date": _apply_parse_date,
    "trim_text": _apply_trim_text,
    "standardize_case": _apply_standardize_case,
    "fill_null": _apply_fill_null,
    "drop_duplicates": _apply_drop_duplicates,
    "drop_column": _apply_drop_column,
    "map_values": _apply_map_values,
    "split_column": _apply_split_column,
}


def _is_consumed_downstream(
    column: str,
    remaining_ops: list[CleaningOperation],
    current_columns: set[str],
) -> bool:
    """Return True if ``column`` was already renamed/split/dropped by a later step in the plan."""
    aliases: set[str] = {column}
    for later in remaining_ops:
        if later.op == "rename_column" and later.column in aliases:
            if later.new_name in current_columns:
                return True
            aliases.add(later.new_name)
        elif later.op == "drop_column" and later.column in aliases:
            return True
        elif later.op == "split_column" and later.drop_source and later.column in aliases:
            if all(target in current_columns for target in later.into):
                return True
    return False


def execute_cleaning_plan(
    df: pd.DataFrame,
    plan: CleaningPlan,
) -> tuple[pd.DataFrame, CleaningExecutionResult]:
    """Execute a validated ``CleaningPlan`` purely in memory and emit full lineage events.

    Pure function: never mutates ``df``, performs zero file I/O, never invokes
    ``eval``/``exec``, and dispatches strictly over ``_OPERATION_ALLOWLIST``.
    """
    current = df.copy()
    columns_before = [str(c) for c in current.columns]
    rows_before_total = len(current)

    step_logs: list[CleaningStepLog] = []
    lineage_events: list[LineageEvent] = []

    for step_index, op in enumerate(plan.operations):
        handler = _OPERATION_ALLOWLIST.get(op.op)
        if handler is None:
            raise CleaningExecutionError(f"Operation '{op.op}' is not in the allowlist.")

        rows_before = len(current)
        target_col = getattr(op, "column", None)
        current_col_set = {str(c) for c in current.columns}

        if (
            isinstance(target_col, str)
            and target_col not in current_col_set
            and _is_consumed_downstream(
                target_col,
                plan.operations[step_index:],
                current_col_set,
            )
        ):
            src_cols = [target_col]
            tgt_cols = (
                [op.new_name]
                if isinstance(op, RenameColumnOp)
                else (list(op.into) if isinstance(op, SplitColumnOp) else [target_col])
            )
            affected = 0
            failed_idx: list[int] = []
            applied = False
            note = f"Idempotent skip: '{target_col}' already processed and consumed downstream."
        else:
            current, src_cols, tgt_cols, affected, failed_idx, applied, note = handler(current, op)
        rows_after = len(current)

        params = op.model_dump(exclude={"op"})
        lineage_events.append(
            build_lineage_event(
                table_name=plan.table_name,
                step_index=step_index,
                operation=op.op,
                source_columns=src_cols,
                target_columns=tgt_cols,
                parameters=params,
                rows_before=rows_before,
                rows_after=rows_after,
                rows_affected=affected,
                applied=applied,
                note=note,
            )
        )
        step_logs.append(
            CleaningStepLog(
                step_index=step_index,
                operation=op.op,
                applied=applied,
                rows_before=rows_before,
                rows_after=rows_after,
                rows_affected=affected,
                failed_row_indices=failed_idx,
                message=note,
            )
        )

    result = CleaningExecutionResult(
        table_name=plan.table_name,
        rows_before=rows_before_total,
        rows_after=len(current),
        columns_before=columns_before,
        columns_after=[str(c) for c in current.columns],
        steps=step_logs,
        lineage_events=lineage_events,
    )
    return current, result


def supported_cleaning_operations() -> tuple[str, ...]:
    """Return the immutable tuple of allowlisted cleaning operation names."""
    return tuple(_OPERATION_ALLOWLIST.keys())


__all__ = [
    "CleaningExecutionError",
    "CleaningOperation",
    "execute_cleaning_plan",
    "parse_cleaning_plan_json",
    "supported_cleaning_operations",
]
