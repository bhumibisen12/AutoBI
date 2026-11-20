"""DAX validator: syntax sanity, table/column existence, DIVIDE vs '/', and no implicit measures."""

from __future__ import annotations

import re
from collections.abc import Sequence

from core.schemas.dax import DAXMeasureSpec, DAXValidationResult
from core.schemas.semantic import SemanticModel

_QUOTED_TABLE_COL_RE = re.compile(r"'([^']+)'\[([^\]]+)\]")
_UNQUOTED_TABLE_COL_RE = re.compile(r"(?<![A-Za-z0-9_'])\b([A-Za-z_][A-Za-z0-9_]*)\[([^\]]+)\]")
_BARE_COL_ONLY_RE = re.compile(r"^\s*(?:'[^']+'|[A-Za-z_][A-Za-z0-9_]*)\[[^\]]+\]\s*$")
_CALC_BARE_COL_RE = re.compile(
    r"\bCALCULATE\s*\(\s*(?:'[^']+'|[A-Za-z_][A-Za-z0-9_]*)\[[^\]]+\]\s*[,)]",
    re.IGNORECASE,
)
_TIME_INTEL_RE = re.compile(
    r"\b(?:SAMEPERIODLASTYEAR|DATEADD|DATESINPERIOD|TOTALYTD|DATESYTD)\s*\(\s*'([^']+)'\[([^\]]+)\]",
    re.IGNORECASE,
)
_EXPLICIT_AGG_RE = re.compile(
    r"\b(?:SUM|SUMX|AVERAGE|AVERAGEX|COUNT|COUNTA|COUNTROWS|DISTINCTCOUNT|MIN|MINX|MAX|MAXX|"
    r"DIVIDE|CALCULATE|RANKX|IF|SWITCH)\s*\(",
    re.IGNORECASE,
)


def _strip_comments_and_strings(dax: str) -> str:
    """Remove DAX comments and double-quoted string literals prior to operator/syntax checks."""
    out: list[str] = []
    idx = 0
    n = len(dax)
    in_string = False
    while idx < n:
        ch = dax[idx]
        if in_string:
            if ch == '"':
                if idx + 1 < n and dax[idx + 1] == '"':
                    idx += 2
                    continue
                in_string = False
            idx += 1
            continue

        if ch == '"':
            in_string = True
            out.append('""')
            idx += 1
            continue

        # Block comment /* ... */
        if ch == "/" and idx + 1 < n and dax[idx + 1] == "*":
            end = dax.find("*/", idx + 2)
            idx = n if end == -1 else end + 2
            out.append(" ")
            continue

        # Single-line comments // or --
        if (ch == "/" and idx + 1 < n and dax[idx + 1] == "/") or (
            ch == "-" and idx + 1 < n and dax[idx + 1] == "-"
        ):
            nl = dax.find("\n", idx + 2)
            idx = n if nl == -1 else nl + 1
            out.append("\n")
            continue

        out.append(ch)
        idx += 1
    return "".join(out)


def _check_balanced_delimiters(dax: str) -> list[str]:
    """Verify balanced parentheses, square brackets, single quotes, and double quotes."""
    errors: list[str] = []
    parens = 0
    brackets = 0
    in_double = False
    in_single = False
    idx = 0
    n = len(dax)

    while idx < n:
        ch = dax[idx]
        if in_double:
            if ch == '"':
                if idx + 1 < n and dax[idx + 1] == '"':
                    idx += 2
                    continue
                in_double = False
            idx += 1
            continue

        if in_single:
            if ch == "'":
                if idx + 1 < n and dax[idx + 1] == "'":
                    idx += 2
                    continue
                in_single = False
            idx += 1
            continue

        if ch == '"':
            in_double = True
        elif ch == "'":
            in_single = True
        elif ch == "(":
            parens += 1
        elif ch == ")":
            parens -= 1
            if parens < 0:
                errors.append("Unmatched closing parenthesis ')' in DAX expression.")
                break
        elif ch == "[":
            brackets += 1
        elif ch == "]":
            brackets -= 1
            if brackets < 0:
                errors.append("Unmatched closing bracket ']' in DAX expression.")
                break
        idx += 1

    if in_double:
        errors.append("Unterminated double-quoted string literal in DAX expression.")
    if in_single:
        errors.append("Unterminated single-quoted table identifier in DAX expression.")
    if parens > 0:
        errors.append(f"Unclosed parenthesis '(' ({parens} missing ')') in DAX expression.")
    if brackets > 0:
        errors.append(f"Unclosed bracket '[' ({brackets} missing ']') in DAX expression.")
    return errors


def validate_semantic_model_no_implicit_measures(model: SemanticModel) -> list[str]:
    """Verify every column in ``SemanticModel`` sets ``summarize_by='none'``."""
    errors: list[str] = []
    for tbl in model.tables:
        for col in tbl.columns:
            if col.summarize_by != "none":
                errors.append(
                    f"Implicit measure detected on '{tbl.table_name}.{col.name}' "
                    f"(summarize_by='{col.summarize_by}'); must be 'none'."
                )
    return errors


def validate_dax_measure(
    measure: DAXMeasureSpec,
    model: SemanticModel,
    *,
    known_measures: Sequence[str] = (),
) -> DAXValidationResult:
    """Validate a ``DAXMeasureSpec`` for syntax, schema existence, DIVIDE, and explicit agg."""
    errors: list[str] = []
    warnings: list[str] = []
    expr = measure.expression.strip()

    if not expr:
        return DAXValidationResult(
            measure_name=measure.name,
            is_valid=False,
            errors=["DAX expression is empty."],
        )

    # 1. Home table exists
    if model.get_table(measure.table_name) is None:
        errors.append(
            f"Measure home table '{measure.table_name}' does not exist in SemanticModel."
        )

    # 2. Syntax balance & VAR/RETURN sanity
    errors.extend(_check_balanced_delimiters(expr))
    stripped = _strip_comments_and_strings(expr)

    if re.search(r"\bVAR\b", stripped, re.IGNORECASE) and not re.search(
        r"\bRETURN\b", stripped, re.IGNORECASE
    ):
        errors.append("DAX expression declares VAR without a matching RETURN clause.")

    # 3. Raw '/' division operator check (DIVIDE must be used instead; ignore '/' inside names)
    expr_without_idents = re.sub(r"\[[^\]]+\]|'[^']+'", "", stripped)
    if "/" in expr_without_idents:
        errors.append(
            "Raw '/' division operator is forbidden in DAX measures; "
            "use DIVIDE(numerator, denominator, 0)."
        )

    # 4. No implicit measures (bare column expression or bare column inside CALCULATE)
    if _BARE_COL_ONLY_RE.match(stripped):
        errors.append(
            "Implicit measure forbidden: bare column reference must be wrapped in an "
            "explicit aggregation (e.g., SUM, AVERAGE, DISTINCTCOUNT)."
        )
    if _CALC_BARE_COL_RE.search(stripped):
        errors.append(
            "Implicit measure forbidden: first argument to CALCULATE must be an explicit "
            "measure or aggregation, not a bare column reference."
        )
    if not _EXPLICIT_AGG_RE.search(stripped) and not any(
        f"[{m}]" in stripped for m in known_measures
    ):
        errors.append(
            "DAX measure does not contain an explicit aggregation function or measure reference."
        )

    # 5. Table & column existence validation
    referenced_tables: set[str] = set()
    referenced_columns: set[str] = set()

    for tbl_name, col_name in _QUOTED_TABLE_COL_RE.findall(stripped):
        referenced_tables.add(tbl_name)
        referenced_columns.add(f"{tbl_name}.{col_name}")
        sem_tbl = model.get_table(tbl_name)
        if sem_tbl is None:
            errors.append(f"Referenced table '{tbl_name}' does not exist in curated SemanticModel.")
        elif sem_tbl.get_column(col_name) is None:
            errors.append(
                f"Referenced column '{tbl_name}.{col_name}' does not exist in SemanticModel."
            )

    for tbl_name, col_name in _UNQUOTED_TABLE_COL_RE.findall(stripped):
        referenced_tables.add(tbl_name)
        referenced_columns.add(f"{tbl_name}.{col_name}")
        sem_tbl = model.get_table(tbl_name)
        if sem_tbl is None:
            errors.append(f"Referenced table '{tbl_name}' does not exist in curated SemanticModel.")
        elif sem_tbl.get_column(col_name) is None:
            errors.append(
                f"Referenced column '{tbl_name}.{col_name}' does not exist in SemanticModel."
            )

    # 6. Time-intelligence date table check
    for ti_table, ti_col in _TIME_INTEL_RE.findall(stripped):
        sem_tbl = model.get_table(ti_table)
        if sem_tbl is None or not sem_tbl.is_date_table:
            errors.append(
                f"Time-intelligence function references '{ti_table}.{ti_col}', "
                "which is not a marked date table."
            )

    # 7. Model-level implicit measure hygiene warning/check
    implicit_errors = validate_semantic_model_no_implicit_measures(model)
    errors.extend(implicit_errors)

    return DAXValidationResult(
        measure_name=measure.name,
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        referenced_tables=sorted(referenced_tables),
        referenced_columns=sorted(referenced_columns),
    )
