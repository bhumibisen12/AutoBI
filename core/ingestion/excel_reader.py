"""Excel workbook extraction across all sheets with automatic header-row detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

from core.ingestion.exceptions import RowLimitExceededError, UnsupportedSourceError

_HEADER_SCAN_ROWS = 30


@dataclass(frozen=True)
class ExcelSheetExtraction:
    """Structured result for a single extracted Excel worksheet."""

    sheet_name: str
    dataframe: pd.DataFrame
    header_row_index: int
    warnings: list[str]


def _is_non_empty(value: Any) -> bool:
    """Return True if an Excel cell value is neither None nor whitespace-only text."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _row_non_empty_indices(row: tuple[Any, ...]) -> list[int]:
    """Return column indices containing non-empty values in a raw worksheet row."""
    return [idx for idx, val in enumerate(row) if _is_non_empty(val)]


def _string_ratio(row: tuple[Any, ...], indices: list[int]) -> float:
    """Return the fraction of non-empty cells in ``row`` that are strings."""
    if not indices:
        return 0.0
    string_count = sum(1 for idx in indices if isinstance(row[idx], str))
    return string_count / len(indices)


def detect_excel_header_row(preview_rows: list[tuple[Any, ...]]) -> int | None:
    """Detect the zero-based header row index in a preview of worksheet rows.

    Skips empty rows and single-cell title/banner rows above the tabular region.
    Returns ``None`` if all preview rows are empty.
    """
    non_empty_rows: list[tuple[int, tuple[Any, ...], list[int]]] = []
    for idx, row in enumerate(preview_rows):
        indices = _row_non_empty_indices(row)
        if indices:
            non_empty_rows.append((idx, row, indices))

    if not non_empty_rows:
        return None

    multi_col_widths = [len(indices) for _, _, indices in non_empty_rows if len(indices) >= 2]
    if not multi_col_widths:
        # Single-column sheet: use the first non-empty row.
        return non_empty_rows[0][0]

    # Determine dominant table width (prefer most frequent width, tie-breaking by wider).
    width_counts = Counter(multi_col_widths)
    target_width = max(width_counts.keys(), key=lambda w: (width_counts[w], w))
    min_header_width = max(2, int(target_width * 0.6))

    # Find the earliest row matching the table width; prefer predominantly string headers
    # when multiple rows in the top of the block share the target width.
    candidates: list[tuple[int, tuple[Any, ...], list[int]]] = [
        item for item in non_empty_rows if len(item[2]) >= min_header_width
    ]
    if not candidates:
        return non_empty_rows[0][0]

    first_candidate_idx, first_row, first_indices = candidates[0]
    if _string_ratio(first_row, first_indices) >= 0.5:
        return first_candidate_idx

    # Check if the immediately following candidate looks more like a text header.
    for row_idx, row, indices in candidates[:3]:
        if _string_ratio(row, indices) >= 0.5:
            return row_idx

    return first_candidate_idx


def _normalize_header_names(raw_header: tuple[Any, ...], col_span: int) -> list[str]:
    """Convert raw header cells into unique non-empty column names without mutating data."""
    names: list[str] = []
    seen: dict[str, int] = {}
    for idx in range(col_span):
        val = raw_header[idx] if idx < len(raw_header) else None
        base = str(val) if _is_non_empty(val) else f"unnamed_{idx}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count}")
    return names


def read_excel_file(file_path: Path, *, max_rows: int) -> list[ExcelSheetExtraction]:
    """Extract every non-empty worksheet from an Excel workbook without cleaning cell values."""
    if file_path.suffix.lower() == ".xls":
        return _read_legacy_xls_file(file_path, max_rows=max_rows)

    workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    results: list[ExcelSheetExtraction] = []
    try:
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            warnings: list[str] = []

            preview_rows: list[tuple[Any, ...]] = []
            for row in sheet.iter_rows(min_row=1, max_row=_HEADER_SCAN_ROWS, values_only=True):
                preview_rows.append(tuple(row))

            header_idx = detect_excel_header_row(preview_rows)
            if header_idx is None:
                warnings.append(
                    f"Skipped empty Excel sheet '{sheet_name}' in '{file_path.name}'."
                )
                results.append(
                    ExcelSheetExtraction(
                        sheet_name=sheet_name,
                        dataframe=pd.DataFrame(),
                        header_row_index=0,
                        warnings=warnings,
                    )
                )
                continue

            if header_idx > 0:
                warnings.append(
                    f"Skipped {header_idx} preamble row(s) before detected header in "
                    f"sheet '{sheet_name}' of '{file_path.name}'."
                )

            header_row = preview_rows[header_idx]
            header_indices = _row_non_empty_indices(header_row)
            col_span = max(header_indices) + 1 if header_indices else len(header_row)
            columns = _normalize_header_names(header_row, col_span)

            data_rows: list[tuple[Any, ...]] = []
            for raw_row in sheet.iter_rows(min_row=header_idx + 2, values_only=True):
                sliced = tuple(raw_row[:col_span])
                if len(sliced) < col_span:
                    sliced = sliced + (None,) * (col_span - len(sliced))
                # Skip completely empty Excel padding rows.
                if all(val is None for val in sliced):
                    continue
                data_rows.append(sliced)
                if len(data_rows) > max_rows:
                    raise RowLimitExceededError(
                        f"Excel sheet '{sheet_name}' in '{file_path.name}' exceeds "
                        f"the configured row limit of {max_rows:,} rows."
                    )

            df = pd.DataFrame(data_rows, columns=columns)
            results.append(
                ExcelSheetExtraction(
                    sheet_name=sheet_name,
                    dataframe=df,
                    header_row_index=header_idx,
                    warnings=warnings,
                )
            )
    finally:
        workbook.close()

    return results


def _read_legacy_xls_file(file_path: Path, *, max_rows: int) -> list[ExcelSheetExtraction]:
    """Extract every worksheet from a legacy ``.xls`` workbook if ``xlrd`` is installed."""
    try:
        import xlrd  # noqa: F401
    except ImportError as exc:
        raise UnsupportedSourceError(
            f"Legacy '.xls' file '{file_path.name}' requires the optional 'xlrd' package. "
            "Please save as '.xlsx' or install 'xlrd'."
        ) from exc

    results: list[ExcelSheetExtraction] = []
    with pd.ExcelFile(file_path, engine="xlrd") as xls:
        for sheet_name in xls.sheet_names:
            raw_preview_df = pd.read_excel(
                xls,
                sheet_name=sheet_name,
                header=None,
                nrows=_HEADER_SCAN_ROWS,
            )
            preview_rows = [
                tuple(None if pd.isna(v) else v for v in row)
                for row in raw_preview_df.itertuples(index=False, name=None)
            ]
            warnings: list[str] = []
            header_idx = detect_excel_header_row(preview_rows)
            if header_idx is None:
                warnings.append(
                    f"Skipped empty Excel sheet '{sheet_name}' in '{file_path.name}'."
                )
                results.append(
                    ExcelSheetExtraction(
                        sheet_name=str(sheet_name),
                        dataframe=pd.DataFrame(),
                        header_row_index=0,
                        warnings=warnings,
                    )
                )
                continue

            if header_idx > 0:
                warnings.append(
                    f"Skipped {header_idx} preamble row(s) before detected header in "
                    f"sheet '{sheet_name}' of '{file_path.name}'."
                )

            df = pd.read_excel(
                xls,
                sheet_name=sheet_name,
                header=header_idx,
                nrows=max_rows + 1,
            )
            if len(df) > max_rows:
                raise RowLimitExceededError(
                    f"Excel sheet '{sheet_name}' in '{file_path.name}' exceeds "
                    f"the configured row limit of {max_rows:,} rows."
                )
            df.columns = [str(col) for col in df.columns]
            results.append(
                ExcelSheetExtraction(
                    sheet_name=str(sheet_name),
                    dataframe=df,
                    header_row_index=header_idx,
                    warnings=warnings,
                )
            )
    return results

