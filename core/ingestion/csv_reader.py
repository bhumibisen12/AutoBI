"""CSV extraction with automatic encoding, delimiter, and header-row detection."""

from __future__ import annotations

import codecs
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

from core.ingestion.exceptions import RowLimitExceededError

_CANDIDATE_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")
_PREFERRED_ENCODINGS: tuple[str, ...] = ("utf_8", "utf_8_sig", "cp1252", "latin_1", "utf_16")
_ENCODING_SAMPLE_BYTES = 262_144
_HEADER_SCAN_LINES = 50


@dataclass(frozen=True)
class CsvExtractionResult:
    """Structured result of raw CSV extraction."""

    dataframe: pd.DataFrame
    encoding: str
    delimiter: str
    header_row_index: int
    warnings: list[str]


def detect_encoding(
    file_path: Path,
    sample_bytes: int = _ENCODING_SAMPLE_BYTES,
) -> tuple[str, list[str]]:
    """Detect text encoding for a CSV file without mutating its contents.

    Returns:
        Tuple of ``(encoding_name, warnings)``.
    """
    warnings: list[str] = []
    with file_path.open("rb") as handle:
        raw_sample = handle.read(sample_bytes)

    if not raw_sample:
        return "utf-8", warnings

    if raw_sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig", warnings
    if raw_sample.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        warnings.append(f"Detected UTF-16 BOM encoding for '{file_path.name}'.")
        return "utf-16", warnings

    try:
        raw_sample.decode("utf-8")
        return "utf-8", warnings
    except UnicodeDecodeError:
        pass

    matches = from_bytes(raw_sample)
    best = matches.best()
    if best is not None and best.encoding:
        # Break near-identical confidence ties in favor of standard CSV encodings (e.g. cp1252).
        tied = [
            match
            for match in matches
            if abs(match.chaos - best.chaos) <= 0.001
            and abs(match.coherence - best.coherence) <= 0.01
        ]
        chosen = best
        for preferred in _PREFERRED_ENCODINGS:
            for candidate in tied:
                if candidate.encoding.lower().replace("-", "_") == preferred:
                    chosen = candidate
                    break
            else:
                continue
            break

        detected = chosen.encoding
        try:
            raw_sample.decode(detected)
            warnings.append(
                f"Non-UTF-8 encoding '{detected}' detected for '{file_path.name}'."
            )
            return detected, warnings
        except (LookupError, UnicodeDecodeError):
            pass

    for fallback in ("cp1252", "latin-1"):
        try:
            raw_sample.decode(fallback)
            warnings.append(
                f"Non-UTF-8 encoding in '{file_path.name}'; fell back to '{fallback}'."
            )
            return fallback, warnings
        except UnicodeDecodeError:
            continue

    warnings.append(f"Encoding detection fell back to 'latin-1' for '{file_path.name}'.")
    return "latin-1", warnings


def _score_delimiter(lines: list[str], delimiter: str) -> tuple[int, int, int, int]:
    """Return ``(run_length, width, header_idx, non_empty_hits)`` for a candidate delimiter."""
    parsed_rows: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            cells = next(csv.reader([line], delimiter=delimiter), [])
        except csv.Error:
            continue
        parsed_rows.append((idx, len(cells)))

    if not parsed_rows:
        return (0, 0, 0, 0)

    best_run_len = 0
    best_width = 0
    best_start_idx = parsed_rows[0][0]

    # Identify contiguous blocks (ignoring empty lines) with identical column width >= 2.
    current_width = 0
    current_run_len = 0
    current_start_idx = parsed_rows[0][0]

    for idx, width in parsed_rows:
        if width >= 2 and width == current_width:
            current_run_len += 1
        elif width >= 2:
            current_width = width
            current_run_len = 1
            current_start_idx = idx
        else:
            current_width = 0
            current_run_len = 0

        if (current_run_len, current_width) > (best_run_len, best_width):
            best_run_len = current_run_len
            best_width = current_width
            best_start_idx = current_start_idx

    multi_col_hits = sum(1 for _, width in parsed_rows if width == best_width and width >= 2)
    return (best_run_len, best_width, best_start_idx, multi_col_hits)


def detect_delimiter_and_header(
    lines: list[str],
    filename: str,
) -> tuple[str, int, list[str]]:
    """Detect the field delimiter and zero-based physical header row index.

    Handles messy CSV files that contain title lines, metadata comments, or blank
    preamble rows above the actual tabular header.
    """
    warnings: list[str] = []
    non_empty = [(idx, line) for idx, line in enumerate(lines) if line.strip()]
    if not non_empty:
        return ",", 0, warnings

    best_choice: tuple[tuple[int, int, int, int, int], str, int] | None = None
    for priority, delim in enumerate(_CANDIDATE_DELIMITERS):
        run_len, width, start_idx, hits = _score_delimiter(lines, delim)
        # Prefer delimiters with >=2 consecutive rows sharing width >=2,
        # then more rows*width coverage, then common delimiter order.
        has_table_block = 1 if (run_len >= 2 and width >= 2) else 0
        has_multi_col = 1 if width >= 2 else 0
        score = (
            has_table_block,
            has_multi_col,
            run_len * min(width, 25),
            hits,
            -priority,
        )
        if best_choice is None or score > best_choice[0]:
            best_choice = (score, delim, start_idx)

    assert best_choice is not None
    score_tuple, chosen_delim, header_idx = best_choice
    if score_tuple[1] == 0:
        # Single-column CSV fallback: start at first non-empty line.
        chosen_delim = ","
        header_idx = non_empty[0][0]

    if header_idx > 0:
        warnings.append(
            f"Skipped {header_idx} preamble row(s) before detected header in '{filename}'."
        )
    return chosen_delim, header_idx, warnings


def read_csv_file(file_path: Path, *, max_rows: int) -> CsvExtractionResult:
    """Extract a CSV file into a raw DataFrame with auto-detected encoding/delimiter/header.

    Performs extraction only: cell values are never cleaned or modified.
    """
    encoding, warnings = detect_encoding(file_path)

    with file_path.open("r", encoding=encoding, newline="") as handle:
        preview_lines: list[str] = []
        for _ in range(_HEADER_SCAN_LINES):
            line = handle.readline()
            if not line:
                break
            preview_lines.append(line)

    if not any(line.strip() for line in preview_lines):
        warnings.append(f"CSV file '{file_path.name}' is empty.")
        return CsvExtractionResult(
            dataframe=pd.DataFrame(),
            encoding=encoding,
            delimiter=",",
            header_row_index=0,
            warnings=warnings,
        )

    delimiter, header_row_index, header_warnings = detect_delimiter_and_header(
        preview_lines,
        filename=file_path.name,
    )
    warnings.extend(header_warnings)

    df = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        skiprows=header_row_index,
        header=0,
        nrows=max_rows + 1,
    )

    if len(df) > max_rows:
        raise RowLimitExceededError(
            f"CSV file '{file_path.name}' exceeds the configured row limit of {max_rows:,} rows."
        )

    # Ensure column labels are string-typed without renaming or mutating data.
    df.columns = [str(col) for col in df.columns]

    return CsvExtractionResult(
        dataframe=df,
        encoding=encoding,
        delimiter=delimiter,
        header_row_index=header_row_index,
        warnings=warnings,
    )
