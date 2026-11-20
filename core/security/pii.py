"""PII detection (column heuristics + regex/Luhn) and masking/hashing helpers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from core.schemas.security import PIIDetectionMethod, PIIFinding, PIIReport, PIIType
from core.security.sanitization import sanitize_for_llm

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_RE = re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[\s-]?){13,19}(?!\d)")
# Matches formatted international/US phones or 10-digit Indian mobile numbers; excludes ISO dates.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:"
    r"\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,5}[\s.-]?\d{4}"
    r"|\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}"
    r"|\d{3}[\s.-]\d{3}[\s.-]\d{4}"
    r"|[6-9]\d{9}"
    r")(?!\d)"
)
_ISO_DATE_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}(?:[T\s].*)?$")

_COLUMN_NAME_PATTERNS: dict[PIIType, tuple[re.Pattern[str], ...]] = {
    "email": (re.compile(r"(?:^|_)(?:e?mail|email_addr(?:ess)?)(?:$|_)"),),
    "phone": (
        re.compile(r"(?:^|_)(?:phone|mobile|tel|telephone|cell|contact_no|contact_number)(?:$|_)"),
    ),
    "credit_card": (
        re.compile(r"(?:^|_)(?:credit_card|card_number|card_no|cc_num|cc_number)(?:$|_)"),
    ),
    "pan": (re.compile(r"(?:^|_)(?:pan|pan_no|pan_number|pan_card)(?:$|_)"),),
    "aadhaar": (re.compile(r"(?:^|_)(?:aadhaar|aadhar|uidai)(?:$|_)"),),
    "ssn": (re.compile(r"(?:^|_)(?:ssn|social_security)(?:$|_)"),),
    "person_name": (
        re.compile(
            r"(?:^|_)(?:full_name|first_name|last_name|customer_name|"
            r"employee_name|patient_name|person_name|client_name|contact_name)(?:$|_)"
        ),
    ),
    "address": (
        re.compile(r"(?:^|_)(?:street_address|home_address|mailing_address|residential_address)(?:$|_)"),
    ),
    "date_of_birth": (re.compile(r"(?:^|_)(?:dob|date_of_birth|birth_date|birthdate)(?:$|_)"),),
}


def is_valid_luhn(number_str: str) -> bool:
    """Return True if the digits in ``number_str`` pass the Luhn checksum."""
    digits = [int(ch) for ch in number_str if ch.isdigit()]
    if not (13 <= len(digits) <= 19):
        return False
    # Reject trivial all-zero strings.
    if all(d == 0 for d in digits):
        return False

    checksum = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        val = digit
        if idx % 2 == parity:
            val *= 2
            if val > 9:
                val -= 9
        checksum += val
    return checksum % 10 == 0


def _contains_credit_card(text: str) -> bool:
    """Return True if ``text`` contains a 13-19 digit sequence passing the Luhn check."""
    for match in _CC_CANDIDATE_RE.finditer(text):
        candidate = match.group(0)
        digit_count = sum(ch.isdigit() for ch in candidate)
        if 13 <= digit_count <= 19 and is_valid_luhn(candidate):
            return True
    return False


def _matches_regex_pii(text: str, pii_type: PIIType) -> bool:
    """Return True if a single cell text matches the regex/validator for ``pii_type``."""
    stripped = text.strip()
    if not stripped:
        return False
    if pii_type == "email":
        return bool(_EMAIL_RE.search(stripped))
    if pii_type == "credit_card":
        return _contains_credit_card(stripped)
    if pii_type == "pan":
        return bool(_PAN_RE.search(stripped))
    if pii_type == "aadhaar":
        if _contains_credit_card(stripped):
            return False
        return bool(_AADHAAR_RE.search(stripped))
    if pii_type == "ssn":
        return bool(_SSN_RE.search(stripped))
    if pii_type == "phone":
        if _ISO_DATE_RE.match(stripped) or _contains_credit_card(stripped):
            return False
        if _AADHAAR_RE.search(stripped):
            return False
        return bool(_PHONE_RE.search(stripped))
    return False


def _detect_column_name_pii_types(column_name: str) -> list[PIIType]:
    """Return PII types suggested by column-name heuristics."""
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower()).strip("_")
    matched: list[PIIType] = []
    for pii_type, patterns in _COLUMN_NAME_PATTERNS.items():
        if any(pat.search(normalized) for pat in patterns):
            matched.append(pii_type)
    return matched


def detect_pii(
    df: pd.DataFrame,
    *,
    table_name: str = "dataset",
    max_scan_rows: int = 5_000,
) -> PIIReport:
    """Scan a DataFrame using column-name heuristics and regex/Luhn checks.

    Returns a ``PIIReport`` detailing each flagged column, PII type, confidence,
    and percentage of non-null rows matched.
    """
    total_rows = len(df)
    sample_df = df.head(max_scan_rows) if total_rows > max_scan_rows else df
    findings: list[PIIFinding] = []

    regex_types: tuple[PIIType, ...] = (
        "email",
        "credit_card",
        "pan",
        "aadhaar",
        "ssn",
        "phone",
    )

    for col in sample_df.columns:
        col_name = str(col)
        heuristic_types = set(_detect_column_name_pii_types(col_name))

        non_null_series = sample_df[col].dropna()
        non_null_count = len(non_null_series)
        str_values = [str(val) for val in non_null_series.tolist()]

        regex_detected_types: set[PIIType] = set()
        for pii_type in regex_types:
            if non_null_count == 0:
                continue
            matched = sum(1 for val in str_values if _matches_regex_pii(val, pii_type))
            if matched == 0:
                continue

            row_pct = round((matched / non_null_count) * 100.0, 2)
            in_heuristic = pii_type in heuristic_types

            # Require either >=5% match rate, a column-name match, or high-precision ID match.
            high_precision = pii_type in {"email", "credit_card", "pan", "aadhaar", "ssn"}
            if not (in_heuristic or row_pct >= 5.0 or (high_precision and matched >= 1)):
                continue

            regex_detected_types.add(pii_type)
            method: PIIDetectionMethod = "both" if in_heuristic else "regex"
            if in_heuristic:
                confidence = min(0.99, round(0.85 + 0.14 * (row_pct / 100.0), 3))
            elif high_precision:
                confidence = min(0.96, round(0.75 + 0.21 * (row_pct / 100.0), 3))
            else:
                confidence = min(0.90, round(0.65 + 0.25 * (row_pct / 100.0), 3))

            findings.append(
                PIIFinding(
                    column=col_name,
                    pii_type=pii_type,
                    confidence=confidence,
                    row_percentage=row_pct,
                    matched_rows=matched,
                    detection_method=method,
                )
            )

        # Emit column-name heuristic findings not already confirmed by regex.
        for h_type in sorted(heuristic_types - regex_detected_types):
            findings.append(
                PIIFinding(
                    column=col_name,
                    pii_type=h_type,
                    confidence=0.65,
                    row_percentage=0.0,
                    matched_rows=0,
                    detection_method="column_name",
                )
            )

    return PIIReport(table_name=table_name, total_rows=total_rows, findings=findings)


def mask_pii_value(value: Any, pii_type: PIIType) -> str | None:
    """Mask a single PII value according to its detected ``PIIType``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return ""

    if pii_type == "email" and "@" in text:
        local, domain = text.split("@", 1)
        prefix = local[0] if local else "*"
        return f"{prefix}***@{domain}"
    if pii_type == "credit_card":
        digits = "".join(ch for ch in text if ch.isdigit())
        tail = digits[-4:] if len(digits) >= 4 else "****"
        return f"****-****-****-{tail}"
    if pii_type == "aadhaar":
        digits = "".join(ch for ch in text if ch.isdigit())
        tail = digits[-4:] if len(digits) >= 4 else "XXXX"
        return f"XXXX-XXXX-{tail}"
    if pii_type == "pan":
        return f"{text[:2]}*****{text[-1]}" if len(text) >= 4 else "[MASKED_PAN]"
    if pii_type == "phone":
        digits = "".join(ch for ch in text if ch.isdigit())
        tail = digits[-2:] if len(digits) >= 2 else "**"
        return f"***-***-**{tail}"
    if pii_type == "ssn":
        return "***-**-****"
    return f"[MASKED_{pii_type.upper()}]"


def hash_pii_value(value: Any, *, salt: str = "") -> str | None:
    """Return a deterministic salted SHA-256 pseudonym for a PII value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    payload = f"{salt}:{str(value).strip()}".encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()[:16]}"


def redact_inline_pii(text: str) -> str:
    """Mask stray inline PII patterns inside free-text cells before sending to an LLM."""
    sanitized = _EMAIL_RE.sub("[MASKED_EMAIL]", text)
    sanitized = _PAN_RE.sub("[MASKED_PAN]", sanitized)
    sanitized = _SSN_RE.sub("[MASKED_SSN]", sanitized)

    def _mask_cc_match(match: re.Match[str]) -> str:
        raw = match.group(0)
        return "[MASKED_CREDIT_CARD]" if is_valid_luhn(raw) else raw

    sanitized = _CC_CANDIDATE_RE.sub(_mask_cc_match, sanitized)
    sanitized = _AADHAAR_RE.sub("[MASKED_AADHAAR]", sanitized)
    sanitized = _PHONE_RE.sub("[MASKED_PHONE]", sanitized)
    return sanitized


def mask_sample_rows_for_llm(
    df: pd.DataFrame,
    pii_report: PIIReport | None = None,
    *,
    max_rows: int = 5,
    max_cell_length: int = 120,
    extra_secrets: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Build masked, sanitised sample rows safe to include in an LLM dataset profile.

    Every flagged PII column is masked via ``mask_pii_value``, unflagged string cells
    are scanned for stray inline PII, control characters are stripped, secrets are
    redacted, and lengths are capped.
    """
    report = pii_report or detect_pii(df)
    flagged = report.flagged_columns
    secrets_tuple = tuple(extra_secrets)

    sample = df.head(max_rows)
    masked_rows: list[dict[str, Any]] = []
    for _, row in sample.iterrows():
        record: dict[str, Any] = {}
        for col in sample.columns:
            col_name = str(col)
            safe_col = sanitize_for_llm(col_name, max_length=120, extra_secrets=secrets_tuple)
            val = row[col]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                record[safe_col] = None
                continue

            if col_name in flagged:
                masked = mask_pii_value(val, flagged[col_name])
                record[safe_col] = (
                    sanitize_for_llm(
                        masked,
                        max_length=max_cell_length,
                        extra_secrets=secrets_tuple,
                    )
                    if masked is not None
                    else None
                )
            elif isinstance(val, str):
                inline_masked = redact_inline_pii(val)
                record[safe_col] = sanitize_for_llm(
                    inline_masked,
                    max_length=max_cell_length,
                    extra_secrets=secrets_tuple,
                )
            else:
                record[safe_col] = val.item() if hasattr(val, "item") else val
        masked_rows.append(record)
    return masked_rows

