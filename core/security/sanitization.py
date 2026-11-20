"""Text sanitisation for LLM prompts and spreadsheet formula-injection neutralisation."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Any

import pandas as pd

from core.logging import REDACTED, redact_text

_FORMULA_TRIGGER_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@")
_SENSITIVE_PROMPT_KEYS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "authorization",
    "credential",
    "private_key",
    "connection_url",
)


def strip_control_chars(text: str) -> str:
    """Remove ASCII/Unicode control and directional-override characters from untrusted text.

    Preserves printable characters and collapses tabs/newlines into single spaces to
    prevent prompt-structure spoofing.
    """
    cleaned_chars: list[str] = []
    for ch in text:
        if ch in ("\n", "\r", "\t"):
            cleaned_chars.append(" ")
            continue
        category = unicodedata.category(ch)
        # Drop Cc (control), Cf (format/zero-width/bidi override), Cs (surrogates)
        if category in ("Cc", "Cf", "Cs"):
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars)


def sanitize_for_llm(
    text: str,
    *,
    max_length: int = 500,
    extra_secrets: Iterable[str] = (),
) -> str:
    """Sanitise untrusted text before including it in any LLM prompt.

    Strips control/format characters, redacts known secrets and API-key patterns,
    and caps total length to ``max_length``.
    """
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    cleaned = strip_control_chars(text)
    redacted = redact_text(cleaned, extra_secrets=extra_secrets)

    if len(redacted) <= max_length:
        return redacted
    if max_length <= 3:
        return redacted[:max_length]
    return redacted[: max_length - 3] + "..."


def sanitize_prompt_payload(
    payload: Any,
    *,
    max_str_length: int = 500,
    max_key_length: int = 120,
    extra_secrets: Iterable[str] = (),
) -> Any:
    """Recursively sanitise dicts, lists, and strings destined for an LLM prompt."""
    secrets_tuple = tuple(extra_secrets)
    if isinstance(payload, str):
        return sanitize_for_llm(
            payload,
            max_length=max_str_length,
            extra_secrets=secrets_tuple,
        )
    if isinstance(payload, dict):
        sanitized_dict: dict[Any, Any] = {}
        for key, value in payload.items():
            safe_key = (
                sanitize_for_llm(key, max_length=max_key_length, extra_secrets=secrets_tuple)
                if isinstance(key, str)
                else key
            )
            if isinstance(safe_key, str) and any(
                marker in safe_key.lower() for marker in _SENSITIVE_PROMPT_KEYS
            ):
                sanitized_dict[safe_key] = REDACTED
            else:
                sanitized_dict[safe_key] = sanitize_prompt_payload(
                    value,
                    max_str_length=max_str_length,
                    max_key_length=max_key_length,
                    extra_secrets=secrets_tuple,
                )
        return sanitized_dict
    if isinstance(payload, list):
        return [
            sanitize_prompt_payload(
                item,
                max_str_length=max_str_length,
                max_key_length=max_key_length,
                extra_secrets=secrets_tuple,
            )
            for item in payload
        ]
    if isinstance(payload, tuple):
        return tuple(
            sanitize_prompt_payload(
                item,
                max_str_length=max_str_length,
                max_key_length=max_key_length,
                extra_secrets=secrets_tuple,
            )
            for item in payload
        )
    return payload


def neutralize_formula_injection(value: Any) -> Any:
    """Neutralise spreadsheet formula injection for string values exported to CSV/Excel.

    Cells whose text begins with ``=``, ``+``, ``-``, or ``@`` (including after
    leading tabs/whitespace/carriage returns) are prefixed with a leading single
    quote ``'`` so Excel/Sheets treats them strictly as literal text.
    """
    if not isinstance(value, str) or not value:
        return value
    if value.startswith("'"):
        return value
    stripped_leading = value.lstrip(" \t\r\n")
    if stripped_leading.startswith(_FORMULA_TRIGGER_PREFIXES):
        return f"'{value}"
    return value


def neutralize_dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with column headers and string cells safe for spreadsheet export."""
    out = df.copy()
    out.columns = [
        neutralize_formula_injection(col) if isinstance(col, str) else col
        for col in out.columns
    ]
    for col in out.columns:
        if out[col].dtype == "object" or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(
                lambda val: val if pd.isna(val) else neutralize_formula_injection(val)
            )
    return out

