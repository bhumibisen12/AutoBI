"""Smoke tests for package imports, configuration loading, and secret-redacting logging."""

from __future__ import annotations

import importlib
import io
import json
from pathlib import Path

import pytest

from core.config import Settings, clear_settings_cache, get_settings
from core.logging import REDACTED, configure_logging, get_logger, redact_text

ALL_PACKAGES: tuple[str, ...] = (
    "app",
    "app.components",
    "app.main",
    "core",
    "core.config",
    "core.logging",
    "core.schemas",
    "core.etl",
    "core.ingestion",
    "core.profiling",
    "core.cleaning",
    "core.transformation",
    "core.validation",
    "core.storage",
    "core.lineage",
    "core.security",
    "core.llm",
    "core.rag",
    "core.semantic",
    "core.dax",
    "core.insights",
    "core.insights.anomaly",
    "core.insights.forecasting",
    "core.insights.supervised",
    "core.insights.clustering",
    "core.insights.dl",
    "core.report_spec",
    "core.pbip",
    "core.pbip.tmdl",
    "core.pbip.pbir",
    "core.pbip.templates",
    "core.pbip.validators",
    "core.orchestrator",
)


@pytest.mark.parametrize("package_name", ALL_PACKAGES)
def test_package_imports(package_name: str) -> None:
    """Every scaffolded AutoBI package and core module must import cleanly."""
    module = importlib.import_module(package_name)
    assert module is not None


def test_settings_defaults_and_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings load typed values from .env and keep secrets masked in string representations."""
    for env_key in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_MODEL",
        "EMBEDDING_MODEL",
        "MAX_UPLOAD_MB",
        "MAX_ROWS",
        "RETENTION_HOURS",
        "ENABLE_DL",
    ):
        monkeypatch.delenv(env_key, raising=False)

    env_file = tmp_path / ".env"
    secret_value = "sk-test-supersecretkey1234567890"
    env_file.write_text(
        "\n".join(
            [
                "LLM_PROVIDER=OpenAI",
                f"LLM_API_KEY={secret_value}",
                "LLM_MODEL=gpt-4o",
                "EMBEDDING_MODEL=text-embedding-3-large",
                "MAX_UPLOAD_MB=75",
                "MAX_ROWS=250000",
                "RETENTION_HOURS=12",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.max_upload_mb == 75
    assert settings.max_upload_bytes == 75 * 1024 * 1024
    assert settings.max_rows == 250_000
    assert settings.retention_hours == 12
    assert settings.enable_dl is False
    assert settings.raw_dir == Path("data/raw")
    assert settings.staging_dir == Path("data/staging")
    assert settings.curated_dir == Path("data/curated")

    # API key must never leak via str() or repr()
    assert secret_value not in repr(settings)
    assert secret_value not in str(settings)
    assert settings.known_secret_values() == (secret_value,)


def test_get_settings_cache_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing the settings cache picks up updated environment variables."""
    clear_settings_cache()
    monkeypatch.setenv("MAX_UPLOAD_MB", "42")
    try:
        assert get_settings().max_upload_mb == 42
        monkeypatch.setenv("MAX_UPLOAD_MB", "99")
        clear_settings_cache()
        assert get_settings().max_upload_mb == 99
    finally:
        clear_settings_cache()


def test_structured_logging_redacts_secrets() -> None:
    """Structured JSON logs redact configured API keys, token patterns, and sensitive extras."""
    configured_secret = "custom-provider-secret-987654"
    settings = Settings(
        _env_file=None,
        llm_api_key=configured_secret,
        log_level="INFO",
        log_json=True,
    )
    buffer = io.StringIO()
    configure_logging(settings, stream=buffer)
    logger = get_logger("tests.redaction")

    logger.info(
        "Calling provider with key=%s and bearer Bearer tok_abcdef1234567890",
        configured_secret,
        extra={
            "api_key": "sk-proj-abcdefghijklmnopqrstuvwxyz1234",
            "nested": {"password": "my-db-password", "safe": f"leaked {configured_secret}"},
            "request_id": "req-001",
        },
    )

    raw_output = buffer.getvalue().strip()
    record = json.loads(raw_output)

    assert record["level"] == "INFO"
    assert record["logger"] == "autobi.tests.redaction"
    assert record["request_id"] == "req-001"
    assert record["api_key"] == REDACTED
    assert record["nested"]["password"] == REDACTED
    assert configured_secret not in raw_output
    assert "tok_abcdef1234567890" not in raw_output
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz1234" not in raw_output
    assert "my-db-password" not in raw_output
    assert REDACTED in record["message"]


def test_redact_text_patterns() -> None:
    """Direct pattern redaction masks provider key formats and key=value pairs."""
    sample = (
        "LLM_API_KEY=mysecretvalue123 "
        "openai=sk-1234567890abcdefghijklmnop "
        "google=AIzaSyD-1234567890abcdefghijklmno"
    )
    cleaned = redact_text(sample)

    assert "mysecretvalue123" not in cleaned
    assert "sk-1234567890abcdefghijklmnop" not in cleaned
    assert "AIzaSyD-1234567890abcdefghijklmno" not in cleaned
    assert cleaned.count(REDACTED) >= 3


def test_streamlit_placeholder_page_renders() -> None:
    """The placeholder Streamlit entry point renders without exceptions or leaked secrets."""
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file("app/main.py").run()
    assert not app_test.exception
    assert [title.value for title in app_test.title] == ["AutoBI"]

