"""Typed application configuration backed by environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Typed AutoBI runtime settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = Field(
        default="mock",
        min_length=1,
        description="LLM provider identifier (e.g., openai, anthropic, gemini, ollama, mock).",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Provider API key loaded from .env; never logged or exposed in prompts.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="Default chat/structured-output model identifier.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        min_length=1,
        description="Embedding model identifier used for local RAG.",
    )

    max_upload_mb: int = Field(
        default=50,
        ge=1,
        le=2048,
        description="Maximum permitted upload file size in megabytes.",
    )
    max_rows: int = Field(
        default=500_000,
        ge=1,
        description="Maximum permitted row count per uploaded dataset.",
    )
    retention_hours: int = Field(
        default=24,
        ge=1,
        description="Session and uploaded artifact retention window in hours.",
    )

    enable_dl: bool = Field(
        default=False,
        description="Enable optional deep-learning insights module (OFF by default).",
    )

    log_level: LogLevel = Field(
        default="INFO",
        description="Application log level.",
    )
    log_json: bool = Field(
        default=True,
        description="Emit structured JSON log records when True.",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for raw, staging, and curated data layers.",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Root directory for generated PBIP outputs and reports.",
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        """Normalize provider names to stripped lowercase strings."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Normalize log level names to stripped uppercase strings."""
        return value.strip().upper() if isinstance(value, str) else value

    @property
    def max_upload_bytes(self) -> int:
        """Return the maximum upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def raw_dir(self) -> Path:
        """Return path to the immutable raw data layer."""
        return self.data_dir / "raw"

    @property
    def staging_dir(self) -> Path:
        """Return path to the typed staging data layer."""
        return self.data_dir / "staging"

    @property
    def curated_dir(self) -> Path:
        """Return path to the curated data layer."""
        return self.data_dir / "curated"

    def known_secret_values(self) -> tuple[str, ...]:
        """Return non-empty secret values for log redaction without exposing them directly."""
        raw_key = self.llm_api_key.get_secret_value().strip()
        return (raw_key,) if raw_key else ()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance loaded from the current environment."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached `Settings` instance (primarily for test isolation)."""
    get_settings.cache_clear()

