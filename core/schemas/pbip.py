"""Pydantic contracts for PBIP generation, validation, packaging, and fallback export."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DataSourceFormat = Literal["parquet", "csv"]
ExportMode = Literal["pbip", "fallback"]
ValidationSeverity = Literal["error", "warning"]


class PBIPBuildConfig(BaseModel):
    """Configuration options controlling PBIP project generation and packaging."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(default="AutoBI_Project", min_length=1)
    source_format: DataSourceFormat = "parquet"
    data_folder_param_default: str = Field(default="../data", min_length=1)
    large_data_warning_mb: float = Field(default=100.0, gt=0.0)


class PBIPValidationIssue(BaseModel):
    """Single schema or cross-reference issue found while validating a PBIP folder."""

    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(min_length=1)
    severity: ValidationSeverity = "error"
    message: str = Field(min_length=1)


class PBIPValidationResult(BaseModel):
    """Validation outcome across all generated PBIP, PBIR, and TMDL files."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    files_checked: int = Field(default=0, ge=0)
    issues: list[PBIPValidationIssue] = Field(default_factory=list)


class PBIPExportResult(BaseModel):
    """Final export receipt for either a validated PBIP bundle or a fallback export bundle."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=1)
    mode: ExportMode
    zip_path: Path
    output_dir: Path
    pbip_file_path: Path | None = None
    files_generated: list[str] = Field(default_factory=list)
    total_size_bytes: int = Field(default=0, ge=0)
    data_size_bytes: int = Field(default=0, ge=0)
    large_data_warning: str | None = None
    unverified_measures: list[str] = Field(default_factory=list)
    validation_result: PBIPValidationResult | None = None
    fallback_reason: str | None = None

