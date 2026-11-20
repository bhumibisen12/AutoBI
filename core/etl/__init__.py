"""ETL pipeline orchestration across raw, staging, quarantine, and curated layers."""

from core.etl.pipeline import run_etl_pipeline

__all__ = ["run_etl_pipeline"]

