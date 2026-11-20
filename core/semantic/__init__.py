"""Curated-layer Power BI semantic model builder, metrics registry, and KPI selector."""

from core.semantic.builder import SemanticModelError, build_semantic_model
from core.semantic.kpi_selector import (
    InvalidMetricSelectionError,
    build_kpi_selection_prompt,
    select_kpis,
    suggest_default_metric_selections,
    validate_metric_selection_plan,
)
from core.semantic.registry import default_metrics_registry

__all__ = [
    "InvalidMetricSelectionError",
    "SemanticModelError",
    "build_kpi_selection_prompt",
    "build_semantic_model",
    "default_metrics_registry",
    "select_kpis",
    "suggest_default_metric_selections",
    "validate_metric_selection_plan",
]

