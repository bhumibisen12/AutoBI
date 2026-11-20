"""Curated-layer opt-in ML insights, anomaly detection, forecasting, clustering, and grounding."""

from core.insights.anomaly import detect_time_series_anomalies
from core.insights.clustering import cluster_dimension_members
from core.insights.dl import DLDisabledError, forecast_with_lstm
from core.insights.forecasting import compute_forecast_metrics, forecast_measure
from core.insights.narrator import (
    build_grounded_narrative,
    phrase_insights_with_llm,
    verify_narrative_numbers,
)
from core.insights.runner import insights_request_from_intent, run_insights
from core.insights.supervised import train_supervised_model

__all__ = [
    "DLDisabledError",
    "build_grounded_narrative",
    "cluster_dimension_members",
    "compute_forecast_metrics",
    "detect_time_series_anomalies",
    "forecast_measure",
    "forecast_with_lstm",
    "insights_request_from_intent",
    "phrase_insights_with_llm",
    "run_insights",
    "train_supervised_model",
    "verify_narrative_numbers",
]

