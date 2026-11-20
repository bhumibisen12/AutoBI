"""Curated-layer-only, opt-in insights runner combining ML modules, lineage, and grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import Settings, get_settings
from core.insights.anomaly import detect_time_series_anomalies
from core.insights.clustering import cluster_dimension_members
from core.insights.dl import forecast_with_lstm
from core.insights.forecasting import forecast_measure
from core.insights.narrator import (
    build_grounded_narrative,
    phrase_insights_with_llm,
    verify_narrative_numbers,
)
from core.insights.supervised import train_supervised_model
from core.llm import BaseLLMClient
from core.schemas.insights import (
    AnomalyResult,
    ClusteringResult,
    DLForecastResult,
    ForecastResult,
    InsightsRequest,
    InsightsSummary,
    SupervisedResult,
)
from core.schemas.lineage import LineageEvent
from core.schemas.llm import UserIntent
from core.storage import read_curated_table


def insights_request_from_intent(
    intent: UserIntent,
    *,
    date_column: str | None = None,
    measure_column: str | None = None,
    target_column: str | None = None,
    dimension_column: str | None = None,
) -> InsightsRequest:
    """Derive opt-in ``InsightsRequest`` toggles from explicit keywords in ``UserIntent``."""
    parts = [intent.goal, *intent.requested_visuals, *intent.focus_kpis]
    text = " ".join(parts).lower()

    run_anomaly = any(k in text for k in ("anomal", "outlier", "spike", "unusual"))
    forecast_kw = ("forecast", "predict future", "projection", "next month")
    run_forecast = any(k in text for k in forecast_kw)
    run_clustering = any(k in text for k in ("cluster", "segment", "kmeans", "cohort group"))
    run_supervised = target_column is not None and any(
        k in text for k in ("predict", "driver", "feature importance", "classify", "regression")
    )
    run_dl = any(k in text for k in ("lstm", "deep learning", "neural forecast"))

    return InsightsRequest(
        run_anomaly=run_anomaly,
        run_forecasting=run_forecast,
        run_supervised=run_supervised,
        run_clustering=run_clustering,
        run_dl_forecast=run_dl,
        date_column=date_column,
        measure_column=measure_column,
        target_column=target_column,
        dimension_column=dimension_column,
    )


def run_insights(
    curated_dir: Path,
    table_name: str,
    request: InsightsRequest,
    *,
    llm_client: BaseLLMClient | None = None,
    settings: Settings | None = None,
    batch_id: str | None = None,
) -> InsightsSummary:
    """Execute ONLY opted-in ML insight modules on a curated table, with lineage and grounding.

    Reads exclusively from ``curated_dir`` via ``core.storage.read_curated_table``.
    """
    cfg = settings or get_settings()
    if not any(
        (
            request.run_anomaly,
            request.run_forecasting,
            request.run_supervised,
            request.run_clustering,
            request.run_dl_forecast,
        )
    ):
        return InsightsSummary(batch_id=batch_id, table_name=table_name, modules_executed=[])

    df = read_curated_table(curated_dir, table_name)

    modules_executed: list[str] = []
    lineage_events: list[LineageEvent] = []
    evidence: dict[str, Any] = {"table_name": table_name}

    anomaly_res: AnomalyResult | None = None
    forecast_res: ForecastResult | None = None
    supervised_res: SupervisedResult | None = None
    clustering_res: ClusteringResult | None = None
    dl_res: DLForecastResult | None = None

    if request.run_anomaly:
        if not request.date_column or not request.measure_column:
            raise ValueError("Anomaly detection requires date_column and measure_column.")
        anomaly_res = detect_time_series_anomalies(
            df,
            table_name=table_name,
            date_column=request.date_column,
            measure_column=request.measure_column,
        )
        modules_executed.append("anomaly")
        lineage_events.append(anomaly_res.lineage_event)
        evidence["anomaly"] = anomaly_res.evidence

    if request.run_forecasting:
        if not request.date_column or not request.measure_column:
            raise ValueError("Forecasting requires date_column and measure_column.")
        forecast_res = forecast_measure(
            df,
            table_name=table_name,
            date_column=request.date_column,
            measure_column=request.measure_column,
            horizon=request.forecast_horizon,
            holdout_size=request.holdout_size,
        )
        modules_executed.append("forecasting")
        lineage_events.append(forecast_res.lineage_event)
        evidence["forecast"] = forecast_res.evidence

    if request.run_supervised:
        if not request.target_column:
            raise ValueError("Supervised learning requires an explicit target_column.")
        supervised_res = train_supervised_model(
            df,
            table_name=table_name,
            target_column=request.target_column,
        )
        modules_executed.append("supervised")
        lineage_events.append(supervised_res.lineage_event)
        evidence["supervised"] = supervised_res.evidence

    if request.run_clustering:
        if not request.dimension_column:
            raise ValueError("Clustering requires a dimension_column.")
        clustering_res = cluster_dimension_members(
            df,
            table_name=table_name,
            dimension_column=request.dimension_column,
            n_clusters=request.n_clusters,
        )
        modules_executed.append("clustering")
        lineage_events.append(clustering_res.lineage_event)
        evidence["clustering"] = clustering_res.evidence

    if request.run_dl_forecast:
        if not request.date_column or not request.measure_column:
            raise ValueError("DL forecasting requires date_column and measure_column.")
        dl_res = forecast_with_lstm(
            df,
            table_name=table_name,
            date_column=request.date_column,
            measure_column=request.measure_column,
            horizon=request.forecast_horizon,
            holdout_size=request.holdout_size,
            settings=cfg,
        )
        modules_executed.append("dl_forecast")
        if dl_res.lineage_event is not None:
            lineage_events.append(dl_res.lineage_event)
        evidence["dl_forecast"] = dl_res.evidence

    if llm_client is not None:
        narrative, grounding = phrase_insights_with_llm(llm_client, evidence)
    else:
        narrative = build_grounded_narrative(evidence)
        grounding = verify_narrative_numbers(narrative, evidence)

    return InsightsSummary(
        batch_id=batch_id,
        table_name=table_name,
        modules_executed=modules_executed,
        anomaly=anomaly_res,
        forecast=forecast_res,
        supervised=supervised_res,
        clustering=clustering_res,
        dl_forecast=dl_res,
        evidence=evidence,
        narrative=narrative,
        grounding_check=grounding,
        lineage_events=lineage_events,
    )

