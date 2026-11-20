"""Pydantic contracts for curated-layer ML insights, evidence grounding, and narratives."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.schemas.lineage import LineageEvent

AnomalyDetectorSource = Literal["robust_z", "isolation_forest", "both"]
SupervisedTaskType = Literal["regression", "classification"]


class InsightsRequest(BaseModel):
    """Explicit opt-in toggles and parameters controlling which ML insight modules run."""

    model_config = ConfigDict(extra="forbid")

    run_anomaly: bool = False
    run_forecasting: bool = False
    run_supervised: bool = False
    run_clustering: bool = False
    run_dl_forecast: bool = False

    date_column: str | None = None
    measure_column: str | None = None
    target_column: str | None = None
    dimension_column: str | None = None
    forecast_horizon: int = Field(default=4, ge=1, le=36)
    holdout_size: int | None = Field(default=None, ge=1)
    n_clusters: int | None = Field(default=None, ge=2, le=20)


class AnomalyPoint(BaseModel):
    """Single flagged anomalous time period with exact statistical scores."""

    model_config = ConfigDict(extra="forbid")

    period: str
    value: float
    robust_z_score: float
    isolation_forest_score: float
    flagged_by: AnomalyDetectorSource


class AnomalyResult(BaseModel):
    """Output of time-series anomaly detection (IsolationForest + robust z-score baseline)."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    date_column: str
    measure_column: str
    total_points: int = Field(ge=0)
    anomaly_count: int = Field(ge=0)
    baseline_median: float
    baseline_mad: float
    anomalies: list[AnomalyPoint] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    lineage_event: LineageEvent


class ForecastMetrics(BaseModel):
    """Time-based holdout error metrics."""

    model_config = ConfigDict(extra="forbid")

    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    mape: float | None = Field(default=None, ge=0.0)


class ForecastPoint(BaseModel):
    """Single forecasted future period."""

    model_config = ConfigDict(extra="forbid")

    period: str
    predicted: float
    lower: float | None = None
    upper: float | None = None


class ForecastResult(BaseModel):
    """Holdout-evaluated forecast comparing lag-feature GBR against a naive baseline."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    date_column: str
    measure_column: str
    model_name: str
    baseline_name: str
    holdout_size: int = Field(ge=1)
    model_metrics: ForecastMetrics
    baseline_metrics: ForecastMetrics
    beats_baseline: bool
    selected_model: str
    baseline_comparison_note: str
    forecasts: list[ForecastPoint] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    lineage_event: LineageEvent


class FeatureImportanceItem(BaseModel):
    """Feature importance score from a trained supervised model."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    importance: float = Field(ge=0.0)


class LeakageWarning(BaseModel):
    """Detected potential target leakage feature."""

    model_config = ConfigDict(extra="forbid")

    column: str
    reason: str
    score: float | None = None


class SupervisedResult(BaseModel):
    """Supervised learning training, validation, leakage-check, and feature-importance report."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    target_column: str
    task_type: SupervisedTaskType
    model_name: str
    train_size: int = Field(ge=1)
    test_size: int = Field(ge=1)
    cv_folds: int = Field(ge=2)
    metrics: dict[str, float] = Field(default_factory=dict)
    feature_importances: list[FeatureImportanceItem] = Field(default_factory=list)
    leakage_warnings: list[LeakageWarning] = Field(default_factory=list)
    dropped_leaky_columns: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    plain_report: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    lineage_event: LineageEvent


class ClusterProfile(BaseModel):
    """Summary statistics for a single KMeans cluster of dimension members."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: int = Field(ge=0)
    member_count: int = Field(ge=1)
    members: list[str] = Field(default_factory=list)
    centroid: dict[str, float] = Field(default_factory=dict)


class ClusteringResult(BaseModel):
    """KMeans clustering outcome over dimension members with silhouette score."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    dimension_column: str
    feature_columns: list[str] = Field(default_factory=list)
    n_clusters: int = Field(ge=2)
    silhouette_score: float = Field(ge=-1.0, le=1.0)
    clusters: list[ClusterProfile] = Field(default_factory=list)
    member_assignments: dict[str, int] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    lineage_event: LineageEvent


class DLForecastResult(BaseModel):
    """Optional PyTorch LSTM forecaster output (disabled by default behind ENABLE_DL)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    table_name: str
    date_column: str
    measure_column: str
    model_name: str = "pytorch_lstm"
    holdout_size: int = Field(default=0, ge=0)
    model_metrics: ForecastMetrics | None = None
    baseline_metrics: ForecastMetrics | None = None
    beats_baseline: bool | None = None
    forecasts: list[ForecastPoint] = Field(default_factory=list)
    note: str = Field(default="")
    evidence: dict[str, Any] = Field(default_factory=dict)
    lineage_event: LineageEvent | None = None


class InsightNarrative(BaseModel):
    """LLM-phrased executive insights strictly grounded in code-computed evidence numbers."""

    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1)
    bullet_points: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class NumberGroundingCheck(BaseModel):
    """Verification proving every numeric value in the narrative appears in computed evidence."""

    model_config = ConfigDict(extra="forbid")

    all_numbers_grounded: bool
    narrative_numbers: list[float] = Field(default_factory=list)
    ungrounded_numbers: list[float] = Field(default_factory=list)


class InsightsSummary(BaseModel):
    """Combined output of all opted-in insight modules plus verified narrative and lineage."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str | None = None
    table_name: str
    modules_executed: list[str] = Field(default_factory=list)
    anomaly: AnomalyResult | None = None
    forecast: ForecastResult | None = None
    supervised: SupervisedResult | None = None
    clustering: ClusteringResult | None = None
    dl_forecast: DLForecastResult | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    narrative: InsightNarrative | None = None
    grounding_check: NumberGroundingCheck | None = None
    lineage_events: list[LineageEvent] = Field(default_factory=list)

