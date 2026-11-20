"""Unit tests for Step 7: curated ML insights, planted anomaly/trend, DL flag, and grounding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.config import Settings
from core.etl import run_etl_pipeline
from core.insights import (
    DLDisabledError,
    cluster_dimension_members,
    forecast_measure,
    forecast_with_lstm,
    insights_request_from_intent,
    phrase_insights_with_llm,
    run_insights,
    train_supervised_model,
    verify_narrative_numbers,
)
from core.llm import MockLLM
from core.schemas import InsightNarrative, InsightsRequest, UserIntent


def _make_settings(tmp_path: Path, *, enable_dl: bool = False) -> Settings:
    """Create isolated Settings rooted under pytest's temporary directory."""
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        enable_dl=enable_dl,
    )


def test_opt_in_gating_runs_nothing_unless_requested(tmp_path: Path) -> None:
    """Insight modules do not execute unless requested by UserIntent or explicit toggles."""
    plain_intent = UserIntent(goal="Show total revenue by region", time_grain="month")
    req = insights_request_from_intent(plain_intent)
    assert req.run_anomaly is False
    assert req.run_forecasting is False
    assert req.run_supervised is False
    assert req.run_clustering is False
    assert req.run_dl_forecast is False

    summary = run_insights(tmp_path, "sales", req)
    assert summary.modules_executed == []
    assert summary.lineage_events == []

    ml_intent = UserIntent(
        goal="Detect anomalies and forecast next month revenue by segment clusters",
        time_grain="month",
    )
    ml_req = insights_request_from_intent(
        ml_intent,
        date_column="dt",
        measure_column="revenue",
        dimension_column="region",
    )
    assert ml_req.run_anomaly is True
    assert ml_req.run_forecasting is True
    assert ml_req.run_clustering is True
    assert ml_req.run_dl_forecast is False


def test_planted_anomaly_and_planted_trend_on_curated_synthetic_data(tmp_path: Path) -> None:
    """Curated synthetic series catches planted anomaly and GBR beats naive on planted trend."""
    # Build SYNTHETIC 24-month series with a clear linear trend + one planted anomaly spike
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)

    anomaly_vals = 100.0 + rng.normal(0.0, 2.0, size=24)
    anomaly_vals[12] = 450.0  # Planted anomaly on 2025-01-01

    trend_vals = np.array([100.0 + 18.0 * i for i in range(24)], dtype=float)

    synth_csv = tmp_path / "SYNTHETIC_series.csv"
    pd.DataFrame(
        {
            "period_id": [f"P-{i:02d}" for i in range(24)],
            "month_date": dates.strftime("%Y-%m-%d"),
            "stable_metric": np.round(anomaly_vals, 2),
            "trending_revenue": np.round(trend_vals, 2),
        }
    ).to_csv(synth_csv, index=False)

    settings = _make_settings(tmp_path)
    etl_res = run_etl_pipeline([synth_csv], settings=settings)

    # 1. Run anomaly detection via curated-layer runner
    anom_req = InsightsRequest(
        run_anomaly=True,
        date_column="month_date",
        measure_column="stable_metric",
    )
    anom_summary = run_insights(
        etl_res.curated_dir,
        "synthetic_series",
        anom_req,
        settings=settings,
        batch_id=etl_res.batch_id,
    )
    assert anom_summary.anomaly is not None
    flagged_periods = {pt.period for pt in anom_summary.anomaly.anomalies}
    assert "2025-01-01" in flagged_periods
    planted = next(pt for pt in anom_summary.anomaly.anomalies if pt.period == "2025-01-01")
    assert planted.flagged_by == "both"
    assert planted.value == pytest.approx(450.0, abs=0.1)
    assert len(anom_summary.lineage_events) == 1
    assert anom_summary.grounding_check is not None
    assert anom_summary.grounding_check.all_numbers_grounded is True

    # 2. Run forecasting on the planted upward trend: lag GBR must beat naive baseline
    fc_req = InsightsRequest(
        run_forecasting=True,
        date_column="month_date",
        measure_column="trending_revenue",
        forecast_horizon=3,
        holdout_size=4,
    )
    fc_summary = run_insights(
        etl_res.curated_dir,
        "synthetic_series",
        fc_req,
        settings=settings,
        batch_id=etl_res.batch_id,
    )
    assert fc_summary.forecast is not None
    assert fc_summary.forecast.beats_baseline is True
    assert fc_summary.forecast.model_metrics.rmse < fc_summary.forecast.baseline_metrics.rmse
    assert fc_summary.forecast.selected_model == "lag_gradient_boosting"
    assert len(fc_summary.forecast.forecasts) == 3
    # Forecasted values must continue rising above the last observed point (100 + 18*23 = 514)
    assert fc_summary.forecast.forecasts[0].predicted > 520.0
    assert fc_summary.grounding_check is not None
    assert fc_summary.grounding_check.all_numbers_grounded is True


def test_forecast_explicitly_reports_when_it_does_not_beat_baseline() -> None:
    """When a flat constant series makes naive optimal, forecaster states it did not beat naive."""
    dates = pd.date_range("2024-01-01", periods=18, freq="MS")
    flat_df = pd.DataFrame({"dt": dates, "metric": [100.0] * 18})

    res = forecast_measure(
        flat_df,
        table_name="synthetic_flat",
        date_column="dt",
        measure_column="metric",
        horizon=2,
        holdout_size=4,
    )
    assert res.beats_baseline is False
    assert "did not beat" in res.baseline_comparison_note.lower()
    assert res.selected_model == res.baseline_name


def test_supervised_regression_leakage_detection_and_classification() -> None:
    """Supervised trainer catches leakage, reports R2/RMSE (reg) and Acc/F1/AUC (clf) + report."""
    rng = np.random.default_rng(7)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    units = rng.integers(1, 20, size=n)
    unit_price = rng.choice([50.0, 100.0, 150.0], size=n)
    region = rng.choice(["North", "South", "East"], size=n)
    revenue = units * unit_price + rng.normal(0, 15.0, size=n)

    reg_df = pd.DataFrame(
        {
            "order_id": [f"SYN-{i}" for i in range(n)],
            "order_date": dates,
            "units": units,
            "unit_price": unit_price,
            "region": region,
            "revenue": np.round(revenue, 2),
            "leaky_double_revenue": np.round(revenue * 2.0, 2),  # Planted target leakage
        }
    )

    reg_res = train_supervised_model(reg_df, table_name="synthetic_reg", target_column="revenue")
    assert reg_res.task_type == "regression"
    assert "leaky_double_revenue" in reg_res.dropped_leaky_columns
    assert "order_id" in reg_res.dropped_leaky_columns
    assert {"r2", "rmse", "cv_r2_mean"} <= set(reg_res.metrics.keys())
    assert reg_res.metrics["r2"] > 0.75
    assert len(reg_res.feature_importances) >= 2
    assert len(reg_res.limitations) >= 2
    assert "Limitations:" in reg_res.plain_report

    # Binary classification test
    churn_flag = np.where((units < 8) & (unit_price > 75.0), "Yes", "No")
    clf_df = pd.DataFrame(
        {
            "units": units,
            "unit_price": unit_price,
            "region": region,
            "churned": churn_flag,
        }
    )
    clf_res = train_supervised_model(clf_df, table_name="synthetic_clf", target_column="churned")
    assert clf_res.task_type == "classification"
    assert {"accuracy", "f1", "roc_auc", "cv_f1_mean"} <= set(clf_res.metrics.keys())
    assert clf_res.metrics["accuracy"] >= 0.75


def test_kmeans_clustering_with_silhouette_score() -> None:
    """KMeans separates clearly distinct dimension member tiers with high silhouette score."""
    df = pd.DataFrame(
        {
            "store": [f"S{i}" for i in range(1, 10)],
            "revenue": [100, 110, 105, 500, 520, 510, 950, 980, 960],
            "margin": [10, 12, 11, 35, 38, 36, 65, 68, 66],
        }
    )
    res = cluster_dimension_members(
        df,
        table_name="synthetic_stores",
        dimension_column="store",
        n_clusters=3,
    )
    assert res.n_clusters == 3
    assert res.silhouette_score > 0.7
    assert len(res.clusters) == 3
    assert len(res.member_assignments) == 9
    assert res.lineage_event.operation == "insights_clustering_kmeans"


@pytest.mark.slow
def test_dl_lstm_disabled_by_default_and_runs_when_enabled(tmp_path: Path) -> None:
    """Optional PyTorch LSTM raises DLDisabledError by default and runs when ENABLE_DL=true."""
    dates = pd.date_range("2024-01-01", periods=16, freq="MS")
    df = pd.DataFrame({"dt": dates, "rev": [100.0 + 10.0 * i for i in range(16)]})

    default_settings = _make_settings(tmp_path, enable_dl=False)
    with pytest.raises(DLDisabledError, match="ENABLE_DL=true"):
        forecast_with_lstm(
            df,
            table_name="synthetic_dl",
            date_column="dt",
            measure_column="rev",
            settings=default_settings,
        )

    enabled_settings = _make_settings(tmp_path, enable_dl=True)
    dl_res = forecast_with_lstm(
        df,
        table_name="synthetic_dl",
        date_column="dt",
        measure_column="rev",
        horizon=2,
        holdout_size=3,
        epochs=15,
        settings=enabled_settings,
    )
    assert dl_res.enabled is True
    assert dl_res.model_metrics is not None
    assert dl_res.baseline_metrics is not None
    assert len(dl_res.forecasts) == 2
    assert dl_res.lineage_event is not None


def test_llm_narrative_number_grounding_check_catches_hallucinations() -> None:
    """Every number in LLM narrative is checked against computed evidence; hallucinations fail."""
    evidence = {
        "table_name": "synthetic_sales",
        "anomaly": {
            "total_points": 24,
            "anomaly_count": 1,
            "baseline_median": 100.25,
            "measure_column": "revenue",
            "anomalies": [{"period": "2025-01-01", "value": 450.0, "robust_z_score": 18.5}],
        },
    }

    grounded_story = InsightNarrative(
        headline="Detected 1 anomaly across 24 periods",
        bullet_points=[
            "Revenue on 2025-01-01 reached 450.0 against a median of 100.25 (z-score 18.5)."
        ],
        caveats=[],
    )
    good_mock = MockLLM(responses_by_model={InsightNarrative: grounded_story})
    narrative_ok, check_ok = phrase_insights_with_llm(good_mock, evidence)
    assert narrative_ok == grounded_story
    assert check_ok.all_numbers_grounded is True
    assert check_ok.ungrounded_numbers == []

    hallucinated_story = InsightNarrative(
        headline="Revenue surged by 9999.50 across 24 periods",
        bullet_points=["Unexpected jump to 450.0 and fake gain of 777.0."],
        caveats=[],
    )
    bad_check = verify_narrative_numbers(hallucinated_story, evidence)
    assert bad_check.all_numbers_grounded is False
    assert 9999.5 in bad_check.ungrounded_numbers
    assert 777.0 in bad_check.ungrounded_numbers

