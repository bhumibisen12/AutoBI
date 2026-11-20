"""Time-series forecasting: lag-feature gradient boosting evaluated against a naive baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from core.lineage import build_lineage_event
from core.schemas.insights import ForecastMetrics, ForecastPoint, ForecastResult


def compute_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
    """Compute MAE, RMSE, and MAPE (skipping zero actuals) on a holdout window."""
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    non_zero = np.abs(actual) > 1e-9
    mape = (
        round(float(np.mean(np.abs(errors[non_zero] / actual[non_zero])) * 100.0), 4)
        if np.any(non_zero)
        else None
    )
    return ForecastMetrics(mae=round(mae, 4), rmse=round(rmse, 4), mape=mape)


def _seasonal_naive_forecast(
    history: np.ndarray,
    steps: int,
    season_length: int,
) -> tuple[np.ndarray, str]:
    """Produce seasonal-naive predictions when enough history exists, else last-value naive."""
    if len(history) >= season_length and season_length >= 2:
        preds = np.array(
            [history[len(history) - season_length + (i % season_length)] for i in range(steps)],
            dtype=float,
        )
        return preds, f"seasonal_naive(s={season_length})"
    preds = np.full(steps, float(history[-1]), dtype=float)
    return preds, "naive_last_value"


def _fit_and_predict_lag_gbr(
    train_values: np.ndarray,
    steps: int,
    *,
    n_lags: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Fit a linear-detrended lag-feature GradientBoostingRegressor and forecast recursively."""
    n = len(train_values)
    effective_lags = min(n_lags, max(1, n - 3))
    t_train = np.arange(n, dtype=float).reshape(-1, 1)

    # Fit regularized linear trend so tree residuals extrapolate monotonic trends cleanly.
    trend_model = Ridge(alpha=1.0)
    trend_model.fit(t_train, train_values)
    residuals = train_values - trend_model.predict(t_train)

    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    for idx in range(effective_lags, n):
        lag_feats = [float(residuals[idx - lag]) for lag in range(1, effective_lags + 1)]
        rolling_mean = float(np.mean(residuals[idx - effective_lags : idx]))
        x_rows.append([*lag_feats, rolling_mean])
        y_vals.append(float(residuals[idx]))

    gbr = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        random_state=random_state,
    )
    gbr.fit(np.array(x_rows, dtype=float), np.array(y_vals, dtype=float))

    # Recursive multi-step forecast
    resid_buf = list(residuals.astype(float))
    preds: list[float] = []
    for step in range(steps):
        t_next = np.array([[float(n + step)]], dtype=float)
        trend_next = float(trend_model.predict(t_next)[0])
        lag_feats = [resid_buf[-lag] for lag in range(1, effective_lags + 1)]
        rolling_mean = float(np.mean(resid_buf[-effective_lags:]))
        resid_next = float(gbr.predict(np.array([[*lag_feats, rolling_mean]], dtype=float))[0])
        resid_buf.append(resid_next)
        preds.append(trend_next + resid_next)

    return np.array(preds, dtype=float)


def _infer_step_delta(dates: pd.Series) -> pd.Timedelta:
    """Estimate typical spacing between consecutive time periods for future labels."""
    diffs = dates.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(days=30)
    median_diff = diffs.median()
    return median_diff if median_diff > pd.Timedelta(0) else pd.Timedelta(days=30)


def forecast_measure(
    df: pd.DataFrame,
    *,
    table_name: str,
    date_column: str,
    measure_column: str,
    horizon: int = 4,
    holdout_size: int | None = None,
    season_length: int = 12,
    random_state: int = 42,
) -> ForecastResult:
    """Forecast a curated time-series measure with time-based holdout evaluation vs baseline."""
    if date_column not in df.columns or measure_column not in df.columns:
        raise ValueError(
            f"Columns '{date_column}' and '{measure_column}' must exist in curated table "
            f"'{table_name}'."
        )

    work = df[[date_column, measure_column]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[measure_column] = pd.to_numeric(work[measure_column], errors="coerce")
    work = work.dropna(subset=[date_column, measure_column])

    series_df = (
        work.groupby(date_column, as_index=False)[measure_column]
        .sum()
        .sort_values(date_column)
        .reset_index(drop=True)
    )
    n_points = len(series_df)
    if n_points < 8:
        raise ValueError(
            f"Forecasting requires at least 8 time periods for train/holdout; found {n_points}."
        )

    resolved_holdout = (
        holdout_size if holdout_size is not None else max(2, min(6, n_points // 4))
    )
    if n_points - resolved_holdout < 5:
        resolved_holdout = max(1, n_points - 5)

    values = series_df[measure_column].to_numpy(dtype=float)
    train_vals = values[:-resolved_holdout]
    holdout_actual = values[-resolved_holdout:]

    baseline_holdout_preds, baseline_name = _seasonal_naive_forecast(
        train_vals,
        resolved_holdout,
        season_length=season_length,
    )
    gbr_holdout_preds = _fit_and_predict_lag_gbr(
        train_vals,
        resolved_holdout,
        random_state=random_state,
    )

    baseline_metrics = compute_forecast_metrics(holdout_actual, baseline_holdout_preds)
    model_metrics = compute_forecast_metrics(holdout_actual, gbr_holdout_preds)
    beats_baseline = model_metrics.rmse < baseline_metrics.rmse

    model_name = "lag_gradient_boosting"
    if beats_baseline:
        selected_model = model_name
        comparison_note = (
            f"{model_name} beat {baseline_name} on time-based holdout "
            f"(RMSE {model_metrics.rmse} vs {baseline_metrics.rmse}; "
            f"MAE {model_metrics.mae} vs {baseline_metrics.mae})."
        )
        future_preds = _fit_and_predict_lag_gbr(values, horizon, random_state=random_state)
        residual_std = float(np.std(holdout_actual - gbr_holdout_preds))
    else:
        selected_model = baseline_name
        comparison_note = (
            f"{model_name} did not beat {baseline_name} on time-based holdout "
            f"(RMSE {model_metrics.rmse} vs {baseline_metrics.rmse}; "
            f"MAE {model_metrics.mae} vs {baseline_metrics.mae}); "
            f"using {baseline_name}."
        )
        future_preds, _ = _seasonal_naive_forecast(values, horizon, season_length=season_length)
        residual_std = float(np.std(holdout_actual - baseline_holdout_preds))

    last_date = pd.Timestamp(series_df[date_column].iloc[-1])
    step_delta = _infer_step_delta(series_df[date_column])
    forecasts: list[ForecastPoint] = []
    for step_idx in range(1, horizon + 1):
        future_dt = (last_date + step_delta * step_idx).strftime("%Y-%m-%d")
        pred_val = round(float(future_preds[step_idx - 1]), 4)
        band = round(1.96 * residual_std, 4)
        forecasts.append(
            ForecastPoint(
                period=future_dt,
                predicted=pred_val,
                lower=round(pred_val - band, 4),
                upper=round(pred_val + band, 4),
            )
        )

    evidence = {
        "module": "forecasting",
        "table_name": table_name,
        "measure_column": measure_column,
        "history_points": n_points,
        "holdout_size": resolved_holdout,
        "model_name": model_name,
        "baseline_name": baseline_name,
        "beats_baseline": beats_baseline,
        "selected_model": selected_model,
        "model_metrics": model_metrics.model_dump(),
        "baseline_metrics": baseline_metrics.model_dump(),
        "forecasts": [pt.model_dump() for pt in forecasts],
    }

    lineage_event = build_lineage_event(
        table_name=table_name,
        step_index=0,
        operation="insights_forecasting",
        source_columns=[date_column, measure_column],
        target_columns=[],
        parameters={"horizon": horizon, "holdout_size": resolved_holdout},
        rows_before=len(df),
        rows_after=n_points,
        rows_affected=horizon,
        applied=True,
        note=comparison_note,
    )

    return ForecastResult(
        table_name=table_name,
        date_column=date_column,
        measure_column=measure_column,
        model_name=model_name,
        baseline_name=baseline_name,
        holdout_size=resolved_holdout,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        beats_baseline=beats_baseline,
        selected_model=selected_model,
        baseline_comparison_note=comparison_note,
        forecasts=forecasts,
        evidence=evidence,
        lineage_event=lineage_event,
    )

