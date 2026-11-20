"""Optional PyTorch LSTM forecaster — OFF by default behind ``Settings.enable_dl``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import Settings, get_settings
from core.insights.forecasting import compute_forecast_metrics
from core.lineage import build_lineage_event
from core.schemas.insights import DLForecastResult, ForecastPoint


class DLDisabledError(Exception):
    """Raised when the optional deep-learning module is invoked while ``ENABLE_DL=false``."""


def forecast_with_lstm(
    df: pd.DataFrame,
    *,
    table_name: str,
    date_column: str,
    measure_column: str,
    horizon: int = 4,
    holdout_size: int | None = None,
    lookback: int = 4,
    epochs: int = 40,
    settings: Settings | None = None,
) -> DLForecastResult:
    """Run an optional PyTorch LSTM forecaster ONLY when ``Settings.enable_dl`` is True."""
    cfg = settings or get_settings()
    if not cfg.enable_dl:
        raise DLDisabledError(
            "Optional deep-learning LSTM forecaster is disabled by default. "
            "Set ENABLE_DL=true in .env to enable it."
        )

    # Lazy import torch so standard AutoBI runs never load PyTorch unless enabled.
    import torch
    from torch import nn

    if date_column not in df.columns or measure_column not in df.columns:
        raise ValueError(
            f"Columns '{date_column}' and '{measure_column}' must exist in curated table."
        )

    work = df[[date_column, measure_column]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[measure_column] = pd.to_numeric(work[measure_column], errors="coerce")
    series_df = (
        work.dropna(subset=[date_column, measure_column])
        .groupby(date_column, as_index=False)[measure_column]
        .sum()
        .sort_values(date_column)
        .reset_index(drop=True)
    )
    n_points = len(series_df)
    if n_points < lookback + 4:
        raise ValueError(
            f"LSTM forecasting requires at least {lookback + 4} periods; found {n_points}."
        )

    resolved_holdout = holdout_size if holdout_size is not None else max(2, min(4, n_points // 4))
    values = series_df[measure_column].to_numpy(dtype=np.float32)
    train_vals = values[:-resolved_holdout]
    holdout_actual = values[-resolved_holdout:].astype(float)

    mean_v = float(np.mean(train_vals))
    std_v = float(np.std(train_vals)) or 1.0
    scaled_train = (train_vals - mean_v) / std_v

    x_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    eff_lookback = min(lookback, max(2, len(scaled_train) - 2))
    for i in range(eff_lookback, len(scaled_train)):
        x_seq.append(scaled_train[i - eff_lookback : i])
        y_seq.append(float(scaled_train[i]))

    torch.manual_seed(42)

    class _TinyLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
            self.fc = nn.Linear(16, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = _TinyLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()

    x_tensor = torch.tensor(np.array(x_seq, dtype=np.float32)).unsqueeze(-1)
    y_tensor = torch.tensor(np.array(y_seq, dtype=np.float32)).unsqueeze(-1)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

    def _roll_forecast(seed_scaled: list[float], steps: int) -> np.ndarray:
        model.eval()
        buf = list(seed_scaled)
        preds_out: list[float] = []
        with torch.no_grad():
            for _ in range(steps):
                window = np.array(buf[-eff_lookback:], dtype=np.float32).reshape(1, eff_lookback, 1)
                nxt = float(model(torch.tensor(window)).item())
                buf.append(nxt)
                preds_out.append(nxt * std_v + mean_v)
        return np.array(preds_out, dtype=float)

    holdout_preds = _roll_forecast(scaled_train.tolist(), resolved_holdout)
    naive_preds = np.full(resolved_holdout, float(train_vals[-1]), dtype=float)
    model_metrics = compute_forecast_metrics(holdout_actual, holdout_preds)
    baseline_metrics = compute_forecast_metrics(holdout_actual, naive_preds)
    beats = model_metrics.rmse < baseline_metrics.rmse

    full_scaled = ((values - mean_v) / std_v).tolist()
    future_preds = _roll_forecast(full_scaled, horizon)

    diffs = series_df[date_column].diff().dropna()
    step_delta = diffs.median() if not diffs.empty else pd.Timedelta(days=30)
    last_date = pd.Timestamp(series_df[date_column].iloc[-1])
    forecasts = [
        ForecastPoint(
            period=(last_date + step_delta * (i + 1)).strftime("%Y-%m-%d"),
            predicted=round(float(future_preds[i]), 4),
        )
        for i in range(horizon)
    ]

    note = (
        f"PyTorch LSTM holdout RMSE={model_metrics.rmse} vs naive RMSE={baseline_metrics.rmse} "
        f"({'beat' if beats else 'did not beat'} naive baseline)."
    )
    evidence = {
        "module": "dl_forecast",
        "table_name": table_name,
        "measure_column": measure_column,
        "holdout_size": resolved_holdout,
        "beats_baseline": beats,
        "model_metrics": model_metrics.model_dump(),
        "baseline_metrics": baseline_metrics.model_dump(),
        "forecasts": [f.model_dump() for f in forecasts],
    }
    lineage_event = build_lineage_event(
        table_name=table_name,
        step_index=0,
        operation="insights_dl_lstm_forecast",
        source_columns=[date_column, measure_column],
        target_columns=[],
        parameters={"lookback": eff_lookback, "epochs": epochs, "horizon": horizon},
        rows_before=len(df),
        rows_after=n_points,
        rows_affected=horizon,
        applied=True,
        note=note,
    )

    return DLForecastResult(
        enabled=True,
        table_name=table_name,
        date_column=date_column,
        measure_column=measure_column,
        holdout_size=resolved_holdout,
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        beats_baseline=beats,
        forecasts=forecasts,
        note=note,
        evidence=evidence,
        lineage_event=lineage_event,
    )

