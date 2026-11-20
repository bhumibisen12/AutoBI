"""Time-series anomaly detection: IsolationForest plus robust z-score baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from core.lineage import build_lineage_event
from core.schemas.insights import AnomalyDetectorSource, AnomalyPoint, AnomalyResult


def detect_time_series_anomalies(
    df: pd.DataFrame,
    *,
    table_name: str,
    date_column: str,
    measure_column: str,
    z_threshold: float = 3.5,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyResult:
    """Detect anomalies on a curated measure over time using IsolationForest + robust z-score."""
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
    total_points = len(series_df)
    if total_points < 5:
        raise ValueError(
            f"Anomaly detection requires at least 5 time points; found {total_points}."
        )

    values = series_df[measure_column].to_numpy(dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        robust_z = 0.6745 * (values - median) / mad
    else:
        std_val = float(np.std(values))
        robust_z = (values - median) / std_val if std_val > 0 else np.zeros_like(values)

    # Build value + first-difference features for IsolationForest
    deltas = np.diff(values, prepend=values[0])
    features = np.column_stack([values, deltas])
    safe_contamination = min(0.25, max(1.0 / total_points, contamination))
    iso = IsolationForest(
        n_estimators=100,
        contamination=safe_contamination,
        random_state=random_state,
    )
    iso_preds = iso.fit_predict(features)
    iso_scores = iso.decision_function(features)

    anomalies: list[AnomalyPoint] = []
    for idx in range(total_points):
        rz_flag = abs(float(robust_z[idx])) > z_threshold
        iso_flag = int(iso_preds[idx]) == -1
        if not (rz_flag or iso_flag):
            continue
        source: AnomalyDetectorSource = (
            "both" if (rz_flag and iso_flag) else ("robust_z" if rz_flag else "isolation_forest")
        )
        period_str = pd.Timestamp(series_df.loc[idx, date_column]).strftime("%Y-%m-%d")
        anomalies.append(
            AnomalyPoint(
                period=period_str,
                value=round(float(values[idx]), 4),
                robust_z_score=round(float(robust_z[idx]), 4),
                isolation_forest_score=round(float(iso_scores[idx]), 4),
                flagged_by=source,
            )
        )

    evidence = {
        "module": "anomaly",
        "table_name": table_name,
        "measure_column": measure_column,
        "total_points": total_points,
        "anomaly_count": len(anomalies),
        "baseline_median": round(median, 4),
        "baseline_mad": round(mad, 4),
        "z_threshold": z_threshold,
        "anomalies": [a.model_dump() for a in anomalies],
    }

    lineage_event = build_lineage_event(
        table_name=table_name,
        step_index=0,
        operation="insights_anomaly_detection",
        source_columns=[date_column, measure_column],
        target_columns=[],
        parameters={
            "z_threshold": z_threshold,
            "contamination": round(safe_contamination, 4),
        },
        rows_before=len(df),
        rows_after=total_points,
        rows_affected=len(anomalies),
        applied=True,
        note=f"Detected {len(anomalies)} anomaly point(s) across {total_points} periods.",
    )

    return AnomalyResult(
        table_name=table_name,
        date_column=date_column,
        measure_column=measure_column,
        total_points=total_points,
        anomaly_count=len(anomalies),
        baseline_median=round(median, 4),
        baseline_mad=round(mad, 4),
        anomalies=anomalies,
        evidence=evidence,
        lineage_event=lineage_event,
    )

