"""Supervised ML trainer with auto feature engineering, leakage checks, CV, and plain report."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from core.lineage import build_lineage_event
from core.schemas.insights import (
    FeatureImportanceItem,
    LeakageWarning,
    SupervisedResult,
    SupervisedTaskType,
)


def _infer_task_type(target: pd.Series) -> SupervisedTaskType:
    """Classify target column as ``classification`` or ``regression``."""
    is_cat = (
        pd.api.types.is_bool_dtype(target)
        or target.dtype == "object"
        or pd.api.types.is_string_dtype(target)
    )
    if is_cat:
        return "classification"
    if pd.api.types.is_integer_dtype(target):
        n_unique = int(target.nunique(dropna=True))
        if n_unique <= 10 and n_unique < max(3, len(target) // 3):
            return "classification"
    return "regression"


def _detect_and_drop_leakage(
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    task_type: SupervisedTaskType,
) -> tuple[pd.DataFrame, list[LeakageWarning], list[str]]:
    """Detect ID columns and target-leakage columns before splitting/training."""
    warnings_list: list[LeakageWarning] = []
    drop_cols: list[str] = []
    n_rows = len(features)

    for col in list(features.columns):
        col_name = str(col)
        series = features[col]
        lowered = col_name.lower()

        # 1. Pure high-cardinality identifier columns
        if (
            lowered.endswith(("_id", "_sk", "_key", "_uuid")) or lowered == "id"
        ) and int(series.nunique(dropna=True)) >= int(0.95 * max(1, n_rows)):
            warnings_list.append(
                LeakageWarning(
                    column=col_name,
                    reason="High-cardinality identifier column dropped to prevent memorization.",
                )
            )
            drop_cols.append(col_name)
            continue

        # 2. Exact/near-perfect linear leakage for regression targets
        if task_type == "regression" and pd.api.types.is_numeric_dtype(series):
            valid = pd.concat([series, target], axis=1).dropna()
            if len(valid) >= 5 and valid.iloc[:, 0].std() > 0 and valid.iloc[:, 1].std() > 0:
                corr = abs(float(valid.iloc[:, 0].corr(valid.iloc[:, 1])))
                if corr >= 0.995:
                    warnings_list.append(
                        LeakageWarning(
                            column=col_name,
                            reason=(
                                f"Near-perfect correlation (|r|={corr:.4f}) with target "
                                f"'{target_column}' indicates target leakage."
                            ),
                            score=round(corr, 4),
                        )
                    )
                    drop_cols.append(col_name)
                    continue

        # 3. Deterministic 1-to-1 duplicate of classification target
        if task_type == "classification":
            paired = pd.DataFrame({"f": series.astype(str), "t": target.astype(str)})
            max_targets_per_val = (
                int(paired.groupby("f")["t"].nunique().max()) if not paired.empty else 2
            )
            n_feat_uniq = int(series.nunique(dropna=True))
            n_tgt_uniq = int(target.nunique(dropna=True))
            if (
                max_targets_per_val == 1
                and n_feat_uniq == n_tgt_uniq
                and n_tgt_uniq >= 2
                and (target_column.lower() in lowered or "leak" in lowered or "outcome" in lowered)
            ):
                warnings_list.append(
                    LeakageWarning(
                        column=col_name,
                        reason=(
                            f"Deterministic mapping to classification target '{target_column}' "
                            "indicates target leakage."
                        ),
                        score=1.0,
                    )
                )
                drop_cols.append(col_name)

    cleaned = features.drop(columns=drop_cols, errors="ignore")
    return cleaned, warnings_list, drop_cols


def _expand_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Expand datetime columns into calendar numeric parts (year, month, day, dayofweek)."""
    out = df.copy()
    for col in list(out.columns):
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            dt = out[col]
            out[f"{col}__year"] = dt.dt.year
            out[f"{col}__month"] = dt.dt.month
            out[f"{col}__day"] = dt.dt.day
            out[f"{col}__dayofweek"] = dt.dt.dayofweek
            out = out.drop(columns=[col])
    return out


def train_supervised_model(
    df: pd.DataFrame,
    *,
    table_name: str,
    target_column: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> SupervisedResult:
    """Train a supervised regression or classification model on a curated table."""
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in curated table '{table_name}'."
        )

    work = df.dropna(subset=[target_column]).reset_index(drop=True)
    if len(work) < 12:
        raise ValueError(
            f"Supervised training requires at least 12 non-null rows; found {len(work)}."
        )

    y_raw = work[target_column]
    task_type = _infer_task_type(y_raw)
    raw_features = work.drop(columns=[target_column])
    input_feature_cols = [str(c) for c in raw_features.columns]

    safe_features, leakage_warnings, dropped_leaky = _detect_and_drop_leakage(
        raw_features,
        y_raw,
        target_column=target_column,
        task_type=task_type,
    )
    engineered = _expand_date_features(safe_features)
    if engineered.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after leakage checks.")

    numeric_cols = [
        str(c)
        for c in engineered.columns
        if pd.api.types.is_numeric_dtype(engineered[c])
        and not pd.api.types.is_bool_dtype(engineered[c])
    ]
    categorical_cols = [str(c) for c in engineered.columns if str(c) not in numeric_cols]
    for c_col in categorical_cols:
        engineered[c_col] = engineered[c_col].astype(str)

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    limitations: list[str] = [
        "Feature importances reflect statistical association, not causal impact.",
        (
            f"Evaluated on {len(work)} curated rows; performance may shift on "
            "out-of-distribution data."
        ),
    ]
    if dropped_leaky:
        limitations.append(
            "Excluded potential leakage/ID column(s) prior to training: "
            f"{', '.join(dropped_leaky)}."
        )

    metrics: dict[str, float] = {}
    if task_type == "regression":
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        x_train, x_test, y_train, y_test = train_test_split(
            engineered, y, test_size=test_size, random_state=random_state
        )
        estimator = GradientBoostingRegressor(
            n_estimators=80,
            max_depth=3,
            random_state=random_state,
        )
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        cv_folds = min(5, max(2, len(x_train) // 4))
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_r2 = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="r2")

        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)
        metrics = {
            "r2": round(float(r2_score(y_test, preds)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, preds)), 4),
            "cv_r2_mean": round(float(np.mean(cv_r2)), 4),
        }
        model_name = "GradientBoostingRegressor"
    else:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw.astype(str))
        class_counts = np.bincount(y)
        min_class = int(class_counts.min())
        can_stratify = min_class >= 2 and len(class_counts) >= 2
        if min_class < 5:
            limitations.append(
                f"Small minority class count ({min_class} samples) limits reliability."
            )

        x_train, x_test, y_train, y_test = train_test_split(
            engineered,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if can_stratify else None,
        )
        estimator = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            random_state=random_state,
        )
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        train_min_class = int(np.bincount(y_train).min())
        cv_folds = min(5, max(2, train_min_class))
        if train_min_class >= 2:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_f1 = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="f1_weighted")
        else:
            cv_folds = 2
            cv = KFold(n_splits=2, shuffle=True, random_state=random_state)
            cv_f1 = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="f1_weighted")

        pipe.fit(x_train, y_train)
        preds = pipe.predict(x_test)
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        f1_val = f1_score(y_test, preds, average="weighted", zero_division=0)
        metrics["f1"] = round(float(f1_val), 4)
        metrics["cv_f1_mean"] = round(float(np.mean(cv_f1)), 4)

        if len(np.unique(y_test)) >= 2 and hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(x_test)
            if len(encoder.classes_) == 2 and proba.shape[1] == 2:
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba[:, 1])), 4)
            elif proba.shape[1] == len(np.unique(y_test)):
                metrics["roc_auc"] = round(
                    float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")),
                    4,
                )
        model_name = "RandomForestClassifier"

    feature_names = [str(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
    raw_importances = pipe.named_steps["model"].feature_importances_
    importances = sorted(
        [
            FeatureImportanceItem(feature=fname, importance=round(float(imp), 4))
            for fname, imp in zip(feature_names, raw_importances, strict=True)
        ],
        key=lambda item: (-item.importance, item.feature),
    )[:15]

    metric_summary = ", ".join(f"{k}={v}" for k, v in metrics.items())
    top_feats = ", ".join(f"{fi.feature} ({fi.importance})" for fi in importances[:3])
    plain_report = (
        f"Trained {model_name} for {task_type} on '{table_name}.{target_column}' "
        f"(train={len(x_train)}, test={len(x_test)}, {cv_folds}-fold CV). "
        f"Holdout metrics: {metric_summary}. Top features: {top_feats}. "
        f"Limitations: {' '.join(limitations)}"
    )

    evidence = {
        "module": "supervised",
        "table_name": table_name,
        "target_column": target_column,
        "task_type": task_type,
        "model_name": model_name,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "cv_folds": cv_folds,
        "metrics": metrics,
        "top_features": [fi.model_dump() for fi in importances[:5]],
        "dropped_leaky_columns": dropped_leaky,
    }

    lineage_event = build_lineage_event(
        table_name=table_name,
        step_index=0,
        operation=f"insights_supervised_{task_type}",
        source_columns=[target_column, *input_feature_cols],
        target_columns=[],
        parameters={"target_column": target_column, "model_name": model_name, "cv_folds": cv_folds},
        rows_before=len(df),
        rows_after=len(work),
        rows_affected=len(x_test),
        applied=True,
        note=f"{model_name} evaluated ({metric_summary}).",
    )

    return SupervisedResult(
        table_name=table_name,
        target_column=target_column,
        task_type=task_type,
        model_name=model_name,
        train_size=len(x_train),
        test_size=len(x_test),
        cv_folds=cv_folds,
        metrics=metrics,
        feature_importances=importances,
        leakage_warnings=leakage_warnings,
        dropped_leaky_columns=dropped_leaky,
        limitations=limitations,
        plain_report=plain_report,
        evidence=evidence,
        lineage_event=lineage_event,
    )

