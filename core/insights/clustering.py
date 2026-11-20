"""KMeans clustering over dimension members with silhouette score evaluation."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.lineage import build_lineage_event
from core.schemas.insights import ClusteringResult, ClusterProfile


def cluster_dimension_members(
    df: pd.DataFrame,
    *,
    table_name: str,
    dimension_column: str,
    feature_columns: list[str] | None = None,
    n_clusters: int | None = None,
    random_state: int = 42,
) -> ClusteringResult:
    """Cluster distinct ``dimension_column`` members with aggregated numeric features + KMeans."""
    if dimension_column not in df.columns:
        raise ValueError(
            f"Dimension column '{dimension_column}' not found in curated table '{table_name}'."
        )

    if feature_columns is None:
        numeric_cols = [
            str(c)
            for c in df.select_dtypes(include="number").columns
            if str(c) != dimension_column and not str(c).endswith(("_sk", "_id", "_key"))
        ]
    else:
        numeric_cols = [c for c in feature_columns if c in df.columns]

    if not numeric_cols:
        raise ValueError("Clustering requires at least one numeric feature column.")

    grouped = (
        df.dropna(subset=[dimension_column])
        .groupby(dimension_column, as_index=True)[numeric_cols]
        .mean()
        .dropna()
    )
    n_members = len(grouped)
    if n_members < 3:
        raise ValueError(
            f"Clustering requires at least 3 distinct members in '{dimension_column}'; "
            f"found {n_members}."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(grouped.to_numpy(dtype=float))

    if n_clusters is not None:
        if not (2 <= n_clusters < n_members):
            raise ValueError(
                f"n_clusters ({n_clusters}) must satisfy 2 <= n_clusters < {n_members}."
            )
        best_k = n_clusters
        best_model = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
        best_labels = best_model.fit_predict(scaled)
        best_sil = float(silhouette_score(scaled, best_labels))
    else:
        best_k = 2
        best_sil = -2.0
        best_labels = None
        for candidate_k in range(2, min(6, n_members)):
            km = KMeans(n_clusters=candidate_k, n_init=10, random_state=random_state)
            labels = km.fit_predict(scaled)
            if len(set(labels)) < 2:
                continue
            sil = float(silhouette_score(scaled, labels))
            if sil > best_sil:
                best_sil = sil
                best_k = candidate_k
                best_labels = labels
        if best_labels is None:
            raise ValueError("Unable to form at least 2 distinct clusters from input features.")

    member_names = [str(idx) for idx in grouped.index.tolist()]
    assignments = {
        name: int(label) for name, label in zip(member_names, best_labels.tolist(), strict=True)
    }

    clusters: list[ClusterProfile] = []
    for cid in sorted(set(assignments.values())):
        c_members = sorted(name for name, lbl in assignments.items() if lbl == cid)
        sub_df = grouped.loc[grouped.index.astype(str).isin(c_members)]
        centroid = {col: round(float(sub_df[col].mean()), 4) for col in numeric_cols}
        clusters.append(
            ClusterProfile(
                cluster_id=cid,
                member_count=len(c_members),
                members=c_members,
                centroid=centroid,
            )
        )

    sil_rounded = round(best_sil, 4)
    evidence = {
        "module": "clustering",
        "table_name": table_name,
        "dimension_column": dimension_column,
        "feature_columns": numeric_cols,
        "n_members": n_members,
        "n_clusters": best_k,
        "silhouette_score": sil_rounded,
        "clusters": [c.model_dump() for c in clusters],
    }

    lineage_event = build_lineage_event(
        table_name=table_name,
        step_index=0,
        operation="insights_clustering_kmeans",
        source_columns=[dimension_column, *numeric_cols],
        target_columns=[],
        parameters={"n_clusters": best_k},
        rows_before=len(df),
        rows_after=n_members,
        rows_affected=n_members,
        applied=True,
        note=(
            f"Clustered {n_members} '{dimension_column}' members into {best_k} clusters "
            f"(silhouette={sil_rounded})."
        ),
    )

    return ClusteringResult(
        table_name=table_name,
        dimension_column=dimension_column,
        feature_columns=numeric_cols,
        n_clusters=best_k,
        silhouette_score=sil_rounded,
        clusters=clusters,
        member_assignments=assignments,
        evidence=evidence,
        lineage_event=lineage_event,
    )

