"""Exploratory Data Analysis (EDA) plotting helpers using Matplotlib and Seaborn."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _maybe_save(fig: Figure, output_path: Path | None) -> Figure:
    """Save ``fig`` to ``output_path`` when provided and return the figure."""
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
    return fig


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    *,
    bins: int = 20,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Plot a histogram + KDE distribution with mean and median markers for a numeric column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Column '{column}' has no valid numeric values to plot.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(series, bins=bins, kde=len(series) > 2, ax=ax, color="#2563EB", alpha=0.75)
    mean_val = float(series.mean())
    median_val = float(series.median())
    ax.axvline(
        mean_val,
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:,.2f}",
    )
    ax.axvline(
        median_val,
        color="#059669",
        linestyle="-.",
        linewidth=1.5,
        label=f"Median: {median_val:,.2f}",
    )
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    return _maybe_save(fig, output_path)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Plot a Seaborn correlation heatmap across numeric columns."""
    subset = df[columns] if columns is not None else df.select_dtypes(include="number")
    numeric_df = subset.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numeric columns are required for a correlation heatmap.")

    corr = numeric_df.corr(method=method)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title(title or f"{method.title()} Correlation Heatmap")
    fig.tight_layout()
    return _maybe_save(fig, output_path)


def plot_time_trend(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    *,
    freq: str = "ME",
    agg: Literal["sum", "mean", "count"] = "sum",
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Plot an aggregated time-series trend over a parsed date column."""
    if date_column not in df.columns or value_column not in df.columns:
        raise ValueError(f"Columns '{date_column}' and '{value_column}' must exist in DataFrame.")

    work = df[[date_column, value_column]].copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[date_column, value_column]).sort_values(date_column)
    if work.empty:
        raise ValueError("No valid rows available for time trend plot.")

    grouped = work.set_index(date_column)[value_column].resample(freq).agg(agg).reset_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.lineplot(
        data=grouped,
        x=date_column,
        y=value_column,
        marker="o",
        linewidth=2.0,
        color="#2563EB",
        ax=ax,
    )
    ax.set_title(title or f"{agg.title()} of {value_column} over {date_column} ({freq})")
    ax.set_xlabel(date_column)
    ax.set_ylabel(f"{agg}({value_column})")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _maybe_save(fig, output_path)


def plot_category_breakdown(
    df: pd.DataFrame,
    category_column: str,
    value_column: str | None = None,
    *,
    agg: Literal["sum", "mean", "count"] = "sum",
    top_n: int = 10,
    title: str | None = None,
    output_path: Path | None = None,
) -> Figure:
    """Plot a horizontal bar breakdown of top-N categories by count or numeric aggregation."""
    if category_column not in df.columns:
        raise ValueError(f"Column '{category_column}' not found in DataFrame.")

    if value_column is None or agg == "count":
        summary = (
            df[category_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        summary.columns = [category_column, "metric"]
        ylabel = "Count"
    else:
        if value_column not in df.columns:
            raise ValueError(f"Column '{value_column}' not found in DataFrame.")
        work = df[[category_column, value_column]].copy()
        work[category_column] = work[category_column].fillna("Missing").astype(str)
        work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
        summary = (
            work.groupby(category_column, as_index=False)[value_column]
            .agg(agg)
            .sort_values(value_column, ascending=False)
            .head(top_n)
        )
        summary.columns = [category_column, "metric"]
        ylabel = f"{agg}({value_column})"

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    sns.barplot(
        data=summary,
        y=category_column,
        x="metric",
        hue=category_column,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title(title or f"Top {len(summary)} {category_column} by {ylabel}")
    ax.set_xlabel(ylabel)
    ax.set_ylabel(category_column)
    fig.tight_layout()
    return _maybe_save(fig, output_path)
