"""Dataset profiling and Matplotlib/Seaborn EDA helpers."""

from core.profiling.eda import (
    plot_category_breakdown,
    plot_correlation_heatmap,
    plot_distribution,
    plot_time_trend,
)
from core.profiling.profiler import profile_dataset, profile_table

__all__ = [
    "plot_category_breakdown",
    "plot_correlation_heatmap",
    "plot_distribution",
    "plot_time_trend",
    "profile_dataset",
    "profile_table",
]

