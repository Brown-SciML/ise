"""Evaluation metrics for ice sheet emulator predictions.

This package provides metrics (e.g. R², MSE, CRPS, ECE, sector-wise sums)
for assessing point predictions and uncertainty quantification.
"""

from .metrics import (
    calculate_ece,
    crps,
    js_divergence,
    kl_divergence,
    kolmogorov_smirnov,
    mape,
    mean_absolute_error,
    mean_prediction_interval_width,
    mean_squared_error,
    mean_squared_error_sector,
    r2_score,
    relative_squared_error,
    sum_by_sector,
    t_test,
    winkler_score,
)

__all__ = [
    "calculate_ece",
    "crps",
    "js_divergence",
    "kl_divergence",
    "kolmogorov_smirnov",
    "mape",
    "mean_absolute_error",
    "mean_prediction_interval_width",
    "mean_squared_error",
    "mean_squared_error_sector",
    "r2_score",
    "relative_squared_error",
    "sum_by_sector",
    "t_test",
    "winkler_score",
]
