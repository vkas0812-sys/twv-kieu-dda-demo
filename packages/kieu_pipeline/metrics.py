"""Forecast accuracy metrics (Wave 1).

Nan-safe: only finite actual/predicted pairs are used.
"""
from __future__ import annotations

import numpy as np

def _finite_pair(actual, predicted):
    """Align and drop non-finite values so metrics never return nan from dirty rows."""
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]
    mask = np.isfinite(actual) & np.isfinite(predicted)
    return actual[mask], predicted[mask]


def calculate_wape(actual, predicted):
    """Weighted Absolute Percentage Error: sum|e| / sum|y| (finite pairs only)."""
    actual, predicted = _finite_pair(actual, predicted)
    denom = np.sum(np.abs(actual))
    if denom == 0 or len(actual) == 0:
        return 0.0
    return float(np.sum(np.abs(actual - predicted)) / denom)


def calculate_mape(actual, predicted):
    """Mean Absolute Percentage Error on non-zero finite actuals only."""
    actual, predicted = _finite_pair(actual, predicted)
    mask = np.abs(actual) > 1e-9
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / np.abs(actual[mask])))


def calculate_rmse(actual, predicted):
    """Root Mean Squared Error (finite pairs only)."""
    actual, predicted = _finite_pair(actual, predicted)
    if len(actual) == 0:
        return 0.0
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def calculate_bias(actual, predicted):
    """Mean(forecast - actual). >0 = systematic over-forecast (finite pairs only)."""
    actual, predicted = _finite_pair(actual, predicted)
    if len(actual) == 0:
        return 0.0
    return float(np.mean(predicted - actual))


def forecast_metric_summary(actual, predicted, label=""):
    """Return dict of WAPE / MAPE / RMSE / Bias for one series pair."""
    return {
        "label": label,
        "wape": calculate_wape(actual, predicted),
        "mape": calculate_mape(actual, predicted),
        "rmse": calculate_rmse(actual, predicted),
        "bias": calculate_bias(actual, predicted),
        "n": int(len(np.asarray(actual))),
    }


def print_forecast_metrics(actual, predicted, title="Forecast metrics"):
    """Pretty-print metric block (Phase 1 — used in lite & full)."""
    s = forecast_metric_summary(actual, predicted)
    print(f"{title}")
    print(f"   WAPE  : {s['wape']*100:6.2f}%")
    print(f"   MAPE  : {s['mape']*100:6.2f}%  (non-zero actuals only)")
    print(f"   RMSE  : {s['rmse']:8.2f}")
    print(f"   Bias  : {s['bias']:8.2f}  (>0 over-forecast)")
    return s
