"""Tests for forecast model explainability and feature engineering."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps")]

import numpy as np
import pandas as pd
import pytest

from api.services.explainability import (
    FEATURE_DEFINITIONS,
    build_feature_table,
    calculate_acf_pacf,
    calculate_hierarchical_shares,
    calculate_train_correlation,
)


def test_feature_definitions_list():
    features = [f["Feature"] for f in FEATURE_DEFINITIONS]
    assert features == ["lag1", "lag2", "lag3", "lag12", "roll4", "month"]


def test_build_feature_table_structure():
    idx = pd.period_range("2021-01", "2024-12", freq="M")
    np.random.seed(42)
    y = pd.Series(np.random.exponential(50, len(idx)), index=idx)
    cutoff = pd.Period("2024-08", freq="M")

    frame = build_feature_table(y, cutoff)
    assert "lag1" in frame.columns
    assert "lag12" in frame.columns
    assert "roll4" in frame.columns
    assert "month" in frame.columns
    assert "split" in frame.columns

    # First 12 months must have lag12 as NaN
    assert frame.iloc[:12]["lag12"].isna().all()

    # After 12 months, lag12 has values
    assert frame.iloc[12:]["lag12"].notna().all()

    # Train vs OOS split verification
    assert (frame.loc[:cutoff, "split"] == "Train (Huấn luyện)").all()
    assert (frame.loc[cutoff + 1:, "split"] == "OOS (Kiểm định)").all()


def test_calculate_train_correlation_strictly_train():
    idx = pd.period_range("2021-01", "2024-12", freq="M")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    cutoff = pd.Period("2024-08", freq="M")

    corr = calculate_train_correlation(y, cutoff)
    assert not corr.empty
    assert "actual_y" in corr.columns
    assert "lag1" in corr.columns
    # Check bounds
    for col in corr.columns:
        for val in corr[col].dropna():
            assert -1.0001 <= val <= 1.0001


def test_acf_pacf_calculation():
    idx = pd.period_range("2021-01", "2024-08", freq="M")
    y = pd.Series(np.sin(np.linspace(0, 8 * np.pi, len(idx))) * 50 + 100, index=idx)
    cutoff = pd.Period("2024-08", freq="M")

    df = calculate_acf_pacf(y, cutoff, max_lags=12)
    assert not df.empty
    assert "Lag" in df.columns
    assert "ACF" in df.columns
    assert "PACF" in df.columns
    assert len(df) == 12


def test_hierarchical_shares_sum_to_100():
    df = calculate_hierarchical_shares("preclean", "W100133", cutoff="2024-08")
    assert not df.empty
    assert "Share_Pct" in df.columns
    assert abs(df["Share_Pct"].sum() - 100.0) < 0.01


def test_lgb_artifact_saving_and_loading_and_shap():
    from api.services.explainability import compute_shap_values, load_lgb_artifact
    from api.services.forecast_lab import run_forecast

    # Run small forecast
    out = run_forecast(["W100133"], baseline="active", cutoff="2024-08")
    run_id = out["run_id"]

    # Verify load_lgb_artifact
    art = load_lgb_artifact(run_id, "W100133", "RT")
    assert art is not None
    assert "importance_split" in art
    assert "importance_gain" in art
    assert len(art["importance_split"]) == 6
    assert len(art["importance_gain"]) == 6

    # Verify SHAP computation
    shap_df = compute_shap_values(run_id, "W100133", "RT", baseline="active", cutoff="2024-08")
    assert shap_df is not None
    assert "Feature" in shap_df.columns
    assert "Mean_Abs_SHAP" in shap_df.columns
    assert len(shap_df) == 6
    if shap_df["Mean_Abs_SHAP"].sum() > 0:
        assert abs(shap_df["Impact_Share_Pct"].sum() - 100.0) < 0.1
    else:
        assert (shap_df["Impact_Share_Pct"] == 0.0).all()
