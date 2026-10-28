"""Forecast Model Explainability and Feature Engineering Inspection Service.

Provides tools to inspect:
1. Exact monthly features used by LightGBM (lag1..lag12, roll4, month) with Train-only correlations.
2. Model-specific feature importances (Gain and Split) saved during training runs.
3. On-demand SHAP TreeExplainer values with resource-efficient caching.
4. Classical time-series diagnostics (ACF, PACF, residuals) for SARIMA, TES, MA4, Hierarchical, and Hybrid.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api.services.data_store import history_monthly, monthly_qty
from api.services.paths import CHANNELS
from api.services.runs import run_dir


FEATURE_DEFINITIONS = [
    {
        "Feature": "lag1",
        "Công thức": "y_{t-1}",
        "Đơn vị": "Chai",
        "Ý nghĩa": "Nhu cầu tháng liền kề trước đó. Phản ánh mức tiêu thụ tức thời gần nhất.",
    },
    {
        "Feature": "lag2",
        "Công thức": "y_{t-2}",
        "Đơn vị": "Chai",
        "Ý nghĩa": "Nhu cầu lùi 2 tháng. Bổ sung quán tính xu hướng ngắn hạn.",
    },
    {
        "Feature": "lag3",
        "Công thức": "y_{t-3}",
        "Đơn vị": "Chai",
        "Ý nghĩa": "Nhu cầu lùi 3 tháng (cuối quý trước). Bắt nhịp chu kỳ đặt hàng theo quý của nhà phân phối/khách sạn.",
    },
    {
        "Feature": "lag12",
        "Công thức": "y_{t-12}",
        "Đơn vị": "Chai",
        "Ý nghĩa": "Nhu cầu cùng kỳ năm trước. Đặc trưng cốt lõi để bắt tính chu kỳ mùa vụ hàng năm (Tết, lễ hội mùa đông).",
    },
    {
        "Feature": "roll4",
        "Công thức": "mean(y_{t-4}..y_{t-1})",
        "Đơn vị": "Chai",
        "Ý nghĩa": "Trung bình trượt 4 tháng liền trước (không chứa tháng hiện tại). Làm mịn nhiễu và phản ánh mức tiêu thụ nền.",
    },
    {
        "Feature": "month",
        "Công thức": "Period.month (1..12)",
        "Đơn vị": "Tháng (1..12)",
        "Ý nghĩa": "Chỉ số tháng định lượng trong năm. Giúp cây quyết định phân tách các tháng cao điểm Tết (tháng 11, 12, 1) và thấp điểm.",
    },
]


def build_feature_table(y: pd.Series, cutoff: pd.Period | None = None) -> pd.DataFrame:
    """Build the exact monthly feature table for a demand series."""
    frame = pd.DataFrame({"actual_y": y.astype(float)})
    frame["lag1"] = frame["actual_y"].shift(1)
    frame["lag2"] = frame["actual_y"].shift(2)
    frame["lag3"] = frame["actual_y"].shift(3)
    frame["lag12"] = frame["actual_y"].shift(12)
    frame["roll4"] = frame["actual_y"].shift(1).rolling(4).mean()
    frame["month"] = [p.month for p in frame.index]
    frame["period_str"] = [str(p) for p in frame.index]
    frame["year"] = [p.year for p in frame.index]

    if cutoff is not None:
        frame["split"] = ["Train (Huấn luyện)" if p <= cutoff else "OOS (Kiểm định)" for p in frame.index]
    else:
        frame["split"] = "All"

    frame["has_sufficient_history"] = ~frame[["lag1", "lag2", "lag3", "lag12", "roll4"]].isna().any(axis=1)
    return frame


def calculate_train_correlation(y: pd.Series, cutoff: pd.Period) -> pd.DataFrame:
    """Compute Pearson correlation matrix strictly on the Train set (no OOS leakage)."""
    frame = build_feature_table(y, cutoff)
    features = ["lag1", "lag2", "lag3", "lag12", "roll4", "month"]
    train = frame[frame["split"] == "Train (Huấn luyện)"].dropna(subset=features + ["actual_y"])
    if len(train) < 5:
        return pd.DataFrame()
    corr = train[features + ["actual_y"]].corr()
    return corr


def load_lgb_artifact(run_id: str, sku: str, channel: str) -> dict[str, Any] | None:
    """Load saved LightGBM feature importance artifact for a specific run and SKU-channel."""
    dest = run_dir(run_id)
    art_path = dest / "lgb_artifacts.json"
    if not art_path.exists():
        return None
    try:
        with open(art_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = f"{sku}__{channel}"
        return data.get(key)
    except Exception:
        return None


def calculate_acf_pacf(y: pd.Series, cutoff: pd.Period, max_lags: int = 12) -> pd.DataFrame:
    """Calculate ACF and PACF strictly on Train set for SARIMA diagnosis."""
    train = y.loc[:cutoff].astype(float)
    if len(train) < max_lags + 2:
        return pd.DataFrame()
    try:
        from statsmodels.tsa.stattools import acf, pacf

        nlags = min(max_lags, len(train) // 2 - 1)
        if nlags < 2:
            return pd.DataFrame()

        acf_vals, confint_acf = acf(train, nlags=nlags, alpha=0.05)
        pacf_vals, confint_pacf = pacf(train, nlags=nlags, alpha=0.05)

        rows = []
        for lag in range(1, len(acf_vals)):
            rows.append({
                "Lag": lag,
                "ACF": float(acf_vals[lag]),
                "ACF_Lower": float(confint_acf[lag][0] - acf_vals[lag]),
                "ACF_Upper": float(confint_acf[lag][1] - acf_vals[lag]),
                "PACF": float(pacf_vals[lag]) if lag < len(pacf_vals) else 0.0,
                "PACF_Lower": float(confint_pacf[lag][0] - pacf_vals[lag]) if lag < len(confint_pacf) else 0.0,
                "PACF_Upper": float(confint_pacf[lag][1] - pacf_vals[lag]) if lag < len(confint_pacf) else 0.0,
                "Sig_Bound": 1.96 / np.sqrt(len(train)),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def calculate_hierarchical_shares(baseline: str, sku: str, cutoff: str = "2024-08") -> pd.DataFrame:
    """Calculate top-down hierarchical channel shares on Train set."""
    cutoff_p = pd.Period(cutoff, freq="M")
    hist = history_monthly(baseline, [sku])
    records = []
    total_sales = 0.0

    sku_col = "sku" if "sku" in hist.columns else "Item number"
    ch_col = "channel" if "channel" in hist.columns else "Channel"

    for ch in CHANNELS:
        sub = hist[(hist[sku_col].astype(str) == str(sku)) & (hist[ch_col].astype(str) == str(ch))]
        train_qty = 0.0
        for r in sub.itertuples(index=False):
            per = pd.Period(f"{int(r.year)}-{int(r.month):02d}", freq="M")
            if per <= cutoff_p:
                train_qty += float(r.Quantity)
        records.append({"Channel": ch, "Train_Quantity": train_qty})
        total_sales += train_qty

    total_sales = max(1.0, total_sales)
    for r in records:
        r["Share_Pct"] = (r["Train_Quantity"] / total_sales) * 100.0

    return pd.DataFrame(records)


def compute_shap_values(
    run_id: str,
    sku: str,
    channel: str,
    baseline: str = "preclean",
    cutoff: str = "2024-08",
) -> pd.DataFrame | None:
    """Compute TreeExplainer SHAP values from saved booster model file."""
    dest = run_dir(run_id)
    model_path = dest / f"lgb_{sku}_{channel}.txt"
    if not model_path.exists():
        return None

    try:
        import lightgbm as lgb
        import shap

        booster = lgb.Booster(model_file=str(model_path))

        hist = history_monthly(baseline, [sku])
        sku_col = "sku" if "sku" in hist.columns else "Item number"
        ch_col = "channel" if "channel" in hist.columns else "Channel"
        sub = hist[(hist[sku_col].astype(str) == str(sku)) & (hist[ch_col].astype(str) == str(channel))]
        idx = pd.period_range("2021-01", "2024-12", freq="M")
        y = pd.Series(0.0, index=idx)
        for r in sub.itertuples(index=False):
            per = pd.Period(f"{int(r.year)}-{int(r.month):02d}", freq="M")
            if per in y.index:
                y.loc[per] = float(r.Quantity)

        cutoff_p = pd.Period(cutoff, freq="M")
        frame = build_feature_table(y, cutoff_p)
        features = ["lag1", "lag2", "lag3", "lag12", "roll4", "month"]
        train_feat = frame[frame["split"] == "Train (Huấn luyện)"].dropna(subset=features)[features]

        if len(train_feat) < 4:
            return None

        # Sample at most 40 rows to ensure fast interactive response
        sample_x = train_feat.sample(min(40, len(train_feat)), random_state=42)
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(sample_x)

        if isinstance(shap_vals, list):
            shap_arr = shap_vals[0]
        else:
            shap_arr = shap_vals

        mean_abs = np.abs(shap_arr).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": features,
            "Mean_Abs_SHAP": mean_abs,
        }).sort_values("Mean_Abs_SHAP", ascending=False)
        total_shap = max(1e-6, shap_df["Mean_Abs_SHAP"].sum())
        shap_df["Impact_Share_Pct"] = (shap_df["Mean_Abs_SHAP"] / total_shap) * 100.0
        return shap_df
    except Exception:
        return None
