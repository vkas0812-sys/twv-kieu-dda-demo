"""Publish targets and evaluation summary (Wave 2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from kieu_pipeline.metrics import calculate_wape


def publish_targets(policy_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Write published_base_stock_targets.csv and return the table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    cols = [c for c in ["sku", "y_star", "beta", "TEC", "fill_rate", "CL", "WC"] if c in policy_df.columns]
    targets = policy_df[cols].copy()
    targets["location"] = "CIW+buffers"
    targets["policy"] = "Kieu-DDA-VP"
    targets["published_at"] = pd.Timestamp.now()
    path = out_dir / "published_base_stock_targets.csv"
    targets.to_csv(path, index=False)
    print(f"Published -> {path}")
    return targets


def evaluate(policy_df, forecasts_monthly=None, stationarity_report=None):
    """Print evaluation KPI block (inventory + optional forecast WAPE + stationary %)."""
    print("Evaluation Summary — Kieu-DDA-VP")
    print(f"  SKUs optimized     : {len(policy_df)}")
    if "TEC" in policy_df.columns:
        print(f"  Avg TEC           : {policy_df['TEC'].mean():.2f}")
    if "y_star" in policy_df.columns:
        print(f"  Avg base-stock y* : {policy_df['y_star'].mean():.1f}")
        print(f"  Total y*          : {policy_df['y_star'].sum():.0f}")
    if "fill_rate" in policy_df.columns:
        print(f"  Avg fill-rate     : {policy_df['fill_rate'].mean()*100:.1f}%")
    if "CL" in policy_df.columns:
        print(f"  Avg CL            : {policy_df['CL'].mean():.2f}")
    if "WC" in policy_df.columns:
        print(f"  Total WC          : {policy_df['WC'].sum():.0f}")
    if forecasts_monthly is not None and "actual" in forecasts_monthly.columns:
        w = calculate_wape(forecasts_monthly["actual"], forecasts_monthly["forecast"])
        print(f"  Forecast WAPE     : {w*100:.2f}%")
    if stationarity_report is not None and len(stationarity_report) > 0:
        raw = stationarity_report[stationarity_report["transform"] == "Raw"]
        if len(raw) > 0:
            pct = (raw["verdict"] == "Stationary").mean() * 100
            print(f"  SKU stationary %  : {pct:.0f}% (Raw, top-{len(raw)} by volume)")
    return policy_df
