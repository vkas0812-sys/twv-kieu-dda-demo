"""Portfolio segmentation: ABC-XYZ and outlier flags (Wave 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Default service targets if caller does not pass SERVICE_TARGETS
DEFAULT_SERVICE_TARGETS = {
    "AX": 0.97, "AY": 0.94, "AZ": 0.93,
    "BX": 0.96, "BY": 0.92, "BZ": 0.90,
    "CX": 0.85, "CY": 0.80, "CZ": 0.75,
}


def classify_portfolio(
    df_grid: pd.DataFrame,
    service_targets: dict | None = None,
) -> pd.DataFrame:
    """
    ABC-XYZ + demand pattern classification on weekly grid.

    Parameters
    ----------
    df_grid : DataFrame with Item number, Channel, Quantity
    service_targets : optional map ABC_XYZ -> beta (default DEFAULT_SERVICE_TARGETS)
    """
    targets = service_targets or DEFAULT_SERVICE_TARGETS

    item_stats = (
        df_grid.groupby(["Item number", "Channel"])["Quantity"]
        .agg(["mean", "std", "sum", "count"])
        .reset_index()
    )
    item_stats["cv2"] = (item_stats["std"] / item_stats["mean"].replace(0, np.nan)) ** 2

    def calc_adi(x):
        pos = np.where(x.values > 0)[0]
        if len(pos) < 2:
            return 2.0
        return float(np.mean(np.diff(pos)))

    adi = (
        df_grid.groupby(["Item number", "Channel"])["Quantity"]
        .apply(calc_adi)
        .reset_index(name="adi")
    )
    item_stats = item_stats.merge(adi, on=["Item number", "Channel"])

    def demand_cat(row):
        if row["adi"] < 1.32:
            return "Smooth" if row["cv2"] < 0.49 else "Erratic"
        return "Intermittent" if row["cv2"] < 0.49 else "Lumpy"

    item_stats["Demand_Category"] = item_stats.apply(demand_cat, axis=1)

    total = item_stats["sum"].sum()
    item_stats = item_stats.sort_values("sum", ascending=False)
    item_stats["cum_pct"] = item_stats["sum"].cumsum() / total
    item_stats["ABC"] = item_stats["cum_pct"].apply(
        lambda x: "A" if x <= 0.80 else ("B" if x <= 0.95 else "C")
    )
    item_stats["XYZ"] = item_stats["cv2"].apply(
        lambda x: "X" if x <= 0.5 else ("Y" if x <= 1.0 else "Z")
    )
    item_stats["ABC_XYZ"] = item_stats["ABC"] + item_stats["XYZ"]
    item_stats["beta"] = item_stats["ABC_XYZ"].map(targets).fillna(0.85)

    print("Portfolio classification:")
    print(item_stats["ABC_XYZ"].value_counts().sort_index())
    print("\nDemand Category:")
    print(item_stats["Demand_Category"].value_counts())
    return item_stats


def flag_outliers(df_grid: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    """Flag |z-score| > threshold within Item×Channel groups (does not drop rows)."""
    df = df_grid.copy()
    g = df.groupby(["Item number", "Channel"])["Quantity"]
    mu = g.transform("mean")
    sigma = g.transform("std").replace(0, 1)
    df["zscore"] = (df["Quantity"] - mu) / sigma
    df["is_outlier"] = df["zscore"].abs() > z_thresh
    n = int(df["is_outlier"].sum())
    print(f"Flagged {n:,} outlier observations ({n/len(df):.1%})")
    return df
