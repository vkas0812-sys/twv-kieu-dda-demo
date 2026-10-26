"""Stationarity diagnostics: ADF + KPSS (Wave 2)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore", category=UserWarning)


def _adf(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 20 or s.std() == 0:
        return {"stat": np.nan, "pvalue": np.nan, "ok": None}
    r = adfuller(s, autolag="AIC")
    return {"stat": round(r[0], 4), "pvalue": round(r[1], 4), "ok": r[1] < 0.05}


def _kpss(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) < 20 or s.std() == 0:
        return {"stat": np.nan, "pvalue": np.nan, "ok": None}
    try:
        r = kpss(s, regression="c", nlags="auto")
        return {"stat": round(r[0], 4), "pvalue": round(r[1], 4), "ok": r[1] >= 0.05}
    except Exception:
        return {"stat": np.nan, "pvalue": np.nan, "ok": None}


def _interpret(adf_ok, kpss_ok) -> str:
    if adf_ok is None or kpss_ok is None:
        return "Insufficient"
    if adf_ok and kpss_ok:
        return "Stationary"
    if (not adf_ok) and (not kpss_ok):
        return "Non-stationary"
    if adf_ok and (not kpss_ok):
        return "Mixed (ADF✓ KPSS✗)"
    return "Mixed (ADF✗ KPSS✓)"


def analyze_stationarity(
    df_grid: pd.DataFrame,
    sku_class: pd.DataFrame = None,
    n_top: int = 15,
    min_weeks: int = 40,
) -> pd.DataFrame:
    """
    Run ADF + KPSS on weekly demand (sum across channels per SKU).

    Tests Raw / log1p / diff1 on top-n SKUs by volume.
    """
    weekly = (
        df_grid.groupby(["Item number", "date"], as_index=False)["Quantity"]
        .sum()
        .sort_values(["Item number", "date"])
    )

    vol = weekly.groupby("Item number")["Quantity"].sum().sort_values(ascending=False)
    top_skus = vol.head(n_top).index.tolist()

    rows = []
    for sku in top_skus:
        s = weekly.loc[weekly["Item number"] == sku, "Quantity"].astype(float).reset_index(drop=True)
        if len(s) < min_weeks:
            continue

        for name, series in [
            ("Raw", s),
            ("log1p", np.log1p(s)),
            ("diff1", s.diff().dropna()),
        ]:
            adf = _adf(series)
            kps = _kpss(series)
            mean_v = float(series.mean())
            std_v = float(series.std())
            cv = (std_v / mean_v) if (name != "diff1" and mean_v > 0) else np.nan
            rows.append({
                "sku": sku,
                "transform": name,
                "n_weeks": len(series),
                "mean": round(mean_v, 2),
                "std": round(std_v, 2),
                "cv": round(cv, 3) if pd.notna(cv) else np.nan,
                "adf_stat": adf["stat"],
                "adf_p": adf["pvalue"],
                "adf_ok": adf["ok"],
                "kpss_stat": kps["stat"],
                "kpss_p": kps["pvalue"],
                "kpss_ok": kps["ok"],
                "verdict": _interpret(adf["ok"], kps["ok"]),
            })

    res = pd.DataFrame(rows)

    print("=" * 70)
    print("STATIONARITY DIAGNOSTICS (ADF + KPSS)")
    print("=" * 70)
    print(f"SKUs tested : {res['sku'].nunique()} (top {n_top} by volume)")
    print(f"Period      : {weekly['date'].min().date()} → {weekly['date'].max().date()}")
    print()

    for tr in ["Raw", "log1p", "diff1"]:
        sub = res[res["transform"] == tr]
        n = len(sub)
        n_stat = (sub["verdict"] == "Stationary").sum()
        n_non = (sub["verdict"] == "Non-stationary").sum()
        n_mix = n - n_stat - n_non
        print(f"[{tr:6s}]  Stationary {n_stat:2d}/{n}  |  Non-stat {n_non:2d}/{n}  |  Mixed {n_mix:2d}/{n}")

    print()
    pivot = (
        res.pivot_table(index="sku", columns="transform", values="verdict", aggfunc="first")
        .reindex(columns=["Raw", "log1p", "diff1"])
    )
    pivot = pivot.join(vol.rename("volume")).sort_values("volume", ascending=False)
    print("Verdict by SKU (top volume):")
    print(pivot.drop(columns=["volume"]).head(n_top).to_string())
    print()
    print("Note: LightGBM (tree-based) không yêu cầu stationarity nghiêm ngặt.")
    print("      diff1 = 100% Stationary → phù hợp nếu sau này thêm ARIMA/ETS baseline.")
    return res
