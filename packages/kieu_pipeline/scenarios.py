"""SAA residual-bootstrap scenarios (Wave 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_saa_scenarios(
    forecasts_monthly: pd.DataFrame,
    resid_scale: pd.DataFrame,
    n_scenarios: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build scenario set Omega from latest monthly point forecast + residual std per SKU.
    """
    rng = np.random.default_rng(seed)
    rows = []

    latest = forecasts_monthly.groupby(["Item number", "Channel"]).tail(1)

    for _, row in latest.iterrows():
        sku, ch, point = row["Item number"], row["Channel"], max(0, row["forecast"])
        std = resid_scale.loc[resid_scale["Item number"] == sku, "resid_std"]
        scale = float(std.values[0]) if len(std) else point * 0.25

        for s in range(n_scenarios):
            noise = rng.normal(0, scale)
            rows.append({
                "sku": sku,
                "channel": ch,
                "scenario": s,
                "demand": max(0.0, point + noise),
                "point_forecast": point,
            })

    sc = pd.DataFrame(rows)
    print(
        f"Scenario set Omega: {len(sc):,} rows | {n_scenarios} scenarios | "
        f"{sc['sku'].nunique()} SKUs"
    )
    return sc


def saa_from_history(monthly_qty: pd.DataFrame, months=None) -> pd.DataFrame:
    """
    One historical month = one SAA scenario (demo / Wang–Minner style).

    monthly_qty columns: Item number, Channel, month (Period/datetime/int 1-12), Quantity.
    """
    df = monthly_qty.copy()
    if "month" not in df.columns:
        raise ValueError("monthly_qty needs column 'month'")
    if np.issubdtype(df["month"].dtype, np.datetime64):
        df["_m"] = pd.to_datetime(df["month"]).dt.month
        df["_key"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    elif hasattr(df["month"].dtype, "name") and str(df["month"].dtype).startswith("period"):
        df["_m"] = df["month"].dt.month
        df["_key"] = df["month"].astype(str)
    else:
        df["_m"] = df["month"].astype(int)
        df["_key"] = df["_m"].astype(str)

    if months is not None:
        months = [int(m) for m in months]
        df = df[df["_m"].isin(months)].copy()

    keys = sorted(df["_key"].unique())
    key_to_w = {k: i for i, k in enumerate(keys)}
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "sku": str(row["Item number"]),
            "channel": str(row["Channel"]),
            "scenario": key_to_w[row["_key"]],
            "month": row["_m"],
            "period": row["_key"],
            "demand": max(0.0, float(row["Quantity"])),
        })
    sc = pd.DataFrame(rows)
    print(
        f"SAA-hist Omega: {len(sc):,} rows | {sc['scenario'].nunique()} months | "
        f"{sc['sku'].nunique()} SKUs"
    )
    return sc


def coverage_window_starts(L: int, month_lo: int, month_hi: int) -> list:
    """Start months whose [s, s+L-1] lies inside [month_lo, month_hi]. No wrap."""
    L = int(L)
    lo, hi = int(month_lo), int(month_hi)
    if L < 1 or hi < lo:
        return []
    return list(range(lo, hi - L + 2))


def saa_from_history_coverage(
    monthly_qty: pd.DataFrame,
    L: int,
    start_months=None,
    month_lo: int | None = None,
    month_hi: int | None = None,
) -> pd.DataFrame:
    """
    One ω = sum of L consecutive months (same channel vector).

    Does not modify saa_from_history. L must be an integer (3, 4, 5, 6).
    Use L=4 or L=5 separately instead of 4.5.
    """
    L = int(L)
    df = monthly_qty.copy()
    if "month" not in df.columns:
        raise ValueError("monthly_qty needs column 'month'")
    if np.issubdtype(df["month"].dtype, np.datetime64):
        df["_m"] = pd.to_datetime(df["month"]).dt.month
    elif hasattr(df["month"].dtype, "name") and str(df["month"].dtype).startswith("period"):
        df["_m"] = df["month"].dt.month
    else:
        df["_m"] = df["month"].astype(int)

    if start_months is None:
        if month_lo is None or month_hi is None:
            raise ValueError("pass start_months or month_lo and month_hi")
        start_months = coverage_window_starts(L, month_lo, month_hi)
    start_months = [int(s) for s in start_months]
    if not start_months:
        raise ValueError(f"no full L={L} windows in requested range")

    df["Item number"] = df["Item number"].astype(str)
    df["Channel"] = df["Channel"].astype(str)
    skus = df["Item number"].unique().tolist()
    chans = df["Channel"].unique().tolist()
    grid = pd.MultiIndex.from_product([skus, chans], names=["Item number", "Channel"])
    rows = []
    for w, s in enumerate(start_months):
        win = list(range(s, s + L))
        sub = df[df["_m"].isin(win)]
        agg = (
            sub.groupby(["Item number", "Channel"])["Quantity"].sum().reindex(grid, fill_value=0.0)
            if len(sub)
            else pd.Series(0.0, index=grid)
        )
        for sku, ch in grid:
            rows.append({
                "sku": sku,
                "channel": ch,
                "scenario": w,
                "start_month": s,
                "end_month": s + L - 1,
                "L": L,
                "demand": max(0.0, float(agg.loc[(sku, ch)])),
            })
    sc = pd.DataFrame(rows)
    print(
        f"SAA-hist coverage L={L}: {len(sc):,} rows | {sc['scenario'].nunique()} windows "
        f"{start_months} | {sc['sku'].nunique()} SKUs"
    )
    return sc


def saa_from_forecast_joint(
    mu: pd.DataFrame,
    residuals: pd.DataFrame,
    clip_zero: bool = True,
) -> pd.DataFrame:
    """
    Ω_fcst for 03c C0: one ω per train month, center = 1-step μ (e.g. month 9).

    ω_{t,c} = max(0, μ_c + e_{t,c}) with e = residual vector of month t
    (same t keeps channel correlation). Does not modify saa_from_history.

    mu columns: sku, channel, mu
    residuals columns: sku, channel, month, residual
    """
    need_mu = {"sku", "channel", "mu"}
    need_e = {"sku", "channel", "month", "residual"}
    if not need_mu.issubset(mu.columns):
        raise ValueError(f"mu needs {need_mu}")
    if not need_e.issubset(residuals.columns):
        raise ValueError(f"residuals needs {need_e}")

    mu_i = mu.copy()
    mu_i["sku"] = mu_i["sku"].astype(str)
    mu_i["channel"] = mu_i["channel"].astype(str)
    mu_i["mu"] = mu_i["mu"].astype(float)
    mu_map = mu_i.set_index(["sku", "channel"])["mu"]

    e = residuals.copy()
    e["sku"] = e["sku"].astype(str)
    e["channel"] = e["channel"].astype(str)
    e["month"] = e["month"].astype(int)
    e["residual"] = e["residual"].astype(float)

    months = sorted(e["month"].unique().tolist())
    month_to_w = {m: i for i, m in enumerate(months)}
    skus = sorted(set(mu_i["sku"]).union(e["sku"]))
    chans = sorted(set(mu_i["channel"]).union(e["channel"]))

    rows = []
    for sku in skus:
        for ch in chans:
            mu_c = float(mu_map.get((sku, ch), 0.0))
            for m in months:
                sub = e[(e["sku"] == sku) & (e["channel"] == ch) & (e["month"] == m)]
                resid = float(sub["residual"].iloc[0]) if len(sub) else 0.0
                dem = mu_c + resid
                if clip_zero:
                    dem = max(0.0, dem)
                rows.append({
                    "sku": sku,
                    "channel": ch,
                    "scenario": month_to_w[m],
                    "month": m,
                    "demand": float(dem),
                    "mu": mu_c,
                    "residual": resid,
                    "omega": "fcst_joint",
                })
    sc = pd.DataFrame(rows)
    print(
        f"SAA fcst-joint: {len(sc):,} rows | {sc['scenario'].nunique()} ω "
        f"(months {months}) | {sc['sku'].nunique()} SKUs | mu=month9_plus_e"
    )
    return sc


def saa_from_forecast_coverage(
    mu: pd.DataFrame,
    residuals: pd.DataFrame,
    L: int,
    clip_zero: bool = True,
) -> pd.DataFrame:
    """
    03d: one ω per train coverage window, center = sum of next-L forecast months.

    ω_{w,c} = max(0, μ_L,c + e_{w,c})
    mu columns: sku, channel, mu
    residuals columns: sku, channel, start_month, residual
    """
    L = int(L)
    need_mu = {"sku", "channel", "mu"}
    need_e = {"sku", "channel", "start_month", "residual"}
    if not need_mu.issubset(mu.columns):
        raise ValueError(f"mu needs {need_mu}")
    if not need_e.issubset(residuals.columns):
        raise ValueError(f"residuals needs {need_e}")

    mu_i = mu.copy()
    mu_i["sku"] = mu_i["sku"].astype(str)
    mu_i["channel"] = mu_i["channel"].astype(str)
    mu_i["mu"] = mu_i["mu"].astype(float)
    mu_map = mu_i.set_index(["sku", "channel"])["mu"]

    e = residuals.copy()
    e["sku"] = e["sku"].astype(str)
    e["channel"] = e["channel"].astype(str)
    e["start_month"] = e["start_month"].astype(int)
    e["residual"] = e["residual"].astype(float)

    starts = sorted(e["start_month"].unique().tolist())
    start_to_w = {s: i for i, s in enumerate(starts)}
    skus = sorted(set(mu_i["sku"]).union(e["sku"]))
    chans = sorted(set(mu_i["channel"]).union(e["channel"]))

    rows = []
    for sku in skus:
        for ch in chans:
            mu_c = float(mu_map.get((sku, ch), 0.0))
            for s in starts:
                sub = e[(e["sku"] == sku) & (e["channel"] == ch) & (e["start_month"] == s)]
                resid = float(sub["residual"].iloc[0]) if len(sub) else 0.0
                dem = mu_c + resid
                if clip_zero:
                    dem = max(0.0, dem)
                rows.append({
                    "sku": sku,
                    "channel": ch,
                    "scenario": start_to_w[s],
                    "start_month": s,
                    "end_month": s + L - 1,
                    "L": L,
                    "demand": float(dem),
                    "mu": mu_c,
                    "residual": resid,
                    "omega": "fcst_coverage",
                })
    sc = pd.DataFrame(rows)
    print(
        f"SAA fcst-coverage L={L}: {len(sc):,} rows | {sc['scenario'].nunique()} ω "
        f"starts {starts} | {sc['sku'].nunique()} SKUs"
    )
    return sc


def saa_from_forecast_gaussian(
    mu: pd.DataFrame,
    resid_std: pd.DataFrame,
    n_scenarios: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """Ablation only: independent Gaussian per channel. Not 03c KPI."""
    rng = np.random.default_rng(seed)
    mu_i = mu.copy()
    mu_i["sku"] = mu_i["sku"].astype(str)
    mu_i["channel"] = mu_i["channel"].astype(str)
    std_map = {}
    if resid_std is not None and len(resid_std):
        tmp = resid_std.copy()
        tmp["sku"] = tmp["sku"].astype(str)
        if "resid_std" in tmp.columns:
            std_map = dict(zip(tmp["sku"], tmp["resid_std"].astype(float)))
    rows = []
    for _, row in mu_i.iterrows():
        sku, ch, point = str(row["sku"]), str(row["channel"]), float(row["mu"])
        scale = float(std_map.get(sku, abs(point) * 0.25))
        for s in range(int(n_scenarios)):
            rows.append({
                "sku": sku,
                "channel": ch,
                "scenario": s,
                "demand": max(0.0, point + float(rng.normal(0.0, scale))),
                "mu": point,
                "omega": "gaussian_ablation",
            })
    sc = pd.DataFrame(rows)
    print(f"SAA gaussian ablation: {len(sc):,} rows | {n_scenarios} ω | NOT KPI")
    return sc

