"""Phase-3 forecast on monthly series. Fit only on ym <= 202408.

Hybrid = mean(LightGBM, SARIMA). Ensemble = LightGBM (03c LGB side).
No 0.3 seasonal-naive mix.
"""
from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd

from kieu_pipeline.metrics import calculate_bias, calculate_mape, calculate_rmse, calculate_wape

from api.services.data_store import history_monthly, sku_abc_map, sku_name
from api.services.paths import CHANNELS, CUTOFF, RANDOM_SEED, S4, S4_CLASS
from api.services.runs import new_run_id, run_dir, write_manifest

warnings.filterwarnings("ignore")


def _ym(year: int, month: int) -> int:
    return int(year) * 100 + int(month)


def _panel(hist: pd.DataFrame, sku: str, channel: str) -> pd.Series:
    sub = hist[(hist["sku"] == sku) & (hist["channel"] == channel)]
    idx = pd.period_range("2021-01", "2024-12", freq="M")
    s = pd.Series(0.0, index=idx)
    for row in sub.itertuples(index=False):
        per = pd.Period(f"{int(row.year)}-{int(row.month):02d}", freq="M")
        if per in s.index:
            s.loc[per] = float(row.Quantity)
    return s


def _ma4(y: pd.Series, cutoff: pd.Period) -> pd.Series:
    out = pd.Series(np.nan, index=y.index)
    vals = y.to_numpy()
    for i in range(len(y)):
        window = vals[max(0, i - 4): i]
        if len(window):
            out.iloc[i] = float(np.mean(window))
    return out


def _snaive(y: pd.Series) -> pd.Series:
    return y.shift(12)


def _tes(y: pd.Series, cutoff: pd.Period) -> tuple[pd.Series, bool]:
    out = pd.Series(np.nan, index=y.index)
    train = y.loc[:cutoff]
    if (train > 0).sum() < 24:
        return out, False
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        fit = ExponentialSmoothing(
            train.astype(float), trend="add", seasonal="add", seasonal_periods=12
        ).fit(optimized=True)
        out.loc[train.index] = np.asarray(fit.fittedvalues, dtype=float)
        fut = fit.forecast(4)
        hold = y.loc[cutoff + 1: cutoff + 4].index
        for i, per in enumerate(hold):
            if i < len(fut):
                out.loc[per] = float(fut.iloc[i])
        return out, True
    except Exception:
        return out, False


def _sarima(y: pd.Series, cutoff: pd.Period) -> tuple[pd.Series, bool]:
    out = pd.Series(np.nan, index=y.index)
    train = y.loc[:cutoff]
    if len(train) < 24:
        return out, False
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            fit = SARIMAX(
                train.astype(float),
                order=(1, 0, 0),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=25)
        out.loc[train.index] = np.asarray(fit.fittedvalues, dtype=float)
        fut = fit.forecast(4)
        hold = y.loc[cutoff + 1: cutoff + 4].index
        for i, per in enumerate(list(hold)):
            out.loc[per] = float(fut.iloc[i])
        return out, True
    except Exception:
        return out, False


def _lgb_one(y: pd.Series, cutoff: pd.Period) -> tuple[pd.Series, bool]:
    out = pd.Series(np.nan, index=y.index)
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        return out, False
    frame = pd.DataFrame({"y": y.astype(float)})
    frame["lag1"] = frame["y"].shift(1)
    frame["lag2"] = frame["y"].shift(2)
    frame["lag3"] = frame["y"].shift(3)
    frame["lag12"] = frame["y"].shift(12)
    frame["roll4"] = frame["y"].shift(1).rolling(4).mean()
    frame["month"] = [p.month for p in frame.index]
    feat = ["lag1", "lag2", "lag3", "lag12", "roll4", "month"]
    train = frame.loc[:cutoff].dropna()
    if len(train) < 16 or float(train["y"].sum()) <= 0 or int((train["y"] > 0).sum()) < 2:
        return out, False
    try:
        model = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            objective="tweedie",
            tweedie_variance_power=1.5,
            random_state=RANDOM_SEED,
            verbose=-1,
        )
        model.fit(train[feat], train["y"])
        pred_idx = frame.dropna(subset=["lag1", "lag12"]).index
        if len(pred_idx) == 0:
            return out, False
        pred = model.predict(frame.loc[pred_idx, feat])
        out.loc[pred_idx] = np.clip(pred, 0, None)
        return out, True
    except Exception:
        return out, False


def rule_tag(abc_xyz: str, smooth: bool, sarima_ok: bool, lgb_ok: bool, tes_ok: bool) -> str:
    abc = abc_xyz[:1]
    if abc_xyz == "AX" and sarima_ok and lgb_ok:
        return "Hybrid"
    if abc_xyz == "AY" or smooth:
        return "TES" if tes_ok else "MA_4"
    if abc == "A":
        if sarima_ok:
            return "SARIMA"
        if tes_ok:
            return "TES"
        return "MA_4"
    if abc == "B":
        return "Ensemble" if lgb_ok else "MA_4"
    if abc_xyz == "CZ":
        return "Hierarchical"
    if abc == "C":
        return "MA_4"
    return "MA_4"


def _pick(tag: str, bundle: dict[str, pd.Series]) -> pd.Series:
    if tag == "Hybrid":
        return (bundle["LightGBM"].fillna(bundle["SARIMA"]) + bundle["SARIMA"].fillna(bundle["LightGBM"])) / 2
    if tag == "Ensemble":
        return bundle["LightGBM"]
    if tag == "Hierarchical":
        return bundle["Hierarchical"]
    if tag == "TES":
        return bundle["TES"].fillna(bundle["MA_4"])
    if tag == "SARIMA":
        return bundle["SARIMA"].fillna(bundle["MA_4"])
    if tag == "SeasonalNaive":
        return bundle["SeasonalNaive"]
    return bundle["MA_4"]


def run_forecast(
    sku_list: list[str] | None = None,
    baseline: str = "preclean",
    cutoff: str = CUTOFF,
    run_id: str | None = None,
) -> dict:
    skus = [str(s) for s in (sku_list or S4)]
    cutoff_p = pd.Period(cutoff, freq="M")
    hist = history_monthly(baseline, skus)
    abc_lookup = sku_abc_map(baseline)
    rows = []
    run_id = run_id or new_run_id("fcst")
    for sku in skus:
        abc = S4_CLASS.get(sku) or abc_lookup.get(sku, "CZ")
        # Hierarchical base: SKU MA4 × channel share of 2024 train.
        shares = {}
        for ch in CHANNELS:
            y = _panel(hist, sku, ch)
            train = y.loc[:cutoff_p]
            shares[ch] = float(train.sum())
        total = sum(shares.values()) or 1.0
        for ch in CHANNELS:
            y = _panel(hist, sku, ch)
            ma4 = _ma4(y, cutoff_p)
            snaive = _snaive(y)
            tes, tes_ok = _tes(y, cutoff_p)
            sarima, sarima_ok = _sarima(y, cutoff_p)
            lgb, lgb_ok = _lgb_one(y, cutoff_p)
            sku_total = sum(_panel(hist, sku, c).loc[:cutoff_p].iloc[-4:].mean() for c in CHANNELS)
            hier = pd.Series(sku_total * shares[ch] / total, index=y.index)
            bundle = {
                "MA_4": ma4,
                "SeasonalNaive": snaive,
                "TES": tes,
                "SARIMA": sarima,
                "LightGBM": lgb,
                "Hierarchical": hier,
            }
            tag = rule_tag(abc, smooth=(abc == "AY"), sarima_ok=sarima_ok, lgb_ok=lgb_ok, tes_ok=tes_ok)
            chosen = _pick(tag, bundle)
            for name, series in {**bundle, "Optimized_Best": chosen}.items():
                for per in y.index:
                    split = "train" if per <= cutoff_p else "oos"
                    if split == "oos" and per > cutoff_p + 4:
                        continue
                    if per.year < 2024 and name != "Optimized_Best":
                        continue
                    actual = float(y.loc[per])
                    yhat = series.loc[per]
                    yhat_f = None if pd.isna(yhat) else float(max(0.0, yhat))
                    resid = None if yhat_f is None else actual - yhat_f
                    rows.append({
                        "sku": sku,
                        "item_name": sku_name(sku),
                        "ABC": abc,
                        "channel": ch,
                        "year": per.year,
                        "month": per.month,
                        "actual": actual,
                        "yhat": yhat_f,
                        "model": name,
                        "rule_tag": tag,
                        "split": "train_1_8" if per <= cutoff_p else "oos_9_12",
                        "residual": resid,
                        "sarima_ok": sarima_ok,
                        "lgb_ok": lgb_ok,
                    })
    df = pd.DataFrame(rows)
    dest = run_dir(run_id)
    df.to_csv(dest / "forecast.csv", index=False)
    best = df[(df["model"] == "Optimized_Best") & (df["year"] == 2024) & (df["month"] <= 8)]
    residuals = best.rename(columns={"residual": "residual"})[
        ["sku", "channel", "month", "residual"]
    ].dropna()
    residuals.to_csv(dest / "residuals.csv", index=False)
    mu_rows = []
    cov_rows = []
    for (sku, ch), g in df[df["model"] == "Optimized_Best"].groupby(["sku", "channel"]):
        g = g.sort_values(["year", "month"])
        m9 = g[(g["year"] == 2024) & (g["month"] == 9)]
        mu_rows.append({"sku": sku, "channel": ch, "mu": float(m9["yhat"].iloc[0] or 0) if len(m9) else 0.0})
        yhat_map = {
            int(r.month): (0.0 if pd.isna(r.yhat) else float(r.yhat))
            for r in g[g["year"] == 2024].itertuples(index=False)
        }
        for L in (3, 4, 5, 6):
            mu_l = sum(yhat_map.get(m, 0.0) for m in range(9, 9 + L))
            cov_rows.append({"sku": sku, "channel": ch, "L": L, "mu": mu_l})
    # Window residuals for L in 3..6 use start_month. Write L-specific files.
    win_rows = []
    for L in (3, 4, 5, 6):
        for (sku, ch), g in best.groupby(["sku", "channel"]):
            rmap = {int(r.month): float(r.residual) for r in g.itertuples(index=False) if pd.notna(r.residual)}
            for start in range(1, 8 - L + 2):
                if any(m not in rmap for m in range(start, start + L)):
                    continue
                win_rows.append({
                    "sku": sku,
                    "channel": ch,
                    "L": L,
                    "start_month": start,
                    "residual": sum(rmap[m] for m in range(start, start + L)),
                })
    pd.DataFrame(mu_rows).to_csv(dest / "mu_joint.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(dest / "mu_coverage.csv", index=False)
    pd.DataFrame(win_rows).to_csv(dest / "resid_coverage.csv", index=False)

    metrics = []
    for keys, g in df.dropna(subset=["yhat"]).groupby(["sku", "channel", "model", "split"]):
        metrics.append({
            "sku": keys[0],
            "item_name": sku_name(keys[0]),
            "ABC": S4_CLASS.get(keys[0]) or abc_lookup.get(keys[0], "CZ"),
            "channel": keys[1],
            "model": keys[2],
            "split": keys[3],
            "wape": calculate_wape(g["actual"], g["yhat"]),
            "mape": calculate_mape(g["actual"], g["yhat"]),
            "rmse": calculate_rmse(g["actual"], g["yhat"]),
            "bias": calculate_bias(g["actual"], g["yhat"]),
            "n": int(len(g)),
        })
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(dest / "wape.csv", index=False)
    write_manifest(
        run_id,
        {
            "kind": "forecast",
            "sku_list": skus,
            "cutoff": cutoff,
            "baseline": baseline,
            "L": None,
            "omega_type": None,
        },
    )
    return {
        "run_id": run_id,
        "cutoff": cutoff,
        "n": int(len(df)),
        "metrics": mdf.where(mdf.notna(), None).to_dict(orient="records"),
        "rule_tags": (
            df[df["model"] == "Optimized_Best"][["sku", "item_name", "ABC", "channel", "rule_tag"]]
            .drop_duplicates()
            .to_dict(orient="records")
        ),
    }


def load_forecast_tables(run_id: str, L: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dest = run_dir(run_id)
    if L is None:
        mu = pd.read_csv(dest / "mu_joint.csv")
        resid = pd.read_csv(dest / "residuals.csv")
        return mu, resid
    mu = pd.read_csv(dest / "mu_coverage.csv")
    mu = mu[mu["L"] == int(L)][["sku", "channel", "mu"]]
    resid = pd.read_csv(dest / "resid_coverage.csv")
    resid = resid[resid["L"] == int(L)][["sku", "channel", "start_month", "residual"]]
    return mu, resid
