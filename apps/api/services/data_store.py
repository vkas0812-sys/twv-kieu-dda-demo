"""Sales and COGS loaders. Column rename stays in memory."""
from __future__ import annotations

import contextlib
import io
from functools import lru_cache
from pathlib import Path

import pandas as pd

from kieu_pipeline.segmentation import DEFAULT_SERVICE_TARGETS, classify_portfolio

from api.services.paths import BASELINES, CHANNELS, COGS_PATH, PARQUET, S4, S4_CLASS, THESIS


def available_baselines() -> list[str]:
    out = ["preclean", "active"]
    if BASELINES["upload"].exists():
        out.append("upload")
    return out


def _sales_path(baseline: str) -> Path:
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {list(BASELINES)}, got {baseline}")
    path = BASELINES[baseline]
    if path.exists():
        return path
    alt_map = {
        "preclean": THESIS / "database" / "01.preclean" / "sales_data_clean_baseline.parquet",
        "active": THESIS / "database" / "02.lifecycle" / "sales_data_active_baseline.parquet",
        "upload": PARQUET / "sales_upload.parquet",
    }
    alt = alt_map.get(baseline)
    if alt and alt.exists():
        return alt
    raise FileNotFoundError(str(path))


def cogs_path() -> Path:
    if COGS_PATH.exists():
        return COGS_PATH
    alt = THESIS / "database" / "00.raw data" / "COGS.parquet"
    if alt.exists():
        return alt
    raise FileNotFoundError(str(COGS_PATH))


@lru_cache(maxsize=4)
def load_sales(baseline: str = "preclean") -> pd.DataFrame:
    df = pd.read_parquet(_sales_path(baseline))
    need = {"Item number", "Channel", "Quantity", "Year", "Month"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"sales missing columns {sorted(missing)}")
    return df


@lru_cache(maxsize=1)
def load_product_catalog() -> pd.DataFrame:
    df = pd.read_parquet(cogs_path()).copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Item number" not in df.columns:
        raise ValueError(f"COGS missing 'Item number': {list(df.columns)}")
    df["sku"] = df["Item number"].astype(str).str.strip()
    name_col = "Item Name" if "Item Name" in df.columns else None
    df["item_name"] = df[name_col].fillna("").astype(str).str.strip() if name_col else df["sku"]
    cogs_col = next((c for c in ("COGS (VND)", "cogs_vnd", "COGS", "cogs") if c in df.columns), None)
    retail_col = next((c for c in ("Retail Price (VND)", "retail_vnd", "Retail Price") if c in df.columns), None)
    df["cogs_vnd"] = pd.to_numeric(df[cogs_col], errors="coerce").fillna(0.0) if cogs_col else 0.0
    df["retail_vnd"] = pd.to_numeric(df[retail_col], errors="coerce").fillna(0.0) if retail_col else 0.0
    return df.drop_duplicates("sku")[["sku", "item_name", "cogs_vnd", "retail_vnd"]].set_index("sku")


@lru_cache(maxsize=1)
def load_cogs() -> pd.Series:
    cat = load_product_catalog()
    return cat["cogs_vnd"].astype(float)


def unit_cost(sku: str) -> float:
    s = load_cogs()
    if sku in s.index and float(s.loc[sku]) > 0:
        return float(s.loc[sku])
    positive = s[s > 0]
    return float(positive.median()) if len(positive) else 1.0


def sku_name(sku: str) -> str:
    try:
        cat = load_product_catalog()
        if str(sku) in cat.index:
            name = str(cat.loc[str(sku), "item_name"]).strip()
            if name and name != "nan":
                return name
    except Exception:
        pass
    return str(sku)


def sku_label(sku: str, abc_map: dict[str, str] | None = None) -> str:
    sku_s = str(sku)
    name = sku_name(sku_s)
    cls = (abc_map or S4_CLASS).get(sku_s, "")
    cls_tag = f" [{cls}]" if cls else ""
    if name and name != sku_s:
        return f"{sku_s}{cls_tag} — {name}"
    return f"{sku_s}{cls_tag}"


@lru_cache(maxsize=4)
def top_skus_2024(baseline: str = "preclean", top_n: int = 40) -> list[str]:
    df = load_sales(baseline)
    sub = df[df["Year"].eq(2024) & df["Channel"].isin(CHANNELS)].copy()
    sub["Quantity"] = pd.to_numeric(sub["Quantity"], errors="coerce").fillna(0).clip(lower=0)
    sub["Item number"] = sub["Item number"].astype(str)
    vol = sub.groupby("Item number")["Quantity"].sum().sort_values(ascending=False)
    top = [str(s) for s in vol.head(int(top_n)).index]
    for s in S4:
        if s not in top:
            top.insert(0, s)
    return top


@lru_cache(maxsize=4)
def sku_abc_map(baseline: str = "preclean") -> dict[str, str]:
    """SKU-level dominant ABC_XYZ class on 2024 data, preserving S4_CLASS overrides."""
    out: dict[str, str] = {}
    try:
        df = load_sales(baseline)
        sub = df[df["Year"].eq(2024) & df["Channel"].isin(CHANNELS)].copy()
        sub["Quantity"] = pd.to_numeric(sub["Quantity"], errors="coerce").fillna(0).clip(lower=0)
        sub["Item number"] = sub["Item number"].astype(str)
        sub["week"] = pd.to_datetime(sub["Invoice date"]).dt.to_period("W").astype(str)
        grid = sub.groupby(["Item number", "Channel", "week"], as_index=False)["Quantity"].sum()
        with contextlib.redirect_stdout(io.StringIO()):
            classified = classify_portfolio(grid, service_targets=DEFAULT_SERVICE_TARGETS)
        # Pick the highest-volume channel's ABC_XYZ class for each SKU
        top_ch = classified.sort_values("sum", ascending=False).drop_duplicates("Item number")
        for r in top_ch.itertuples(index=False):
            out[str(r._0)] = str(r.ABC_XYZ)
    except Exception:
        pass
    out.update(S4_CLASS)
    return out


def monthly_qty(
    baseline: str,
    skus: list[str],
    year: int = 2024,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """Non-negative monthly Quantity. Returns are clipped to 0."""
    channels = channels or CHANNELS
    df = load_sales(baseline)
    sub = df[df["Item number"].astype(str).isin(skus) & df["Year"].eq(year)].copy()
    sub = sub[sub["Channel"].isin(channels)]
    sub["Quantity"] = pd.to_numeric(sub["Quantity"], errors="coerce").fillna(0).clip(lower=0)
    sub["Item number"] = sub["Item number"].astype(str)
    g = (
        sub.groupby(["Item number", "Channel", "Month"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Month": "month"})
    )
    return g


def history_monthly(baseline: str, skus: list[str]) -> pd.DataFrame:
    """All years, for forecast features. Quantity clipped at 0."""
    df = load_sales(baseline)
    sub = df[df["Item number"].astype(str).isin([str(s) for s in skus])].copy()
    sub = sub[sub["Channel"].isin(CHANNELS)]
    sub["Quantity"] = pd.to_numeric(sub["Quantity"], errors="coerce").fillna(0).clip(lower=0)
    sub["Item number"] = sub["Item number"].astype(str)
    g = (
        sub.groupby(["Item number", "Channel", "Year", "Month"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Year": "year", "Month": "month", "Item number": "sku", "Channel": "channel"})
    )
    return g


def save_upload(raw: bytes, name: str = "sales_upload.parquet") -> Path:
    PARQUET.mkdir(parents=True, exist_ok=True)
    dest = PARQUET / name
    dest.write_bytes(raw)
    load_sales.cache_clear()
    top_skus_2024.cache_clear()
    sku_abc_map.cache_clear()
    return dest


def sku_counts(baseline: str) -> dict:
    df = load_sales(baseline)
    y2024 = df[df["Year"].eq(2024)]
    present = set(y2024["Item number"].astype(str).unique())
    years = sorted(df["Year"].dropna().astype(int).unique().tolist())

    return {
        "baseline": baseline,
        "path": str(_sales_path(baseline)),
        "n_rows": int(len(df)),
        "n_rows_2024": int(len(y2024)),
        "years": years,
        "n_sku_all": int(df["Item number"].nunique()),
        "n_sku_2024": int(y2024["Item number"].nunique()),
        "total_qty_2024": float(pd.to_numeric(y2024["Quantity"], errors="coerce").fillna(0).clip(lower=0).sum()),
        "s4_present": {sku: sku in present for sku in S4},
        "channels_2024": sorted(y2024["Channel"].dropna().astype(str).unique().tolist()),
    }
