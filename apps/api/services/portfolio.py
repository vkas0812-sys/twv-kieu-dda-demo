"""ABC–XYZ on a weekly grid of the loaded baseline."""
from __future__ import annotations

import contextlib
import io

import pandas as pd

from kieu_pipeline.segmentation import DEFAULT_SERVICE_TARGETS, classify_portfolio

from api.services.data_store import load_sales, sku_name, unit_cost
from api.services.paths import CHANNELS, S4


def weekly_grid(baseline: str, year: int = 2024, channels: list[str] | None = None) -> pd.DataFrame:
    channels = channels or CHANNELS
    df = load_sales(baseline)
    sub = df[df["Year"].eq(year) & df["Channel"].isin(channels)].copy()
    sub["Quantity"] = pd.to_numeric(sub["Quantity"], errors="coerce").fillna(0).clip(lower=0)
    sub["Item number"] = sub["Item number"].astype(str)
    sub["week"] = pd.to_datetime(sub["Invoice date"]).dt.to_period("W").astype(str)
    return sub.groupby(["Item number", "Channel", "week"], as_index=False)["Quantity"].sum()


def build_portfolio(
    baseline: str = "preclean",
    year: int = 2024,
    channels: list[str] | None = None,
    top_n: int = 40,
) -> dict:
    grid = weekly_grid(baseline, year, channels)
    if grid.empty:
        return {"n": 0, "n_series": 0, "rows": [], "scatter": [], "sku_rank": [], "service_targets": DEFAULT_SERVICE_TARGETS}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        classified = classify_portfolio(grid, service_targets=DEFAULT_SERVICE_TARGETS)
    volume = (
        classified.groupby("Item number")["sum"]
        .sum()
        .sort_values(ascending=False)
    )
    top = list(volume.head(int(top_n)).index.astype(str))
    for sku in S4:
        if sku not in top and sku in set(volume.index.astype(str)):
            top.append(sku)
    view = classified[classified["Item number"].astype(str).isin(top)].copy()
    view["item_name"] = view["Item number"].astype(str).map(sku_name)
    view["unit_cost"] = view["Item number"].astype(str).map(unit_cost)
    scatter = view[
        ["Item number", "item_name", "Channel", "adi", "cv2", "ABC_XYZ", "Demand_Category", "sum", "beta"]
    ].copy()
    scatter = scatter.replace([float("inf"), float("-inf")], None)
    return {
        "baseline": baseline,
        "year": year,
        "grain": "tuần (Invoice date → tuần ISO), Quantity < 0 đã cắt về 0",
        "n_series": int(len(classified)),
        "n_sku": int(classified["Item number"].nunique()),
        "service_targets": DEFAULT_SERVICE_TARGETS,
        "sku_rank": [
            {"sku": str(i), "item_name": sku_name(str(i)), "qty": float(volume.loc[i])}
            for i in volume.head(max(top_n, 40)).index
        ],
        "rows": view.where(view.notna(), None).to_dict(orient="records"),
        "scatter": scatter.where(scatter.notna(), None).to_dict(orient="records"),
    }
