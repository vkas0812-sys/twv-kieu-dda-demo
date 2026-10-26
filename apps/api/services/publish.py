"""Publish base-stock CSV and also call kieu_pipeline.publish_targets."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from kieu_pipeline.evaluation import publish_targets

from api.services.paths import APP_RUNS
from api.services.runs import read_run, run_dir


PUBLISH_COLS = [
    "sku", "location", "policy", "y_CIW", "y_Hub_Retail", "y_Hub_B2B",
    "beta", "TEC", "fill_rate", "WC", "published_at", "run_id",
]


def _y(row: dict, key: str) -> float:
    y = row.get("y_star") or {}
    return float(y.get(key, 0.0) or 0.0)


def publish_run(run_id: str, policies: list[str] | None = None) -> dict:
    data = read_run(run_id)
    results = data.get("results") or {}
    rows_in = results.get("rows") or []
    if not rows_in:
        blob = APP_RUNS / run_id / "results.json"
        if blob.exists():
            results = json.loads(blob.read_text(encoding="utf-8"))
            rows_in = results.get("rows") or []
    train = [r for r in rows_in if r.get("split") == "train" and r.get("status") == "Optimal"]
    if policies:
        train = [r for r in train if r.get("policy") in policies]
    if not train:
        raise ValueError("run không có dòng train Optimal để xuất")
    now = datetime.now(timezone.utc).isoformat()
    out_rows = []
    for r in train:
        policy = r["policy"]
        if policy == "Decentralized":
            y_ciw = y_retail = y_b2b = 0.0
            location = "per-channel"
        else:
            y_ciw, y_retail, y_b2b = _y(r, "CIW"), _y(r, "Hub_Retail"), _y(r, "Hub_B2B")
            location = "dual-hub"
        out_rows.append({
            "sku": r["sku"],
            "location": location,
            "policy": policy,
            "y_CIW": y_ciw,
            "y_Hub_Retail": y_retail,
            "y_Hub_B2B": y_b2b,
            "beta": r.get("beta_used"),
            "TEC": r.get("TEC"),
            "fill_rate": r.get("fill_rate"),
            "WC": r.get("WC"),
            "published_at": now,
            "run_id": run_id,
        })
    df = pd.DataFrame(out_rows)[PUBLISH_COLS]
    dest = run_dir(run_id)
    path = dest / "published_planner.csv"
    df.to_csv(path, index=False)

    vp = df[df["policy"] == "Kieu-DDA-VP"].copy()
    if len(vp):
        pkg = pd.DataFrame({
            "sku": vp["sku"],
            "y_star": vp["y_CIW"] + vp["y_Hub_Retail"] + vp["y_Hub_B2B"],
            "beta": vp["beta"],
            "TEC": vp["TEC"],
            "fill_rate": vp["fill_rate"],
            "CL": 0.0,
            "WC": vp["WC"],
        })
        publish_targets(pkg, dest)
    return {"run_id": run_id, "n": int(len(df)), "path": str(path), "rows": df.to_dict(orient="records")}
