"""Sensitivity: replay 03e, plus a short live one-at-a-time sweep and MC ablation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from kieu_pipeline.inventory import solve_kieu_dda_vp
from kieu_pipeline.scenarios import saa_from_forecast_gaussian, saa_from_forecast_joint, saa_from_history

from api.services.data_store import monthly_qty
from api.services.inventory_lab import _costs, _demand_map, beta_for
from api.services.jsonutil import scrub
from api.services.paths import RANDOM_SEED, S4
from api.services.replay import replay_payload
from api.services.runs import new_run_id, write_manifest, write_results


def replay_sa() -> dict:
    return replay_payload("03e", "03e_sa_param.csv")


def replay_mc() -> dict:
    return replay_payload("03e", "03e_mc.csv")


def replay_jackknife() -> dict:
    return replay_payload("03e", "03e_jackknife.csv")


def live_sa(
    sku_list: list[str] | None = None,
    baseline: str = "preclean",
    h_grid: list[float] | None = None,
    beta_grid: list[float] | None = None,
) -> dict:
    """One factor at a time: vary h_mult with beta_mult=1, then the reverse. VP only."""
    skus = [str(s) for s in (sku_list or S4[:1])]
    h_grid = h_grid or [0.75, 1.0, 1.25]
    beta_grid = beta_grid or [0.75, 1.0, 1.25]
    monthly = monthly_qty(baseline, skus)
    omega = saa_from_history(monthly, months=list(range(1, 9)))
    rows = []
    run_id = new_run_id("sa")
    for sku in skus:
        demand = _demand_map(omega, sku)
        beta0 = beta_for(sku, None, 1.0)
        for h_mult in h_grid:
            cost = _costs(sku, "dual_hub", h_mult)
            penalty = (beta0 / (1 - beta0)) * cost["h_ciw"]
            solved = solve_kieu_dda_vp(
                demand, cost["h_ciw"], penalty, cost["unit_cost"],
                locations=cost["locations"], channels=cost["channels"],
                holding=cost["holding"], fulfill_cost=cost["fulfill"],
                feasible_arcs=cost["arcs"],
            )
            rows.append({
                "sku": sku, "factor": "h_mult", "h_mult": h_mult, "beta_mult": 1.0,
                "beta_used": beta0, "policy": "Kieu-DDA-VP", "TEC": solved["TEC"],
                "fill_rate": solved["fill_rate"], "status": solved["status"], "split": "train",
            })
        cost = _costs(sku, "dual_hub", 1.0)
        for b_mult in beta_grid:
            beta = beta_for(sku, None, b_mult)
            penalty = (beta / (1 - beta)) * cost["h_ciw"]
            solved = solve_kieu_dda_vp(
                demand, cost["h_ciw"], penalty, cost["unit_cost"],
                locations=cost["locations"], channels=cost["channels"],
                holding=cost["holding"], fulfill_cost=cost["fulfill"],
                feasible_arcs=cost["arcs"],
            )
            rows.append({
                "sku": sku, "factor": "beta_mult", "h_mult": 1.0, "beta_mult": b_mult,
                "beta_used": beta, "policy": "Kieu-DDA-VP", "TEC": solved["TEC"],
                "fill_rate": solved["fill_rate"], "status": solved["status"], "split": "train",
            })
    body = scrub({
        "run_id": run_id, "rows": rows, "ablation": False,
        "note": "Một yếu tố mỗi lượt. Không phải lưới đầy đủ.",
    })
    write_manifest(run_id, {"kind": "sa", "sku_list": skus, "cutoff": "2024-08", "omega_type": "hist", "L": None})
    write_results(run_id, body)
    return body


def live_mc(sku: str = "W100133", n: int = 8, baseline: str = "preclean") -> dict:
    """Ablation only: i.i.d. Gaussian vs joint residual of the same train months."""
    monthly = monthly_qty(baseline, [sku], year=2024)
    train = monthly[monthly["month"].between(1, 8)]
    mu_rows = []
    resid_rows = []
    std_rows = []
    for ch, g in train.groupby("Channel"):
        mu = float(g["Quantity"].mean())
        mu_rows.append({"sku": sku, "channel": ch, "mu": mu})
        resid = g["Quantity"].astype(float) - mu
        std_rows.append({"sku": sku, "resid_std": float(resid.std(ddof=0) or abs(mu) * 0.25)})
        for month, val in zip(g["month"], resid):
            resid_rows.append({"sku": sku, "channel": ch, "month": int(month), "residual": float(val)})
    mu = pd.DataFrame(mu_rows)
    # std map is per sku; gaussian helper reads resid_std by sku (one scale).
    scale = float(np.mean([r["resid_std"] for r in std_rows])) if std_rows else 1.0
    std = pd.DataFrame([{"sku": sku, "resid_std": scale}])
    gauss = saa_from_forecast_gaussian(mu, std, n_scenarios=n, seed=RANDOM_SEED)
    joint = saa_from_forecast_joint(mu, pd.DataFrame(resid_rows))
    cost = _costs(sku, "dual_hub", 1.0)
    beta = beta_for(sku, None, 1.0)
    penalty = (beta / (1 - beta)) * cost["h_ciw"]
    rows = []
    for label, frame in (("mc_indep", gauss), ("mc_joint", joint)):
        demand = _demand_map(frame, sku)
        solved = solve_kieu_dda_vp(
            demand, cost["h_ciw"], penalty, cost["unit_cost"],
            locations=cost["locations"], channels=cost["channels"],
            holding=cost["holding"], fulfill_cost=cost["fulfill"],
            feasible_arcs=cost["arcs"],
        )
        rows.append({
            "sku": sku, "omega": label, "n_omega": int(frame["scenario"].nunique()),
            "policy": "Kieu-DDA-VP", "TEC": solved["TEC"], "fill_rate": solved["fill_rate"],
            "beta_used": beta, "status": solved["status"],
            "ablation": True,
        })
    run_id = new_run_id("mc")
    body = scrub({
        "run_id": run_id,
        "rows": rows,
        "ablation": True,
        "note": "Ablation — không phải KPI chính. Seed 42.",
        "beta_used": beta,
    })
    write_manifest(run_id, {"kind": "mc", "sku_list": [sku], "cutoff": "2024-08", "omega_type": "mc", "L": None})
    write_results(run_id, body)
    return body


def live_jackknife(sku: str = "W100133", baseline: str = "preclean") -> dict:
    monthly = monthly_qty(baseline, [sku])
    rows = []
    cost = _costs(sku, "dual_hub", 1.0)
    beta = beta_for(sku, None, 1.0)
    penalty = (beta / (1 - beta)) * cost["h_ciw"]
    for left in range(1, 9):
        keep = [m for m in range(1, 9) if m != left]
        omega = saa_from_history(monthly, months=keep)
        demand = _demand_map(omega, sku)
        solved = solve_kieu_dda_vp(
            demand, cost["h_ciw"], penalty, cost["unit_cost"],
            locations=cost["locations"], channels=cost["channels"],
            holding=cost["holding"], fulfill_cost=cost["fulfill"],
            feasible_arcs=cost["arcs"],
        )
        rows.append({
            "sku": sku, "left_out_month": left, "n_omega": len(keep),
            "policy": "Kieu-DDA-VP", "TEC": solved["TEC"], "fill_rate": solved["fill_rate"],
            "beta_used": beta, "status": solved["status"],
        })
    run_id = new_run_id("jk")
    body = scrub({"run_id": run_id, "rows": rows, "beta_used": beta})
    write_manifest(run_id, {"kind": "jackknife", "sku_list": [sku], "cutoff": "2024-08", "omega_type": "hist", "L": None})
    write_results(run_id, body)
    return body
