"""Live inventory: dual-hub solvers + Protocol A evaluate. No LP edits."""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from kieu_pipeline.inventory import (
    CHANNELS as LEGACY_CHANNELS,
    DUAL_HUB_CHANNELS,
    DUAL_HUB_LOCATIONS,
    LOCATIONS,
    dual_hub_arcs,
    dual_hub_fulfill,
    dual_hub_holding,
    evaluate_decentralized_given_y,
    evaluate_given_y,
    make_fulfill,
    make_holding,
    solve_decentralized_newsvendor,
    solve_kieu_dda_fp,
    solve_kieu_dda_vp,
)
from kieu_pipeline.scenarios import (
    coverage_window_starts,
    saa_from_forecast_coverage,
    saa_from_forecast_joint,
    saa_from_history,
    saa_from_history_coverage,
)
from kieu_pipeline.segmentation import DEFAULT_SERVICE_TARGETS

from api.services.data_store import monthly_qty, sku_abc_map, sku_name, unit_cost
from api.services.jsonutil import scrub
from api.services.paths import CHANNELS, MAX_SKUS, OMEGA_TYPES, POLICIES, S4, S4_CLASS
from api.services.runs import new_run_id, write_manifest, write_results


def class_of(sku: str, portfolio_class: dict[str, str] | None = None, baseline: str = "preclean") -> str:
    if sku in S4_CLASS:
        return S4_CLASS[sku]
    if portfolio_class and sku in portfolio_class:
        return portfolio_class[sku]
    lookup = sku_abc_map(baseline)
    if sku in lookup:
        return lookup[sku]
    return "CZ"


def beta_for(sku: str, overrides: dict[str, float] | None, beta_mult: float, baseline: str = "preclean") -> float:
    overrides = overrides or {}
    if sku in overrides:
        base = float(overrides[sku])
    else:
        abc = class_of(sku, baseline=baseline)
        base = float(DEFAULT_SERVICE_TARGETS.get(abc, 0.85))
    used = base * float(beta_mult)
    if used >= 0.999:
        used = 0.999
    if used <= 0.01:
        used = 0.01
    return used


def _demand_map(frame: pd.DataFrame, sku: str) -> dict[tuple[str, int], float]:
    sub = frame[frame["sku"].astype(str) == str(sku)]
    return {
        (str(r.channel), int(r.scenario)): float(r.demand)
        for r in sub.itertuples(index=False)
    }


def _mean_period_demand(demand: dict[tuple[str, int], float]) -> float:
    scenarios = {w for (_, w) in demand.keys()}
    if not scenarios:
        return 0.0
    return sum(demand.values()) / len(scenarios)


def _remap_legacy(demand: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Thesis HO/HC → legacy EC/HR. Not used for thesis KPIs."""
    alias = {"HO": "EC", "HC": "HR"}
    out: dict[tuple[str, int], float] = {}
    for (ch, w), q in demand.items():
        out[(alias.get(ch, ch), w)] = out.get((alias.get(ch, ch), w), 0.0) + q
    return out


def build_omega(
    monthly: pd.DataFrame,
    omega_type: str,
    L: int | None,
    month_lo: int,
    month_hi: int,
    mu: pd.DataFrame | None = None,
    residuals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame | None, bool, str]:
    if omega_type not in OMEGA_TYPES:
        raise ValueError(f"unknown omega_type {omega_type}")
    if omega_type == "hist":
        return saa_from_history(monthly, months=list(range(month_lo, month_hi + 1))), True, ""
    if omega_type == "hist_coverage":
        if L is None:
            raise ValueError("L required for hist_coverage")
        starts = coverage_window_starts(int(L), month_lo, month_hi)
        if not starts:
            return None, False, f"no window L={L} trong tháng {month_lo}–{month_hi}"
        try:
            frame = saa_from_history_coverage(
                monthly, L=int(L), month_lo=month_lo, month_hi=month_hi
            )
        except ValueError as exc:
            return None, False, str(exc)
        return frame, True, ""
    if mu is None or residuals is None:
        raise ValueError("fcst omega cần forecast run (mu + residual)")
    if omega_type == "fcst_joint":
        return saa_from_forecast_joint(mu, residuals), True, ""
    if L is None:
        raise ValueError("L required for fcst_coverage")
    return saa_from_forecast_coverage(mu, residuals, L=int(L)), True, ""


def _costs(sku: str, network: str, h_mult: float) -> dict[str, Any]:
    cogs = unit_cost(sku)
    if network == "dual_hub":
        holding = {k: v * h_mult for k, v in dual_hub_holding(cogs).items()}
        fulfill = dual_hub_fulfill()
        locations = list(DUAL_HUB_LOCATIONS)
        channels = list(DUAL_HUB_CHANNELS)
        arcs = dual_hub_arcs()
    elif network == "legacy_6buffer":
        h_ciw = 0.02 * cogs * h_mult
        holding = make_holding(h_ciw)
        fulfill = make_fulfill(h_ciw)
        locations = list(LOCATIONS)
        channels = list(LEGACY_CHANNELS)
        arcs = None
    else:
        raise ValueError(network)
    h_ciw = float(holding["CIW"])
    return {
        "unit_cost": cogs,
        "holding": holding,
        "fulfill": fulfill,
        "locations": locations,
        "channels": channels,
        "arcs": arcs,
        "h_ciw": h_ciw,
    }


def _solve_one(
    policy: str,
    demand: dict[tuple[str, int], float],
    cost: dict[str, Any],
    penalty: float,
    budget_B: float | None,
) -> dict[str, Any]:
    if policy == "Decentralized":
        return solve_decentralized_newsvendor(
            demand, cost["h_ciw"], penalty, cost["unit_cost"], channels=cost["channels"]
        )
    kwargs = dict(
        locations=cost["locations"],
        channels=cost["channels"],
        holding=cost["holding"],
        fulfill_cost=cost["fulfill"],
        feasible_arcs=cost["arcs"],
        budget_B=budget_B,
    )
    if policy == "Kieu-DDA-VP":
        return solve_kieu_dda_vp(demand, cost["h_ciw"], penalty, cost["unit_cost"], **kwargs)
    if policy == "Kieu-DDA-FP":
        return solve_kieu_dda_fp(demand, cost["h_ciw"], penalty, cost["unit_cost"], **kwargs)
    raise ValueError(policy)


def _evaluate_one(policy: str, solved: dict, demand: dict, cost: dict, penalty: float) -> dict:
    y = {k: float(v) for k, v in solved["y_star"].items()}
    if policy == "Decentralized":
        return evaluate_decentralized_given_y(
            y, demand, cost["h_ciw"], penalty, cost["unit_cost"], channels=cost["channels"]
        )
    assignment = solved.get("assignment") if policy == "Kieu-DDA-FP" else None
    return evaluate_given_y(
        y,
        demand,
        cost["holding"],
        penalty,
        cost["fulfill"],
        cost["locations"],
        cost["channels"],
        cost["arcs"],
        cost["unit_cost"],
        assignment=assignment,
    )


def _row(
    sku: str,
    policy: str,
    split: str,
    beta: float,
    abc: str,
    payload: dict,
    y_star: dict,
    mean_demand: float = 0.0,
) -> dict:
    tec = payload.get("TEC")
    e_ful = payload.get("E_fulfill")
    cpu = (float(tec) / mean_demand) if (tec is not None and mean_demand > 0) else payload.get("cost_per_unit")
    return {
        "sku": sku,
        "item_name": sku_name(sku),
        "ABC": abc,
        "policy": policy,
        "split": split,
        "beta_target": DEFAULT_SERVICE_TARGETS.get(abc, 0.85),
        "beta_used": beta,
        "status": payload.get("status"),
        "TEC": tec,
        "cost_per_unit": cpu,
        "E_holding": payload.get("E_holding"),
        "E_penalty": payload.get("E_penalty"),
        "E_fulfill": e_ful,
        "TEC_ex_fulfill": (None if tec is None or e_ful is None else float(tec) - float(e_ful or 0)),
        "fill_rate": payload.get("fill_rate"),
        "WC": payload.get("WC"),
        "total_y": payload.get("total_y", sum(y_star.values())),
        "CL_stock": payload.get("CL_stock", payload.get("CL")),
        "CL_open": payload.get("CL_open"),
        "SA": payload.get("SA"),
        "y_star": {str(k): float(v) for k, v in y_star.items()},
        "assignment": payload.get("assignment"),
        "note": (
            "Dec không trả cước dual-hub (E_fulfill = 0). So sánh với VP hãy nhìn TEC_ex_fulfill."
            if policy == "Decentralized"
            else ""
        ),
    }


def solve_protocol(
    *,
    sku_list: list[str] | None = None,
    omega_type: str = "hist",
    L: int | None = None,
    policies: list[str] | None = None,
    beta_overrides: dict[str, float] | None = None,
    h_mult: float = 1.0,
    beta_mult: float = 1.0,
    budget_on: bool = False,
    budget_mult: float = 1.0,
    network: str = "dual_hub",
    baseline: str = "preclean",
    mu: pd.DataFrame | None = None,
    residuals: pd.DataFrame | None = None,
    evaluate_oos: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    skus = [str(s) for s in (sku_list or S4)]
    if len(skus) > MAX_SKUS:
        raise ValueError(f"tối đa {MAX_SKUS} SKU")
    policies = list(policies or POLICIES)
    for p in policies:
        if p not in POLICIES:
            raise ValueError(f"unknown policy {p}")
    if omega_type not in OMEGA_TYPES:
        raise ValueError(f"unknown omega_type {omega_type}")

    monthly = monthly_qty(baseline, skus, year=2024)
    train, train_ok, train_note = build_omega(
        monthly, omega_type, L, 1, 8, mu=mu, residuals=residuals
    )
    oos_frame = None
    oos_ok = False
    oos_note = ""
    if evaluate_oos and omega_type in ("hist", "hist_coverage"):
        oos_frame, oos_ok, oos_note = build_omega(monthly, omega_type, L, 9, 12)
    elif evaluate_oos and omega_type.startswith("fcst"):
        # Score frozen y* on actual Sep–Dec (1-month) or coverage windows.
        if omega_type == "fcst_joint":
            oos_frame, oos_ok, oos_note = build_omega(monthly, "hist", None, 9, 12)
        else:
            oos_frame, oos_ok, oos_note = build_omega(monthly, "hist_coverage", L, 9, 12)

    run_id = run_id or new_run_id("inv")
    rows: list[dict] = []
    if not train_ok or train is None:
        body = {
            "run_id": run_id,
            "oos_available": False,
            "oos_note": train_note or oos_note,
            "rows": [],
            "omega_type": omega_type,
            "L": L,
            "cutoff": "2024-08",
        }
        write_manifest(run_id, {k: body[k] for k in ("omega_type", "L", "cutoff")} | {"sku_list": skus, "kind": "inventory"})
        write_results(run_id, body)
        return body

    for sku in skus:
        abc = class_of(sku, baseline=baseline)
        beta = beta_for(sku, beta_overrides, beta_mult, baseline=baseline)
        cost = _costs(sku, network, h_mult)
        penalty = (beta / (1.0 - beta)) * cost["h_ciw"]
        demand = _demand_map(train, sku)
        if network == "legacy_6buffer":
            demand = _remap_legacy(demand)
        mean_train_d = _mean_period_demand(demand)
        budget_B = None
        if budget_on:
            budget_B = float(budget_mult) * cost["unit_cost"] * max(mean_train_d, 1.0)
        for policy in policies:
            try:
                solved = _solve_one(policy, demand, cost, penalty, budget_B)
                y_before = copy.deepcopy(solved.get("y_star") or {})
                rows.append(_row(sku, policy, "train", beta, abc, solved, y_before, mean_train_d))
                if oos_ok and oos_frame is not None:
                    oos_dem = _demand_map(oos_frame, sku)
                    if network == "legacy_6buffer":
                        oos_dem = _remap_legacy(oos_dem)
                    mean_oos_d = _mean_period_demand(oos_dem)
                    ev = _evaluate_one(policy, solved, oos_dem, cost, penalty)
                    y_after = solved.get("y_star") or {}
                    if {k: float(v) for k, v in y_before.items()} != {
                        k: float(v) for k, v in y_after.items()
                    }:
                        raise RuntimeError("evaluate_given_y changed y_star")
                    rows.append(_row(sku, policy, "oos", beta, abc, ev, y_before, mean_oos_d))
            except Exception as exc:  # one SKU must not drop the batch
                rows.append({
                    "sku": sku,
                    "item_name": sku_name(sku),
                    "policy": policy,
                    "split": "train",
                    "status": f"Failed: {exc}",
                    "beta_used": beta,
                    "ABC": abc,
                })

    body = scrub({
        "run_id": run_id,
        "omega_type": omega_type,
        "L": L if omega_type.endswith("coverage") else None,
        "period_label": "cửa sổ L tháng" if omega_type.endswith("coverage") else "một tháng",
        "cutoff": "2024-08",
        "network": network,
        "network_note": (
            "6-buffer legacy, map HO→EC và HC→HR. Không phải số luận văn."
            if network == "legacy_6buffer"
            else "dual-hub: CIW / Hub_Retail / Hub_B2B"
        ),
        "h_mult": h_mult,
        "beta_mult": beta_mult,
        "baseline": baseline,
        "n_sku": len(skus),
        "oos_available": bool(oos_ok),
        "oos_note": oos_note,
        "rows": rows,
    })
    write_manifest(
        run_id,
        {
            "kind": "inventory",
            "sku_list": skus,
            "omega_type": omega_type,
            "L": body["L"],
            "cutoff": "2024-08",
            "policies": policies,
            "network": network,
            "baseline": baseline,
        },
    )
    write_results(run_id, body)
    return body


def compare_hist_fcst(
    *,
    sku_list: list[str],
    coverage: bool,
    L: int | None,
    policies: list[str],
    baseline: str,
    network: str,
    mu: pd.DataFrame | None,
    residuals: pd.DataFrame | None,
) -> dict[str, Any]:
    """RQ2 style: same policies on hist Ω and forecast Ω. Do not add TEC across grains."""
    hist_type = "hist_coverage" if coverage else "hist"
    fcst_type = "fcst_coverage" if coverage else "fcst_joint"
    hist = solve_protocol(
        sku_list=sku_list, omega_type=hist_type, L=L, policies=policies,
        baseline=baseline, network=network, evaluate_oos=True,
    )
    fcst = solve_protocol(
        sku_list=sku_list, omega_type=fcst_type, L=L, policies=policies,
        baseline=baseline, network=network, mu=mu, residuals=residuals, evaluate_oos=True,
    )
    key = ["sku", "policy"]
    h = pd.DataFrame([r for r in hist["rows"] if r.get("split") == "oos"])
    f = pd.DataFrame([r for r in fcst["rows"] if r.get("split") == "oos"])
    merged = []
    if len(h) and len(f):
        both = h.merge(f, on=key, suffixes=("_hist", "_fcst"))
        for r in both.itertuples(index=False):
            cpu_h = getattr(r, "cost_per_unit_hist", None)
            cpu_f = getattr(r, "cost_per_unit_fcst", None)
            cpu_delta = (float(cpu_f) - float(cpu_h)) if (cpu_h is not None and cpu_f is not None) else None
            merged.append({
                "sku": r.sku,
                "item_name": sku_name(r.sku),
                "ABC": getattr(r, "ABC_hist", class_of(r.sku, baseline=baseline)),
                "policy": r.policy,
                "TEC_hist": r.TEC_hist,
                "TEC_fcst": r.TEC_fcst,
                "TEC_delta": (float(r.TEC_fcst) - float(r.TEC_hist)) if (r.TEC_hist is not None and r.TEC_fcst is not None) else None,
                "cpu_hist": cpu_h,
                "cpu_fcst": cpu_f,
                "cpu_delta": cpu_delta,
                "FR_hist": r.fill_rate_hist,
                "FR_fcst": r.fill_rate_fcst,
                "E_fulfill_hist": r.E_fulfill_hist,
                "E_fulfill_fcst": r.E_fulfill_fcst,
                "TEC_ex_fulfill_hist": r.TEC_ex_fulfill_hist,
                "TEC_ex_fulfill_fcst": r.TEC_ex_fulfill_fcst,
                "grain": hist["period_label"],
                "L": L if coverage else None,
            })
    return scrub({
        "hist_run_id": hist["run_id"],
        "fcst_run_id": fcst["run_id"],
        "oos_available": hist["oos_available"] and fcst["oos_available"],
        "oos_note": hist["oos_note"] or fcst["oos_note"],
        "rows": merged,
        "caption": "So sánh cùng grain. Không cộng TEC tháng (03a) với TEC cửa sổ L (03d).",
    })
