"""Kieu-DDA inventory solvers and cost helpers (Wave 3).

Models:
  - Kieu-DDA-VP (Virtual Pooling, LP)
  - Kieu-DDA-FP (Fixed Pooling, MILP)
  - Decentralized newsvendor

NEXT SESSION (2026-09-10): M6 hiện là SAA *một kỳ*, lead time mô hình = 0.
Không dùng LEAD_TIME_DAYS=(45,90) trong Config — biến đó chưa vào ràng buộc.
Phiên tới: 3 scenario planning lead time 3 / 4.5 / 6 tháng (coverage demand
trên cửa sổ L, rồi VP/FP/Dec như cũ). Chi tiết:
  4. Code/NEXT_SESSION_LEADTIME.md
"""
from __future__ import annotations

# Planning lead-time scenarios — CHƯA dùng trong solver (placeholder phiên tới)
LEAD_TIME_MONTHS = (3.0, 4.5, 6.0)
LEAD_TIME_WEEKS = (13, 20, 26)  # 3 / 4.5 / 6 tháng, grain tuần

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pulp

CARRYING_RATE_ANNUAL = 0.25   # 25% / year
BUF_HOLDING_MULT = 1.4        # channel buffer more expensive than CIW
FULFILL_CIW = 0.15            # relative to h_CIW (transport from central)
FULFILL_BUF = 0.03            # local fulfillment cheap

# ---------------------------------------------------------------
#  Network
# ---------------------------------------------------------------
LOCATIONS = ["CIW", "RT_buf", "EC_buf", "HR_buf", "GT_buf", "MT_buf"]
CHANNELS  = ["RT", "EC", "HR", "GT", "MT"]

def default_feasible_arcs() -> Set[Tuple[str, str]]:
    arcs = {("CIW", ch) for ch in CHANNELS}
    for ch in CHANNELS:
        arcs.add((f"{ch}_buf", ch))
    return arcs

def make_holding(h_ciw: float) -> Dict[str, float]:
    h = {"CIW": h_ciw}
    for ch in CHANNELS:
        h[f"{ch}_buf"] = h_ciw * BUF_HOLDING_MULT
    return h

def make_fulfill(h_ciw: float) -> Dict[Tuple[str, str], float]:
    c = {}
    for ch in CHANNELS:
        c[("CIW", ch)] = h_ciw * FULFILL_CIW
        c[(f"{ch}_buf", ch)] = h_ciw * FULFILL_BUF
    return c


# ---------------------------------------------------------------
#  Kieu-DDA-VP (LP)
# ---------------------------------------------------------------
def solve_kieu_dda_vp(
    demand_scenarios: Dict[Tuple[str, int], float],
    h_ciw: float,
    penalty: float,
    unit_cost: float,
    locations: List[str] = None,
    channels: List[str] = None,
    capacity: Optional[Dict[str, float]] = None,
    budget_B: Optional[float] = None,
    feasible_arcs: Optional[Set[Tuple[str, str]]] = None,
    n_scenarios: int = None,
    holding: Optional[Dict[str, float]] = None,
    fulfill_cost: Optional[Dict[Tuple[str, str], float]] = None,
    msg: bool = False,
) -> dict:
    locations = locations or LOCATIONS
    channels  = channels or CHANNELS
    holding   = holding if holding is not None else make_holding(h_ciw)
    fulfill_cost = fulfill_cost if fulfill_cost is not None else make_fulfill(h_ciw)
    feasible_arcs = feasible_arcs or default_feasible_arcs()

    sc_ids = sorted({s for (_, s) in demand_scenarios.keys()})
    if n_scenarios is not None:
        sc_ids = sc_ids[:n_scenarios]
    pi = 1.0 / max(len(sc_ids), 1)

    prob = pulp.LpProblem("Kieu_DDA_VP", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", locations, lowBound=0)
    z_keys = [(l, r, w) for l in locations for r in channels for w in sc_ids if (l, r) in feasible_arcs]
    z = pulp.LpVariable.dicts("z", z_keys, lowBound=0)
    s = pulp.LpVariable.dicts("s", [(l, w) for l in locations for w in sc_ids], lowBound=0)
    b = pulp.LpVariable.dicts("b", [(r, w) for r in channels for w in sc_ids], lowBound=0)

    holding_term = pi * pulp.lpSum(holding[l] * s[(l, w)] for l in locations for w in sc_ids)
    penalty_term = pi * pulp.lpSum(penalty * b[(r, w)] for r in channels for w in sc_ids)
    fulfill_term = pi * pulp.lpSum(fulfill_cost.get((l, r), 0) * z[(l, r, w)] for (l, r, w) in z_keys)
    prob += holding_term + penalty_term + fulfill_term

    for w in sc_ids:
        for l in locations:
            outflow = pulp.lpSum(z[(l, r, w)] for r in channels if (l, r, w) in z)
            prob += s[(l, w)] == y[l] - outflow
        for r in channels:
            inflow = pulp.lpSum(z[(l, r, w)] for l in locations if (l, r, w) in z)
            d = demand_scenarios.get((r, w), 0.0)
            prob += b[(r, w)] == d - inflow

    if capacity:
        for l, cap in capacity.items():
            if l in y:
                prob += y[l] <= cap
    if budget_B is not None:
        prob += unit_cost * pulp.lpSum(y[l] for l in locations) <= budget_B

    status = prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=60))
    y_star = {l: (pulp.value(y[l]) or 0.0) for l in locations}
    total_y = sum(y_star.values())
    tec = pulp.value(prob.objective) or 0.0

    total_demand = sum(demand_scenarios.get((r, w), 0) for r in channels for w in sc_ids)
    total_shortage = sum((pulp.value(b[(r, w)]) or 0) for r in channels for w in sc_ids)
    fill_rate = 1.0 - total_shortage / total_demand if total_demand > 0 else 1.0
    served = {
        (l, r) for (l, r, w) in z_keys
        if (pulp.value(z[(l, r, w)]) or 0.0) > 1e-6
    }
    metrics = placement_metrics(
        y_star, channels, served=served, demand_scenarios=demand_scenarios
    )

    return {
        "status": pulp.LpStatus[status],
        "y_star": y_star,
        "TEC": tec,
        "E_holding": pulp.value(holding_term) or 0,
        "E_penalty": pulp.value(penalty_term) or 0,
        "E_fulfill": pulp.value(fulfill_term) or 0,
        "fill_rate": fill_rate,
        "CL": metrics["CL_stock"],
        "CL_stock": metrics["CL_stock"],
        "CL_open": metrics["CL_open"],
        "CL_open_paper": metrics["CL_open_paper"],
        "SA": metrics["SA"],
        "served": served,
        "total_y": total_y,
        "WC": unit_cost * total_y,
        "policy": "Kieu-DDA-VP",
    }


# ---------------------------------------------------------------
#  Kieu-DDA-FP (MILP)
# ---------------------------------------------------------------
def solve_kieu_dda_fp(
    demand_scenarios: Dict[Tuple[str, int], float],
    h_ciw: float,
    penalty: float,
    unit_cost: float,
    locations: List[str] = None,
    channels: List[str] = None,
    capacity: Optional[Dict[str, float]] = None,
    budget_B: Optional[float] = None,
    feasible_arcs: Optional[Set[Tuple[str, str]]] = None,
    n_scenarios: int = None,
    holding: Optional[Dict[str, float]] = None,
    fulfill_cost: Optional[Dict[Tuple[str, str], float]] = None,
    msg: bool = False,
) -> dict:
    locations = locations or LOCATIONS
    channels  = channels or CHANNELS
    holding   = holding if holding is not None else make_holding(h_ciw)
    fulfill_cost = fulfill_cost if fulfill_cost is not None else make_fulfill(h_ciw)
    feasible_arcs = feasible_arcs or default_feasible_arcs()

    sc_ids = sorted({s for (_, s) in demand_scenarios.keys()})
    if n_scenarios is not None:
        sc_ids = sc_ids[:n_scenarios]
    pi = 1.0 / max(len(sc_ids), 1)
    assign_arcs = [(l, r) for l in locations for r in channels if (l, r) in feasible_arcs]

    prob = pulp.LpProblem("Kieu_DDA_FP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", assign_arcs, cat="Binary")
    y = pulp.LpVariable.dicts("y", locations, lowBound=0)
    open_loc = pulp.LpVariable.dicts("open", locations, cat="Binary")
    z = pulp.LpVariable.dicts("z", [(l, r, w) for (l, r) in assign_arcs for w in sc_ids], lowBound=0)
    s = pulp.LpVariable.dicts("s", [(l, w) for l in locations for w in sc_ids], lowBound=0)
    b = pulp.LpVariable.dicts("b", [(r, w) for r in channels for w in sc_ids], lowBound=0)

    holding_term = pi * pulp.lpSum(holding[l] * s[(l, w)] for l in locations for w in sc_ids)
    penalty_term = pi * pulp.lpSum(penalty * b[(r, w)] for r in channels for w in sc_ids)
    fulfill_term = pi * pulp.lpSum(
        fulfill_cost.get((l, r), 0) * z[(l, r, w)] for (l, r) in assign_arcs for w in sc_ids
    )
    prob += holding_term + penalty_term + fulfill_term

    for r in channels:
        cand = [l for l in locations if (l, r) in feasible_arcs]
        if cand:
            prob += pulp.lpSum(x[(l, r)] for l in cand) == 1

    for l in locations:
        assigned = [r for r in channels if (l, r) in feasible_arcs]
        if assigned:
            for r in assigned:
                prob += x[(l, r)] <= open_loc[l]
            big_M = capacity.get(l, 1e6) if capacity else 1e6
            prob += y[l] <= big_M * open_loc[l]

    for w in sc_ids:
        for l in locations:
            outflow = pulp.lpSum(z[(l, r, w)] for r in channels if (l, r) in feasible_arcs)
            prob += s[(l, w)] == y[l] - outflow
        for r in channels:
            inflow = pulp.lpSum(z[(l, r, w)] for l in locations if (l, r) in feasible_arcs)
            d = demand_scenarios.get((r, w), 0.0)
            prob += b[(r, w)] == d - inflow
            for l in locations:
                if (l, r) in feasible_arcs:
                    prob += z[(l, r, w)] <= d * x[(l, r)]

    if capacity:
        for l, cap in capacity.items():
            if l in y:
                prob += y[l] <= cap
    if budget_B is not None:
        prob += unit_cost * pulp.lpSum(y[l] for l in locations) <= budget_B

    status = prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=90))
    y_star = {l: (pulp.value(y[l]) or 0.0) for l in locations}
    x_star = {(l, r): (pulp.value(x[(l, r)]) or 0) for (l, r) in assign_arcs}
    assignment = {r: l for (l, r), v in x_star.items() if v > 0.5}
    total_y = sum(y_star.values())
    tec = pulp.value(prob.objective) or 0.0

    total_demand = sum(demand_scenarios.get((r, w), 0) for r in channels for w in sc_ids)
    total_shortage = sum((pulp.value(b[(r, w)]) or 0) for r in channels for w in sc_ids)
    fill_rate = 1.0 - total_shortage / total_demand if total_demand > 0 else 1.0
    served = {(l, r) for (l, r), v in x_star.items() if v > 0.5}
    metrics = placement_metrics(
        y_star, channels, served=served, assignment=assignment,
        demand_scenarios=demand_scenarios,
    )

    return {
        "status": pulp.LpStatus[status],
        "y_star": y_star,
        "assignment": assignment,
        "TEC": tec,
        "E_holding": pulp.value(holding_term) or 0,
        "E_penalty": pulp.value(penalty_term) or 0,
        "E_fulfill": pulp.value(fulfill_term) or 0,
        "fill_rate": fill_rate,
        "CL": metrics["CL_stock"],
        "CL_stock": metrics["CL_stock"],
        "CL_open": metrics["CL_open"],
        "CL_open_paper": metrics["CL_open_paper"],
        "SA": metrics["SA"],
        "served": served,
        "total_y": total_y,
        "WC": unit_cost * total_y,
        "policy": "Kieu-DDA-FP",
    }


# ---------------------------------------------------------------
#  Decentralized newsvendor (same cost scale + fill-rate)
# ---------------------------------------------------------------
def solve_decentralized_newsvendor(
    demand_scenarios: Dict[Tuple[str, int], float],
    h_ciw: float,
    penalty: float,
    unit_cost: float,
    channels: List[str] = None,
    n_scenarios: int = None,
) -> dict:
    """Each channel independently solves a newsvendor with holding = h_ciw."""
    channels = channels or CHANNELS
    sc_ids = sorted({s for (_, s) in demand_scenarios.keys()})
    if n_scenarios:
        sc_ids = sc_ids[:n_scenarios]

    y_star = {}
    tec = 0.0
    total_demand = 0.0
    total_fulfilled = 0.0

    for r in channels:
        demands = np.array([demand_scenarios.get((r, w), 0.0) for w in sc_ids], dtype=float)
        total_demand += demands.sum()
        if demands.sum() <= 0:
            y_star[r] = 0.0
            continue
        beta = penalty / (penalty + h_ciw)
        sorted_d = np.sort(demands)
        idx = min(int(np.ceil(beta * len(sorted_d))) - 1, len(sorted_d) - 1)
        y_r = float(sorted_d[max(0, idx)])
        y_star[r] = y_r
        holding_c = h_ciw * np.mean(np.maximum(y_r - demands, 0))
        shortage_c = penalty * np.mean(np.maximum(demands - y_r, 0))
        tec += holding_c + shortage_c
        total_fulfilled += np.minimum(demands, y_r).sum()

    total_y = sum(y_star.values())
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 1.0

    served = {(r, r) for r, yv in y_star.items() if yv > 1e-6}
    metrics = placement_metrics(
        y_star, channels, served=served, demand_scenarios=demand_scenarios
    )

    return {
        "status": "Optimal",
        "y_star": y_star,
        "TEC": tec,
        "E_holding": None,
        "E_penalty": None,
        "E_fulfill": 0.0,
        "fill_rate": fill_rate,
        "CL": 0.0,
        "CL_stock": 0.0,
        "CL_open": metrics["CL_open"],
        "CL_open_paper": metrics["CL_open_paper"],
        "SA": metrics["SA"],
        "served": served,
        "total_y": total_y,
        "WC": unit_cost * total_y,
        "policy": "Decentralized",
    }


# ---------------------------------------------------------------
#  Wave 1 align — metrics + dual-hub network (paper / demo)
# ---------------------------------------------------------------
DUAL_HUB_LOCATIONS = ["CIW", "Hub_Retail", "Hub_B2B"]
DUAL_HUB_CHANNELS = ["RT", "GT", "HC", "HO", "MT"]
DEMO_HOLDING_RATES = {"CIW": 0.02, "Hub_Retail": 0.025, "Hub_B2B": 0.023}
DEMO_FULFILL_VND = {"CIW": 5000.0, "Hub_Retail": 2000.0, "Hub_B2B": 2500.0}


def dual_hub_arcs() -> Set[Tuple[str, str]]:
    arcs = {("CIW", ch) for ch in DUAL_HUB_CHANNELS}
    for ch in ("RT", "HO"):
        arcs.add(("Hub_Retail", ch))
    for ch in ("HC", "GT", "MT"):
        arcs.add(("Hub_B2B", ch))
    return arcs


def dual_hub_holding(unit_cost: float, rates: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    rates = rates or DEMO_HOLDING_RATES
    return {loc: float(rates[loc]) * float(unit_cost) for loc in DUAL_HUB_LOCATIONS}


def dual_hub_fulfill() -> Dict[Tuple[str, str], float]:
    c: Dict[Tuple[str, str], float] = {}
    for ch in DUAL_HUB_CHANNELS:
        c[("CIW", ch)] = DEMO_FULFILL_VND["CIW"]
    for ch in ("RT", "HO"):
        c[("Hub_Retail", ch)] = DEMO_FULFILL_VND["Hub_Retail"]
    for ch in ("HC", "GT", "MT"):
        c[("Hub_B2B", ch)] = DEMO_FULFILL_VND["Hub_B2B"]
    return c


def placement_metrics(
    y_star: Dict[str, float],
    channels: List[str],
    served: Optional[Set[Tuple[str, str]]] = None,
    assignment: Optional[Dict[str, str]] = None,
    demand_scenarios: Optional[Dict[Tuple[str, int], float]] = None,
) -> Dict[str, float]:
    """CL_stock = y_CIW / sum y (03). CL_open uses all channels.

    CL_open_paper: eq. (18) Wang & Minner — denominator = regions with demand.
    Extra key only; CL_open unchanged so 03a CSV cũ vẫn đọc được.
    """
    total_y = float(sum(y_star.values()))
    cl_stock = (float(y_star.get("CIW", 0.0)) / total_y) if total_y > 1e-9 else 0.0
    n_region = max(len(channels), 1)
    n_open = sum(1 for v in y_star.values() if float(v) > 1e-6)
    cl_open = 1.0 - n_open / n_region

    if demand_scenarios is not None:
        with_d = {
            r for (r, _w), q in demand_scenarios.items()
            if r in channels and float(q) > 1e-9
        }
        n_paper = max(len(with_d), 1)
    else:
        n_paper = n_region
    cl_open_paper = 1.0 - n_open / n_paper

    loc_to_ch: Dict[str, Set[str]] = {}
    if assignment:
        for r, loc in assignment.items():
            loc_to_ch.setdefault(loc, set()).add(r)
    elif served:
        for loc, r in served:
            loc_to_ch.setdefault(loc, set()).add(r)
    opened = [loc for loc, v in y_star.items() if float(v) > 1e-6]
    if opened and loc_to_ch:
        sa = float(np.mean([len(loc_to_ch.get(loc, set())) for loc in opened]))
    elif opened and not loc_to_ch:
        sa = 1.0
    else:
        sa = 0.0
    return {
        "CL_stock": cl_stock,
        "CL_open": cl_open,
        "CL_open_paper": cl_open_paper,
        "SA": sa,
    }


def evaluate_given_y(
    y_star: Dict[str, float],
    demand_scenarios: Dict[Tuple[str, int], float],
    holding: Dict[str, float],
    penalty: float,
    fulfill_cost: Dict[Tuple[str, str], float],
    locations: List[str],
    channels: List[str],
    feasible_arcs: Set[Tuple[str, str]],
    unit_cost: float,
    assignment: Optional[Dict[str, str]] = None,
    msg: bool = False,
) -> dict:
    """Second-stage only: fix y, optimize z on a (possibly OOS) scenario set.

    assignment=None: any feasible arc (VP / old 03a FP OOS).
    assignment={channel: location}: paper (6) — FP chỉ ship từ kho đã gán.
    """
    sc_ids = sorted({s for (_, s) in demand_scenarios.keys()})
    pi = 1.0 / max(len(sc_ids), 1)
    z_keys = [(l, r, w) for l in locations for r in channels for w in sc_ids if (l, r) in feasible_arcs]

    prob = pulp.LpProblem("Kieu_DDA_eval", pulp.LpMinimize)
    z = pulp.LpVariable.dicts("z", z_keys, lowBound=0)
    s = pulp.LpVariable.dicts("s", [(l, w) for l in locations for w in sc_ids], lowBound=0)
    b = pulp.LpVariable.dicts("b", [(r, w) for r in channels for w in sc_ids], lowBound=0)

    holding_term = pi * pulp.lpSum(holding.get(l, 0.0) * s[(l, w)] for l in locations for w in sc_ids)
    penalty_term = pi * pulp.lpSum(penalty * b[(r, w)] for r in channels for w in sc_ids)
    fulfill_term = pi * pulp.lpSum(fulfill_cost.get((l, r), 0.0) * z[(l, r, w)] for (l, r, w) in z_keys)
    prob += holding_term + penalty_term + fulfill_term

    for w in sc_ids:
        for l in locations:
            y_l = float(y_star.get(l, 0.0))
            outflow = pulp.lpSum(z[(l, r, w)] for r in channels if (l, r, w) in z)
            prob += s[(l, w)] == y_l - outflow
        for r in channels:
            inflow = pulp.lpSum(z[(l, r, w)] for l in locations if (l, r, w) in z)
            d = demand_scenarios.get((r, w), 0.0)
            prob += b[(r, w)] == d - inflow
            if assignment is not None:
                loc_r = assignment.get(r)
                for l in locations:
                    if (l, r, w) in z and loc_r is not None and l != loc_r:
                        prob += z[(l, r, w)] == 0

    status = prob.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=60))
    total_demand = sum(demand_scenarios.get((r, w), 0.0) for r in channels for w in sc_ids)
    total_shortage = sum((pulp.value(b[(r, w)]) or 0.0) for r in channels for w in sc_ids)
    fill_rate = 1.0 - total_shortage / total_demand if total_demand > 0 else 1.0
    tec = pulp.value(prob.objective) or 0.0
    mean_d = total_demand / max(len(sc_ids), 1)
    return {
        "status": pulp.LpStatus[status],
        "TEC": tec,
        "E_holding": pulp.value(holding_term) or 0.0,
        "E_penalty": pulp.value(penalty_term) or 0.0,
        "E_fulfill": pulp.value(fulfill_term) or 0.0,
        "fill_rate": fill_rate,
        "cost_per_unit": tec / mean_d if mean_d > 1e-9 else None,
        "total_demand": total_demand,
        "WC": unit_cost * sum(float(y_star.get(l, 0.0)) for l in locations),
    }


def evaluate_decentralized_given_y(
    y_star: Dict[str, float],
    demand_scenarios: Dict[Tuple[str, int], float],
    h_ciw: float,
    penalty: float,
    unit_cost: float,
    channels: Optional[List[str]] = None,
    fulfill_cost: Optional[Dict[Tuple[str, str], float]] = None,
    msg: bool = False,
) -> dict:
    """OOS Dec on identity network (channel = fake DC). Does not change solve_decentralized_newsvendor.

    Default fulfill = 0 → cùng thước train Dec (holding + penalty). Truyền fulfill_cost nếu muốn cộng cước local.
    """
    channels = list(channels or CHANNELS)
    locations = list(channels)
    holding = {r: float(h_ciw) for r in channels}
    arcs = {(r, r) for r in channels}
    ful = fulfill_cost if fulfill_cost is not None else {(r, r): 0.0 for r in channels}
    y_loc = {r: float(y_star.get(r, 0.0)) for r in channels}
    ev = evaluate_given_y(
        y_loc, demand_scenarios, holding, penalty, ful,
        locations, channels, arcs, unit_cost, assignment=None, msg=msg,
    )
    ev["policy"] = "Decentralized"
    return ev




