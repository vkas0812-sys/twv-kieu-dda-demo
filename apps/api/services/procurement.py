"""Procurement and Case Allocation Module.

Calculates purchase order quantities (PO) to Central Imported Warehouse (CIW)
and downstream fulfillment allocation to 5 distribution channels (RT, HO, HC, GT, MT),
enforcing:
1. Integer pack size constraint: 6 bottles/carton (no partial cartons allowed).
2. Supplier Minimum Order Quantity (MOQ): >= 600 bottles (>= 100 cartons) per shipment.
3. Top-up heuristics when initial demand is below MOQ (priority by ABC-XYZ velocity or proportional).
4. Fair integer carton allocation across channels using the Largest Remainder Method (Hare-Niemeyer).
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from api.services.data_store import monthly_qty, sku_abc_map, sku_name, unit_cost
from api.services.paths import CHANNELS, S4_CLASS


ABC_PRIORITY = {
    "AX": 1,
    "AY": 2,
    "AZ": 3,
    "BX": 4,
    "BY": 5,
    "BZ": 6,
    "CX": 7,
    "CY": 8,
    "CZ": 9,
}


def allocate_cartons_to_channels(
    total_cartons: int,
    channel_weights: dict[str, float],
    channels: list[str] = CHANNELS,
) -> dict[str, int]:
    """Distribute integer cartons to channels using Largest Remainder (Hare-Niemeyer).

    Guarantees:
    - sum(result.values()) == total_cartons exactly.
    - Every channel receives an integer number of cartons >= 0.
    """
    if total_cartons <= 0:
        return {ch: 0 for ch in channels}

    weight_sum = sum(channel_weights.get(ch, 0.0) for ch in channels)
    if weight_sum <= 0:
        # If no historical/target preference, split as evenly as possible
        shares = {ch: 1.0 / len(channels) for ch in channels}
    else:
        shares = {ch: max(0.0, channel_weights.get(ch, 0.0)) / weight_sum for ch in channels}

    ideals = {ch: total_cartons * shares[ch] for ch in channels}
    floors = {ch: int(math.floor(ideals[ch])) for ch in channels}
    remainders = {ch: ideals[ch] - floors[ch] for ch in channels}

    deficit = total_cartons - sum(floors.values())
    # Sort channels by highest remainder first, tie-break by channel name
    ranked_channels = sorted(channels, key=lambda ch: (remainders[ch], shares[ch]), reverse=True)

    result = dict(floors)
    for i in range(deficit):
        ch = ranked_channels[i % len(ranked_channels)]
        result[ch] += 1

    return result


def get_channel_weights_for_sku(
    sku: str,
    baseline: str = "preclean",
    y_star: dict[str, float] | None = None,
    policy: str = "Kieu-DDA-VP",
) -> dict[str, float]:
    """Derive channel fulfillment weights for a SKU.

    If Decentralized policy, uses direct channel targets y_star[channel].
    Otherwise, uses historical 2024 channel sales shares from monthly_qty.
    """
    weights = {ch: 0.0 for ch in CHANNELS}

    # If Decentralized has channel-specific y_star
    if policy == "Decentralized" and y_star:
        for ch in CHANNELS:
            weights[ch] = float(y_star.get(ch, 0.0) or 0.0)
        if sum(weights.values()) > 0:
            return weights

    # Otherwise fallback to 2024 channel distribution
    try:
        hist = monthly_qty(baseline, [sku], year=2024)
        if not hist.empty and "Quantity" in hist.columns and "Channel" in hist.columns:
            ch_sum = hist.groupby("Channel")["Quantity"].sum().to_dict()
            for ch in CHANNELS:
                weights[ch] = max(0.0, float(ch_sum.get(ch, 0.0)))
    except Exception:
        pass

    # If still empty, return equal weights
    if sum(weights.values()) <= 0:
        for ch in CHANNELS:
            weights[ch] = 1.0

    return weights


def calculate_procurement_plan(
    sku_rows: list[dict[str, Any]],
    moq: int = 600,
    pack_size: int = 6,
    topup_strategy: str = "velocity",
    current_stocks: dict[str, float] | None = None,
    baseline: str = "preclean",
) -> dict[str, Any]:
    """Calculate procurement order quantities and channel allocations.

    Parameters
    ----------
    sku_rows : list[dict]
        List of result dictionaries from solve_protocol or UI (must contain 'sku',
        and optionally 'total_y', 'ABC', 'item_name', 'policy', 'y_star').
    moq : int
        Minimum order quantity in bottles per shipment (default 600).
    pack_size : int
        Bottles per carton / thùng (default 6, no partial cartons).
    topup_strategy : str
        'velocity' (prioritize top ABC classes and high demand),
        'proportional' (distribute deficit proportionally to demand),
        'none' (do not top up, flag under MOQ).
    current_stocks : dict[str, float] | None
        Current on-hand inventory per SKU (bottles). Defaults to 0.
    baseline : str
        Data baseline for SKU lookups (default 'preclean').

    Returns
    -------
    dict[str, Any]
        Summary KPIs, per-SKU purchase orders, and channel allocation plans.
    """
    pack_size = max(1, int(pack_size))
    moq = max(0, int(moq))
    target_cartons = int(math.ceil(moq / pack_size)) if pack_size > 0 else 0
    current_stocks = current_stocks or {}

    abc_map = sku_abc_map(baseline)

    items: list[dict[str, Any]] = []
    for r in sku_rows:
        sku = str(r.get("sku", ""))
        if not sku:
            continue

        item_title = str(r.get("item_name") or sku_name(sku))
        abc = str(r.get("ABC") or S4_CLASS.get(sku) or abc_map.get(sku) or "CZ")
        policy = str(r.get("policy") or "Kieu-DDA-VP")
        target_y = float(r.get("total_y") or 0.0)
        on_hand = float(current_stocks.get(sku, 0.0))

        net_req = max(0.0, target_y - on_hand)
        base_k = int(math.ceil(net_req / pack_size)) if pack_size > 0 else 0
        base_bottles = base_k * pack_size

        cost = unit_cost(sku)
        y_star = r.get("y_star") if isinstance(r.get("y_star"), dict) else {}

        ch_weights = get_channel_weights_for_sku(sku, baseline=baseline, y_star=y_star, policy=policy)

        items.append({
            "sku": sku,
            "item_name": item_title,
            "ABC": abc,
            "abc_rank": ABC_PRIORITY.get(abc, 99),
            "policy": policy,
            "target_y": target_y,
            "current_stock": on_hand,
            "net_requirement": net_req,
            "base_cartons": base_k,
            "base_bottles": base_bottles,
            "topup_cartons": 0,
            "final_cartons": base_k,
            "final_bottles": base_bottles,
            "unit_cost_vnd": cost,
            "po_amount_vnd": base_bottles * cost,
            "y_star": y_star,
            "channel_weights": ch_weights,
        })

    total_base_cartons = sum(it["base_cartons"] for it in items)
    total_base_bottles = sum(it["base_bottles"] for it in items)

    deficit_cartons = max(0, target_cartons - total_base_cartons)
    deficit_bottles = deficit_cartons * pack_size
    moq_met = deficit_cartons == 0

    # Top-up logic if below MOQ
    if deficit_cartons > 0 and items and topup_strategy != "none":
        if topup_strategy == "proportional":
            # Allocate deficit cartons proportionally to base_cartons (or target_y)
            weights = {it["sku"]: max(0.1, it["base_cartons"] if it["base_cartons"] > 0 else it["target_y"]) for it in items}
            topup_alloc = allocate_cartons_to_channels(
                deficit_cartons, weights, channels=[it["sku"] for it in items]
            )
            for it in items:
                it["topup_cartons"] = topup_alloc.get(it["sku"], 0)
        else:
            # Velocity / ABC priority:
            # Sort items by: 1) ABC rank (AX first), 2) target_y descending, 3) SKU name
            ranked_items = sorted(
                items, key=lambda it: (it["abc_rank"], -it["target_y"], it["sku"])
            )
            # Round-robin allocation among the top available items
            # Give to highest ranked first
            num_eligible = max(1, min(len(ranked_items), 4))
            for i in range(deficit_cartons):
                chosen = ranked_items[i % num_eligible]
                chosen["topup_cartons"] += 1

        # Recompute finals
        for it in items:
            it["final_cartons"] = it["base_cartons"] + it["topup_cartons"]
            it["final_bottles"] = it["final_cartons"] * pack_size
            it["po_amount_vnd"] = it["final_bottles"] * it["unit_cost_vnd"]

        moq_met = True

    total_final_cartons = sum(it["final_cartons"] for it in items)
    total_final_bottles = sum(it["final_bottles"] for it in items)
    total_po_amount = sum(it["po_amount_vnd"] for it in items)

    # Channel allocations per SKU
    channel_totals_cartons = {ch: 0 for ch in CHANNELS}
    channel_totals_bottles = {ch: 0 for ch in CHANNELS}

    for it in items:
        ch_cartons = allocate_cartons_to_channels(
            it["final_cartons"], it["channel_weights"], channels=CHANNELS
        )
        ch_bottles = {ch: ch_cartons[ch] * pack_size for ch in CHANNELS}
        it["channel_cartons"] = ch_cartons
        it["channel_bottles"] = ch_bottles

        for ch in CHANNELS:
            channel_totals_cartons[ch] += ch_cartons[ch]
            channel_totals_bottles[ch] += ch_bottles[ch]

    summary = {
        "moq_bottles": moq,
        "pack_size": pack_size,
        "target_cartons": target_cartons,
        "base_cartons": total_base_cartons,
        "base_bottles": total_base_bottles,
        "topup_cartons": total_final_cartons - total_base_cartons,
        "final_cartons": total_final_cartons,
        "final_bottles": total_final_bottles,
        "deficit_cartons": 0 if moq_met else deficit_cartons,
        "deficit_bottles": 0 if moq_met else deficit_bottles,
        "moq_met": moq_met,
        "total_po_amount_vnd": total_po_amount,
        "topup_strategy": topup_strategy,
        "channel_totals_cartons": channel_totals_cartons,
        "channel_totals_bottles": channel_totals_bottles,
    }

    return {
        "summary": summary,
        "items": items,
    }
