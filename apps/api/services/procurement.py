"""Procurement and Case Allocation Module.

Calculates purchase order quantities (PO) to Central Imported Warehouse (CIW)
and downstream fulfillment allocation to 5 distribution channels (RT, HO, HC, GT, MT),
enforcing:
1. Strict integer pack size constraint: 6 bottles/carton (strictly 1 single SKU per carton, no mixed-SKU cartons, no broken cartons).
2. Minimum Order Quantity (MOQ): >= 600 bottles (>= 100 cartons), configurable per-SKU (default) or per-shipment.
3. Top-up heuristics when initial demand is below MOQ (priority by ABC-XYZ velocity or proportional),
   strictly avoiding overstocking SKUs that already have zero net demand.
4. Conservation of integer cartons across channels using the Largest Remainder Method (Hare-Niemeyer).
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

VALID_STRATEGIES = {"velocity", "proportional", "none"}
VALID_MOQ_LEVELS = {"sku", "shipment"}


def allocate_cartons_to_channels(
    total_cartons: int,
    channel_weights: dict[str, float],
    channels: list[str] = CHANNELS,
) -> dict[str, int]:
    """Distribute integer cartons of a single SKU to channels using Largest Remainder (Hare-Niemeyer).

    Guarantees:
    - sum(result.values()) == total_cartons exactly.
    - Every channel receives an integer number of cartons >= 0.
    - Negative weights are clamped to 0.0 and cannot artificially inflate totals.
    """
    if total_cartons <= 0:
        return {ch: 0 for ch in channels}

    clamped_weights = {ch: max(0.0, float(channel_weights.get(ch, 0.0) or 0.0)) for ch in channels}
    weight_sum = sum(clamped_weights.values())

    if weight_sum <= 0:
        shares = {ch: 1.0 / len(channels) for ch in channels}
    else:
        shares = {ch: clamped_weights[ch] / weight_sum for ch in channels}

    ideals = {ch: total_cartons * shares[ch] for ch in channels}
    floors = {ch: int(math.floor(ideals[ch])) for ch in channels}
    remainders = {ch: ideals[ch] - floors[ch] for ch in channels}

    deficit = total_cartons - sum(floors.values())
    ranked = sorted(channels, key=lambda ch: (remainders[ch], shares[ch], ch), reverse=True)

    result = dict(floors)
    for i in range(deficit):
        ch = ranked[i % len(ranked)]
        result[ch] += 1

    return result


def get_channel_weights_for_sku(
    sku: str,
    baseline: str = "preclean",
    y_star: dict[str, float] | None = None,
    policy: str = "Kieu-DDA-VP",
    split: str = "train",
) -> dict[str, float]:
    """Derive channel fulfillment weights for a SKU respecting data partitions.

    - If Decentralized: uses direct channel targets y_star[channel].
    - Otherwise: uses historical 2024 channel sales shares filtered by split
      (train=months 1-8, oos=months 9-12, all=months 1-12) to prevent information leakage.
    """
    weights = {ch: 0.0 for ch in CHANNELS}

    if policy == "Decentralized" and y_star:
        for ch in CHANNELS:
            weights[ch] = max(0.0, float(y_star.get(ch, 0.0) or 0.0))
        if sum(weights.values()) > 0:
            return weights

    try:
        hist = monthly_qty(baseline, [sku], year=2024)
        if not hist.empty and "Quantity" in hist.columns and "Channel" in hist.columns:
            if split == "train" and "month" in hist.columns:
                hist = hist[hist["month"].between(1, 8)]
            elif split == "oos" and "month" in hist.columns:
                hist = hist[hist["month"].between(9, 12)]

            ch_sum = hist.groupby("Channel")["Quantity"].sum().to_dict()
            for ch in CHANNELS:
                weights[ch] = max(0.0, float(ch_sum.get(ch, 0.0)))
    except Exception:
        pass

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
    split: str = "train",
    moq_level: str = "sku",
) -> dict[str, Any]:
    """Calculate procurement order quantities and channel allocations.

    Parameters
    ----------
    sku_rows : list[dict]
        List of result dictionaries. Must contain 'sku'.
    moq : int
        Minimum order quantity in bottles (default 600).
    pack_size : int
        Bottles per carton / thùng (strictly 6 for wine, 1 single SKU per carton).
    topup_strategy : str
        'velocity' (prioritize top ABC classes and high demand),
        'proportional' (distribute deficit proportionally to demand),
        'none' (do not top up, flag under MOQ).
    current_stocks : dict[str, float] | None
        Current on-hand inventory per SKU (bottles). Defaults to 0.
    baseline : str
        Data baseline for SKU lookups (default 'preclean').
    split : str
        Partition split ('train' or 'oos').
    moq_level : str
        'sku' (each ordered SKU must meet MOQ >= 600 bottles / 100 cartons) or
        'shipment' (total supplier shipment must meet MOQ >= 600 bottles).

    Returns
    -------
    dict[str, Any]
        Summary KPIs, per-SKU purchase orders, and channel allocation plans.
    """
    if not isinstance(pack_size, int) or pack_size != 6:
        raise ValueError(f"pack_size phải là 6 chai/thùng nguyên (nhận được: {pack_size})")

    if topup_strategy not in VALID_STRATEGIES:
        raise ValueError(f"Chiến lược top-up không hợp lệ: '{topup_strategy}'. Hợp lệ: {sorted(VALID_STRATEGIES)}")

    if moq_level not in VALID_MOQ_LEVELS:
        raise ValueError(f"moq_level phải là 'sku' hoặc 'shipment', nhận được: '{moq_level}'")

    moq = max(0, int(moq))
    target_cartons = int(math.ceil(moq / pack_size)) if pack_size > 0 else 0
    current_stocks = current_stocks or {}

    # Validate duplicate SKUs
    seen_skus: set[str] = set()
    for r in sku_rows:
        s = str(r.get("sku", "")).strip()
        if not s:
            continue
        if s in seen_skus:
            raise ValueError(f"Phát hiện SKU bị trùng lặp trong đơn hàng: '{s}'")
        seen_skus.add(s)

    abc_map = sku_abc_map(baseline)

    items: list[dict[str, Any]] = []
    for r in sku_rows:
        sku = str(r.get("sku", "")).strip()
        if not sku:
            continue

        item_title = str(r.get("item_name") or sku_name(sku))
        abc = str(r.get("ABC") or S4_CLASS.get(sku) or abc_map.get(sku) or "CZ")
        policy = str(r.get("policy") or "Kieu-DDA-VP")
        supplier_id = str(r.get("supplier_id") or r.get("supplier") or "DEFAULT")

        # Resolve target_y: use total_y if present; if total_y is missing or zero, use sum(y_star)
        target_y_raw = r.get("total_y")
        y_star = r.get("y_star") if isinstance(r.get("y_star"), dict) else {}
        if target_y_raw is not None and float(target_y_raw) > 0:
            target_y = float(target_y_raw)
        elif y_star and sum(y_star.values()) > 0:
            target_y = float(sum(y_star.values()))
        else:
            target_y = float(target_y_raw or 0.0)

        on_hand = max(0.0, float(current_stocks.get(sku, 0.0)))
        net_req = max(0.0, target_y - on_hand)
        base_k = int(math.ceil(net_req / pack_size)) if pack_size > 0 else 0
        base_bottles = base_k * pack_size

        cost = unit_cost(sku)
        ch_weights = get_channel_weights_for_sku(
            sku, baseline=baseline, y_star=y_star, policy=policy, split=split
        )

        items.append({
            "sku": sku,
            "item_name": item_title,
            "ABC": abc,
            "abc_rank": ABC_PRIORITY.get(abc, 99),
            "policy": policy,
            "supplier_id": supplier_id,
            "target_y": target_y,
            "current_stock": on_hand,
            "net_requirement": net_req,
            "base_cartons": base_k,
            "base_bottles": base_bottles,
            "topup_cartons": 0,
            "final_cartons": base_k,
            "final_bottles": base_bottles,
            "deficit_cartons": 0,
            "moq_met": True,
            "unit_cost_vnd": cost,
            "po_amount_vnd": base_bottles * cost,
            "y_star": y_star,
            "channel_weights": ch_weights,
            "channel_cartons": {ch: 0 for ch in CHANNELS},
            "channel_bottles": {ch: 0 for ch in CHANNELS},
        })

    # Empty plan guard
    if not items:
        return {
            "summary": {
                "moq_bottles": moq,
                "pack_size": pack_size,
                "target_cartons": target_cartons,
                "moq_level": moq_level,
                "base_cartons": 0,
                "base_bottles": 0,
                "topup_cartons": 0,
                "final_cartons": 0,
                "final_bottles": 0,
                "deficit_cartons": 0,
                "deficit_bottles": 0,
                "moq_met": True,
                "total_po_amount_vnd": 0.0,
                "topup_strategy": topup_strategy,
                "channel_totals_cartons": {ch: 0 for ch in CHANNELS},
                "channel_totals_bottles": {ch: 0 for ch in CHANNELS},
                "suppliers": {},
            },
            "items": [],
        }

    total_base_cartons = sum(it["base_cartons"] for it in items)
    total_net_req = sum(it["net_requirement"] for it in items)

    # If total demand across all items is zero (e.g. all fully covered by stock or target=0):
    # Do NOT create a shipment! No purchase order required!
    if total_net_req <= 0 or total_base_cartons <= 0:
        for it in items:
            it["final_cartons"] = 0
            it["final_bottles"] = 0
            it["topup_cartons"] = 0
            it["po_amount_vnd"] = 0.0
            it["moq_met"] = True
            it["deficit_cartons"] = 0

        return {
            "summary": {
                "moq_bottles": moq,
                "pack_size": pack_size,
                "target_cartons": target_cartons,
                "moq_level": moq_level,
                "base_cartons": 0,
                "base_bottles": 0,
                "topup_cartons": 0,
                "final_cartons": 0,
                "final_bottles": 0,
                "deficit_cartons": 0,
                "deficit_bottles": 0,
                "moq_met": True,
                "total_po_amount_vnd": 0.0,
                "topup_strategy": topup_strategy,
                "channel_totals_cartons": {ch: 0 for ch in CHANNELS},
                "channel_totals_bottles": {ch: 0 for ch in CHANNELS},
                "suppliers": {},
            },
            "items": items,
        }

    # =========================================================================
    # CASE 1: PER-SKU MOQ (Every SKU with demand > 0 must meet MOQ >= 600 bottles)
    # =========================================================================
    if moq_level == "sku":
        all_skus_met = True
        total_deficit_cartons = 0

        for it in items:
            if it["net_requirement"] <= 0 or it["base_cartons"] <= 0:
                it["topup_cartons"] = 0
                it["final_cartons"] = 0
                it["final_bottles"] = 0
                it["po_amount_vnd"] = 0.0
                it["moq_met"] = True
                it["deficit_cartons"] = 0
            else:
                if it["base_cartons"] < target_cartons:
                    deficit_k = target_cartons - it["base_cartons"]
                    if topup_strategy != "none":
                        # Auto top-up this SKU to 100 cartons (600 bottles)
                        it["topup_cartons"] = deficit_k
                        it["final_cartons"] = target_cartons
                        it["final_bottles"] = target_cartons * pack_size
                        it["po_amount_vnd"] = it["final_bottles"] * it["unit_cost_vnd"]
                        it["deficit_cartons"] = 0
                        it["moq_met"] = True
                    else:
                        it["topup_cartons"] = 0
                        it["final_cartons"] = it["base_cartons"]
                        it["final_bottles"] = it["base_bottles"]
                        it["po_amount_vnd"] = it["final_bottles"] * it["unit_cost_vnd"]
                        it["deficit_cartons"] = deficit_k
                        it["moq_met"] = False
                        all_skus_met = False
                        total_deficit_cartons += deficit_k
                else:
                    it["topup_cartons"] = 0
                    it["final_cartons"] = it["base_cartons"]
                    it["final_bottles"] = it["base_bottles"]
                    it["po_amount_vnd"] = it["final_bottles"] * it["unit_cost_vnd"]
                    it["deficit_cartons"] = 0
                    it["moq_met"] = True

        supplier_summaries: dict[str, dict[str, Any]] = {}
        for it in items:
            s_id = it["supplier_id"]
            if s_id not in supplier_summaries:
                supplier_summaries[s_id] = {
                    "supplier_id": s_id,
                    "base_cartons": 0,
                    "final_cartons": 0,
                    "deficit_cartons": 0,
                    "moq_met": True,
                }
            supplier_summaries[s_id]["base_cartons"] += it["base_cartons"]
            supplier_summaries[s_id]["final_cartons"] += it["final_cartons"]
            supplier_summaries[s_id]["deficit_cartons"] += it.get("deficit_cartons", 0)
            if not it.get("moq_met", True):
                supplier_summaries[s_id]["moq_met"] = False

        all_suppliers_met = all_skus_met

    # =========================================================================
    # CASE 2: PER-SHIPMENT MOQ (Total supplier shipment must meet MOQ >= 600 bottles)
    # =========================================================================
    else:
        supplier_groups: dict[str, list[dict[str, Any]]] = {}
        for it in items:
            supplier_groups.setdefault(it["supplier_id"], []).append(it)

        supplier_summaries = {}
        all_suppliers_met = True
        total_deficit_cartons = 0

        for supp_id, supp_items in supplier_groups.items():
            supp_base_k = sum(it["base_cartons"] for it in supp_items)
            supp_deficit_k = max(0, target_cartons - supp_base_k)

            # Top-up is only allowed on SKUs that genuinely have net_requirement > 0 (not overstocked!)
            eligible_items = [it for it in supp_items if it["net_requirement"] > 0]

            if supp_deficit_k > 0 and eligible_items and topup_strategy != "none":
                if topup_strategy == "proportional":
                    weights = {it["sku"]: max(1.0, float(it["base_cartons"])) for it in eligible_items}
                    topup_alloc = allocate_cartons_to_channels(
                        supp_deficit_k, weights, channels=[it["sku"] for it in eligible_items]
                    )
                    for it in eligible_items:
                        it["topup_cartons"] += topup_alloc.get(it["sku"], 0)
                else:
                    # Velocity / ABC priority on eligible items
                    ranked = sorted(
                        eligible_items, key=lambda it: (it["abc_rank"], -it["base_cartons"], it["sku"])
                    )
                    num_eligible = max(1, min(len(ranked), 4))
                    for i in range(supp_deficit_k):
                        chosen = ranked[i % num_eligible]
                        chosen["topup_cartons"] += 1

                for it in supp_items:
                    it["final_cartons"] = it["base_cartons"] + it["topup_cartons"]
                    it["final_bottles"] = it["final_cartons"] * pack_size
                    it["po_amount_vnd"] = it["final_bottles"] * it["unit_cost_vnd"]

                supp_final_k = sum(it["final_cartons"] for it in supp_items)
                supp_met = supp_final_k >= target_cartons
                supp_deficit_k = max(0, target_cartons - supp_final_k)
            else:
                supp_final_k = sum(it["final_cartons"] for it in supp_items)
                supp_met = supp_final_k >= target_cartons
                if not supp_met:
                    all_suppliers_met = False

            if not supp_met:
                all_suppliers_met = False

            total_deficit_cartons += supp_deficit_k
            supplier_summaries[supp_id] = {
                "supplier_id": supp_id,
                "base_cartons": supp_base_k,
                "final_cartons": supp_final_k,
                "deficit_cartons": supp_deficit_k,
                "moq_met": supp_met,
            }

    total_final_cartons = sum(it["final_cartons"] for it in items)
    total_final_bottles = sum(it["final_bottles"] for it in items)
    total_po_amount = sum(it["po_amount_vnd"] for it in items)

    # Downstream channel allocation per SKU (100% Single-SKU Cartons)
    channel_totals_cartons = {ch: 0 for ch in CHANNELS}
    channel_totals_bottles = {ch: 0 for ch in CHANNELS}

    for it in items:
        if it["final_cartons"] > 0:
            ch_cartons = allocate_cartons_to_channels(
                it["final_cartons"], it["channel_weights"], channels=CHANNELS
            )
        else:
            ch_cartons = {ch: 0 for ch in CHANNELS}

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
        "moq_level": moq_level,
        "base_cartons": total_base_cartons,
        "base_bottles": total_base_cartons * pack_size,
        "topup_cartons": total_final_cartons - total_base_cartons,
        "final_cartons": total_final_cartons,
        "final_bottles": total_final_bottles,
        "deficit_cartons": total_deficit_cartons,
        "deficit_bottles": total_deficit_cartons * pack_size,
        "moq_met": all_suppliers_met,
        "total_po_amount_vnd": total_po_amount,
        "topup_strategy": topup_strategy,
        "channel_totals_cartons": channel_totals_cartons,
        "channel_totals_bottles": channel_totals_bottles,
        "suppliers": supplier_summaries,
    }

    return {
        "summary": summary,
        "items": items,
    }
