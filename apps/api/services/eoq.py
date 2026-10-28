"""Economic Order Quantity (EOQ) Analysis Service.

Calculates theoretical and practical integer-carton EOQ, performs cost trade-off
analysis (floor vs ceil rounding), models supplier MOQ impact, and provides
curve decomposition points for visual analytics.

Notes:
- EOQ is a static, deterministic lot-sizing baseline.
- It does NOT replace two-stage stochastic base-stock (y*) from Kieu-DDA.
- It does NOT automatically modify Purchase Orders or deduct on-hand stock.
"""
from __future__ import annotations

import math
from typing import Any


def calculate_eoq(
    D: float,
    K: float,
    H: float,
    pack_size: int = 6,
    moq: int = 600,
    supplier_id: str | None = None,
) -> dict[str, Any]:
    """Calculate Economic Order Quantity (EOQ) with integer case constraints.

    Parameters
    ----------
    D : float
        Annual demand in bottles/year (must be >= 0).
    K : float
        Fixed ordering cost in VND/order (must be > 0).
    H : float
        Annual inventory holding cost in VND/bottle/year (must be > 0).
    pack_size : int
        Bottles per carton (strictly 6 for wine).
    moq : int
        Supplier minimum order quantity per shipment (default 600 bottles = 100 cartons).
    supplier_id : str | None
        Optional supplier ID for shipment pooling context.

    Returns
    -------
    dict[str, Any]
        Comprehensive EOQ metrics, floor/ceil comparisons, and cost curve points.
    """
    # 1. Validation
    for name, val in [("D", D), ("K", K), ("H", H)]:
        if val is None or math.isnan(val) or math.isinf(val):
            raise ValueError(f"Tham số {name} không được là NaN hoặc Infinity.")

    if D < 0:
        raise ValueError("Nhu cầu năm D không được âm.")
    if K <= 0:
        raise ValueError("Chi phí đặt hàng K phải lớn hơn 0 (VND/đơn).")
    if H <= 0:
        raise ValueError("Chi phí lưu kho H phải lớn hơn 0 (VND/chai/năm).")
    if not isinstance(pack_size, int) or pack_size != 6:
        raise ValueError(f"Quy cách đóng thùng pack_size phải là 6 chai/thùng (nhận được: {pack_size}).")

    # 2. Edge Case: D == 0
    if D == 0.0:
        return {
            "status": "no_demand",
            "message": "Không có nhu cầu đặt hàng",
            "D": 0.0,
            "K": float(K),
            "H": float(H),
            "pack_size": pack_size,
            "moq": moq,
            "supplier_id": supplier_id,
            "eoq_continuous": 0.0,
            "eoq_cartons_exact": 0.0,
            "floor_cartons": 0,
            "floor_bottles": 0,
            "floor_ordering_cost": 0.0,
            "floor_holding_cost": 0.0,
            "floor_trc": 0.0,
            "ceil_cartons": 0,
            "ceil_bottles": 0,
            "ceil_ordering_cost": 0.0,
            "ceil_holding_cost": 0.0,
            "ceil_trc": 0.0,
            "recommended_cartons": 0,
            "recommended_bottles": 0,
            "recommended_rounding": "none",
            "ordering_cost": 0.0,
            "holding_cost": 0.0,
            "total_relevant_cost": 0.0,
            "orders_per_year": 0.0,
            "cycle_days": 0.0,
            "moq_met": True,
            "moq_bottles": moq,
            "moq_cartons": int(math.ceil(moq / pack_size)),
            "cost_penalty_moq": 0.0,
            "curve_points": [],
            "assumptions": [
                "Nhu cầu D = 0: Không phát sinh lệnh đặt hàng.",
                "Không áp dụng bù MOQ khi sản phẩm không có nhu cầu.",
            ],
        }

    # 3. Theoretical continuous EOQ
    eoq_cont = math.sqrt((2.0 * D * K) / H)
    k_exact = eoq_cont / pack_size

    # 4. Integer Carton Options: Floor vs Ceil
    k_floor = max(1, int(math.floor(k_exact)))
    q_floor = k_floor * pack_size
    oc_floor = (D / q_floor) * K
    hc_floor = (q_floor / 2.0) * H
    trc_floor = oc_floor + hc_floor

    k_ceil = max(1, int(math.ceil(k_exact)))
    q_ceil = k_ceil * pack_size
    oc_ceil = (D / q_ceil) * K
    hc_ceil = (q_ceil / 2.0) * H
    trc_ceil = oc_ceil + hc_ceil

    # Cost comparison: do NOT assume ceil is always better!
    if trc_floor <= trc_ceil:
        rec_k = k_floor
        rec_q = q_floor
        rec_oc = oc_floor
        rec_hc = hc_floor
        rec_trc = trc_floor
        rec_rounding = "floor"
    else:
        rec_k = k_ceil
        rec_q = q_ceil
        rec_oc = oc_ceil
        rec_hc = hc_ceil
        rec_trc = trc_ceil
        rec_rounding = "ceil"

    orders_per_year = D / rec_q
    cycle_days = (365.0 / orders_per_year) if orders_per_year > 0 else 0.0

    # Theoretical continuous costs
    theo_oc = (D / eoq_cont) * K
    theo_hc = (eoq_cont / 2.0) * H
    theo_trc = theo_oc + theo_hc

    # 5. MOQ Analysis (Per Supplier / Shipment)
    target_moq_k = int(math.ceil(moq / pack_size)) if pack_size > 0 else 0
    moq_met_single = rec_q >= moq

    # Calculate penalty if forced to order 600 standalone
    q_moq = max(rec_q, moq)
    oc_moq = (D / q_moq) * K
    hc_moq = (q_moq / 2.0) * H
    trc_moq = oc_moq + hc_moq
    cost_penalty_moq = max(0.0, trc_moq - rec_trc)

    # 6. Cost Curves Generation (for plotting TRC, OC, HC vs Q)
    q_min = max(6, int(round(min(q_floor * 0.3, eoq_cont * 0.3))))
    q_max = max(1200, int(round(max(q_ceil * 2.5, eoq_cont * 2.5, 900))))
    q_min = (q_min // 6) * 6 or 6
    q_max = (q_max // 6) * 6

    step = max(6, ((q_max - q_min) // 80 // 6) * 6 or 6)
    curve_points: list[dict[str, Any]] = []

    for q in range(q_min, q_max + step, step):
        if q <= 0:
            continue
        oc = (D / q) * K
        hc = (q / 2.0) * H
        curve_points.append({
            "Q_bottles": q,
            "Q_cartons": q // pack_size,
            "Ordering_Cost": oc,
            "Holding_Cost": hc,
            "Total_Relevant_Cost": oc + hc,
        })

    return {
        "status": "success",
        "message": "Tính toán thành công",
        "D": float(D),
        "K": float(K),
        "H": float(H),
        "pack_size": pack_size,
        "moq": moq,
        "supplier_id": supplier_id,
        "eoq_continuous": eoq_cont,
        "eoq_cartons_exact": k_exact,
        "floor_cartons": k_floor,
        "floor_bottles": q_floor,
        "floor_ordering_cost": oc_floor,
        "floor_holding_cost": hc_floor,
        "floor_trc": trc_floor,
        "ceil_cartons": k_ceil,
        "ceil_bottles": q_ceil,
        "ceil_ordering_cost": oc_ceil,
        "ceil_holding_cost": hc_ceil,
        "ceil_trc": trc_ceil,
        "recommended_cartons": rec_k,
        "recommended_bottles": rec_q,
        "recommended_rounding": rec_rounding,
        "ordering_cost": rec_oc,
        "holding_cost": rec_hc,
        "total_relevant_cost": rec_trc,
        "theo_trc": theo_trc,
        "orders_per_year": orders_per_year,
        "cycle_days": cycle_days,
        "moq_met_single": moq_met_single,
        "moq_bottles": moq,
        "moq_cartons": target_moq_k,
        "trc_at_moq": trc_moq,
        "cost_penalty_moq": cost_penalty_moq,
        "curve_points": curve_points,
        "assumptions": [
            "Tốc độ tiêu thụ nhu cầu D là hằng số đều đặn theo thời gian.",
            "Thời gian giao hàng (Lead time) coi như cố định hoặc đã được bù bằng điểm đặt lại ROP.",
            "Giá mua đơn vị không đổi (chưa xét chiết khấu theo số lượng Quantity Discount).",
            "Không cho phép thiếu hụt hàng trong chu kỳ EOQ chuẩn (No planned backorders).",
            "Chi phí lưu kho H tính trên tồn kho bình quân chu kỳ (Q / 2).",
            "Quy cách đóng thùng cố định 6 chai/thùng nguyên (Single-SKU cartons).",
        ],
    }
