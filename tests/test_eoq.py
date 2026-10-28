"""Unit tests for the EOQ (Economic Order Quantity) service."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps")]

import pytest

from api.services.eoq import calculate_eoq


def test_eoq_theoretical_formula():
    # D = 2400 bottles/yr, K = 1,000,000 VND, H = 30,000 VND/bottle/yr
    # EOQ_cont = sqrt(2 * 2400 * 1_000_000 / 30_000) = sqrt(160,000) = 400 bottles
    res = calculate_eoq(D=2400, K=1_000_000, H=30_000, pack_size=6)
    assert math.isclose(res["eoq_continuous"], 400.0, rel_tol=1e-5)
    assert res["status"] == "success"

    # Exact cartons = 400 / 6 = 66.6667
    assert math.isclose(res["eoq_cartons_exact"], 66.6667, rel_tol=1e-3)
    assert res["floor_cartons"] == 66
    assert res["floor_bottles"] == 396
    assert res["ceil_cartons"] == 67
    assert res["ceil_bottles"] == 402

    # Recommended must be one of floor or ceil with min TRC
    assert res["recommended_cartons"] in (66, 67)
    assert res["total_relevant_cost"] == min(res["floor_trc"], res["ceil_trc"])


def test_eoq_floor_vs_ceil_selection():
    # Test a case where floor is strictly cheaper than ceil
    # D = 120, K = 500_000, H = 20_000
    # EOQ_cont = sqrt(2 * 120 * 500_000 / 20_000) = sqrt(6_000) = 77.4597 bottles
    # k_exact = 77.4597 / 6 = 12.90995 -> floor=12 (72 bottles), ceil=13 (78 bottles)
    res = calculate_eoq(D=120, K=500_000, H=20_000, pack_size=6)
    assert res["floor_cartons"] == 12
    assert res["ceil_cartons"] == 13
    assert res["total_relevant_cost"] <= max(res["floor_trc"], res["ceil_trc"])


def test_eoq_zero_demand_must_not_topup():
    res = calculate_eoq(D=0, K=2_000_000, H=50_000, pack_size=6, moq=600)
    assert res["status"] == "no_demand"
    assert res["recommended_bottles"] == 0
    assert res["recommended_cartons"] == 0
    assert res["total_relevant_cost"] == 0.0
    assert res["moq_met"] is True


def test_eoq_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_eoq(D=-100, K=1_000_000, H=20_000)

    with pytest.raises(ValueError):
        calculate_eoq(D=1000, K=0, H=20_000)

    with pytest.raises(ValueError):
        calculate_eoq(D=1000, K=1_000_000, H=-10)

    with pytest.raises(ValueError):
        calculate_eoq(D=float("nan"), K=1_000_000, H=20_000)

    with pytest.raises(ValueError):
        calculate_eoq(D=1000, K=1_000_000, H=20_000, pack_size=12)


def test_eoq_curve_generation():
    res = calculate_eoq(D=1200, K=1_500_000, H=30_000, pack_size=6)
    pts = res["curve_points"]
    assert len(pts) > 20
    for p in pts:
        assert p["Q_bottles"] % 6 == 0
        assert math.isclose(p["Total_Relevant_Cost"], p["Ordering_Cost"] + p["Holding_Cost"], rel_tol=1e-5)
