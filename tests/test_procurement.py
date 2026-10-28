"""Tests for the procurement and case allocation service."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps")]

import pytest

from api.services.procurement import (
    allocate_cartons_to_channels,
    calculate_procurement_plan,
)


def test_allocate_cartons_to_channels_conservation():
    weights = {"RT": 40.0, "HO": 20.0, "HC": 15.0, "GT": 15.0, "MT": 10.0}
    for total_cartons in [0, 1, 5, 17, 33, 100, 245]:
        alloc = allocate_cartons_to_channels(total_cartons, weights)
        assert sum(alloc.values()) == total_cartons
        for ch, count in alloc.items():
            assert isinstance(count, int)
            assert count >= 0


def test_procurement_plan_moq_topup_velocity():
    # S4 demo rows with small targets that sum to < 100 cartons
    sku_rows = [
        {"sku": "W100133", "item_name": "Allan Scott", "ABC": "AX", "total_y": 70.0, "policy": "Kieu-DDA-VP"},
        {"sku": "W103028", "item_name": "Chevalier Brut", "ABC": "AY", "total_y": 50.0, "policy": "Kieu-DDA-VP"},
        {"sku": "W131012", "item_name": "Cantine Ronco Bianco", "ABC": "BY", "total_y": 30.0, "policy": "Kieu-DDA-VP"},
        {"sku": "W100627", "item_name": "Cantine Ronco Rosso", "ABC": "CZ", "total_y": 20.0, "policy": "Kieu-DDA-VP"},
    ]

    res = calculate_procurement_plan(sku_rows, moq=600, pack_size=6, topup_strategy="velocity")
    summary = res["summary"]
    items = res["items"]

    assert summary["moq_bottles"] == 600
    assert summary["pack_size"] == 6
    assert summary["target_cartons"] == 100

    # Total cartons must meet or exceed MOQ (100 cartons)
    assert summary["final_cartons"] >= 100
    assert summary["final_bottles"] >= 600
    assert summary["moq_met"] is True

    # Every SKU must have integer multiples of 6 bottles
    for it in items:
        assert it["final_bottles"] == it["final_cartons"] * 6
        assert it["base_bottles"] == it["base_cartons"] * 6
        assert sum(it["channel_cartons"].values()) == it["final_cartons"]
        assert sum(it["channel_bottles"].values()) == it["final_bottles"]

        # Channel bottles are multiples of 6
        for ch, b_count in it["channel_bottles"].items():
            assert b_count % 6 == 0


def test_procurement_plan_no_topup():
    sku_rows = [
        {"sku": "W100133", "item_name": "Allan Scott", "ABC": "AX", "total_y": 70.0, "policy": "Kieu-DDA-VP"},
    ]
    res = calculate_procurement_plan(sku_rows, moq=600, pack_size=6, topup_strategy="none")
    summary = res["summary"]

    assert summary["moq_met"] is False
    assert summary["deficit_cartons"] > 0
    assert summary["deficit_bottles"] > 0
    assert summary["final_bottles"] < 600


def test_procurement_plan_per_sku_moq_w102705():
    # User's exact scenario: W102705 with y*=544.7, 0 on hand
    sku_rows = [
        {"sku": "W102705", "item_name": "Cantine Ronco Romance", "ABC": "AY", "total_y": 544.7, "policy": "Kieu-DDA-VP"},
    ]
    res = calculate_procurement_plan(sku_rows, moq=600, pack_size=6, moq_level="sku", topup_strategy="velocity")
    item = res["items"][0]

    # Base: ceil(544.7 / 6) = 91 cartons = 546 bottles
    assert item["base_cartons"] == 91
    assert item["base_bottles"] == 546

    # Must be topped up to 100 cartons = 600 bottles
    assert item["topup_cartons"] == 9
    assert item["final_cartons"] == 100
    assert item["final_bottles"] == 600
    assert item["moq_met"] is True
    assert sum(item["channel_cartons"].values()) == 100
    assert sum(item["channel_bottles"].values()) == 600
    for ch, b in item["channel_bottles"].items():
        assert b % 6 == 0

