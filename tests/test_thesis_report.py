"""Tests for the Thesis Report Service and 3-Tier Governance."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps")]

import pandas as pd
import pytest

from api.services.thesis_report import (
    calculate_po_waterfall,
    generate_three_tier_governance,
    get_data_health_metrics,
    get_forecast_tournament_summary,
    get_inventory_tradeoff_summary,
    get_portfolio_segmentation_summary,
    render_full_markdown_report,
)


def test_data_health_metrics():
    health = get_data_health_metrics("active")
    assert health["n_skus"] > 0
    assert health["total_volume"] > 0
    assert health["train_months"] == 32
    assert health["oos_months"] == 4
    assert len(health["intermittency_stats"]) > 0


def test_portfolio_segmentation_summary():
    port = get_portfolio_segmentation_summary("active")
    assert len(port["s4_classes"]) == 4
    skus = [s["sku"] for s in port["s4_classes"]]
    assert set(skus) == {"W100133", "W103028", "W131012", "W100627"}


def test_forecast_tournament_summary():
    fc = get_forecast_tournament_summary()
    assert not fc.empty
    assert "sku" in fc.columns
    assert "wape" in fc.columns
    assert "rmse" in fc.columns
    assert "bias" in fc.columns


def test_inventory_tradeoff_summary():
    inv = get_inventory_tradeoff_summary()
    assert not inv.empty
    assert "policy" in inv.columns
    assert "TEC" in inv.columns
    assert "fill_rate" in inv.columns


def test_po_waterfall_calculation():
    # Test normal positive case
    wf = calculate_po_waterfall("W100133", y_target=750, on_hand=200, on_order=0, pack_size=6, moq=600)
    assert wf["net_requirement"] == 550.0
    assert wf["carton_bottles"] == 552.0  # ceil(550/6)*6 = 92*6
    assert wf["final_po"] == 600          # Top-up to 600 MOQ
    assert wf["moq_topup"] == 48

    # Test zero net requirement (on hand >= target)
    wf_zero = calculate_po_waterfall("W100133", y_target=100, on_hand=200, on_order=0, pack_size=6, moq=600)
    assert wf_zero["net_requirement"] == 0.0
    assert wf_zero["final_po"] == 0
    assert wf_zero["moq_topup"] == 0


def test_three_tier_governance_and_markdown():
    gov = generate_three_tier_governance()
    assert len(gov["tier1_descriptive"]) >= 3
    assert len(gov["tier2_conditional"]) >= 3
    assert len(gov["tier3_inconclusive"]) >= 3

    md = render_full_markdown_report("active")
    assert "# BÁO CÁO TỔNG HỢP KHOA HỌC" in md
    assert "CHƯƠNG 1:" in md
    assert "CHƯƠNG 6:" in md
