"""Independent audit checks; kept outside the user's checkout."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps")]
from api.services import procurement as p


@pytest.fixture(autouse=True)
def deterministic_master(monkeypatch):
    monkeypatch.setattr(p, "sku_abc_map", lambda baseline: {})
    monkeypatch.setattr(p, "sku_name", lambda sku: sku)
    monkeypatch.setattr(p, "unit_cost", lambda sku: 100_000.0)
    monkeypatch.setattr(p, "get_channel_weights_for_sku", lambda *a, **k: {"RT": 1.0})


def row(sku="A", target=60.0, abc="AX", **extras):
    return {"sku": sku, "total_y": target, "ABC": abc, "policy": "Kieu-DDA-VP", **extras}


@pytest.mark.parametrize("strategy", ["velocity", "proportional"])
def test_zero_demand_must_not_create_a_shipment(strategy):
    plan = p.calculate_procurement_plan([row(target=0)], topup_strategy=strategy)
    assert plan["summary"]["final_bottles"] == 0


@pytest.mark.parametrize("strategy", ["velocity", "proportional"])
def test_fully_covered_stock_must_not_create_a_shipment(strategy):
    plan = p.calculate_procurement_plan([row(target=60)], current_stocks={"A": 600}, topup_strategy=strategy)
    assert plan["items"][0]["net_requirement"] == 0
    assert plan["summary"]["final_bottles"] == 0


@pytest.mark.parametrize("strategy", ["velocity", "proportional"])
def test_moq_topup_must_not_buy_an_already_overstocked_sku(strategy):
    plan = p.calculate_procurement_plan(
        [row("A", 600, "AX"), row("B", 6, "BY")],
        current_stocks={"A": 1000, "B": 0}, topup_strategy=strategy,
    )
    items = {it["sku"]: it for it in plan["items"]}
    assert items["A"]["final_bottles"] == 0


def test_moq_is_separate_for_different_suppliers():
    plan = p.calculate_procurement_plan(
        [row("A", 300, supplier_id="SUP_A"), row("B", 300, supplier_id="SUP_B")],
        topup_strategy="none",
    )
    assert plan["summary"]["moq_met"] is False


@pytest.mark.parametrize("pack", [0, -6, 1, 5, 6.5])
def test_invalid_six_bottle_pack_is_rejected(pack):
    with pytest.raises(ValueError):
        p.calculate_procurement_plan([row(target=7)], pack_size=pack, moq=0)


def test_negative_weights_cannot_increase_the_allocated_total():
    alloc = p.allocate_cartons_to_channels(10, {"RT": 2, "HO": -1})
    assert sum(alloc.values()) == 10


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError):
        p.calculate_procurement_plan([row()], topup_strategy="veloctiy")


def test_missing_total_y_must_not_silently_become_zero():
    plan = p.calculate_procurement_plan(
        [{"sku": "A", "y_star": {"CIW": 120, "Hub_Retail": 180, "Hub_B2B": 300}}],
        topup_strategy="none",
    )
    assert plan["items"][0]["base_bottles"] == 600


def test_duplicate_sku_rows_must_be_rejected():
    with pytest.raises(ValueError):
        p.calculate_procurement_plan([row(), row()], topup_strategy="proportional")


@pytest.mark.parametrize("strategy", ["velocity", "proportional", "none"])
def test_orders_already_above_moq_are_not_topped_up(strategy):
    plan = p.calculate_procurement_plan([row(target=601)], topup_strategy=strategy)
    assert plan["summary"]["final_bottles"] == 606
    assert plan["summary"]["topup_cartons"] == 0


def test_non_case_multiple_moq_rounds_to_next_case():
    plan = p.calculate_procurement_plan([row(target=6)], moq=601)
    assert plan["summary"]["final_bottles"] == 606


def test_none_reports_the_exact_shortfall():
    plan = p.calculate_procurement_plan([row(target=70)], topup_strategy="none")
    assert plan["summary"]["base_bottles"] == 72
    assert plan["summary"]["deficit_bottles"] == 528
    assert plan["summary"]["moq_met"] is False


def test_empty_input_does_not_crash():
    plan = p.calculate_procurement_plan([])
    assert plan["summary"]["final_bottles"] == 0


def test_zero_moq_and_financial_conservation():
    plan = p.calculate_procurement_plan([row(target=7)], moq=0)
    assert plan["summary"]["final_bottles"] == 12
    assert plan["summary"]["total_po_amount_vnd"] == 1_200_000
    assert sum(plan["summary"]["channel_totals_bottles"].values()) == 12


def test_all_zero_weights_and_fractional_weights_conserve_cartons():
    for total in range(101):
        for weights in [{}, {"RT": 0.1, "GT": 0.2, "HC": 0.7}]:
            alloc = p.allocate_cartons_to_channels(total, weights)
            assert sum(alloc.values()) == total
            assert all(isinstance(v, int) and v >= 0 for v in alloc.values())
