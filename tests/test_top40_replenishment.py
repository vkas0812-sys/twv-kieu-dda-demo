"""Unit tests for Top-40 SKU Replenishment & Channel Allocation service."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps")]

import pytest
import pandas as pd
from api.services.top40_replenishment import (
    load_primary_plan,
    load_upper_bound_plan,
    load_channel_allocation,
    load_channel_economic_summary,
    load_abc_economic_summary,
    load_sku_economic_metrics,
    load_assumption_audit,
    load_validation_table,
    get_top40_kpis,
    get_figure_path,
    FIGURES_METADATA,
    CHANNELS,
)


def test_load_primary_plan():
    df = load_primary_plan()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40
    assert "sku" in df.columns
    assert "final_order_bottles" in df.columns
    assert "final_order_cases" in df.columns
    # Check that all quantities are multiple of 6
    assert (df["final_order_bottles"] % 6 == 0).all()
    # Check that all quantities are >= 600
    assert (df["final_order_bottles"] >= 600).all()


def test_load_upper_bound_plan():
    df = load_upper_bound_plan()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40
    assert "sku" in df.columns


def test_load_channel_allocation():
    long_df, wide_df = load_channel_allocation()
    assert isinstance(long_df, pd.DataFrame)
    assert isinstance(wide_df, pd.DataFrame)
    assert len(long_df) == 40 * 5  # 40 SKUs x 5 channels
    assert list(wide_df.columns) == CHANNELS
    assert len(wide_df) == 40
    # Every channel allocation must be whole cases
    assert (long_df["allocated_bottles"] % 6 == 0).all()


def test_economic_summaries():
    ch_df = load_channel_economic_summary()
    assert len(ch_df) == 5
    assert set(ch_df["channel"]) == set(CHANNELS)
    assert ch_df["volume_share"].sum() == pytest.approx(1.0, 1e-4)

    abc_df = load_abc_economic_summary()
    assert len(abc_df) > 0
    assert "ABC" in abc_df.columns
    assert abc_df["order_volume_share"].sum() == pytest.approx(1.0, 1e-4)

    sku_econ = load_sku_economic_metrics()
    assert len(sku_econ) == 40


def test_assumption_audit_and_validation():
    audit_df = load_assumption_audit()
    assert len(audit_df) >= 7

    val_df = load_validation_table()
    assert len(val_df) == 8
    # All validation checks must pass
    assert val_df["result"].all()


def test_get_top40_kpis():
    kpis = get_top40_kpis()
    assert kpis["n_skus"] == 40
    assert kpis["final_bottles"] == 93552
    assert kpis["final_cases"] == 15592
    assert kpis["largest_channel"] == "HC"
    assert kpis["all_skus_clear_600"] is True
    assert kpis["rounded_working_capital_vnd"] > 1e10


def test_figures_metadata_and_paths():
    assert len(FIGURES_METADATA) == 27
    for fig_meta in FIGURES_METADATA:
        path = get_figure_path(fig_meta["filename"])
        assert path is not None, f"Figure path missing: {fig_meta['filename']}"
        assert path.exists(), f"Figure file does not exist: {path}"
