"""Streamlit AppTest for Page 11: Top-40 Replenishment, MOQ & Channel Allocation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps"), str(ROOT / "apps" / "web")]

import pytest
from streamlit.testing.v1 import AppTest


def test_top40_page_apptest():
    page_path = ROOT / "apps" / "web" / "pages" / "11_Top40_Bo_sung_MOQ_Kenh.py"
    at = AppTest.from_file(str(page_path), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"
    # Verify metric cards are present
    assert len(at.metric) >= 5
    # Verify tables or dataframes are present
    assert len(at.dataframe) >= 2 or len(at.table) >= 1
