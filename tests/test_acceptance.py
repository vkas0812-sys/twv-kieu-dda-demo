"""Acceptance checks for the TWV planning workbench."""
from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps"), str(ROOT / "apps" / "web")]

from kieu_pipeline.inventory import dual_hub_arcs
from kieu_pipeline.scenarios import saa_from_history_coverage

from api.main import app
from api.services.data_store import load_product_catalog, sku_abc_map, sku_name, top_skus_2024
from api.services.forecast_lab import load_forecast_tables, run_forecast
from api.services.inventory_lab import class_of, solve_protocol
from api.services.paths import FIGURES_DIR
from api.services.publish import publish_run
from api.services.replay import replay_frame


def test_dual_hub_arc_count() -> None:
    arcs = dual_hub_arcs()
    assert arcs == {
        ("CIW", "RT"), ("CIW", "GT"), ("CIW", "HC"), ("CIW", "HO"), ("CIW", "MT"),
        ("Hub_Retail", "RT"), ("Hub_Retail", "HO"),
        ("Hub_B2B", "HC"), ("Hub_B2B", "GT"), ("Hub_B2B", "MT"),
    }
    assert len(arcs) == 10


def test_s4_hist_solve_finite() -> None:
    out = solve_protocol(
        sku_list=["W100133", "W103028", "W131012", "W100627"],
        omega_type="hist",
        policies=["Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized"],
        evaluate_oos=True,
    )
    train = [r for r in out["rows"] if r["split"] == "train"]
    assert len(train) == 12
    for row in train:
        assert row["status"] == "Optimal"
        assert math.isfinite(float(row["TEC"]))
        assert row["cost_per_unit"] is not None and math.isfinite(float(row["cost_per_unit"]))
        assert row["item_name"]
        assert row["y_star"]
        assert all(math.isfinite(float(v)) for v in row["y_star"].values())
    oos = [r for r in out["rows"] if r["split"] == "oos"]
    assert len(oos) == 12
    publish = publish_run(out["run_id"])
    assert publish["n"] == 12
    skus = {r["sku"] for r in publish["rows"]}
    policies = {r["policy"] for r in publish["rows"]}
    assert skus == {"W100133", "W103028", "W131012", "W100627"}
    assert policies == {"Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized"}


def test_l6_oos_empty_and_api() -> None:
    monthly = pd.DataFrame({
        "Item number": ["W100133"] * 4,
        "Channel": ["RT"] * 4,
        "month": [9, 10, 11, 12],
        "Quantity": [1, 1, 1, 1],
    })
    try:
        saa_from_history_coverage(monthly, L=6, month_lo=9, month_hi=12)
        raised = False
    except ValueError:
        raised = True
    assert raised
    client = TestClient(app)
    res = client.post("/inventory/evaluate", json={"omega_type": "hist_coverage", "L": 6})
    assert res.status_code == 200
    body = res.json()
    assert body["oos_available"] is False


def test_replay_wape_rows() -> None:
    df = replay_frame("03c", "03c_wape_sku.csv")
    assert len(df) > 0


def test_replay_03a_and_03c_dashboard_files() -> None:
    assert len(replay_frame("03a", "03a_align_oos.csv")) > 0
    assert len(replay_frame("03c", "03c_hist_vs_fcst.csv")) > 0


def test_catalog_and_non_s4_abc_lookup() -> None:
    cat = load_product_catalog()
    assert "W100133" in cat.index
    assert "Allan Scott" in sku_name("W100133")
    top40 = top_skus_2024("preclean", 40)
    assert len(top40) >= 40
    abc_map = sku_abc_map("preclean")
    non_s4 = next(s for s in top40 if s not in {"W100133", "W103028", "W131012", "W100627"})
    cls = class_of(non_s4, baseline="preclean")
    assert len(cls) == 2 and cls in abc_map.values()


def test_streamlit_pages_syntax_and_figures() -> None:
    web_root = ROOT / "apps" / "web"
    page_files = sorted(p for p in (web_root / "pages").glob("*.py") if not p.name.startswith("._"))
    py_files = [web_root / "bootstrap.py", web_root / "streamlit_app.py"] + page_files
    assert len(py_files) == 10
    for f in py_files:
        source = f.read_text(encoding="utf-8")
        ast.parse(source, filename=str(f))
    assert FIGURES_DIR.exists()


def test_forecast_and_fcst_joint_pipeline() -> None:
    fc_out = run_forecast(["W100133", "W103028", "W131012", "W100627"], baseline="active", cutoff="2024-08")
    assert fc_out["run_id"]
    assert fc_out["n"] > 0
    mu, resid = load_forecast_tables(fc_out["run_id"], None)
    assert len(mu) == 20
    inv_out = solve_protocol(
        sku_list=["W100133", "W103028", "W131012", "W100627"],
        omega_type="fcst_joint",
        baseline="active",
        mu=mu,
        residuals=resid,
        evaluate_oos=True,
    )
    assert inv_out["oos_available"] is True
    assert len([r for r in inv_out["rows"] if r["split"] == "oos"]) == 12
