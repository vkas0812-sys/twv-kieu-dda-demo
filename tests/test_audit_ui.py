"""Run Streamlit's UI harness against the isolated audit copy."""
import copy
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "apps/web/pages/4_Ton_kho.py"
RUN = ROOT / "data/results/app_runs/20261026T161330-inv-3542775b/results.json"
payload = json.loads(RUN.read_text())


def app(data, selected):
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.session_state["inv_run"] = data
    at.session_state["sku_list"] = selected
    at.run(timeout=60)
    return at


def describe(at):
    tables = [x.value for x in at.dataframe]
    po = next((x for x in tables if "Tổng chai đặt (Bottles)" in x.columns), None)
    return {
        "exceptions": [x.message for x in at.exception],
        "stock_fields": [x.label for x in at.number_input if x.label.startswith("Tồn kho ")],
        "po_rows": [] if po is None else po[["Mã SKU", "Nhu cầu ròng (chai)", "Tổng chai đặt (Bottles)"]].to_dict("records"),
        "po_ready_messages": [x.value for x in at.success if "phát lệnh PO" in x.value],
        "downloads": [x.label for x in at.get("download_button")],
    }


at = app(payload, ["W103028"])
print("SAVED_RUN_SKUS_VS_CURRENT_SELECTION", json.dumps(describe(at), ensure_ascii=False))

failed = copy.deepcopy(payload)
first = copy.deepcopy(next(r for r in payload["rows"] if r["split"] == "train" and r["policy"] == "Kieu-DDA-VP"))
first.update(status="Failed: forced audit scenario", total_y=0, y_star={}, TEC=0, WC=0)
failed["rows"] = [first]
failed["n_sku"] = 1
at_failed = app(failed, [first["sku"]])
print("FAILED_SOLVER_ROW", json.dumps(describe(at_failed), ensure_ascii=False))

empty = copy.deepcopy(payload)
empty["rows"] = [copy.deepcopy(first)]
empty["rows"][0].update(status="Optimal", total_y=0, y_star={"CIW": 0, "Hub_Retail": 0, "Hub_B2B": 0})
at_zero = app(empty, [first["sku"]])
print("ZERO_NEED_UI", json.dumps(describe(at_zero), ensure_ascii=False))

from api.services.procurement import get_channel_weights_for_sku
from api.services.data_store import monthly_qty
hist = monthly_qty("active", ["W100133"], year=2024)
train = hist[hist["month"].between(1, 8)].groupby("Channel")["Quantity"].sum().to_dict()
full_year = hist.groupby("Channel")["Quantity"].sum().to_dict()
weights = get_channel_weights_for_sku("W100133", baseline="active")
print("WEIGHTS_USE_FULL_2024", json.dumps({"weights": weights, "train_only": train, "full_year": full_year}, ensure_ascii=False))
