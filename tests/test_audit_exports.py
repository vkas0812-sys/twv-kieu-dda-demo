"""Verify download bytes and PO arithmetic through the isolated Streamlit page."""
import io
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "data/results/app_runs/20261026T161330-inv-3542775b/results.json").read_text())
downloads = []
original = st.download_button


def capture(label, *args, **kwargs):
    data = kwargs.get("data", args[0] if args else None)
    if isinstance(data, bytes) and kwargs.get("file_name", "").startswith("TWV_"):
        frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
        downloads.append({"file": kwargs["file_name"], "has_bom": data.startswith(b"\xef\xbb\xbf"), "rows": len(frame), "columns": list(frame.columns)})
    return original(label, *args, **kwargs)


at = AppTest.from_file(ROOT / "apps/web/pages/4_Ton_kho.py", default_timeout=60)
at.session_state["inv_run"] = payload
with patch.object(st, "download_button", side_effect=capture):
    at.run(timeout=60)
assert not at.exception, [x.message for x in at.exception]
assert len(downloads) == 2
assert all(x["has_bom"] for x in downloads)
po = next(x.value for x in at.dataframe if "Thành tiền PO (VND)" in x.value.columns)
calculated = po["Tổng chai đặt (Bottles)"] * po["Đơn giá COGS (VND)"]
assert (calculated == po["Thành tiền PO (VND)"]).all()
assert all(int(v) % 6 == 0 for v in po["Tổng chai đặt (Bottles)"])
print(json.dumps({"downloads": downloads, "financial_checks": "PASS", "po_six_bottle_checks": "PASS"}, ensure_ascii=False))
