from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

from api.services.data_store import sku_name
from api.services.paths import S4
from api.services.replay import replay_frame
from bootstrap import POLICY_COLOR, caption, gate, guardrail_banner, page_header

gate("TWV — 5. Lead Time Phủ Cửa sổ L (RQ3)")

page_header(
    "5. Phân tích Độ trễ Nhập khẩu Đường biển — Cửa sổ Phủ L ∈ {3, 4, 5, 6} Tháng (RQ3)",
    "Đánh giá sự thay đổi của định mức tồn kho y*, Tổng chi phí kỳ vọng (TEC), Cost per Unit và Vốn lưu động (WC) khi thời gian bảo vệ kéo dài từ 3 đến 6 tháng.",
    badge="MODULE 05 • IMPORT LEAD-TIME COVERAGE (WAVE 03b & 03d)",
)

guardrail_banner(
    "<strong>Nguyên tắc tách biệt độ phân giải (Grain):</strong> Mỗi giá trị <code>L ∈ {3, 4, 5, 6}</code> đại diện cho tổng nhu cầu tích lũy trượt trên cửa sổ <code>L</code> tháng liên tiếp. "
    "Tuyệt đối <strong>không cộng hoặc lấy trung bình cộng TEC giữa các mức L khác nhau</strong>. Với tập kiểm định OOS tháng 9–12/2024 (4 tháng), "
    "chỉ có <code>L = 3</code> (2 cửa sổ: T9–11, T10–12) và <code>L = 4</code> (1 cửa sổ: T9–12) đủ độ dài kiểm định; với <code>L = 5, 6</code> hệ thống báo <code>no window</code> minh bạch."
)

c1, c2 = st.columns([1.4, 1.0])
with c1:
    src = st.radio(
        "Chọn nguồn dữ liệu Replay:",
        ["03b — Kịch bản Lịch sử tích lũy trượt (03b_leadtime_L.csv)", "03d — So sánh Lịch sử vs Dự báo trên cửa sổ L (03d_hist_vs_fcst.csv)"],
        horizontal=False,
    )
    src_code = "03b" if src.startswith("03b") else "03d"
with c2:
    only_s4 = st.radio(
        "Phạm vi SKU:",
        ["Chỉ 4 mã đại diện chuẩn S4", "Toàn bộ SKU trong file kết quả"],
        horizontal=False,
    ).startswith("Chỉ")

try:
    df = replay_frame(src_code, "03b_leadtime_L.csv" if src_code == "03b" else "03d_hist_vs_fcst.csv").copy()
    if "sku" in df.columns:
        df["item_name"] = df["sku"].map(sku_name)
except Exception as exc:
    st.error(f"Không đọc được file Replay `{src_code}`: {exc}")
    st.stop()

if only_s4 and "sku" in df.columns:
    df = df[df["sku"].isin(S4)].copy()

# Overview comparison chart between L=3 and L=4
if src_code == "03b" and "split" in df.columns:
    oos_all = df[df["split"].astype(str).str.contains("test")].copy()
    if len(oos_all):
        st.subheader("📊 Biểu đồ So sánh Chi phí OOS mỗi Chai (`cost_per_unit`) giữa $L = 3$ và $L = 4$ Tháng")
        oos_all["L_label"] = "L = " + oos_all["L"].astype(int).astype(str) + " tháng"
        chart_l = (
            alt.Chart(oos_all)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("sku:N", title="Mã SKU"),
                xOffset=alt.XOffset("policy:N", sort=["Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized"]),
                y=alt.Y("cost_per_unit:Q", title="Chi phí OOS mỗi chai (VND/chai)"),
                color=alt.Color(
                    "policy:N",
                    title="Chính sách",
                    scale=alt.Scale(
                        domain=["Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized"],
                        range=[POLICY_COLOR["Kieu-DDA-VP"], POLICY_COLOR["Kieu-DDA-FP"], POLICY_COLOR["Decentralized"]],
                    ),
                ),
                column=alt.Column("L_label:N", title="Độ bao phủ Lead Time (Tháng)"),
                tooltip=[
                    alt.Tooltip("L:Q", title="L (tháng)"),
                    alt.Tooltip("sku:N", title="SKU"),
                    alt.Tooltip("item_name:N", title="Tên rượu"),
                    alt.Tooltip("ABC:N", title="Lớp"),
                    alt.Tooltip("policy:N", title="Chính sách"),
                    alt.Tooltip("TEC:Q", title="TEC cửa sổ L (VND)", format=",.0f"),
                    alt.Tooltip("cost_per_unit:Q", title="Cost/Unit (VND/chai)", format=",.0f"),
                    alt.Tooltip("fill_rate:Q", title="Fill Rate", format=".1%"),
                    alt.Tooltip("WC:Q", title="Vốn lưu động WC (VND)", format=",.0f"),
                ],
            )
            .properties(height=320, width=380)
        )
        st.altair_chart(chart_l)

present = set(int(x) for x in df["L"].dropna().unique()) if "L" in df.columns else set()

st.subheader(r"📑 Chi tiết Kết quả theo từng Cửa sổ Độ trễ $L \in \{3, 4, 5, 6\}$")
tabs_l = st.tabs(["L = 3 tháng (13 tuần)", "L = 4 tháng (17 tuần)", "L = 5 tháng (22 tuần)", "L = 6 tháng (26 tuần)"])

for idx, L in enumerate((3, 4, 5, 6)):
    with tabs_l[idx]:
        if L not in present:
            if L == 6:
                st.warning(
                    "⚠️ **no window** — Với `L = 6` tháng trên tập kiểm định tháng 9–12/2024 (chỉ có 4 tháng), không tồn tại cửa sổ trượt 6 tháng đầy đủ. "
                    "Hệ thống tuân thủ nguyên tắc minh bạch học thuật: **không ngoại suy hay điền số giả**."
                )
            else:
                st.info(
                    f"ℹ️ **no window** — Tập kiểm định tháng 9–12/2024 (4 tháng) không đủ độ dài cho cửa sổ trượt `L = {L}` tháng. "
                    "Không gộp trung bình với các mức L khác."
                )
            caption("cửa sổ L tháng", src_code, L, "2024-08", 0)
            continue

        block = df[df["L"] == L].copy()
        if src_code == "03b" and "split" in block.columns:
            oos = block[block["split"].astype(str).str.contains("test")].copy()
            if L == 6 and oos.empty:
                st.warning("⚠️ **no window** — `L = 6` trên tháng 9–12 không có cửa sổ OOS. Không điền số.")
            else:
                cols_show = [
                    c for c in ["L", "sku", "item_name", "ABC", "beta", "n_omega", "policy", "TEC", "cost_per_unit", "fill_rate", "WC", "E_holding", "E_penalty", "E_fulfill"]
                    if c in oos.columns
                ]
                st.dataframe(
                    oos[cols_show],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "TEC": st.column_config.NumberColumn("TEC cửa sổ L (VND)", format="%,.0f"),
                        "cost_per_unit": st.column_config.NumberColumn("Cost/Unit (VND/chai)", format="%,.0f"),
                        "fill_rate": st.column_config.NumberColumn("Fill Rate", format="%.3f"),
                        "WC": st.column_config.NumberColumn("Vốn lưu động WC (VND)", format="%,.0f"),
                        "E_holding": st.column_config.NumberColumn("CP Lưu kho (VND)", format="%,.0f"),
                        "E_penalty": st.column_config.NumberColumn("CP Phạt thiếu (VND)", format="%,.0f"),
                        "E_fulfill": st.column_config.NumberColumn("CP Vận chuyển (VND)", format="%,.0f"),
                    },
                )
        else:
            cols_03d = [
                c for c in ["L", "sku", "item_name", "ABC", "policy", "cpu_hist", "cpu_fcst", "cpu_delta", "cpu_delta_pct", "FR_hist", "FR_fcst", "rank_change"]
                if c in block.columns
            ]
            st.dataframe(
                block[cols_03d],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "cpu_hist": st.column_config.NumberColumn("Cost/Unit Lịch sử (VND)", format="%,.0f"),
                    "cpu_fcst": st.column_config.NumberColumn("Cost/Unit Dự báo (VND)", format="%,.0f"),
                    "cpu_delta": st.column_config.NumberColumn("Δ Cost/Unit (VND)", format="%+,.0f"),
                    "cpu_delta_pct": st.column_config.NumberColumn("Δ Cost/Unit (%)", format="%+.2f%%"),
                    "FR_hist": st.column_config.NumberColumn("Fill Rate (Hist)", format="%.3f"),
                    "FR_fcst": st.column_config.NumberColumn("Fill Rate (Fcst)", format="%.3f"),
                },
            )
        caption(
            "cửa sổ L tháng" if src_code == "03b" else "cửa sổ L tháng, hist so với fcst",
            src_code,
            L,
            "2024-08",
            int(block["sku"].nunique()) if "sku" in block else 0,
        )
