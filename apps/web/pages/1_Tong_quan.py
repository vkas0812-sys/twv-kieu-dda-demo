from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

from api.services.data_store import sku_abc_map, sku_counts, sku_name
from api.services.paths import CHANNEL_LABELS, S4, S4_CLASS
from api.services.replay import replay_frame
from api.services.runs import list_runs
from bootstrap import POLICY_COLOR, POLICY_LABEL, caption, gate, guardrail_banner, page_header, pct, vnd

gate("TWV — 1. Tổng quan & Đối chứng 03a/03c")

page_header(
    "1. Tổng quan Mạng lưới Dual-Hub & Kết quả Đối chứng Luận văn (03a / 03c)",
    "Kiến trúc phân phối 3 nút – 5 kênh (10 cung hợp lệ), giao thức kiểm định Out-of-Sample Protocol A và kết quả đối chứng RQ1 & RQ2.",
    badge="MODULE 01 • EXECUTIVE OVERVIEW & THESIS REPLAY",
)

guardrail_banner(
    "<strong>Giao thức kiểm định Protocol A:</strong> Huấn luyện tìm định mức tồn kho tối ưu <code>y*</code> trên tháng 1–8/2024 (Cutoff <code>2024-08</code>), "
    "khóa cứng <code>y*</code> (và ma trận gán kênh đối với <code>FP</code>), sau đó chấm điểm chi phí thực tế trên tập kiểm định độc lập tháng 9–12/2024."
)

baseline = st.session_state.get("baseline", "preclean")
try:
    counts = sku_counts(baseline)
except Exception as exc:
    counts = None
    st.warning(f"Chưa đọc được dữ liệu Parquet ({baseline}): {exc}")

runs = list_runs()
c1, c2, c3, c4 = st.columns(4)
if counts:
    c1.metric("SKU hoạt động năm 2024", f"{counts['n_sku_2024']:,}", f"Baseline: {baseline}")
    missing = [s for s, ok in counts["s4_present"].items() if not ok]
    c2.metric("Bộ mẫu chuẩn S4 (4 lớp)", "Đầy đủ 4/4 SKU" if not missing else "Thiếu " + ",".join(missing), "W100133, W103028, W131012, W100627")
    c3.metric("Kênh phân phối", f"{len(counts['channels_2024'])} kênh", ", ".join(counts["channels_2024"]))
c4.metric("Phiên chạy gần nhất", runs[0]["run_id"][:18] if runs else "Chưa có run mới", f"Tổng cộng {len(runs)} runs")

with st.expander("🕸️ Xem chi tiết Kiến trúc Mạng lưới Dual-Hub & Thông số Chi phí chuẩn (Wang & Minner Adapted)", expanded=True):
    col_net, col_cost = st.columns([1.15, 1.0])
    with col_net:
        st.markdown("#### Cấu trúc 10 cung phân phối hợp lệ (`feasible_arcs`)")
        st.code(
            "Kho Nhập Trung Tâm (CIW) ──► RT, HO, HC, GT, MT  (5 cung trực tiếp — cước 5.000đ/chai)\n"
            "Kho Đệm Bán Lẻ (Hub_Retail) ► RT, HO              (2 cung chuyên biệt — cước 2.000đ/chai)\n"
            "Kho Đệm Bán Buôn (Hub_B2B) ─► HC, GT, MT          (3 cung chuyên biệt — cước 2.500đ/chai)",
            language="text",
        )
        st.caption(
            "• **Kênh nhu cầu:** "
            + " | ".join(f"`{k}`: {v}" for k, v in CHANNEL_LABELS.items())
        )
    with col_cost:
        st.markdown("#### Thông số chi phí mỗi kỳ & quy tắc phạt thiếu hàng")
        cost_spec = pd.DataFrame([
            {"Nút lưu kho (Location)": "CIW (Kho nhập trung tâm)", "Chi phí giữ tồn / kỳ (h_l)": "2,0% × COGS", "Cước xuất hàng (c_lr)": "5.000 VND / chai", "Kênh phục vụ": "Tất cả 5 kênh (RT, HO, HC, GT, MT)"},
            {"Nút lưu kho (Location)": "Hub_Retail (Kho đệm B2C)", "Chi phí giữ tồn / kỳ (h_l)": "2,5% × COGS", "Cước xuất hàng (c_lr)": "2.000 VND / chai", "Kênh phục vụ": "RT, HO"},
            {"Nút lưu kho (Location)": "Hub_B2B (Kho đệm B2B)", "Chi phí giữ tồn / kỳ (h_l)": "2,3% × COGS", "Cước xuất hàng (c_lr)": "2.500 VND / chai", "Kênh phục vụ": "HC, GT, MT"},
        ])
        st.dataframe(cost_spec, use_container_width=True, hide_index=True)
        st.caption("Chi phí phạt thiếu hàng: $p = \\frac{\\beta}{1-\\beta} \\cdot h_{\\text{CIW}}$, trong đó $\\beta$ là *Critical Ratio* theo lớp ABC–XYZ.")

st.divider()
st.subheader("📊 Replay 03a (RQ1) — Đối chứng 3 Chính sách Tồn kho trên Tập Kiểm định OOS (Tháng 9–12/2024)")

scope_col, metric_col = st.columns([1.2, 1.2])
scope_03a = scope_col.radio(
    "Phạm vi danh mục hiển thị (03a):",
    ["Bộ mẫu chuẩn S4 (4 SKU đại diện)", "Toàn bộ mẫu Top 40 SKU trong file 03a"],
    horizontal=True,
)
metric_03a = metric_col.radio(
    "Chỉ số biểu đồ so sánh chính:",
    ["TEC (Tổng chi phí kỳ vọng - VND)", "cost_per_unit (Chi phí / chai - VND)", "fill_rate (Tỷ lệ đáp ứng đơn hàng)"],
    horizontal=True,
)
metric_key = metric_03a.split(" ")[0]

try:
    oos = replay_frame("03a", "03a_align_oos.csv").copy()
    abc_map = sku_abc_map(baseline)
    oos["ABC"] = oos["sku"].map(lambda s: S4_CLASS.get(s) or abc_map.get(s, "—"))
    oos["item_name"] = oos["sku"].map(sku_name)
    oos["TEC_ex_fulfill"] = oos["TEC"] - oos["E_fulfill"].fillna(0.0)
    oos["policy_label"] = oos["policy"].map(lambda p: POLICY_LABEL.get(p, p))

    view_03a = oos[oos["sku"].isin(S4)].copy() if scope_03a.startswith("Bộ mẫu") else oos.copy()

    # Summary KPI cards across selected view
    piv_tec = view_03a.pivot_table(index="sku", columns="policy", values="TEC", aggfunc="mean")
    piv_cpu = view_03a.pivot_table(index="sku", columns="policy", values="cost_per_unit", aggfunc="mean")
    piv_fr = view_03a.pivot_table(index="sku", columns="policy", values="fill_rate", aggfunc="mean")

    m1, m2, m3, m4 = st.columns(4)
    if {"Kieu-DDA-VP", "Decentralized"}.issubset(piv_tec.columns):
        vp_wins = int((piv_tec["Kieu-DDA-VP"] <= piv_tec["Decentralized"]).sum())
        med_saving = float(((piv_tec["Decentralized"] - piv_tec["Kieu-DDA-VP"]) / piv_tec["Decentralized"]).median())
        m1.metric("Tỷ lệ SKU VP thắng Dec (OOS)", f"{vp_wins}/{len(piv_tec)} SKU", f"Trung vị giảm TEC: {pct(med_saving)}")
    if "Kieu-DDA-VP" in piv_cpu.columns:
        m2.metric("Trung vị Cost/Unit (VP)", vnd(piv_cpu["Kieu-DDA-VP"].median()), "Kieu-DDA-VP (Virtual Pooling)")
    if "Kieu-DDA-FP" in piv_cpu.columns:
        m3.metric("Trung vị Cost/Unit (FP)", vnd(piv_cpu["Kieu-DDA-FP"].median()), "Kieu-DDA-FP (Fixed Pooling)")
    if "Kieu-DDA-VP" in piv_fr.columns:
        m4.metric("Trung vị Fill Rate OOS (VP)", pct(piv_fr["Kieu-DDA-VP"].median()), "Type-II Service Level thực tế")

    chart_df = view_03a.copy()
    chart_df["sku_display"] = chart_df["sku"] + " (" + chart_df["ABC"] + ")"
    color_scale = alt.Scale(
        domain=["Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized"],
        range=[POLICY_COLOR["Kieu-DDA-VP"], POLICY_COLOR["Kieu-DDA-FP"], POLICY_COLOR["Decentralized"]],
    )
    bar = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("sku_display:N", title="Mã SKU (Lớp ABC–XYZ)", sort=None),
            xOffset=alt.XOffset("policy:N", sort=["Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized"]),
            y=alt.Y(f"{metric_key}:Q", title=metric_03a),
            color=alt.Color("policy:N", title="Chính sách", scale=color_scale),
            tooltip=[
                alt.Tooltip("sku:N", title="Mã SKU"),
                alt.Tooltip("item_name:N", title="Tên rượu"),
                alt.Tooltip("ABC:N", title="Lớp ABC–XYZ"),
                alt.Tooltip("policy:N", title="Chính sách"),
                alt.Tooltip("TEC:Q", title="TEC (VND)", format=",.0f"),
                alt.Tooltip("cost_per_unit:Q", title="Cost/Unit (VND)", format=",.0f"),
                alt.Tooltip("fill_rate:Q", title="Fill Rate", format=".1%"),
                alt.Tooltip("WC:Q", title="Vốn lưu động WC (VND)", format=",.0f"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(bar, use_container_width=True)

    disp_03a = view_03a[
        ["sku", "item_name", "ABC", "policy", "TEC", "TEC_ex_fulfill", "cost_per_unit", "fill_rate", "E_holding", "E_penalty", "E_fulfill", "WC"]
    ].copy()
    st.dataframe(
        disp_03a,
        use_container_width=True,
        hide_index=True,
        column_config={
            "sku": "Mã SKU",
            "item_name": "Tên sản phẩm",
            "ABC": "Lớp",
            "policy": "Chính sách",
            "TEC": st.column_config.NumberColumn("TEC (VND)", format="%,.0f"),
            "TEC_ex_fulfill": st.column_config.NumberColumn("TEC trừ cước (VND)", format="%,.0f"),
            "cost_per_unit": st.column_config.NumberColumn("Cost/Unit (VND/chai)", format="%,.0f"),
            "fill_rate": st.column_config.NumberColumn("Fill Rate", format="%.3f"),
            "E_holding": st.column_config.NumberColumn("CP Lưu kho (VND)", format="%,.0f"),
            "E_penalty": st.column_config.NumberColumn("CP Phạt thiếu (VND)", format="%,.0f"),
            "E_fulfill": st.column_config.NumberColumn("CP Vận chuyển (VND)", format="%,.0f"),
            "WC": st.column_config.NumberColumn("Vốn lưu động WC (VND)", format="%,.0f"),
        },
    )
    caption("một tháng, OOS 9–12", "hist (03a)", None, "2024-08", int(view_03a["sku"].nunique()))
except Exception as exc:
    st.info(f"Chưa đọc được 03a_align_oos.csv ({exc})")

st.divider()
st.subheader("🔬 Replay 03c (RQ2) — Giá trị Kinh tế của Dự báo (SAA-History vs SAA-Forecast Joint)")

scope_03c = st.radio(
    "Phạm vi danh mục hiển thị (03c):",
    ["Bộ mẫu chuẩn S4 (4 SKU đại diện)", "Toàn bộ SKU trong file 03c"],
    horizontal=True,
    key="_scope_03c",
)

try:
    cmp = replay_frame("03c", "03c_hist_vs_fcst.csv").copy()
    cmp["item_name"] = cmp["sku"].map(sku_name)
    view_03c = cmp[cmp["sku"].isin(S4)].copy() if scope_03c.startswith("Bộ mẫu") else cmp.copy()

    # Melt for side-by-side comparison chart of cpu_hist vs cpu_fcst
    melt_cpu = view_03c.melt(
        id_vars=["sku", "item_name", "ABC", "policy"],
        value_vars=["cpu_hist", "cpu_fcst"],
        var_name="Nhánh kịch bản Ω",
        value_name="Cost per Unit (VND/chai)",
    )
    melt_cpu["Nhánh kịch bản Ω"] = melt_cpu["Nhánh kịch bản Ω"].map({
        "cpu_hist": "1. SAA từ Lịch sử (cpu_hist)",
        "cpu_fcst": "2. SAA từ Dự báo + Joint Residual (cpu_fcst)",
    })
    melt_cpu["sku_policy"] = melt_cpu["sku"] + " • " + melt_cpu["policy"].str.replace("Kieu-DDA-", "")

    chart_03c = (
        alt.Chart(melt_cpu)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("sku_policy:N", title="SKU • Chính sách", sort=None),
            xOffset=alt.XOffset("Nhánh kịch bản Ω:N"),
            y=alt.Y("Cost per Unit (VND/chai):Q", title="Chi phí OOS mỗi chai (VND/chai)"),
            color=alt.Color(
                "Nhánh kịch bản Ω:N",
                scale=alt.Scale(range=["#64748b", "#0072B2"]),
            ),
            tooltip=[
                alt.Tooltip("sku:N", title="SKU"),
                alt.Tooltip("item_name:N", title="Tên rượu"),
                alt.Tooltip("ABC:N", title="Lớp"),
                alt.Tooltip("policy:N", title="Chính sách"),
                alt.Tooltip("Nhánh kịch bản Ω:N", title="Nguồn Ω"),
                alt.Tooltip("Cost per Unit (VND/chai):Q", format=",.0f"),
            ],
        )
        .properties(height=340)
    )
    st.altair_chart(chart_03c, use_container_width=True)

    cols_03c = [
        c for c in ["sku", "item_name", "ABC", "policy", "cpu_hist", "cpu_fcst", "cpu_delta", "cpu_delta_pct", "FR_hist", "FR_fcst", "rank_change"]
        if c in view_03c.columns
    ]
    st.dataframe(
        view_03c[cols_03c],
        use_container_width=True,
        hide_index=True,
        column_config={
            "sku": "Mã SKU",
            "item_name": "Tên sản phẩm",
            "ABC": "Lớp",
            "policy": "Chính sách",
            "cpu_hist": st.column_config.NumberColumn("Cost/Unit Lịch sử (VND)", format="%,.0f"),
            "cpu_fcst": st.column_config.NumberColumn("Cost/Unit Dự báo (VND)", format="%,.0f"),
            "cpu_delta": st.column_config.NumberColumn("Δ Cost/Unit (VND)", format="%+,.0f"),
            "cpu_delta_pct": st.column_config.NumberColumn("Δ Cost/Unit (%)", format="%+.2f%%"),
            "FR_hist": st.column_config.NumberColumn("Fill Rate (Hist)", format="%.3f"),
            "FR_fcst": st.column_config.NumberColumn("Fill Rate (Fcst)", format="%.3f"),
            "rank_change": "Thay đổi hạng VP vs Dec",
        },
    )
    caption("một tháng (L=1)", "hist so với fcst_joint (03c)", 1, "2024-08", int(view_03c["sku"].nunique()))
except Exception as exc:
    st.info(f"Chưa đọc được 03c_hist_vs_fcst.csv ({exc})")
