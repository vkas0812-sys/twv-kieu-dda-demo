from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

from api.services.data_store import sku_name
from api.services.replay import replay_frame
from api.services.sensitivity import live_jackknife, live_mc, live_sa, replay_jackknife, replay_mc, replay_sa
from bootstrap import caption, format_sku, gate, guardrail_banner, page_header, sku_options

gate("TWV — 6. Độ nhạy & Kiểm định Độ bền (03e)")

page_header(
    "6. Phân tích Độ nhạy Tham số, Đường cong Pareto Vốn & Kiểm định Độ bền (03e)",
    "Kiểm chứng độ vững chắc (Robustness) của chính sách Kieu-DDA qua quét tham số (h_mult, beta_mult), giới hạn ngân sách vốn lưu động (Pareto B), Monte Carlo và Jackknife.",
    badge="MODULE 06 • SENSITIVITY ANALYSIS & ROBUSTNESS SUITE",
)

guardrail_banner(
    "<strong>Nguyên tắc giải thích:</strong> (1) Cột <code>beta_used</code> là tỷ lệ tới hạn mục tiêu (Critical Ratio) đưa vào hàm phạt, không phải ràng buộc cứng CSL; "
    "(2) Thí nghiệm <strong>Monte Carlo Gaussian i.i.d.</strong> là bài kiểm chứng đối조 (Ablation) để chỉ ra sai lệch khi bỏ qua tương quan kênh so với <strong>Joint Residual Bootstrap</strong>."
)

baseline = st.session_state.get("baseline", "preclean")
opts = sku_options(baseline)

tab_sa, tab_pareto, tab_mc, tab_jk = st.tabs([
    "🎛️ 1. Độ nhạy Tham số (h_mult & beta_mult)",
    "💹 2. Đường cong Pareto Ngân sách Vốn (B_mult)",
    "🎲 3. Monte Carlo Ablation (Joint vs i.i.d.)",
    "🔄 4. Kiểm định Jackknife (Leave-One-Month-Out)",
])

with tab_sa:
    st.subheader("Replay 03e — Quét Tham số Chi phí Giữ tồn (`h_mult`) & Mức Phục vụ (`beta_mult`)")
    try:
        payload = replay_sa()
        sa_df = pd.DataFrame(payload["rows"])
        sa_df["item_name"] = sa_df["sku"].map(sku_name)

        c_sa1, c_sa2 = st.columns([1.2, 1.2])
        sku_sa = c_sa1.selectbox("Lọc theo SKU trong 03e:", sorted(sa_df["sku"].unique()), format_func=lambda s: format_sku(s, baseline), key="_sa_sku")
        sub_sa = sa_df[sa_df["sku"] == sku_sa].copy()
        sub_sa["param_combo"] = "h×" + sub_sa["h_mult"].astype(str) + " | β×" + sub_sa["beta_mult"].astype(str)

        chart_sa = (
            alt.Chart(sub_sa)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("param_combo:N", title="Tổ hợp hệ số (h_mult | beta_mult)", sort=None),
                xOffset=alt.XOffset("policy:N"),
                y=alt.Y("cost_per_unit:Q", title="Chi phí OOS mỗi chai (VND/chai)"),
                color=alt.Color("policy:N", title="Chính sách"),
                tooltip=[
                    "sku:N", "policy:N", "h_mult:Q", "beta_mult:Q",
                    alt.Tooltip("beta_used:Q", format=".3f"),
                    alt.Tooltip("cost_per_unit:Q", format=",.0f"),
                    alt.Tooltip("fill_rate:Q", format=".1%"),
                    alt.Tooltip("CL_stock:Q" if "CL_stock" in sub_sa.columns else "total_y:Q", format=".2f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(chart_sa, use_container_width=True)
        st.dataframe(sa_df, use_container_width=True, hide_index=True)
        caption("cửa sổ L trong file 03e", "fcst_joint (cột fit_omega)", "xem cột L", "2024-08", int(sa_df["sku"].nunique()))
    except Exception as exc:
        st.info(f"Thiếu 03e_sa_param.csv ({exc})")

    st.divider()
    st.markdown("#### ⚡ Chạy Quét Độ nhạy Live (One-Factor-at-a-Time trên chính sách VP)")
    live_sa_sku = st.selectbox("Chọn SKU chạy Live SA:", opts, index=0, format_func=lambda s: format_sku(s, baseline), key="_live_sa_sku")
    if st.button("🚀 Chạy Live SA (Quét h_mult & beta_mult)", type="primary"):
        with st.spinner(f"Đang giải VP qua các mức h_mult và beta_mult cho {live_sa_sku}…"):
            live = live_sa([live_sa_sku], baseline=baseline)
        st.dataframe(pd.DataFrame(live["rows"]), use_container_width=True, hide_index=True)
        st.caption(live["note"])

with tab_pareto:
    st.subheader("Replay 03a — Đường cong Đánh đổi Pareto giữa Ngân sách Vốn Lưu động ($B$) và Mức Phục vụ (`Fill Rate`)")
    try:
        par_df = replay_frame("03a", "03a_align_pareto.csv").copy()
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            p_chart1 = (
                alt.Chart(par_df)
                .mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=2.5, color="#0072B2")
                .encode(
                    x=alt.X("B_mult:Q", title="Hệ số ngân sách vốn lưu động (B_mult × WC_VP tối ưu)"),
                    y=alt.Y("fill_rate:Q", title="Mức phục vụ đạt được (Fill Rate)", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("B_mult:Q", title="B_mult"),
                        alt.Tooltip("B:Q", title="Trần ngân sách B (VND)", format=",.0f"),
                        alt.Tooltip("WC:Q", title="Vốn lưu động thực dùng (VND)", format=",.0f"),
                        alt.Tooltip("fill_rate:Q", title="Fill Rate", format=".2%"),
                        alt.Tooltip("CL_stock:Q", title="Tỷ trọng tồn tại CIW (CL_stock)", format=".1%"),
                    ],
                )
                .properties(height=320, title="Đường cong Pareto: Ngân sách Vốn (B_mult) → Fill Rate")
            )
            st.altair_chart(p_chart1, use_container_width=True)
        with c_p2:
            p_chart2 = (
                alt.Chart(par_df)
                .mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=2.5, color="#E69F00")
                .encode(
                    x=alt.X("B_mult:Q", title="Hệ số ngân sách vốn lưu động (B_mult)"),
                    y=alt.Y("CL_stock:Q", title="Mức độ tập trung tồn kho tại CIW (CL_stock)"),
                    tooltip=[
                        alt.Tooltip("B_mult:Q", title="B_mult"),
                        alt.Tooltip("CL_stock:Q", title="CL_stock", format=".1%"),
                        alt.Tooltip("SA:Q", title="Shipment Allocation (SA)", format=".1%"),
                        alt.Tooltip("TEC:Q", title="TEC (VND)", format=",.0f"),
                    ],
                )
                .properties(height=320, title="Dịch chuyển Tồn kho về Kho Tổng CIW khi Thắt chặt Vốn")
            )
            st.altair_chart(p_chart2, use_container_width=True)
        st.dataframe(par_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(f"Không đọc được 03a_align_pareto.csv: {exc}")

with tab_mc:
    st.subheader("Replay 03e — Đối chứng Monte Carlo: Joint Residual Bootstrap vs Gaussian Độc lập (i.i.d.)")
    st.checkbox("Ablation Study — Dùng để chứng minh giá trị của bảo toàn tương quan chéo giữa các kênh (không phải KPI chính)", value=True, disabled=True)
    try:
        payload = replay_mc()
        mc_df = pd.DataFrame(payload["rows"])
        mc_df["item_name"] = mc_df["sku"].map(sku_name)
        mc_chart = (
            alt.Chart(mc_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("sku:N", title="Mã SKU"),
                xOffset=alt.XOffset("omega:N"),
                y=alt.Y("cost_per_unit:Q", title="Chi phí OOS mỗi chai (VND/chai)"),
                color=alt.Color("omega:N", title="Cơ chế sinh mẫu Ω"),
                column=alt.Column("policy:N", title="Chính sách"),
                tooltip=["sku:N", "item_name:N", "policy:N", "omega:N", "n_omega:Q", alt.Tooltip("cost_per_unit:Q", format=",.0f"), alt.Tooltip("fill_rate:Q", format=".1%")],
            )
            .properties(height=300, width=260)
        )
        st.altair_chart(mc_chart)
        st.dataframe(mc_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(str(exc))

    st.divider()
    mc_sku = st.selectbox("Chọn SKU chạy Live Monte Carlo Ablation:", opts, index=0, format_func=lambda s: format_sku(s, baseline), key="_live_mc_sku")
    if st.button("🎲 Chạy Live Monte Carlo i.i.d. vs Joint (Seed 42)", type="primary"):
        live = live_mc(mc_sku, baseline=baseline)
        st.warning(live["note"])
        st.write(f"`beta_used = {live['beta_used']:.3f}`")
        st.dataframe(pd.DataFrame(live["rows"]), use_container_width=True, hide_index=True)

with tab_jk:
    st.subheader("Replay 03e — Kiểm định Độ ổn định Jackknife (Leave-One-Period-Out)")
    try:
        payload = replay_jackknife()
        jk_df = pd.DataFrame(payload["rows"])
        jk_df["item_name"] = jk_df["sku"].map(sku_name)
        jk_chart = (
            alt.Chart(jk_df)
            .mark_boxplot(extent="min-max", size=40)
            .encode(
                x=alt.X("sku:N", title="Mã SKU"),
                y=alt.Y("cost_per_unit:Q", title="Phân phối Cost/Unit OOS qua các lần bỏ 1 kỳ (VND/chai)", scale=alt.Scale(zero=False)),
                color=alt.Color("policy:N", title="Chính sách"),
            )
            .properties(height=320)
        )
        st.altair_chart(jk_chart, use_container_width=True)
        st.dataframe(jk_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(str(exc))

    st.divider()
    jk_sku = st.selectbox("Chọn SKU chạy Live Jackknife (bỏ lần lượt từng tháng 1..8):", opts, index=0, format_func=lambda s: format_sku(s, baseline), key="_live_jk_sku")
    if st.button("🔄 Chạy Live Jackknife (Leave-One-Month-Out)", type="primary"):
        live = live_jackknife(jk_sku, baseline=baseline)
        st.write(f"`beta_used = {live['beta_used']:.3f}`")
        st.dataframe(pd.DataFrame(live["rows"]), use_container_width=True, hide_index=True)
