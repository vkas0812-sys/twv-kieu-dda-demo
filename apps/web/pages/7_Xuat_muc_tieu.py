from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

from api.services.data_store import sku_name, unit_cost
from api.services.paths import THESIS_RESULT
from api.services.publish import publish_run
from api.services.runs import list_runs
from bootstrap import gate, guardrail_banner, page_header, pct, vnd

gate("TWV — 7. Xuất Định mức Tồn kho Mục tiêu (M7)")

page_header(
    "7. Xuất Bảng Định mức Tồn kho Mục tiêu & Kế hoạch Vốn Lưu động (M7 & M8)",
    "Kết xuất định mức Base-stock y* tại Kho Nhập Trung Tâm (CIW), Hub Bán Lẻ (Hub_Retail), Hub Bán Buôn (Hub_B2B) ra chuẩn CSV phục vụ vận hành.",
    badge="MODULE 07 • BASE-STOCK TARGET PUBLISHER & GOVERNANCE",
)

guardrail_banner(
    "<strong>Nguyên tắc quản trị dữ liệu:</strong> Mỗi lần xuất tạo ra file <code>published_planner.csv</code> và <code>published_base_stock_targets.csv</code> "
    "trong thư mục riêng của phiên chạy (<code>data/results/app_runs/&lt;run_id&gt;/</code>), tuyệt đối không ghi đè các file kết quả chuẩn <code>03a–03e</code> của luận văn."
)

inv_runs = list_runs(kind="inventory")

tab_live, tab_thesis = st.tabs([
    "📤 1. Xuất từ Phiên Giải Tồn kho Live (`app_runs`)",
    "🏛️ 2. Xem Định mức Tồn kho Chuẩn Luận văn (`published_base_stock_targets.csv`)",
])

with tab_live:
    if not inv_runs:
        st.info("Chưa có phiên giải Tồn kho Live nào trong `app_runs`. Hãy chạy **Trang 4: Lab Tồn kho** trước, hoặc xem tab **Định mức Tồn kho Chuẩn Luận văn** bên cạnh.")
    else:
        c_sel, c_btn = st.columns([2.2, 1.0])
        with c_sel:
            run_id = st.selectbox(
                "Chọn phiên giải Tồn kho (`run_id`) để xuất định mức:",
                options=[r["run_id"] for r in inv_runs],
                format_func=lambda rid: f"{rid}  •  Ω={next((r.get('omega_type') for r in inv_runs if r['run_id']==rid), '')}  •  {len(next((r.get('sku_list', []) for r in inv_runs if r['run_id']==rid), []))} SKU",
            )
        with c_btn:
            st.write("")
            do_pub = st.button("🚀 Xuất Bảng Định mức CSV", type="primary", use_container_width=True)

        # Auto-publish or load existing published_planner.csv for the selected run
        pub_data = None
        if do_pub:
            try:
                pub_data = publish_run(run_id)
                st.success(f"Đã xuất thành công `{pub_data['path']}` ({pub_data['n']} dòng Optimal).")
            except ValueError as exc:
                st.error(str(exc))
        else:
            try:
                pub_data = publish_run(run_id)
            except Exception:
                pub_data = None

        if pub_data and pub_data.get("rows"):
            df = pd.DataFrame(pub_data["rows"]).copy()
            df["item_name"] = df["sku"].map(sku_name)
            df["unit_cost"] = df["sku"].map(unit_cost)
            df["total_y_hub"] = df["y_CIW"] + df["y_Hub_Retail"] + df["y_Hub_B2B"]

            vp_df = df[df["policy"] == "Kieu-DDA-VP"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Số dòng định mức Optimal", f"{len(df)} dòng", f"{df['sku'].nunique()} SKU × {df['policy'].nunique()} chính sách")
            if len(vp_df):
                k2.metric("Tổng Base-Stock VP (chai)", f"{vp_df['total_y_hub'].sum():,.0f} chai", f"CIW: {vp_df['y_CIW'].sum():,.0f} chai")
                k3.metric("Tổng Vốn lưu động WC (VP)", vnd(vp_df["WC"].sum()), f"TEC kỳ vọng: {vnd(vp_df['TEC'].sum())}")
                k4.metric("Trung vị Fill Rate (VP)", pct(vp_df["fill_rate"].median()), "Chính sách Kieu-DDA-VP")

            # Visual allocation chart for VP & FP
            dual_df = df[df["policy"] != "Decentralized"]
            if len(dual_df):
                melt_y = dual_df.melt(
                    id_vars=["sku", "item_name", "policy"],
                    value_vars=["y_CIW", "y_Hub_Retail", "y_Hub_B2B"],
                    var_name="Nút lưu kho",
                    value_name="Base-stock y* (chai)",
                )
                melt_y["Nút lưu kho"] = melt_y["Nút lưu kho"].map({
                    "y_CIW": "1. Kho Nhập Trung Tâm (CIW)",
                    "y_Hub_Retail": "2. Kho Đệm Bán Lẻ (Hub_Retail)",
                    "y_Hub_B2B": "3. Kho Đệm Bán Buôn (Hub_B2B)",
                })
                chart_pub = (
                    alt.Chart(melt_y)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("sku:N", title="Mã SKU"),
                        xOffset=alt.XOffset("policy:N"),
                        y=alt.Y("Base-stock y* (chai):Q", title="Định mức tồn kho mục tiêu y* (chai)"),
                        color=alt.Color("Nút lưu kho:N", scale=alt.Scale(range=["#0072B2", "#E69F00", "#009E73"])),
                        tooltip=["sku:N", "item_name:N", "policy:N", "Nút lưu kho:N", alt.Tooltip("Base-stock y* (chai):Q", format=",.1f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_pub, use_container_width=True)

            disp_cols = [
                "sku", "item_name", "location", "policy",
                "y_CIW", "y_Hub_Retail", "y_Hub_B2B", "total_y_hub",
                "beta", "fill_rate", "TEC", "WC", "unit_cost", "published_at", "run_id",
            ]
            st.dataframe(
                df[[c for c in disp_cols if c in df.columns]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "sku": "Mã SKU",
                    "item_name": "Tên sản phẩm rượu",
                    "location": "Kiến trúc mạng",
                    "policy": "Chính sách",
                    "y_CIW": st.column_config.NumberColumn("y* CIW (chai)", format="%,.1f"),
                    "y_Hub_Retail": st.column_config.NumberColumn("y* Hub_Retail (chai)", format="%,.1f"),
                    "y_Hub_B2B": st.column_config.NumberColumn("y* Hub_B2B (chai)", format="%,.1f"),
                    "total_y_hub": st.column_config.NumberColumn("Tổng y* Dual-Hub", format="%,.1f"),
                    "beta": st.column_config.NumberColumn("β đã dùng", format="%.2f"),
                    "fill_rate": st.column_config.NumberColumn("Fill Rate", format="%.3f"),
                    "TEC": st.column_config.NumberColumn("TEC (VND)", format="%,.0f"),
                    "WC": st.column_config.NumberColumn("Vốn lưu động WC (VND)", format="%,.0f"),
                    "unit_cost": st.column_config.NumberColumn("COGS / chai (VND)", format="%,.0f"),
                },
            )
            st.download_button(
                "📥 Tải file published_planner.csv",
                pd.DataFrame(pub_data["rows"]).to_csv(index=False),
                file_name=f"{run_id}_published_planner.csv",
                mime="text/csv",
                type="primary",
            )
            st.caption(f"Đã ghi tại `{pub_data['path']}`. Các file kết quả luận văn `03a–03e` được bảo toàn nguyên vẹn.")

with tab_thesis:
    from api.services.paths import RESULTS
    thesis_pub = RESULTS / "published_base_stock_targets.csv"
    if not thesis_pub.exists():
        thesis_pub = THESIS_RESULT / "published_base_stock_targets.csv"
    if thesis_pub.exists():
        tdf = pd.read_csv(thesis_pub)
        if "sku" in tdf.columns:
            tdf["item_name"] = tdf["sku"].map(sku_name)
        st.markdown("#### Bảng Định mức Tồn kho Xuất bản trong Luận văn (`published_base_stock_targets.csv` — Top 40 SKU)")
        st.dataframe(tdf, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Tải bản sao published_base_stock_targets.csv",
            tdf.to_csv(index=False),
            file_name="published_base_stock_targets.csv",
            mime="text/csv",
        )
    else:
        st.info("Không tìm thấy file `published_base_stock_targets.csv` trong thư mục kết quả luận văn.")
