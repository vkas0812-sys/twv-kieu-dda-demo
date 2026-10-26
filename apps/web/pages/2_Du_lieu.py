from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

from api.services.data_store import available_baselines, save_upload, sku_counts
from api.services.paths import CHANNELS, S4
from api.services.portfolio import build_portfolio
from bootstrap import BASELINE_LABEL, caption, format_sku, gate, guardrail_banner, page_header

gate("TWV — 2. Dữ liệu & Danh mục ABC–XYZ")

page_header(
    "2. Nền tảng Dữ liệu ERP & Phân khúc Danh mục ABC–XYZ (M1 & M2)",
    "Chuẩn hóa lưới nhu cầu theo tuần ISO năm 2024, phân loại ma trận giá trị – biến động ABC–XYZ và nhận diện kiểu nhu cầu theo Syntetos–Boylan.",
    badge="MODULE 02 • DATA FOUNDATION & PORTFOLIO SEGMENTATION",
)

guardrail_banner(
    "<strong>Nguyên tắc làm sạch dữ liệu (Bước 02):</strong> Tập <code>preclean</code> đã tách riêng Top 3 khách hàng lớn nhất và 10 khách hàng bán buôn "
    "<em>B2B Whales</em> để loại bỏ nhiễu đơn hàng dự án đột biến; tập <code>active</code> lọc tiếp các SKU đang trong vòng đời hoạt động ổn định."
)

col_base, col_up = st.columns([1.2, 1.0])
with col_base:
    baselines = available_baselines()
    cur_base = st.session_state.get("baseline", "preclean")
    if cur_base not in baselines:
        cur_base = "preclean"
    baseline = st.radio(
        "Chọn tập dữ liệu Baseline:",
        options=baselines,
        index=baselines.index(cur_base),
        format_func=lambda b: BASELINE_LABEL.get(b, b),
    )
    st.session_state["baseline"] = baseline

with col_up:
    upload = st.file_uploader(
        "Tải lên file Parquet bán hàng mới (lưu riêng thành sales_upload.parquet, không ghi đè file luận văn)",
        type=["parquet"],
    )
    if upload is not None and st.button("💾 Lưu và kích hoạt tập dữ liệu Upload", type="primary"):
        path = save_upload(upload.getvalue(), "sales_upload.parquet")
        st.session_state["baseline"] = "upload"
        st.session_state.pop("portfolio", None)
        st.success(f"Đã lưu `{path.name}` và chuyển sang chế độ `upload`.")
        st.rerun()

try:
    info = sku_counts(baseline)
except Exception as exc:
    st.error(f"Không đọc được tập dữ liệu `{baseline}`: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tổng dòng dữ liệu", f"{info['n_rows']:,}", f"Dòng 2024: {info['n_rows_2024']:,}")
m2.metric("Tổng SKU năm 2024", f"{info['n_sku_2024']:,}", f"Toàn bộ lịch sử: {info['n_sku_all']:,}")
m3.metric("Tổng sản lượng 2024", f"{info['total_qty_2024']:,.0f} chai", f"Kênh: {', '.join(info['channels_2024'])}")
m4.metric("Kiểm tra 4 mã chuẩn S4", "4/4 Hợp lệ" if all(info["s4_present"].values()) else "Thiếu mã S4", Path(info["path"]).name)

st.divider()
st.subheader("🎯 Bộ lọc Phân khúc Danh mục ABC–XYZ & Syntetos–Boylan")

f1, f2, f3 = st.columns([1.5, 1.0, 0.8])
with f1:
    channels = st.multiselect("Kênh phân phối đưa vào phân tích", CHANNELS, default=CHANNELS)
with f2:
    top_n = st.slider("Số lượng SKU hiển thị (Top-N theo sản lượng 2024 + S4)", min_value=4, max_value=40, value=20, step=4)
with f3:
    st.write("")
    run_btn = st.button("🚀 Chạy Phân loại ABC–XYZ", type="primary", use_container_width=True)

cache_key = f"{baseline}_{','.join(sorted(channels))}_{top_n}"
if run_btn or st.session_state.get("_port_key") != cache_key:
    if channels:
        with st.spinner("Đang tổng hợp lưới tuần ISO 2024 và tính chỉ số ADI, CV², ABC–XYZ…"):
            port = build_portfolio(baseline, 2024, channels, top_n)
            st.session_state["portfolio"] = port
            st.session_state["_port_key"] = cache_key

port = st.session_state.get("portfolio")
if not port or port.get("n_series", 0) == 0:
    st.info("Hãy chọn ít nhất 1 kênh phân phối và bấm **Chạy Phân loại ABC–XYZ**.")
    st.stop()

rows = pd.DataFrame(port["rows"])
scatter = pd.DataFrame(port["scatter"])

tab_chart, tab_table, tab_basket = st.tabs([
    "📈 Biểu đồ Syntetos–Boylan & Ma trận ABC–XYZ",
    "📋 Bảng Chi tiết Chuỗi SKU × Kênh",
    "🛒 Chọn Giỏ SKU mang sang Lab Dự báo & Tồn kho",
])

with tab_chart:
    c_left, c_right = st.columns([1.35, 1.0])
    with c_left:
        st.markdown(r"#### Phân bố Kiểu nhu cầu theo Syntetos–Boylan ($\text{CV}^2$ vs $\text{ADI}$)")
        if len(scatter):
            sc_df = scatter.dropna(subset=["cv2", "adi"]).copy()
            # Clip extreme values for visual clarity while keeping real values in tooltip
            sc_df["cv2_plot"] = sc_df["cv2"].clip(upper=4.0)
            sc_df["adi_plot"] = sc_df["adi"].clip(upper=6.0)
            pts = (
                alt.Chart(sc_df)
                .mark_circle(size=95, opacity=0.85, stroke="#ffffff", strokeWidth=1)
                .encode(
                    x=alt.X("cv2_plot:Q", title="Hệ số biến thiên bình phương CV² (đã giới hạn trục ở 4.0)"),
                    y=alt.Y("adi_plot:Q", title="Khoảng cách nhu cầu trung bình ADI (tuần)"),
                    color=alt.Color("ABC_XYZ:N", title="Lớp ABC–XYZ"),
                    tooltip=[
                        alt.Tooltip("Item number:N", title="Mã SKU"),
                        alt.Tooltip("item_name:N", title="Tên rượu"),
                        alt.Tooltip("Channel:N", title="Kênh"),
                        alt.Tooltip("ABC_XYZ:N", title="Lớp ABC–XYZ"),
                        alt.Tooltip("Demand_Category:N", title="Kiểu nhu cầu"),
                        alt.Tooltip("sum:Q", title="Sản lượng 2024 (chai)", format=",.0f"),
                        alt.Tooltip("adi:Q", title="ADI thực tế", format=".2f"),
                        alt.Tooltip("cv2:Q", title="CV² thực tế", format=".2f"),
                        alt.Tooltip("beta:Q", title="Critical Ratio mục tiêu (β)", format=".2f"),
                    ],
                )
            )
            hline = alt.Chart(pd.DataFrame({"y": [1.32]})).mark_rule(strokeDash=[5, 5], color="#ef4444", strokeWidth=1.5).encode(y="y:Q")
            vline = alt.Chart(pd.DataFrame({"x": [0.49]})).mark_rule(strokeDash=[5, 5], color="#ef4444", strokeWidth=1.5).encode(x="x:Q")
            st.altair_chart((pts + hline + vline).properties(height=380), use_container_width=True)
            st.caption(r"Đường đứt nét đỏ: Ngưỡng chuẩn Syntetos–Boylan ($\text{ADI} = 1{,}32$; $\text{CV}^2 = 0{,}49$) chia 4 vùng: *Smooth, Erratic, Intermittent, Lumpy*.")

    with c_right:
        st.markdown("#### Thống kê Số chuỗi SKU × Kênh theo Lớp `ABC–XYZ`")
        if len(rows):
            grp = (
                rows.groupby(["ABC_XYZ", "Demand_Category"], as_index=False)
                .agg(so_chuoi=("Item number", "count"), tong_san_luong=("sum", "sum"))
            )
            bar_abc = (
                alt.Chart(grp)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("ABC_XYZ:N", title="Phân lớp ABC–XYZ"),
                    y=alt.Y("so_chuoi:Q", title="Số lượng chuỗi (SKU × Kênh)"),
                    color=alt.Color("Demand_Category:N", title="Kiểu nhu cầu"),
                    tooltip=[
                        alt.Tooltip("ABC_XYZ:N", title="Lớp"),
                        alt.Tooltip("Demand_Category:N", title="Kiểu nhu cầu"),
                        alt.Tooltip("so_chuoi:Q", title="Số chuỗi"),
                        alt.Tooltip("tong_san_luong:Q", title="Tổng sản lượng (chai)", format=",.0f"),
                    ],
                )
                .properties(height=380)
            )
            st.altair_chart(bar_abc, use_container_width=True)

with tab_table:
    disp_cols = [
        c for c in ["Item number", "item_name", "Channel", "ABC_XYZ", "Demand_Category", "sum", "mean", "std", "cv2", "adi", "beta", "unit_cost"]
        if c in rows.columns
    ]
    st.dataframe(
        rows[disp_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Item number": "Mã SKU",
            "item_name": "Tên sản phẩm rượu",
            "Channel": "Kênh",
            "ABC_XYZ": "Lớp ABC–XYZ",
            "Demand_Category": "Kiểu nhu cầu",
            "sum": st.column_config.NumberColumn("Tổng lượng 2024 (chai)", format="%,.0f"),
            "mean": st.column_config.NumberColumn("TB tuần (chai)", format="%.2f"),
            "std": st.column_config.NumberColumn("Độ lệch chuẩn", format="%.2f"),
            "cv2": st.column_config.NumberColumn("CV²", format="%.2f"),
            "adi": st.column_config.NumberColumn("ADI (tuần)", format="%.2f"),
            "beta": st.column_config.NumberColumn("Mục tiêu β", format="%.2f"),
            "unit_cost": st.column_config.NumberColumn("Giá vốn COGS (VND)", format="%,.0f"),
        },
    )

with tab_basket:
    st.markdown("#### Chọn danh sách SKU đưa sang **Lab Dự báo (Trang 3)** và **Lab Tồn kho (Trang 4)**")
    rank = pd.DataFrame(port["sku_rank"])
    options = rank["sku"].astype(str).tolist()
    cur_pick = [s for s in (st.session_state.get("sku_list") or S4) if s in options]
    if not cur_pick:
        cur_pick = [s for s in S4 if s in options] or options[:4]
    pick = st.multiselect(
        "Giỏ SKU nghiên cứu (Tối đa 40 mã):",
        options=options,
        default=cur_pick,
        format_func=lambda s: format_sku(s, baseline),
    )
    if pick:
        st.session_state["sku_list"] = pick
        st.success(f"Đã lưu **{len(pick)} mã SKU** vào giỏ phiên làm việc hiện tại.")
    st.dataframe(
        rank[rank["sku"].isin(pick)],
        use_container_width=True,
        hide_index=True,
        column_config={
            "sku": "Mã SKU",
            "item_name": "Tên sản phẩm rượu",
            "qty": st.column_config.NumberColumn("Tổng sản lượng năm 2024 (chai)", format="%,.0f"),
        },
    )

caption("tuần ISO 2024", "Phân khúc danh mục M2 (chưa tạo Ω tồn kho)", None, "2024-08", int(rows["Item number"].nunique()) if len(rows) else 0)
