"""Page 11 — Top-40 SKU Replenishment, Supplier MOQ & 5-Channel Allocation.

Provides executive visibility and interactive analytics for the thesis notebook:
`06_top40_replenishment_moq_channel_allocation.ipynb`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

from api.services.top40_replenishment import (
    CHANNELS,
    FIGURES_METADATA,
    get_figure_path,
    get_top40_kpis,
    load_abc_economic_summary,
    load_assumption_audit,
    load_channel_allocation,
    load_channel_economic_summary,
    load_primary_plan,
    load_sku_economic_metrics,
    load_upper_bound_plan,
    load_validation_table,
)
from bootstrap import gate, guardrail_banner, page_header, pct, vnd

gate("TWV — 11. Bổ sung Top-40 SKU, MOQ & Phân bổ 5 Kênh")

page_header(
    "11. Kế hoạch Bổ sung Top-40 SKU, Ràng buộc MOQ & Phân bổ 5 Kênh Tiêu thụ",
    "Hiện thực hóa Lời giải DDA-VP (L=3 kỳ) về quy cách Thùng nguyên (6 chai/thùng), Ràng buộc MOQ Nhà cung cấp (600 chai/shipment) và Phân bổ theo 5 Kênh Bán hàng (RT, GT, HC, HO, MT).",
    badge="THESIS REPLENISHMENT EXTENSION • TOP-40 PORTFOLIO & 5-CHANNEL ALLOCATION",
)

guardrail_banner(
    "<strong>Nguyên tắc Nghiệp vụ Chuẩn Luận văn:</strong> "
    "1. <strong>Quy cách Thùng nguyên:</strong> 6 chai/thùng, tuyệt đối không xé lẻ trong chuỗi cung ứng nhập khẩu • "
    "2. <strong>Ràng buộc MOQ Nhà cung cấp:</strong> Tối thiểu 600 chai/shipment (100 thùng), phân bổ bù thiếu bằng thuật toán Phần dư lớn nhất (Largest Remainder) • "
    "3. <strong>Phân bổ 5 Kênh Bán hàng:</strong> Dựa trên tỷ trọng dự báo nhu cầu Sep–Nov 2024, bảo toàn 100% nguyên thùng và khớp hoàn toàn với tổng nhập CIW."
)

# Load Key KPIs
try:
    kpis = get_top40_kpis()
except Exception as exc:
    st.error(f"Lỗi khi tải dữ liệu Top-40 Replenishment: {exc}")
    st.stop()

# Metric Cards Ribbon
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Tổng Nhập Kho CIW",
    f"{kpis['final_bottles']:,} chai",
    f"{kpis['final_cases']:,} thùng nguyên",
)
c2.metric(
    "Làm tròn Thùng 6 chai",
    f"+{kpis['case_uplift_bottles']} chai",
    f"+{kpis['case_uplift_pct']:.2f}% tổng lượng",
)
c3.metric(
    "Bù MOQ Nhà cung cấp",
    f"+{kpis['moq_uplift_bottles']} chai",
    "100% SKU đều ≥ 600 chai",
)
c4.metric(
    "Kênh Chủ lực Lớn nhất",
    f"{kpis['largest_channel']} ({kpis['largest_channel_share']:.1%})",
    f"{kpis['largest_channel_bottles']:,} chai",
)
c5.metric(
    "Tổng Vốn Lưu động",
    vnd(kpis["rounded_working_capital_vnd"]),
    "13.737 tỷ VND (Top-40)",
)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 1. Kế hoạch Bổ sung Top-40 & Ràng buộc Rời rạc",
    "🚚 2. Phân bổ Chi tiết 5 Kênh Tiêu thụ",
    "💰 3. 10 Câu hỏi Ra Quyết định Kinh tế & Vốn lưu động",
    "🛡️ 4. Giám định Giả định & Bàn giao Tác nghiệp",
])

# ==============================================================================
# TAB 1: Kế hoạch Bổ sung Top-40 & Ràng buộc Rời rạc
# ==============================================================================
with tab1:
    st.markdown("### 1.1. Tóm lược Điều hành Kế hoạch Nhập khẩu CIW")
    st.markdown(
        f"""
        - Lời giải mô hình **DDA-VP (L=3 kỳ: Tháng 9 – Tháng 11/2024)** xác định mức tồn kho mục tiêu gộp cho toàn mạng lưới là **{kpis['continuous_target']:,.1f} chai**.
        - Quy tắc đóng thùng **6 chai/thùng** làm tròn số lượng đặt hàng lên nguyên thùng, làm tăng thêm **{kpis['case_uplift_bottles']} chai** (+{kpis['case_uplift_pct']:.2f}%), đưa tổng đơn hàng lên **{kpis['final_bottles']:,} chai ({kpis['final_cases']:,} thùng)**.
        - Ràng buộc **MOQ Nhà cung cấp 600 chai/shipment**: Do quy mô nhu cầu 3 tháng của từng SKU trong nhóm Top-40 đều vượt ngưỡng 600 chai, phần bù MOQ trong kịch bản này là **0 chai**.
        - Tập trung hóa sản lượng: **{kpis['n_skus_for_80_volume']} SKU đầu tiên** đã chiếm **80%** tổng sản lượng nhập khẩu. SKU lớn nhất là `{kpis['top_volume_sku']}` ({kpis['top_volume_bottles']:,} chai).
        """
    )

    st.markdown("### 1.2. Bảng Kế hoạch Bổ sung Đặt hàng 40 SKU")
    primary_df = load_primary_plan()

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        abc_filter = st.multiselect(
            "Lọc phân khúc ABC:",
            options=sorted(primary_df["ABC"].unique()),
            default=sorted(primary_df["ABC"].unique()),
        )
    with col_f2:
        search_query = st.text_input("Tìm kiếm theo mã SKU hoặc Tên sản phẩm:", "")

    filtered_df = primary_df[primary_df["ABC"].isin(abc_filter)]
    if search_query:
        mask = filtered_df["sku"].str.contains(search_query, case=False, na=False) | filtered_df["item_name"].str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]

    display_table = filtered_df[[
        "sku", "item_name", "ABC", "model_target_bottles", "order_pre_moq_cases",
        "final_order_cases", "final_order_bottles", "unit_cost_vnd",
        "rounded_working_capital_vnd", "model_fill_rate", "centralization_level"
    ]].copy()

    display_table.columns = [
        "Mã SKU", "Tên Sản phẩm", "ABC", "Mục tiêu DDA-VP (Chai)", "Đơn sơ bộ (Thùng)",
        "Đơn chốt (Thùng)", "Đơn chốt (Chai)", "Đơn giá COGS", "Vốn lưu động (VND)",
        "Fill Rate", "Mức tập trung CL"
    ]

    st.dataframe(
        display_table.style.format({
            "Mục tiêu DDA-VP (Chai)": "{:,.1f}",
            "Đơn sơ bộ (Thùng)": "{:,.0f}",
            "Đơn chốt (Thùng)": "{:,.0f}",
            "Đơn chốt (Chai)": "{:,.0f}",
            "Đơn giá COGS": "{:,.0f}",
            "Vốn lưu động (VND)": "{:,.0f}",
            "Fill Rate": "{:.3f}",
            "Mức tập trung CL": "{:.2f}",
        }),
        use_container_width=True,
        height=380,
    )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Tải Kế hoạch Sơ cấp (Primary Plan CSV)",
            data=primary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="06_top40_replenishment_primary.csv",
            mime="text/csv",
        )
    with col_dl2:
        upper_df = load_upper_bound_plan()
        st.download_button(
            "📥 Tải Kế hoạch Cận trên (Upper Bound Plan CSV)",
            data=upper_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="06_top40_replenishment_upper_bound.csv",
            mime="text/csv",
        )

    st.markdown("### 1.3. Bộ sưu tập Biểu đồ Phân tích Danh mục Top-40 (Figures 1 – 12)")
    view_mode_1 = st.radio(
        "Chế độ hiển thị biểu đồ:",
        ["Duyệt từng biểu đồ chi tiết (Kèm phân tích)", "Xem toàn bộ theo danh sách"],
        horizontal=True,
        key="view_mode_tab1",
    )

    group1_figs = [f for f in FIGURES_METADATA if f["id"] <= 12]

    if view_mode_1 == "Duyệt từng biểu đồ chi tiết (Kèm phân tích)":
        fig_select = st.selectbox(
            "Chọn biểu đồ cần xem:",
            options=group1_figs,
            format_func=lambda x: f"{x['title']} [{x['category']}]",
            key="select_fig_g1",
        )
        if fig_select:
            st.markdown(f"#### {fig_select['title']}")
            st.info(fig_select["description"])
            img_path = get_figure_path(fig_select["filename"])
            if img_path:
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Không tìm thấy tệp ảnh {fig_select['filename']}.")
    else:
        for f in group1_figs:
            with st.expander(f"{f['title']} — {f['category']}", expanded=(f["id"] in [1, 2, 4, 11])):
                st.caption(f["description"])
                img_path = get_figure_path(f["filename"])
                if img_path:
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning(f"Không tìm thấy tệp ảnh {f['filename']}.")

# ==============================================================================
# TAB 2: Phân bổ Chi tiết 5 Kênh Tiêu thụ
# ==============================================================================
with tab2:
    st.markdown("### 2.1. Ma trận Phân bổ 5 Kênh Tiêu thụ (SKU × Channel Matrix)")
    long_alloc, wide_alloc = load_channel_allocation()
    ch_summary = load_channel_economic_summary()

    unit_mode = st.radio(
        "Đơn vị hiển thị trong ma trận:",
        ["Số Chai (Bottles)", "Số Thùng nguyên (Cases - 6 chai/thùng)"],
        horizontal=True,
    )

    if unit_mode == "Số Thùng nguyên (Cases - 6 chai/thùng)":
        matrix_disp = (wide_alloc / 6).astype(int)
    else:
        matrix_disp = wide_alloc.copy()

    # Add row totals and column totals
    matrix_with_totals = matrix_disp.copy()
    matrix_with_totals["Tổng cộng"] = matrix_with_totals.sum(axis=1)
    totals_row = matrix_with_totals.sum(axis=0)
    totals_row.name = "TỔNG TOÀN BỘ"
    matrix_with_totals = pd.concat([matrix_with_totals, pd.DataFrame(totals_row).T])

    st.dataframe(
        matrix_with_totals.style.format("{:,.0f}").background_gradient(cmap="Blues", subset=CHANNELS),
        use_container_width=True,
        height=400,
    )

    st.markdown("### 2.2. Tổng hợp Kinh tế & Cơ cấu 5 Kênh Tiêu thụ")
    st.dataframe(
        ch_summary.rename(columns={
            "channel": "Kênh Bán hàng",
            "allocated_bottles": "Sản lượng Phân bổ (Chai)",
            "allocated_capital_vnd": "Vốn Lưu động (VND)",
            "volume_share": "Tỷ trọng Sản lượng",
            "capital_share": "Tỷ trọng Vốn Lưu động",
        }).style.format({
            "Sản lượng Phân bổ (Chai)": "{:,.0f}",
            "Vốn Lưu động (VND)": "{:,.0f}",
            "Tỷ trọng Sản lượng": "{:.2%}",
            "Tỷ trọng Vốn Lưu động": "{:.2%}",
        }),
        use_container_width=True,
    )

    st.download_button(
        "📥 Tải Bảng Phân bổ Kênh Chi tiết (CSV)",
        data=long_alloc.to_csv(index=False).encode("utf-8-sig"),
        file_name="06_top40_channel_allocation_primary.csv",
        mime="text/csv",
    )

    st.markdown("### 2.3. Chi tiết Phân bổ Từng Kênh (Figures 13 – 17)")
    ch_tabs = st.tabs([
        "🍽️ HC (HoReCa - 52.6%)",
        "🏪 GT (Đại lý - 18.5%)",
        "🍷 RT (Bán lẻ - 12.4%)",
        "🛒 MT (Siêu thị - 12.3%)",
        "🏢 HO (Doanh nghiệp - 4.2%)",
    ])

    ch_map = {
        0: ("HC", "Fig_06_15_channel_HC_allocation.png", "Kênh Nhà hàng - Khách sạn chiếm vị thế chi phối tuyệt đối với 49,206 chai (6.98 tỷ VND)."),
        1: ("GT", "Fig_06_14_channel_GT_allocation.png", "Kênh Đại lý truyền thống tiếp nhận 17,262 chai (2.90 tỷ VND), thiên về các dòng sản phẩm có thương hiệu mạnh."),
        2: ("RT", "Fig_06_13_channel_RT_allocation.png", "Kênh Cửa hàng bán lẻ tiếp nhận 11,604 chai (1.98 tỷ VND), đặc trưng bởi đơn giá cao và tỷ trọng vốn vượt trội."),
        3: ("MT", "Fig_06_17_channel_MT_allocation.png", "Kênh Chuỗi siêu thị hiện đại tiếp nhận 11,532 chai (1.34 tỷ VND), tập trung vào các dòng rượu phổ thông."),
        4: ("HO", "Fig_06_16_channel_HO_allocation.png", "Kênh Khách hàng Doanh nghiệp / Quà tặng B2B tiếp nhận 3,948 chai (534 triệu VND)."),
    }

    for idx, tab_ch in enumerate(ch_tabs):
        with tab_ch:
            ch_code, fig_name, ch_desc = ch_map[idx]
            st.markdown(f"#### Phân bổ SKU cho Kênh {ch_code}")
            st.info(ch_desc)
            img_path = get_figure_path(fig_name)
            if img_path:
                st.image(str(img_path), use_container_width=True)

# ==============================================================================
# TAB 3: 10 Câu hỏi Ra Quyết định Kinh tế & Vốn lưu động
# ==============================================================================
with tab3:
    st.markdown("### 3.1. Phân tích Kinh tế Quản trị Chuỗi Cung ứng")
    st.markdown(
        """
        Toàn bộ kế hoạch bổ sung hàng và phân bổ kênh được kiểm định dưới **10 câu hỏi kinh tế trọng yếu** nhằm giúp Ban Giám đốc và Bộ phận Mua hàng tối ưu hóa dòng tiền, kiểm soát rủi ro tồn kho và đánh giá đúng bản chất chi phí:
        """
    )

    q_options = [f for f in FIGURES_METADATA if f["id"] >= 18]
    selected_q = st.selectbox(
        "Chọn câu hỏi kinh tế để xem phân tích chuyên sâu:",
        options=q_options,
        format_func=lambda x: f"{x['title']}",
    )

    if selected_q:
        st.markdown(f"#### {selected_q['title']}")
        st.markdown(f"**Ý nghĩa Kinh tế & Quản trị:** {selected_q['description']}")
        img_path = get_figure_path(selected_q["filename"])
        if img_path:
            st.image(str(img_path), use_container_width=True)

    st.markdown("### 3.2. Bảng Chỉ số Kinh tế Chi tiết của 40 SKU")
    sku_econ_df = load_sku_economic_metrics()
    st.dataframe(
        sku_econ_df[[
            "sku", "ABC", "unit_cost_vnd", "final_order_bottles", "sku_demand_proxy_bottles",
            "rounded_working_capital_vnd", "coverage_ratio", "case_rounding_capital_vnd",
            "centralization_level", "service_target_beta", "model_fill_rate"
        ]].rename(columns={
            "sku": "Mã SKU",
            "ABC": "ABC",
            "unit_cost_vnd": "Đơn giá COGS",
            "final_order_bottles": "Đơn hàng (Chai)",
            "sku_demand_proxy_bottles": "Nhu cầu Dự báo (Chai)",
            "rounded_working_capital_vnd": "Vốn lưu động (VND)",
            "coverage_ratio": "Tỷ lệ đệm (Coverage)",
            "case_rounding_capital_vnd": "Vốn dôi do thùng 6 chai",
            "centralization_level": "Mức tập trung CL",
            "service_target_beta": "Service Target Beta",
            "model_fill_rate": "Fill Rate Mô hình",
        }).style.format({
            "Đơn giá COGS": "{:,.0f}",
            "Đơn hàng (Chai)": "{:,.0f}",
            "Nhu cầu Dự báo (Chai)": "{:,.1f}",
            "Vốn lưu động (VND)": "{:,.0f}",
            "Tỷ lệ đệm (Coverage)": "{:.2f}×",
            "Vốn dôi do thùng 6 chai": "{:,.0f}",
            "Mức tập trung CL": "{:.2f}",
            "Service Target Beta": "{:.2f}",
            "Fill Rate Mô hình": "{:.3f}",
        }),
        use_container_width=True,
        height=380,
    )

    st.markdown("### 3.3. Phân khúc ABC-XYZ & Cơ cấu Chi phí Mô hình")
    abc_summary = load_abc_economic_summary()
    st.dataframe(
        abc_summary.rename(columns={
            "ABC": "Phân khúc ABC",
            "sku_count": "Số lượng SKU",
            "order_bottles": "Sản lượng Đặt (Chai)",
            "forecast_bottles": "Nhu cầu Dự báo (Chai)",
            "working_capital_vnd": "Vốn Lưu động (VND)",
            "expected_holding_vnd": "Chi phí Giữ hàng E(H)",
            "expected_penalty_vnd": "Chi phí Phạt Thiếu E(P)",
            "expected_fulfill_vnd": "Chi phí Phục vụ E(F)",
            "order_volume_share": "Tỷ trọng Sản lượng",
            "capital_share": "Tỷ trọng Vốn",
        }).style.format({
            "Sản lượng Đặt (Chai)": "{:,.0f}",
            "Nhu cầu Dự báo (Chai)": "{:,.1f}",
            "Vốn Lưu động (VND)": "{:,.0f}",
            "Chi phí Giữ hàng E(H)": "{:,.0f}",
            "Chi phí Phạt Thiếu E(P)": "{:,.0f}",
            "Chi phí Phục vụ E(F)": "{:,.0f}",
            "Tỷ trọng Sản lượng": "{:.2%}",
            "Tỷ trọng Vốn": "{:.2%}",
        }),
        use_container_width=True,
    )

# ==============================================================================
# TAB 4: Giám định Giả định & Bàn giao Tác nghiệp
# ==============================================================================
with tab4:
    st.markdown("### 4.1. Kiểm định Tính Khả thi Nghiệp vụ & Toàn vẹn Dữ liệu (8/8 Checks)")
    val_table = load_validation_table()

    # Format check results
    val_display = val_table.copy()
    val_display["Trạng thái"] = val_display["result"].apply(lambda x: "✅ ĐẠT CHUẨN" if x else "❌ KHÔNG ĐẠT")
    val_display = val_display.rename(columns={"check": "Tiêu chí Kiểm toán Nghiệp vụ"})[["Tiêu chí Kiểm toán Nghiệp vụ", "Trạng thái"]]

    st.table(val_display)

    st.markdown("### 4.2. Bảng Kiểm toán Tham số & Giả định Mô hình")
    audit_table = load_assumption_audit()
    st.dataframe(
        audit_table.rename(columns={
            "parameter": "Tham số Giả định",
            "value": "Giá trị Áp dụng",
            "status": "Căn cứ Kiểm toán",
        }),
        use_container_width=True,
    )

    st.markdown("### 4.3. Khung Hướng dẫn Bàn giao Tác nghiệp Phát hành Đơn hàng Mua (PO)")
    st.markdown(
        r"""
        | Kịch bản Tác nghiệp | Hiện trạng Mô hình | Hành động Tác nghiệp Cần thực hiện |
        | :--- | :--- | :--- |
        | **Kịch bản 1: Kế hoạch Định mức Gộp (Gross Target)** | Dùng trực tiếp mục tiêu DDA-VP ($y^*$) | Dành cho lập kế hoạch ngân sách và đàm phán phân bổ hạn ngạch nhập khẩu dài hạn. |
        | **Kịch bản 2: Tích hợp Tồn kho (Net Requirement)** | Trừ Tồn hiện có ($OH$) và Đơn đang về ($IT$) | Khi cung cấp tệp `inventory_snapshot.csv`, app tự động chuyển sang $NR = \max(0, y^* - OH - IT)$. |
        | **Kịch bản 3: Ràng buộc Nhà cung cấp (Supplier MOQ)** | Đơn sơ cấp đang ở mức Unmapped | Khi nạp tệp `supplier_map.csv`, hệ thống sẽ nhóm các SKU cùng nhà cung cấp và áp chuẩn 600 chai/shipment. |
        """
    )

    st.markdown("### 4.4. Trung tâm Tải Trọn bộ Dữ liệu Kết quả")
    c_d1, c_d2, c_d3 = st.columns(3)
    with c_d1:
        st.download_button(
            "📥 Kế hoạch Đặt hàng Top-40 (Primary Plan)",
            data=load_primary_plan().to_csv(index=False).encode("utf-8-sig"),
            file_name="06_top40_replenishment_primary.csv",
            mime="text/csv",
        )
    with c_d2:
        st.download_button(
            "📥 Ma trận Phân bổ 5 Kênh Tiêu thụ",
            data=load_channel_allocation()[0].to_csv(index=False).encode("utf-8-sig"),
            file_name="06_top40_channel_allocation_primary.csv",
            mime="text/csv",
        )
    with c_d3:
        st.download_button(
            "📥 Bảng Chỉ số Kinh tế & Chi phí SKU",
            data=load_sku_economic_metrics().to_csv(index=False).encode("utf-8-sig"),
            file_name="06_top40_sku_economic_metrics.csv",
            mime="text/csv",
        )
