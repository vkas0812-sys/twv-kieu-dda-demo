from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

import importlib
import api.services.eoq as _eoq_mod
importlib.reload(_eoq_mod)
from api.services.data_store import monthly_qty, sku_name, unit_cost
from api.services.eoq import calculate_eoq
from api.services.paths import S4
from bootstrap import (
    caption,
    format_sku,
    gate,
    guardrail_banner,
    page_header,
    sku_options,
    vnd,
)

gate("TWV — 9. Phân tích Cỡ lô Đặt hàng Kinh tế (EOQ)")

page_header(
    "9. Phân tích Cỡ lô Đặt hàng Kinh tế (Economic Order Quantity — EOQ)",
    "Đánh giá điểm cân bằng chi phí đặt hàng (Ordering Cost) và chi phí lưu kho (Holding Cost) theo quy cách đóng thùng 6 chai/thùng và đối chiếu với ngưỡng MOQ 600 chai/chuyến hàng.",
    badge="MODULE 09 • STATIC ECONOMIC ORDER QUANTITY & LOT-SIZING",
)

guardrail_banner(
    "<strong>Lưu ý phương pháp luận quan trọng:</strong> EOQ là mô hình cỡ lô đặt hàng kinh tế tĩnh (Lot-Sizing) theo chu kỳ. "
    "EOQ <strong>KHÔNG</strong> phải là định mức tồn kho mục tiêu hai giai đoạn <code>y*</code> của mô hình Kieu-DDA (phòng ngừa rủi ro ngẫu nhiên đa điểm) "
    "và <strong>KHÔNG</strong> tự động áp vào Purchase Order (PO) hay trừ lùi tồn kho hiện có. "
    "Đây là công cụ phân tích độc lập nhằm hỗ trợ nhà quản trị so sánh chi phí đánh đổi giữa tần suất đặt hàng và vốn tồn kho."
)

baseline = st.session_state.get("baseline", "preclean")
opts = sku_options(baseline)
default_sku = (st.session_state.get("sku_list") or S4)[0]
if default_sku not in opts:
    default_sku = opts[0] if opts else S4[0]

# Control Section
st.subheader("⚙️ Cấu hình Tham số Tính toán EOQ")

c1, c2, c3 = st.columns([1.5, 1.2, 1.3])

with c1:
    selected_sku = st.selectbox(
        "Chọn SKU phân tích:",
        options=opts,
        index=opts.index(default_sku) if default_sku in opts else 0,
        format_func=lambda s: format_sku(s, baseline),
    )

# Fetch historical 2024 annual demand for reference
hist_df = monthly_qty(baseline, [selected_sku], year=2024)
hist_annual_demand = float(hist_df["Quantity"].sum()) if not hist_df.empty else 1200.0
cogs_val = unit_cost(selected_sku)

with c2:
    demand_mode = st.radio(
        "Nguồn nhu cầu năm (D):",
        options=["Lịch sử 2024 (Tham khảo)", "Nhập thủ công"],
        horizontal=True,
    )
    if demand_mode.startswith("Lịch sử"):
        D_input = st.number_input(
            "Nhu cầu năm D (chai/năm):",
            min_value=0.0,
            value=float(hist_annual_demand),
            step=60.0,
            help="Dữ liệu tham khảo: Tổng sản lượng tiêu thụ năm 2024 (Lịch sử, không phải dự báo tương lai).",
        )
    else:
        D_input = st.number_input(
            "Nhu cầu năm D (chai/năm):",
            min_value=0.0,
            value=float(hist_annual_demand) if hist_annual_demand > 0 else 1200.0,
            step=60.0,
        )

with c3:
    K_input = st.number_input(
        "Chi phí cố định mỗi lần đặt K (VND/đơn):",
        min_value=10_000.0,
        max_value=100_000_000.0,
        value=2_000_000.0,
        step=200_000.0,
        help="Chi phí mở tờ khai hải quan, chứng từ vận đơn, phí giao dịch ngoại thương mỗi lần đặt hàng.",
    )

c4, c5, c6 = st.columns([1.5, 1.2, 1.3])

with c4:
    holding_mode = st.selectbox(
        "Cơ chế xác định Chi phí Lưu kho (H):",
        options=["Tỷ lệ năm % × COGS (Khuyến nghị)", "Nhập trực tiếp VND/chai/năm"],
    )
    if holding_mode.startswith("Tỷ lệ"):
        holding_rate = st.slider("Tỷ lệ chi phí lưu kho năm (i):", min_value=0.05, max_value=0.50, value=0.20, step=0.01, format="%.2f")
        H_calc = holding_rate * cogs_val
        st.caption(f"Đơn giá vốn COGS: **{vnd(cogs_val)}/chai** $\\to$ H = {holding_rate*100:.1f}% × COGS = **{vnd(H_calc)}/chai/năm**")
    else:
        H_calc = st.number_input(
            "Chi phí lưu kho H (VND/chai/năm):",
            min_value=1_000.0,
            max_value=5_000_000.0,
            value=max(10_000.0, round(cogs_val * 0.20, -2)),
            step=5_000.0,
        )

with c5:
    st.markdown("**Quy cách đóng gói & MOQ**")
    st.info("📦 **6 chai/thùng nguyên** (Single-SKU)\n\n🚢 **MOQ: 600 chai** (= 100 thùng)")

with c6:
    supplier_context = st.text_input(
        "Nhà cung cấp / Winery (Tùy chọn):",
        value="",
        placeholder="VD: Allan Scott Wines (New Zealand)",
        help="Nếu có nhiều SKU cùng nhà cung cấp, MOQ 600 áp dụng chung cho cả chuyến hàng (Shipment Pooling).",
    )

# Execution
try:
    res = calculate_eoq(
        D=float(D_input),
        K=float(K_input),
        H=float(H_calc),
        pack_size=6,
        moq=600,
        supplier_id=supplier_context or None,
    )
except Exception as exc:
    st.error(f"Lỗi tham số EOQ: {exc}")
    st.stop()

# Result Presentation
st.markdown("---")
st.subheader("📊 Kết quả Phân tích EOQ & Phương án Nguyên Thùng")

if res["status"] == "no_demand":
    st.warning("⚠️ **Không có nhu cầu đặt hàng:** Nhu cầu năm $D = 0$. Hệ thống không phát sinh cỡ lô đặt hàng kinh tế và không tự động bù MOQ 600 chai.")
else:
    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "EOQ Lý thuyết (Liên tục)",
        f"{res['eoq_continuous']:,.1f} chai",
        f"{res['eoq_cartons_exact']:,.2f} thùng",
    )
    rec_text = "Làm tròn xuống (Floor)" if res["recommended_rounding"] == "floor" else "Làm tròn lên (Ceil)"
    k2.metric(
        "Phương án Thùng Khả thi Tối ưu",
        f"{res['recommended_bottles']:,} chai",
        f"{res['recommended_cartons']} thùng nguyên ({rec_text})",
    )
    k3.metric(
        "Tần suất Đặt hàng",
        f"{res['orders_per_year']:.1f} lần/năm",
        f"Chu kỳ: ~{res['cycle_days']:.0f} ngày/đơn",
    )
    k4.metric(
        "Tổng Chi phí Liên quan (TRC)",
        vnd(res["total_relevant_cost"]),
        f"Lưu kho: {vnd(res['holding_cost'])}",
    )

    # Comparison Table: Floor vs Ceil vs Theoretical vs MOQ
    st.markdown("#### ⚖️ Bảng So sánh Đánh đổi Chi phí: Làm tròn Thùng vs MOQ 600")

    comp_data = [
        {
            "Phương án": "1. EOQ Lý thuyết (Liên tục)",
            "Số thùng": f"{res['eoq_cartons_exact']:.2f} th",
            "Số chai": f"{res['eoq_continuous']:,.1f}",
            "Chi phí Đặt hàng (VND/năm)": (res['D'] / res['eoq_continuous']) * res['K'],
            "Chi phí Lưu kho (VND/năm)": (res['eoq_continuous'] / 2.0) * res['H'],
            "Tổng chi phí TRC (VND/năm)": res['theo_trc'],
            "Chênh lệch chi phí": "— Cực tiểu lý thuyết —",
            "Ghi chú": "Không khả thi thực tế (bị lẻ chai)",
        },
        {
            "Phương án": f"2. Làm tròn xuống (Floor — {res['floor_cartons']} thùng)",
            "Số thùng": f"{res['floor_cartons']} th",
            "Số chai": f"{res['floor_bottles']:,}",
            "Chi phí Đặt hàng (VND/năm)": res['floor_ordering_cost'],
            "Chi phí Lưu kho (VND/năm)": res['floor_holding_cost'],
            "Tổng chi phí TRC (VND/năm)": res['floor_trc'],
            "Chênh lệch chi phí": f"+{vnd(res['floor_trc'] - res['theo_trc'])} (+{((res['floor_trc']/res['theo_trc'])-1)*100:.2f}%)",
            "Ghi chú": "⭐ Tối ưu chi phí thực tế" if res["recommended_rounding"] == "floor" else "Cao hơn phương án Ceil",
        },
        {
            "Phương án": f"3. Làm tròn lên (Ceil — {res['ceil_cartons']} thùng)",
            "Số thùng": f"{res['ceil_cartons']} th",
            "Số chai": f"{res['ceil_bottles']:,}",
            "Chi phí Đặt hàng (VND/năm)": res['ceil_ordering_cost'],
            "Chi phí Lưu kho (VND/năm)": res['ceil_holding_cost'],
            "Tổng chi phí TRC (VND/năm)": res['ceil_trc'],
            "Chênh lệch chi phí": f"+{vnd(res['ceil_trc'] - res['theo_trc'])} (+{((res['ceil_trc']/res['theo_trc'])-1)*100:.2f}%)",
            "Ghi chú": "⭐ Tối ưu chi phí thực tế" if res["recommended_rounding"] == "ceil" else "Cao hơn phương án Floor",
        },
        {
            "Phương án": "4. Ngưỡng MOQ Đơn lẻ (100 thùng = 600 chai)",
            "Số thùng": f"{res['moq_cartons']} th",
            "Số chai": f"{res['moq_bottles']:,}",
            "Chi phí Đặt hàng (VND/năm)": (res['D'] / res['moq_bottles']) * res['K'],
            "Chi phí Lưu kho (VND/năm)": (res['moq_bottles'] / 2.0) * res['H'],
            "Tổng chi phí TRC (VND/năm)": res['trc_at_moq'],
            "Chênh lệch chi phí": (
                f"+{vnd(res['cost_penalty_moq'])} (+{((res['trc_at_moq']/res['total_relevant_cost'])-1)*100:.1f}%)"
                if res['cost_penalty_moq'] > 0 else "0 đ (Đã đạt MOQ)"
            ),
            "Ghi chú": "Chỉ bắt buộc nếu đi chuyến hàng đơn lẻ 1 SKU không gom được đơn",
        },
    ]

    df_comp = pd.DataFrame(comp_data)
    st.dataframe(
        df_comp,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Chi phí Đặt hàng (VND/năm)": st.column_config.NumberColumn(format="%,.0f đ"),
            "Chi phí Lưu kho (VND/năm)": st.column_config.NumberColumn(format="%,.0f đ"),
            "Tổng chi phí TRC (VND/năm)": st.column_config.NumberColumn(format="%,.0f đ"),
        },
    )

    # Cost Curves Chart
    st.markdown("#### 📈 Đồ thị Đường Cong Chi phí Đặt hàng, Lưu kho & Tổng Chi phí Liên quan (TRC)")

    curve_df = pd.DataFrame(res["curve_points"])
    if not curve_df.empty:
        c_melt = curve_df.melt(
            id_vars=["Q_bottles", "Q_cartons"],
            value_vars=["Ordering_Cost", "Holding_Cost", "Total_Relevant_Cost"],
            var_name="Loại chi phí",
            value_name="Chi phí (VND)",
        )
        c_melt["Loại chi phí"] = c_melt["Loại chi phí"].map({
            "Ordering_Cost": "1. Chi phí Đặt hàng (Ordering Cost — D/Q × K)",
            "Holding_Cost": "2. Chi phí Lưu kho (Holding Cost — Q/2 × H)",
            "Total_Relevant_Cost": "3. Tổng Chi phí Liên quan (TRC = OC + HC)",
        })

        base_chart = alt.Chart(c_melt).encode(
            x=alt.X("Q_bottles:Q", title="Quy mô Lô Đặt hàng Q (Chai)"),
        )

        lines = base_chart.mark_line(strokeWidth=2.4).encode(
            y=alt.Y("Chi phí (VND):Q", title="Chi phí Năm (VND)"),
            color=alt.Color(
                "Loại chi phí:N",
                scale=alt.Scale(range=["#0284c7", "#10b981", "#ef4444"]),
                legend=alt.Legend(title="Thành phần chi phí", orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Q_bottles:Q", title="Cỡ lô Q (chai)", format=",d"),
                alt.Tooltip("Q_cartons:Q", title="Số thùng", format=",d"),
                alt.Tooltip("Loại chi phí:N", title="Loại chi phí"),
                alt.Tooltip("Chi phí (VND):Q", title="Chi phí", format=",.0f"),
            ],
        )

        # Mark EOQ vertical rule
        rule_eoq = alt.Chart(pd.DataFrame([{"Q": res["eoq_continuous"]}])).mark_rule(
            color="#ef4444", strokeDash=[4, 4], strokeWidth=2
        ).encode(x="Q:Q")

        rule_moq = alt.Chart(pd.DataFrame([{"Q": 600}])).mark_rule(
            color="#f59e0b", strokeDash=[2, 2], strokeWidth=1.8
        ).encode(x="Q:Q")

        chart_full = (lines + rule_eoq + rule_moq).properties(height=380)
        st.altair_chart(chart_full, use_container_width=True)
        st.caption("🔴 Đường nét đứt màu đỏ: **EOQ lý thuyết** • 🟡 Đường nét đứt màu vàng: **Ngưỡng MOQ chuyến hàng 600 chai (100 thùng)**")

    # CSV Export
    csv_bytes = df_comp.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Tải Bảng Phân Tích EOQ (CSV UTF-8 BOM)",
        data=csv_bytes,
        file_name=f"TWV_EOQ_Analysis_{selected_sku}.csv",
        mime="text/csv",
    )

# Assumptions & Governance
with st.expander("📖 Cơ sở Lý thuyết, Giả định Mô hình & Giới hạn Áp dụng"):
    st.markdown("""
    ##### 1. Công thức và Cơ sở Toán học
    - **Cỡ lô kinh tế lý thuyết:** $EOQ = \\sqrt{\\frac{2 D K}{H}}$
    - **Chi phí đặt hàng năm:** $Ordering\\ Cost(Q) = \\frac{D}{Q} \\times K$
    - **Chi phí lưu kho năm:** $Holding\\ Cost(Q) = \\frac{Q}{2} \\times H$
    - **Tổng chi phí liên quan:** $TRC(Q) = Ordering\\ Cost + Holding\\ Cost$

    ##### 2. Giả định của Mô hình EOQ Cổ điển
    - Nhu cầu tiêu thụ $D$ diễn ra đều đặn và có tính tất định (deterministic).
    - Thời gian đặt hàng (Lead time) coi như cố định hoặc được bù đắp bởi điểm đặt hàng lại ROP.
    - Đơn giá mua không đổi (không xét chiết khấu theo số lượng).
    - Không cho phép thiếu hàng (No backorders).

    ##### 3. Sự Khác biệt giữa EOQ và Mô hình Luận văn Kieu-DDA
    - **EOQ** là mô hình tĩnh, cục bộ cho 1 điểm và 1 chu kỳ lặp lại nhằm xác định cỡ lô đặt hàng kinh tế.
    - **Kieu-DDA** là mô hình Quy hoạch Ngẫu nhiên Hai Giai đoạn (Two-Stage Stochastic LP/MILP) giải bài toán định vị tồn kho đa điểm (Dual-Hub: CIW, Hub_Retail, Hub_B2B) ứng phó với bất định nhu cầu tương quan đa kênh.
    """)

caption("2024", "EOQ Analysis", None, "2024-08", len(opts))
