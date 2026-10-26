"""Executive Control Tower — TWV Inventory Placement & Forecasting Workbench."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps"), str(ROOT / "apps" / "web")]

from kieu_pipeline.segmentation import DEFAULT_SERVICE_TARGETS

from api.services.data_store import load_product_catalog, sku_counts
from api.services.paths import REPLAY_FILES, S4, S4_CLASS, result_csv
from api.services.runs import list_runs
from bootstrap import gate, guardrail_banner, page_header, pct, vnd

gate("TWV — Trung tâm Điều hành Đặt tồn & Dự báo")

page_header(
    "The Warehouse Vietnam — Hệ thống Hoạch định Dự báo & Tồn kho Đa điểm",
    "Tích hợp Phân khúc Danh mục ABC–XYZ, Dự báo Nhu cầu Phân tầng (Tiered Forecasting) và Tối ưu hóa Tồn kho Hai Giai đoạn Định hướng Dữ liệu (Kieu-DDA / Wang & Minner 2025).",
    badge="LUẬN VĂN THẠC SĨ • NGUYỄN THỊ XUÂN KIỀU (MLSIU24025) • GVHD: TS. TRẦN ĐỨC VĨ",
)

guardrail_banner(
    "<strong>Phạm vi mô hình chuẩn mực:</strong> Đây là bàn lập kế hoạch chiến thuật (Tactical Inventory Placement & Base-stock Allocation), "
    "không phải phần mềm quản lý kho tác nghiệp (WMS). Dữ liệu bán hàng là nhu cầu quan sát được; "
    "độ bao phủ thời gian chờ nhập khẩu <code>L ∈ {3, 4, 5, 6}</code> tháng được đưa vào qua cửa sổ kịch bản tích lũy (LP giữ cấu trúc 1 kỳ)."
)

baseline = st.session_state.get("baseline", "preclean")
try:
    info = sku_counts(baseline)
except Exception as exc:
    info = None
    st.error(f"Không thể đọc dữ liệu Parquet ({baseline}): {exc}")

runs_all = list_runs()
replay_ready = sum(
    1
    for files in REPLAY_FILES.values()
    for f in files
    if (lambda name: (lambda: True)() if Path(result_csv(name)).exists() else False)(f)
)

k1, k2, k3, k4, k5 = st.columns(5)
if info:
    k1.metric("Tổng dòng giao dịch", f"{info['n_rows']:,}", f"Năm {min(info['years'])}–{max(info['years'])}")
    k2.metric("SKU hoạt động (2024)", f"{info['n_sku_2024']:,}", f"Tổng danh mục: {info['n_sku_all']:,}")
    k3.metric("Sản lượng 2024 (chai)", f"{info['total_qty_2024']:,.0f}", f"{len(info['channels_2024'])} kênh phân phối")
k4.metric("Tập kết quả Replay (03a–03e)", f"{replay_ready}/12 file", "Sẵn sàng kiểm chứng")
k5.metric("Phiên chạy Live đã lưu", f"{len(runs_all)} runs", runs_all[0]["run_id"][:15] if runs_all else "Chưa có run mới")

st.markdown("### 1. Bốn Mã Rượu Đại diện Chuẩn Luận văn (`S4` Benchmark Portfolio)")
try:
    cat = load_product_catalog()
    s4_rows = []
    strategy_map = {
        "AX": ("Hybrid (0.5 SARIMA + 0.5 LightGBM)", "Virtual / Fixed Pooling gần kênh"),
        "AY": ("Holt-Winters TES / MA_4", "Virtual Pooling linh hoạt Dual-Hub"),
        "BY": ("Ensemble LightGBM (Tweedie)", "Virtual Pooling cân bằng vốn & cước"),
        "CZ": ("Hierarchical Top-Down", "Tập trung tồn kho về CIW (Risk Pooling)"),
    }
    for sku in S4:
        abc = S4_CLASS[sku]
        name = cat.loc[sku, "item_name"] if sku in cat.index else sku
        cogs = float(cat.loc[sku, "cogs_vnd"]) if sku in cat.index else 0.0
        retail = float(cat.loc[sku, "retail_vnd"]) if sku in cat.index else 0.0
        beta = DEFAULT_SERVICE_TARGETS[abc]
        fc_model, inv_strat = strategy_map[abc]
        s4_rows.append({
            "Mã SKU": sku,
            "Tên rượu vang (Master Data)": name,
            "Phân lớp ABC–XYZ": abc,
            "Critical Ratio mục tiêu (β)": pct(beta, 0),
            "Giá vốn COGS": vnd(cogs),
            "Giá bán lẻ niêm yết": vnd(retail),
            "Mô hình Dự báo (M4)": fc_model,
            "Chiến lược Tồn kho khuyến nghị (M6)": inv_strat,
        })
    st.dataframe(pd.DataFrame(s4_rows), use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning(f"Không tải được danh mục COGS: {exc}")

st.markdown("### 2. Kiến trúc 8 Phân hệ Chức năng (Click để truy cập nhanh)")

row1 = st.columns(4)
with row1[0]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-vp">Module 01</span>
            <h4>📊 Tổng quan & Replay 03a/03c</h4>
            <p style="font-size:0.86rem;color:#475569;">Kiến trúc mạng Dual-Hub (10 cung), đối chứng OOS Protocol A (VP vs FP vs Dec) và giá trị dự báo (RQ1 & RQ2).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Tong_quan.py", label="Mở Trang 1: Tổng quan →", icon="📊")

with row1[1]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-abc">Module 02</span>
            <h4>🗂️ Dữ liệu & Phân khúc ABC–XYZ</h4>
            <p style="font-size:0.86rem;color:#475569;">Khám phá dữ liệu ERP, ma trận ABC–XYZ theo tuần ISO 2024, biểu đồ góc phần tư Syntetos–Boylan (ADI vs CV²).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Du_lieu.py", label="Mở Trang 2: Dữ liệu & ABC–XYZ →", icon="🗂️")

with row1[2]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-fp">Module 03</span>
            <h4>📈 Lab Dự báo Phân tầng (M4)</h4>
            <p style="font-size:0.86rem;color:#475569;">Huấn luyện Hybrid / TES / SARIMA / LightGBM / Hierarchical chống rò rỉ tại mốc 2024-08, xuất WAPE & phần dư SAA.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Du_bao.py", label="Mở Trang 3: Lab Dự báo →", icon="📈")

with row1[3]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-vp">Module 04</span>
            <h4>🧮 Lab Tối ưu Tồn kho (M6)</h4>
            <p style="font-size:0.86rem;color:#475569;">Giải trực tiếp LP/MILP bằng COIN-OR CBC cho VP, FP, Dec. Phân tích vị trí đặt tồn y*, cấu trúc chi phí & đối đầu Hist vs Fcst.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_Ton_kho.py", label="Mở Trang 4: Lab Tồn kho →", icon="🧮")

row2 = st.columns(4)
with row2[0]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-dec">Module 05</span>
            <h4>🚢 Lead Time Phủ Cửa sổ L (RQ3)</h4>
            <p style="font-size:0.86rem;color:#475569;">Phân tích độ trễ nhập khẩu đường biển L ∈ {3, 4, 5, 6} tháng (Replay 03b & 03d) và giới hạn cửa sổ OOS cuối năm.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/5_Lead_time.py", label="Mở Trang 5: Lead Time L →", icon="🚢")

with row2[1]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-fp">Module 06</span>
            <h4>🔬 Phân tích Độ nhạy & Độ bền (03e)</h4>
            <p style="font-size:0.86rem;color:#475569;">Quét tham số (h_mult, beta_mult), đường cong Pareto vốn B, kiểm định Monte Carlo (Joint vs i.i.d.) và Jackknife.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/6_Do_nhay.py", label="Mở Trang 6: Độ nhạy & 03e →", icon="🔬")

with row2[2]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-dec">Module 07</span>
            <h4>📤 Xuất Mức Tồn Mục tiêu (M7)</h4>
            <p style="font-size:0.86rem;color:#475569;">Tổng hợp định mức Base-stock y* tại CIW, Hub_Retail, Hub_B2B kèm nhu cầu vốn lưu động (WC) để bàn giao vận hành.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/7_Xuat_muc_tieu.py", label="Mở Trang 7: Xuất Mục tiêu →", icon="📤")

with row2[3]:
    st.markdown(
        """
        <div class="twv-card">
            <span class="twv-pill pill-abc">Module 08</span>
            <h4>📐 Phương pháp & Thư viện Biểu đồ</h4>
            <p style="font-size:0.86rem;color:#475569;">Hệ phương trình quy hoạch ngẫu nhiên 2 giai đoạn SAA, thông số cước Dual-Hub và toàn bộ biểu đồ Chương 4 & 5.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/8_Phuong_phap.py", label="Mở Trang 8: Phương pháp & Biểu đồ →", icon="📐")
