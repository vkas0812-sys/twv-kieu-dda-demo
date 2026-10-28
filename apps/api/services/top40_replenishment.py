"""Top-40 SKU Replenishment, Supplier MOQ & 5-Channel Allocation Service.

Implements and serves the complete analytical findings from thesis notebook:
`06_top40_replenishment_moq_channel_allocation.ipynb`.
Covers:
- 40 SKU DDA-VP replenishment targets converted to 6-bottle cases.
- Supplier MOQ 600 bottles constraint enforcement.
- 5 sales channels (RT, GT, HC, HO, MT) allocation via largest remainder.
- 27 thesis-ready visual analytics figures and economic decision answers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api.services.paths import RESULTS, WORKSPACE_ROOT

# Resolve Top-40 results directory
_LOCAL_TOP40 = RESULTS / "06_replenishment_moq_top40"
_WORKSPACE_TOP40_BRANCH = (
    WORKSPACE_ROOT
    / "06_Full_project_Branch"
    / "3. Database and result file"
    / "result"
    / "06_replenishment_moq_top40 (1)"
)
_WORKSPACE_TOP40_ORIG = (
    WORKSPACE_ROOT
    / "06_Full_project_Branch"
    / "3. Database and result file"
    / "result"
    / "06_replenishment_moq_top40"
)

if _LOCAL_TOP40.exists():
    TOP40_DIR = _LOCAL_TOP40
elif _WORKSPACE_TOP40_BRANCH.exists():
    TOP40_DIR = _WORKSPACE_TOP40_BRANCH
elif _WORKSPACE_TOP40_ORIG.exists():
    TOP40_DIR = _WORKSPACE_TOP40_ORIG
else:
    TOP40_DIR = _LOCAL_TOP40

TOP40_FIGURES_DIR = TOP40_DIR / "figures"

CHANNELS = ["RT", "GT", "HC", "HO", "MT"]
CASE_PACK = 6
SUPPLIER_MOQ = 600

# Metadata for all 27 figures from the thesis notebook
FIGURES_METADATA = [
    # Group 1: Figures 1-12 (Portfolio & Discretization)
    {
        "id": 1,
        "filename": "Fig_06_01_target_vs_final_order.png",
        "title": "Figure 1. Mục tiêu liên tục DDA-VP vs Đơn đặt hàng rời rạc nguyên thùng",
        "category": "Ràng buộc Rời rạc",
        "description": "So sánh giữa lời giải liên tục DDA-VP (L=3 kỳ) và đơn hàng thực tế sau khi làm tròn nguyên thùng 6 chai cho toàn bộ 40 SKU.",
    },
    {
        "id": 2,
        "filename": "Fig_06_02_discrete_constraint_uplift.png",
        "title": "Figure 2. Lượng chai tăng thêm do ràng buộc rời rạc theo SKU",
        "category": "Ràng buộc Rời rạc",
        "description": "Lượng tồn kho tăng thêm do quy cách đóng thùng 6 chai (tối đa 5 chai/SKU) và bù MOQ 600 chai.",
    },
    {
        "id": 3,
        "filename": "Fig_06_03_target_vs_demand_proxy.png",
        "title": "Figure 3. Mục tiêu mô hình so với Nhu cầu Dự báo 3 tháng (Sep–Nov 2024)",
        "category": "Ràng buộc Rời rạc",
        "description": "Đối chiếu giữa mức tồn mục tiêu của DDA-VP với tổng nhu cầu dự báo điểm trong cửa sổ L=3 kỳ.",
    },
    {
        "id": 4,
        "filename": "Fig_06_04_channel_totals.png",
        "title": "Figure 4. Tổng sản lượng phân bổ theo 5 Kênh Tiêu thụ",
        "category": "Phân bổ Kênh",
        "description": "Phân bổ tổng 93,552 chai theo 5 kênh tiêu thụ: HC chiếm ưu thế vượt trội với 52.6% (49,206 chai).",
    },
    {
        "id": 5,
        "filename": "Fig_06_05_sku_channel_heatmap.png",
        "title": "Figure 5. Ma trận Heatmap Phân bổ SKU × Kênh Tiêu thụ (Chai)",
        "category": "Phân bổ Kênh",
        "description": "Bản đồ nhiệt thể hiện chi tiết mật độ phân bổ 40 SKU vào 5 kênh bán hàng theo quy cách nguyên thùng.",
    },
    {
        "id": 6,
        "filename": "Fig_06_06_channel_mix_by_sku.png",
        "title": "Figure 6. Cơ cấu tỷ trọng Kênh (Channel Mix) trong từng SKU",
        "category": "Phân bổ Kênh",
        "description": "Tỷ lệ phần trăm phân bổ 5 kênh cho mỗi mã rượu, phản ánh cơ cấu tiêu thụ dự báo của từng SKU.",
    },
    {
        "id": 7,
        "filename": "Fig_06_07_functional_hub_split.png",
        "title": "Figure 7. Phân bổ tồn kho chức năng giữa các Hub tiến phương",
        "category": "Phân bổ Kênh",
        "description": "Kế hoạch định vị tồn kho giữa Kho trung tâm CIW và các Hub phân phối chức năng.",
    },
    {
        "id": 8,
        "filename": "Fig_06_08_order_distribution.png",
        "title": "Figure 8. Phân phối quy mô đơn đặt hàng theo thùng nguyên",
        "category": "Ràng buộc Rời rạc",
        "description": "Biểu đồ phân phối tần suất quy mô đặt hàng (số thùng 6 chai) của 40 SKU Top-40.",
    },
    {
        "id": 9,
        "filename": "Fig_06_09_working_capital_by_sku.png",
        "title": "Figure 9. Nhu cầu vốn lưu động (Working Capital) theo SKU",
        "category": "Kinh tế & Vốn",
        "description": "Giá trị vốn lưu động nhập kho theo từng SKU sau khi áp dụng ràng buộc nguyên thùng và MOQ (Tổng 13.74 tỷ VND).",
    },
    {
        "id": 10,
        "filename": "Fig_06_10_centralization_vs_order.png",
        "title": "Figure 10. Mức độ tập trung kho trung tâm (CL) vs Quy mô đơn hàng",
        "category": "Ràng buộc Rời rạc",
        "description": "Mối quan hệ giữa chỉ số Centralization Level (CL) và số lượng đặt hàng CIW của 40 SKU.",
    },
    {
        "id": 11,
        "filename": "Fig_06_11_portfolio_constraint_bridge.png",
        "title": "Figure 11. Biểu đồ cầu nối (Bridge Waterfall) tác động của Thùng và MOQ",
        "category": "Ràng buộc Rời rạc",
        "description": "Phân rã biến thiên từ Mục tiêu DDA-VP (93,508 chai) -> Làm tròn 6 chai (+44 chai) -> MOQ (+0 chai) -> 93,552 chai.",
    },
    {
        "id": 12,
        "filename": "Fig_06_12_order_concentration_pareto.png",
        "title": "Figure 12. Đường cong tích lũy Pareto sản lượng nhập kho Top-40",
        "category": "Ràng buộc Rời rạc",
        "description": "16 SKU đầu tiên đã chiếm 80% tổng sản lượng nhập kho; top 5 SKU chiếm gần 40%.",
    },
    # Group 2: Figures 13-17 (Channel-specific details)
    {
        "id": 13,
        "filename": "Fig_06_13_channel_RT_allocation.png",
        "title": "Figure 13. Chi tiết phân bổ SKU cho Kênh RT (Retail - Cửa hàng bán lẻ)",
        "category": "Chi tiết Kênh",
        "description": "Chi tiết số chai và số thùng nguyên phân bổ cho kênh Bán lẻ (11,604 chai, 1.98 tỷ VND).",
    },
    {
        "id": 14,
        "filename": "Fig_06_14_channel_GT_allocation.png",
        "title": "Figure 14. Chi tiết phân bổ SKU cho Kênh GT (General Trade - Đại lý truyền thống)",
        "category": "Chi tiết Kênh",
        "description": "Chi tiết số chai và số thùng nguyên phân bổ cho kênh Đại lý (17,262 chai, 2.90 tỷ VND).",
    },
    {
        "id": 15,
        "filename": "Fig_06_15_channel_HC_allocation.png",
        "title": "Figure 15. Chi tiết phân bổ SKU cho Kênh HC (HoReCa - Nhà hàng / Khách sạn)",
        "category": "Chi tiết Kênh",
        "description": "Kênh tiêu thụ trọng điểm lớn nhất của TWV (49,206 chai, 6.98 tỷ VND, chiếm 52.6% danh mục).",
    },
    {
        "id": 16,
        "filename": "Fig_06_16_channel_HO_allocation.png",
        "title": "Figure 16. Chi tiết phân bổ SKU cho Kênh HO (Head Office - Khách hàng Doanh nghiệp / B2B)",
        "category": "Chi tiết Kênh",
        "description": "Chi tiết số chai và số thùng nguyên phân bổ cho kênh Trụ sở / B2B (3,948 chai, 534 triệu VND).",
    },
    {
        "id": 17,
        "filename": "Fig_06_17_channel_MT_allocation.png",
        "title": "Figure 17. Chi tiết phân bổ SKU cho Kênh MT (Modern Trade - Chuỗi Siêu thị)",
        "category": "Chi tiết Kênh",
        "description": "Chi tiết số chai và số thùng nguyên phân bổ cho kênh Siêu thị hiện đại (11,532 chai, 1.34 tỷ VND).",
    },
    # Group 3: Figures 18-27 (10 Economic Decision Questions)
    {
        "id": 18,
        "filename": "Fig_06_18_working_capital_pareto.png",
        "title": "Figure 18 (Q1). Những SKU nào hấp thụ nhiều vốn lưu động nhất?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Phân tích Pareto vốn: 17 SKU đầu tiên chiếm 80% vốn lưu động. SKU W101683 dẫn đầu với 1.48 tỷ VND.",
    },
    {
        "id": 19,
        "filename": "Fig_06_19_unit_cost_vs_order_size.png",
        "title": "Figure 19 (Q2). Vốn lưu động bị chi phối bởi Đơn giá COGS hay Sản lượng?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Biểu đồ phân tán kích thước bong bóng phân định nhóm rượu cao cấp đơn giá cao vs rượu phổ thông sản lượng lớn.",
    },
    {
        "id": 20,
        "filename": "Fig_06_20_inventory_coverage_ratio.png",
        "title": "Figure 20 (Q3). Những SKU nào mang bộ đệm tồn kho (Coverage Ratio) lớn nhất?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Tỷ lệ Coverage Ratio = Đơn hàng / Nhu cầu dự báo. Trung vị là 1.08×, phản ánh phần đệm an toàn phòng ngừa biến động.",
    },
    {
        "id": 21,
        "filename": "Fig_06_21_case_rounding_capital_overhead.png",
        "title": "Figure 21 (Q4). Chi phí vốn phát sinh do quy cách đóng thùng 6 chai là bao nhiêu?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Tổng vốn tăng thêm do làm tròn nguyên thùng chỉ là 5.76 triệu VND trên toàn danh mục 13.7 tỷ VND (< 0.05%).",
    },
    {
        "id": 22,
        "filename": "Fig_06_22_channel_capital_exposure.png",
        "title": "Figure 22 (Q5). Kênh nào tiếp nhận giá trị vốn tồn kho lớn nhất?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Định giá hàng hóa theo từng kênh: HC dẫn đầu với 6.98 tỷ VND, GT tiếp theo với 2.90 tỷ VND, RT đạt 1.98 tỷ VND.",
    },
    {
        "id": 23,
        "filename": "Fig_06_23_channel_volume_vs_capital_share.png",
        "title": "Figure 23 (Q6). Kênh nào thiên về Giá trị (Value-heavy) và Kênh nào thiên về Sản lượng (Volume-heavy)?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "RT và GT có tỷ trọng vốn > tỷ trọng lượng (hàng giá trị cao). MT có tỷ trọng lượng > vốn (hàng phổ thông phân khúc thấp hơn).",
    },
    {
        "id": 24,
        "filename": "Fig_06_24_sku_channel_capital_heatmap.png",
        "title": "Figure 24 (Q7). Rủi ro tập trung vốn nằm ở giao điểm SKU × Kênh nào?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Bản đồ nhiệt giá trị vốn lưu động (triệu VND) hỗ trợ kiểm soát trọng điểm vào các mắt xích có giá trị vật chất lớn.",
    },
    {
        "id": 25,
        "filename": "Fig_06_25_abc_volume_vs_capital_share.png",
        "title": "Figure 25 (Q8). Phân khúc ABC-XYZ nào tiêu thụ vốn không tương xứng?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Nhóm AZ và AX chiếm tỷ trọng vốn vượt trội (hơn 85% tổng vốn danh mục Top-40).",
    },
    {
        "id": 26,
        "filename": "Fig_06_26_expected_cost_components_by_abc.png",
        "title": "Figure 26 (Q9). Thành phần chi phí nào chi phối từng phân khúc ABC-XYZ?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Phân rã chi phí kỳ vọng của DDA-VP: Chi phí Fulfill chiếm ưu thế, chi phí giữ hàng Holding vừa phải, chi phí phạt Penalty gần như triệt tiêu.",
    },
    {
        "id": 27,
        "filename": "Fig_06_27_demand_vs_capital_concentration.png",
        "title": "Figure 27 (Q10). Vốn lưu động có đi liền với mức độ tập trung nhu cầu dự báo không?",
        "category": "10 Câu hỏi Kinh tế",
        "description": "Đường cong tích lũy kép so sánh Nhu cầu vs Vốn lưu động với hệ số tương quan thứ hạng Spearman ρ = 0.54.",
    },
]


def load_primary_plan() -> pd.DataFrame:
    """Load primary replenishment plan (40 SKUs, complete cases, MOQ satisfied)."""
    csv_path = TOP40_DIR / "06_top40_replenishment_primary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing primary replenishment file: {csv_path}")
    return pd.read_csv(csv_path)


def load_upper_bound_plan() -> pd.DataFrame:
    """Load conservative upper bound plan (each SKU treated as independent supplier shipment)."""
    csv_path = TOP40_DIR / "06_top40_replenishment_upper_bound.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing upper bound replenishment file: {csv_path}")
    return pd.read_csv(csv_path)


def load_channel_allocation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load channel allocation: long format and wide format (SKU x Channel)."""
    csv_path = TOP40_DIR / "06_top40_channel_allocation_primary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing channel allocation file: {csv_path}")
    long_df = pd.read_csv(csv_path)
    wide_df = long_df.pivot(
        index="sku", columns="channel", values="allocated_bottles"
    ).reindex(columns=CHANNELS, fill_value=0)
    return long_df, wide_df


def load_channel_economic_summary() -> pd.DataFrame:
    """Load channel-level economic summary (allocated bottles, capital, shares)."""
    csv_path = TOP40_DIR / "06_top40_channel_economic_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing channel economic summary: {csv_path}")
    return pd.read_csv(csv_path)


def load_abc_economic_summary() -> pd.DataFrame:
    """Load ABC-XYZ segment economic summary (capital, holding, penalty, fulfillment)."""
    csv_path = TOP40_DIR / "06_top40_abc_economic_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing ABC economic summary: {csv_path}")
    return pd.read_csv(csv_path)


def load_sku_economic_metrics() -> pd.DataFrame:
    """Load detailed economic metrics for all 40 SKUs."""
    csv_path = TOP40_DIR / "06_top40_sku_economic_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing SKU economic metrics: {csv_path}")
    return pd.read_csv(csv_path)


def load_assumption_audit() -> pd.DataFrame:
    """Load assumptions and audit parameters."""
    csv_path = TOP40_DIR / "06_top40_assumption_audit.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing assumption audit file: {csv_path}")
    return pd.read_csv(csv_path)


def load_validation_table() -> pd.DataFrame:
    """Run operational and mathematical validation checks against loaded tables."""
    primary = load_primary_plan()
    upper = load_upper_bound_plan()
    long_df, _ = load_channel_allocation()

    reconciled = long_df.groupby("sku")["allocated_bottles"].sum().reindex(primary["sku"])
    is_reconciled = bool(np.array_equal(reconciled.to_numpy(dtype=int), primary["final_order_bottles"].to_numpy(dtype=int)))
    moq_bounds_match = bool(int(primary["final_order_bottles"].sum()) == int(upper["final_order_bottles"].sum()))

    checks = [
        {"check": "Chính xác 40 SKU đơn nhất trong danh mục", "result": bool(len(primary) == 40 and primary["sku"].nunique() == 40)},
        {"check": "Mục tiêu DDA-VP duy nhất cho mỗi SKU", "result": bool(not primary.duplicated("sku").any())},
        {"check": "Đầy đủ 5 kênh tiêu thụ của luận văn (RT, GT, HC, HO, MT)", "result": bool(set(long_df["channel"]) == set(CHANNELS))},
        {"check": "100% sản lượng nhập kho nguyên thùng 6 chai (Không xé lẻ)", "result": bool((primary["final_order_bottles"] % 6 == 0).all())},
        {"check": "100% phân bổ kênh nguyên thùng 6 chai (Không xé lẻ)", "result": bool((long_df["allocated_bottles"] % 6 == 0).all())},
        {"check": "Thỏa mãn MOQ nhà cung cấp (≥ 600 chai/đơn)", "result": bool((primary["final_order_bottles"] >= 600).all())},
        {"check": "Phân bổ kênh khớp 100% với đơn nhập khẩu CIW", "result": is_reconciled},
        {"check": "Biên dưới (Unmapped) và biên trên (Per-SKU) MOQ trùng khớp", "result": moq_bounds_match},
    ]
    return pd.DataFrame(checks)


def get_top40_kpis() -> dict[str, Any]:
    """Calculate key portfolio performance and replenishment indicators."""
    primary = load_primary_plan()
    long_df, _ = load_channel_allocation()
    channel_totals = long_df.groupby("channel")["allocated_bottles"].sum().reindex(CHANNELS)
    largest_ch = str(channel_totals.idxmax())
    largest_ch_share = float(channel_totals.max() / channel_totals.sum())

    continuous_total = float(primary["model_target_bottles"].sum())
    final_total = int(primary["final_order_bottles"].sum())
    total_cases = int(primary["final_order_cases"].sum())
    case_uplift = final_total - int(primary["order_pre_moq_bottles"].sum())
    moq_uplift = int(primary["moq_uplift_cases"].sum() * 6)
    rounded_capital = float(primary["rounded_working_capital_vnd"].sum())

    # Pareto calculation
    sorted_df = primary.sort_values("final_order_bottles", ascending=False).reset_index(drop=True)
    sorted_df["cum_share"] = sorted_df["final_order_bottles"].cumsum() / sorted_df["final_order_bottles"].sum()
    n_80 = int(sorted_df.loc[sorted_df["cum_share"] >= 0.80].index[0]) + 1

    top_volume_sku = sorted_df.iloc[0]["sku"]
    top_volume_qty = int(sorted_df.iloc[0]["final_order_bottles"])

    cap_sorted = primary.sort_values("rounded_working_capital_vnd", ascending=False).reset_index(drop=True)
    top_cap_sku = cap_sorted.iloc[0]["sku"]
    top_cap_val = float(cap_sorted.iloc[0]["rounded_working_capital_vnd"])

    return {
        "n_skus": len(primary),
        "continuous_target": continuous_total,
        "final_bottles": final_total,
        "final_cases": total_cases,
        "case_pack": CASE_PACK,
        "case_uplift_bottles": case_uplift,
        "case_uplift_pct": (case_uplift / continuous_total) * 100.0,
        "moq_uplift_bottles": moq_uplift,
        "rounded_working_capital_vnd": rounded_capital,
        "largest_channel": largest_ch,
        "largest_channel_share": largest_ch_share,
        "largest_channel_bottles": int(channel_totals.max()),
        "n_skus_for_80_volume": n_80,
        "top_volume_sku": top_volume_sku,
        "top_volume_bottles": top_volume_qty,
        "top_capital_sku": top_cap_sku,
        "top_capital_vnd": top_cap_val,
        "all_skus_clear_600": bool((primary["final_order_bottles"] >= SUPPLIER_MOQ).all()),
    }


def get_figure_path(filename: str) -> Path | None:
    """Retrieve full file path for a specific figure PNG."""
    path = TOP40_FIGURES_DIR / filename
    if path.exists():
        return path
    # Try alternate location in workspace root if needed
    alt = _WORKSPACE_TOP40_BRANCH / "figures" / filename
    if alt.exists():
        return alt
    return None
