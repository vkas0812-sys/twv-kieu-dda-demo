from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

from api.services.paths import FIGURES_DIR
from bootstrap import gate, guardrail_banner, page_header

gate("TWV — 8. Phương pháp luận & Thư viện Biểu đồ")

page_header(
    "8. Phương pháp luận Toán học, Quy tắc Quản trị & Thư viện Biểu đồ Luận văn",
    "Mô hình Quy hoạch Ngẫu nhiên Hai Giai đoạn (Two-Stage Data-Driven SAA) theo Wang & Minner (2025), cấu trúc chi phí Dual-Hub và bộ 25 biểu đồ Chương 4 & 5.",
    badge="MODULE 08 • MATHEMATICAL FORMULATION & THESIS GALLERY",
)

guardrail_banner(
    "<strong>Cam kết minh bạch học thuật:</strong> Mức tiết kiệm <code>−19,2%</code> trong bài báo của Wang & Minner (2025) là kết quả trên ngành thương mại điện tử của tác giả gốc, "
    "không phải KPI của The Warehouse Vietnam. Luận văn kế thừa khung phương pháp luận SAA 2 giai đoạn và thích nghi cho chuỗi cung ứng rượu vang nhập khẩu đa kênh."
)

tab_math, tab_spec, tab_figs = st.tabs([
    "📐 1. Mô hình Toán học Kieu-DDA (VP / FP / Dec)",
    "🛡️ 2. Thiết kế Thực nghiệm Protocol A & Giới hạn Phạm vi",
    "🖼️ 3. Thư viện Biểu đồ Chuẩn Luận văn (Chương 4 & Chương 5)",
])

with tab_math:
    st.markdown(
        r"""
### 1.1. Ký hiệu Tập hợp, Tham số và Biến quyết định

- **Tập hợp:**
  - $\mathcal{L} = \{\text{CIW}, \text{Hub\_Retail}, \text{Hub\_B2B}\}$: Tập các nút lưu giữ tồn kho trong mạng `dual_hub`.
  - $\mathcal{R} = \{\text{RT}, \text{HO}, \text{HC}, \text{GT}, \text{MT}\}$: Tập 5 kênh nhu cầu phân phối rượu vang.
  - $\mathcal{A} \subseteq \mathcal{L} \times \mathcal{R}$: Tập 10 cung phân phối hợp lệ (`CIW` $\to$ 5 kênh; `Hub_Retail` $\to$ `RT, HO`; `Hub_B2B` $\to$ `HC, GT, MT`).
  - $\Omega = \{1, \dots, |\Omega|\}$: Tập kịch bản nhu cầu SAA với xác suất đồng đều $\pi_\omega = \frac{1}{|\Omega|}$.

- **Tham số chi phí (cho từng mã SKU $k$):**
  - $c_k$: Giá vốn nhập khẩu đơn vị (`COGS`, VND/chai).
  - $h_l$: Chi phí lưu kho mỗi kỳ tại nút $l \in \mathcal{L}$ ($h_{\text{CIW}} = 0{,}02 c_k$; $h_{\text{Retail}} = 0{,}025 c_k$; $h_{\text{B2B}} = 0{,}023 c_k$).
  - $c_{lr}$: Chi phí vận chuyển/xử lý đơn hàng từ nút $l$ đến kênh $r$ ($5.000$ / $2.000$ / $2.500$ VND/chai).
  - $\beta_k \in (0,1)$: Tỷ lệ tới hạn mục tiêu (*Target Critical Ratio*) theo lớp `ABC–XYZ`, suy ra chi phí phạt thiếu hàng đơn vị:
    $$p_k = \frac{\beta_k}{1 - \beta_k} h_{\text{CIW}}$$

- **Biến quyết định 2 giai đoạn:**
  - **Giai đoạn 1 (Trước khi biết nhu cầu):** Mức tồn kho mục tiêu $y_l \ge 0$ tại mỗi nút $l \in \mathcal{L}$ (và biến nhị phân gán kênh $x_{lr} \in \{0,1\}$ đối với `FP`).
  - **Giai đoạn 2 (Sau khi quan sát kịch bản $\omega \in \Omega$):** Luồng xuất hàng $z_{lr\omega} \ge 0$ trên cung $(l,r) \in \mathcal{A}$, tồn kho cuối kỳ $s_{l\omega} \ge 0$, và lượng thiếu hàng $b_{r\omega} \ge 0$.

---

### 1.2. Chính sách Kieu-DDA-VP (Virtual Pooling — Quy hoạch Tuyến tính LP)

$$\min_{y, z, s, b} \quad \text{TEC} = \frac{1}{|\Omega|} \sum_{\omega \in \Omega} \left( \sum_{l \in \mathcal{L}} h_l s_{l\omega} + \sum_{r \in \mathcal{R}} p_k b_{r\omega} + \sum_{(l,r) \in \mathcal{A}} c_{lr} z_{lr\omega} \right)$$

**Ràng buộc cân bằng luồng và ngân sách vốn:**
$$s_{l\omega} = y_l - \sum_{r : (l,r) \in \mathcal{A}} z_{lr\omega}, \qquad \forall l \in \mathcal{L}, \forall \omega \in \Omega$$
$$b_{r\omega} = d_{r\omega} - \sum_{l : (l,r) \in \mathcal{A}} z_{lr\omega}, \qquad \forall r \in \mathcal{R}, \forall \omega \in \Omega$$
$$c_k \sum_{l \in \mathcal{L}} y_l \le B \quad (\text{khi bật ràng buộc vốn lưu động } B)$$

---

### 1.3. Chính sách Kieu-DDA-FP (Fixed Pooling — Quy hoạch Nguyên Hỗn hợp MILP)

Bổ sung biến nhị phân mở kho $x_{ll} \in \{0,1\}$ và gán kênh cố định $x_{lr} \in \{0,1\}$ ở Giai đoạn 1:
$$\sum_{l : (l,r) \in \mathcal{A}} x_{lr} = 1 \quad (\forall r \in \mathcal{R}), \qquad x_{lr} \le x_{ll}, \qquad z_{lr\omega} \le d_{r\omega} x_{lr}$$
Khi chấm điểm Out-of-Sample (`evaluate_given_y`), cả vector tồn kho $y^*$ lẫn ma trận gán kênh $x_{lr}^*$ đều được **khóa cứng** từ tập Train.

---

### 1.4. Các Chỉ số Đánh giá Cấu trúc Lời giải
- **Mức phục vụ thực hiện (Type-II Fill Rate):** $\text{FR} = 1 - \frac{\sum_{\omega, r} b_{r\omega}}{\sum_{\omega, r} d_{r\omega}}$
- **Mức độ tập trung lượng tồn (`CL_stock`):** $\text{CL}_{\text{stock}} = \frac{y_{\text{CIW}}^*}{\sum_{l \in \mathcal{L}} y_l^*}$
- **Tỷ lệ phân bổ luồng hàng từ kho tổng (`SA`):** $\text{SA} = \frac{\sum_{\omega, r} z_{\text{CIW}, r, \omega}^*}{\sum_{\omega, l, r} z_{lr\omega}^*}$
        """
    )

with tab_spec:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            r"""
### Thiết kế Thực nghiệm Chuẩn (Protocol A)
1. **Chia tách thời gian (Temporal Split — Không lấy mẫu ngẫu nhiên):**
   - **Tập Huấn luyện (Train):** Tháng `01/2024 – 08/2024` (8 tháng, Cutoff `2024-08`).
   - **Tập Kiểm định Độc lập (Out-of-Sample Test):** Tháng `09/2024 – 12/2024` (4 tháng mùa cao điểm cuối năm).
2. **Bốn nhánh Kịch bản $\Omega$:**
   - `hist`: 8 kịch bản lịch sử 1 tháng (chuẩn bài báo gốc).
   - `hist_coverage`: Kịch bản tích lũy trượt $L \in \{3, 4, 5, 6\}$ tháng.
   - `fcst_joint`: Tâm dự báo tháng 9 (`Optimized_Best`) cộng vector phần dư 5 kênh cùng tháng ($\omega_t = \max(0, \hat{\mu}_9 + e_t)$) để **bảo toàn tương quan chéo giữa các kênh**.
   - `fcst_coverage`: Tổng dự báo trên cửa sổ $L$ tháng cộng tổng phần dư cửa sổ $L$ tháng.
            """
        )
    with c2:
        st.markdown(
            r"""
### Những điều Hệ thống KHÔNG Khẳng định & Giới hạn Phạm vi
- **Không đồng nhất $\beta$ với ràng buộc CSL cứng:** $\beta$ là tỷ lệ tới hạn trong hàm mục tiêu, còn Fill Rate là kết quả thực hiện sau tối ưu hóa.
- **Không bịa số cho $L=5, 6$ trên OOS 2024:** Vì tập OOS tháng 9–12 chỉ dài 4 tháng, các cửa sổ trượt $L \ge 5$ không có cửa sổ kiểm định đầy đủ và được ghi nhận trung thực là `no window`.
- **Không gộp trung bình TEC khác độ phân giải (Grain):** Chi phí 1 tháng (`03a/03c`) không bao giờ cộng gộp với chi phí cửa sổ $L$ tháng (`03b/03d`).
- **Ngoài phạm vi tác nghiệp:** Không bao gồm quản lý vị trí kệ/bin (WMS), quét mã vạch, kiểm soát lô/hạn dùng (FEFO), định tuyến giao hàng (VRP) hay MOQ nhà cung cấp ngoại.
            """
        )

with tab_figs:
    st.subheader("🖼️ Thư viện Biểu đồ Xuất bản cho Chương 4 & Chương 5 (`06/figures/`)")
    if FIGURES_DIR.exists():
        pngs = sorted([p for p in FIGURES_DIR.glob("*.png") if not p.name.startswith("._")])
        if pngs:
            CHAPTER_LABELS = {
                "fig_4_1_pipeline.png": "Hình 4.1 — Sơ đồ Kiến trúc Pipeline Tích hợp M1–M8",
                "fig_4_2_portfolio.png": "Hình 4.2 — Phân bố Danh mục theo Ma trận ABC–XYZ",
                "fig_4_5_residual_quad.png": "Hình 4.5 — Chẩn đoán Phần dư Chuỗi thời gian (4 SKU đại diện S4)",
                "fig_4_6_s4_actual_vs_predicted.png": "Hình 4.6 — So sánh Thực tế vs Dự báo trên 4 mã S4",
                "fig_4_7_dual_hub.png": "Hình 4.7 — Cấu trúc Mạng lưới Phân phối Dual-Hub (10 cung)",
                "fig_4_8_protocol_a.png": "Hình 4.8 — Giao thức Kiểm định Out-of-Sample Protocol A (Train 1–8 / Test 9–12)",
                "fig_4_9_policies.png": "Hình 4.9 — Minh họa Cơ chế 3 Chính sách: VP, FP và Decentralized",
                "fig_4_10_saa.png": "Hình 4.10 — Cơ chế Sinh Kịch bản SAA & Joint Residual Bootstrap",
                "fig_4_11_holdout_split.png": "Hình 4.11 — Thiết kế Chia cắt Dữ liệu Holdout Chống Rò rỉ",
                "fig_5_1_wape_by_model.png": "Hình 5.1 — So sánh Sai số WAPE theo Mô hình Dự báo",
                "fig_5_2_wape_heatmap.png": "Hình 5.2 — Bản đồ nhiệt Sai số WAPE theo SKU × Kênh",
                "fig_5_3_residual_corr.png": "Hình 5.3 — Ma trận Tương quan Phần dư Giữa các Kênh Phân phối",
                "fig_5_4_s4_policy_panels.png": "Hình 5.4 — Kết quả Đối chứng 3 Chính sách trên 4 Mã S4 (RQ1)",
                "fig_5_5_portfolio_medians.png": "Hình 5.5 — Phân phối Trung vị Chi phí & Fill Rate trên Top 40 SKU",
                "fig_5_6_holding_penalty.png": "Hình 5.6 — Cơ cấu Chi phí Lưu kho (Holding) vs Chi phí Phạt (Penalty)",
                "fig_5_7_cl_stock.png": "Hình 5.7 — Mức độ Tập trung Tồn kho tại Kho Tổng CIW (CL_stock)",
                "fig_5_8_cl_sa.png": "Hình 5.8 — Tỷ lệ Phân bổ Luồng hàng từ CIW (Shipment Allocation - SA)",
                "fig_5_9_hist_vs_fcst.png": "Hình 5.9 — So sánh Kịch bản Lịch sử vs Dự báo (RQ2 - 03c)",
                "fig_5_10_leadtime.png": "Hình 5.10 — Tác động của Độ trễ Nhập khẩu L=3 và L=4 Tháng (RQ3 - 03b/03d)",
                "fig_5_11_pareto.png": "Hình 5.11 — Đường cong Pareto Ngân sách Vốn Lưu động B vs Mức Phục vụ",
                "fig_5_12_sa_param.png": "Hình 5.12 — Phân tích Độ nhạy Tham số (h_mult × beta_mult)",
                "fig_5_13_monte_carlo.png": "Hình 5.13 — Đối chứng Monte Carlo: Joint Bootstrap vs Gaussian Độc lập",
                "fig_5_14_jackknife.png": "Hình 5.14 — Kiểm định Độ bền vững Jackknife (Leave-One-Month-Out)",
                "fig_5_15_networks.png": "Hình 5.15 — Đối chiếu Kiến trúc Mạng Dual-Hub và Mạng 6-Buffer",
                "fig_im_cost_structure.png": "Hình Phụ lục IM.1 — Cơ cấu Chi phí Demo Tồn kho",
                "fig_im_service_vs_target.png": "Hình Phụ lục IM.2 — So sánh Fill Rate Thực hiện vs Mục tiêu β",
                "fig_im_budget_panels.png": "Hình Phụ lục IM.3 — Phân tích Giới hạn Ngân sách Vốn",
            }
            col_filter, col_pick = st.columns([1.0, 2.0])
            grp_choice = col_filter.radio(
                "Nhóm biểu đồ:",
                ["Tất cả (27 hình)", "Chương 4 — Phương pháp luận (fig_4_*)", "Chương 5 — Kết quả thực nghiệm (fig_5_*)", "Phụ lục Demo (fig_im_*)"],
            )
            if grp_choice.startswith("Chương 4"):
                filtered = [p for p in pngs if p.name.startswith("fig_4_")]
            elif grp_choice.startswith("Chương 5"):
                filtered = [p for p in pngs if p.name.startswith("fig_5_")]
            elif grp_choice.startswith("Phụ lục"):
                filtered = [p for p in pngs if p.name.startswith("fig_im_")]
            else:
                filtered = pngs

            selected_png = col_pick.selectbox(
                "Chọn biểu đồ để phóng to:",
                options=filtered,
                format_func=lambda p: CHAPTER_LABELS.get(p.name, p.name),
            )
            if selected_png:
                st.image(str(selected_png), caption=CHAPTER_LABELS.get(selected_png.name, selected_png.name), use_container_width=True)

            with st.expander("Xem lưới thu nhỏ toàn bộ biểu đồ trong nhóm đã chọn"):
                cols = st.columns(2)
                for idx, p in enumerate(filtered):
                    with cols[idx % 2]:
                        st.image(str(p), caption=CHAPTER_LABELS.get(p.name, p.name), use_container_width=True)
        else:
            st.info("Không tìm thấy file ảnh PNG trong thư mục `06/figures/`.")
    else:
        st.info(f"Thư mục biểu đồ `{FIGURES_DIR}` chưa tồn tại.")
