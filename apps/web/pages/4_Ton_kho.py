from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_ROOT / "packages"), str(_ROOT / "apps"), str(_ROOT / "apps" / "web")]

import importlib
import api.services.forecast_lab as _fl_mod
import api.services.inventory_lab as _il_mod
import api.services.procurement as _pr_mod
importlib.reload(_fl_mod)
importlib.reload(_il_mod)
importlib.reload(_pr_mod)
from api.services.forecast_lab import load_forecast_tables
from api.services.inventory_lab import compare_hist_fcst, solve_protocol
from api.services.procurement import calculate_procurement_plan
from api.services.paths import CHANNELS, CHANNEL_LABELS, POLICIES, S4
from api.services.runs import list_runs, read_run
from bootstrap import (
    POLICY_COLOR,
    POLICY_LABEL,
    caption,
    format_sku,
    gate,
    guardrail_banner,
    page_header,
    pct,
    sku_options,
    vnd,
)

gate("TWV — 4. Lab Tối ưu Tồn kho Kieu-DDA (M6)")

page_header(
    "4. Lab Tối ưu hóa Vị trí Tồn kho Đa điểm Kieu-DDA (Module M6)",
    "Giải bài toán Quy hoạch Ngẫu nhiên Hai Giai đoạn (SAA LP/MILP) bằng bộ giải COIN-OR CBC cho 3 chính sách: Virtual Pooling (VP), Fixed Pooling (FP) và Decentralized (Dec).",
    badge="MODULE 04 • TWO-STAGE STOCHASTIC INVENTORY PLACEMENT",
)

guardrail_banner(
    "<strong>Lưu ý phương pháp luận khi đọc bảng kết quả:</strong> Chính sách <code>Decentralized</code> giả định đặt hàng trực tiếp tại kênh nên không trả cước vận chuyển "
    "Dual-Hub (<code>E_fulfill = 0</code>). Khi so sánh công bằng hiệu quả gộp rủi ro tồn kho giữa <code>VP</code> và <code>Dec</code>, hãy quan sát cột "
    "<code>TEC_ex_fulfill = TEC − E_fulfill</code> bên cạnh tổng chi phí <code>TEC</code>. Cột <code>beta_used</code> là tỷ lệ tới hạn mục tiêu (Critical Ratio), còn <code>fill_rate</code> là mức phục vụ thực hiện."
)

baseline = st.session_state.get("baseline", "preclean")
opts = sku_options(baseline)
default_skus = [s for s in (st.session_state.get("sku_list") or S4) if s in opts] or list(S4)

OMEGA_LABELS = {
    "hist": "hist — Kịch bản Lịch sử 1 tháng (03a chuẩn Wang & Minner)",
    "hist_coverage": "hist_coverage — Kịch bản Lịch sử tích lũy cửa sổ L tháng (03b)",
    "fcst_joint": "fcst_joint — Dự báo M4 + Vector Residual Bootstrap 1 tháng (03c)",
    "fcst_coverage": "fcst_coverage — Dự báo M4 + Residual tích lũy cửa sổ L tháng (03d)",
}

with st.container():
    st.subheader("⚙️ Cấu hình Tham số Giải Mô hình Tối ưu Tồn kho")
    r1c1, r1c2, r1c3 = st.columns([1.8, 1.4, 0.7])
    with r1c1:
        skus = st.multiselect(
            "Danh sách SKU đưa vào tối ưu hóa (Tối đa 40 SKU):",
            options=opts,
            default=default_skus,
            format_func=lambda s: format_sku(s, baseline),
        )
    with r1c2:
        omega = st.selectbox(
            "Cơ chế sinh kịch bản nhu cầu (Ω):",
            options=["hist", "hist_coverage", "fcst_joint", "fcst_coverage"],
            format_func=lambda o: OMEGA_LABELS.get(o, o),
        )
    with r1c3:
        coverage = omega.endswith("coverage")
        L = st.selectbox("Cửa sổ phủ L (tháng):", [3, 4, 5, 6], disabled=not coverage)

    r2c1, r2c2, r2c3, r2c4 = st.columns([1.6, 0.7, 0.7, 1.0])
    with r2c1:
        policies = st.multiselect(
            "Chính sách tồn kho cần giải:",
            list(POLICIES),
            default=list(POLICIES),
            format_func=lambda p: POLICY_LABEL[p],
        )
    with r2c2:
        h_mult = st.number_input("Hệ số nhân chi phí lưu kho (h_mult)", min_value=0.1, max_value=5.0, value=1.0, step=0.25)
    with r2c3:
        beta_mult = st.number_input("Hệ số nhân tỷ lệ tới hạn (beta_mult)", min_value=0.5, max_value=1.5, value=1.0, step=0.05)
    with r2c4:
        beta_note = st.text_input("Ghi đè β riêng từng SKU (VD: W100133=0.97)", placeholder="Để trống = lấy theo bảng lớp ABC–XYZ")

    r3c1, r3c2, r3c3 = st.columns([1.2, 1.0, 1.6])
    with r3c1:
        budget_on = st.checkbox("Bật ràng buộc trần ngân sách vốn lưu động (B)")
        budget_mult = st.number_input("Hệ số ngân sách (budget_mult × COGS × E[D])", min_value=0.1, max_value=5.0, value=1.0, step=0.1, disabled=not budget_on)
    with r3c2:
        legacy = st.checkbox("Mạng 6-buffer cũ (Legacy 03 — chỉ dùng đối chiếu)")
    with r3c3:
        fcst_runs = list_runs(kind="forecast")
        fcst_ids = [r["run_id"] for r in fcst_runs]
        cur_fcst = st.session_state.get("fcst_run")
        if cur_fcst and cur_fcst not in fcst_ids:
            fcst_ids.insert(0, cur_fcst)
        if omega.startswith("fcst"):
            if fcst_ids:
                fcst_run = st.selectbox("Chọn phiên Dự báo (`forecast_run_id`):", options=fcst_ids, index=0)
            else:
                fcst_run = st.text_input("Nhập `forecast_run_id` (hoặc chạy Trang 3 trước):", value=cur_fcst or "")
        else:
            fcst_run = cur_fcst

overrides: dict[str, float] = {}
if beta_note.strip():
    for part in beta_note.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                overrides[k.strip()] = float(v)
            except ValueError:
                pass

btn_col1, btn_col2 = st.columns([1.2, 1.8])
solve_clicked = btn_col1.button("🚀 Giải Tối ưu y* trên Train (T1–T8) & Chấm OOS (T9–T12)", type="primary", use_container_width=True)

inv_runs = list_runs(kind="inventory")
with btn_col2:
    if inv_runs and not solve_clicked:
        load_prev = st.selectbox(
            "Hoặc nạp lại một phiên giải Tồn kho đã lưu:",
            options=["— Phiên hiện tại —"] + [r["run_id"] for r in inv_runs[:15]],
            index=0,
        )
        if load_prev != "— Phiên hiện tại —":
            try:
                loaded = read_run(load_prev)
                if loaded.get("results"):
                    st.session_state["inv_run"] = loaded["results"]
            except Exception:
                pass

if solve_clicked:
    if not skus or not policies:
        st.error("Vui lòng chọn ít nhất 1 SKU và 1 chính sách tồn kho.")
        st.stop()
    mu = resid = None
    if omega.startswith("fcst"):
        if not fcst_run:
            st.error("Chế độ kịch bản Dự báo (`fcst_*`) yêu cầu phải có `forecast_run_id`. Hãy chạy Trang 3 (Lab Dự báo) trước!")
            st.stop()
        mu, resid = load_forecast_tables(fcst_run, int(L) if omega == "fcst_coverage" else None)
    with st.spinner(f"Bộ giải COIN-OR CBC đang giải quy hoạch LP/MILP cho {len(skus)} SKU × {len(policies)} chính sách…"):
        out = solve_protocol(
            sku_list=skus,
            omega_type=omega,
            L=int(L) if coverage else None,
            policies=policies,
            beta_overrides=overrides,
            h_mult=float(h_mult),
            beta_mult=float(beta_mult),
            budget_on=budget_on,
            budget_mult=float(budget_mult),
            network="legacy_6buffer" if legacy else "dual_hub",
            baseline=baseline,
            mu=mu,
            residuals=resid,
        )
    st.session_state["inv_run"] = out
    st.success(f"Đã giải xong! Mã phiên tồn kho: `{out['run_id']}`")

out = st.session_state.get("inv_run")
if not out:
    st.info("Chưa có phiên giải trong bộ nhớ. Hãy bấm **Giải Tối ưu y*** ở trên để bắt đầu.")
    st.stop()

if not out.get("oos_available"):
    st.warning(out.get("oos_note") or "Không có cửa sổ OOS hợp lệ (ví dụ L=6 trên tháng 9–12). Hệ thống chỉ xuất kết quả Train, không điền số giả.")

rows = pd.DataFrame(out.get("rows") or [])
if rows.empty:
    st.stop()

# Summary KPI row
target_split = "oos" if (rows["split"] == "oos").any() else "train"
sub_kpi = rows[rows["split"] == target_split]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Mã phiên (Run ID)", out.get("run_id", "")[:18], f"Mạng: {out.get('network_note', '')}")
vp_sub = sub_kpi[sub_kpi["policy"] == "Kieu-DDA-VP"]
dec_sub = sub_kpi[sub_kpi["policy"] == "Decentralized"]
if len(vp_sub):
    k2.metric(f"Tổng TEC ({target_split.upper()} • VP)", vnd(vp_sub["TEC"].sum()), f"Trung vị Fill Rate: {pct(vp_sub['fill_rate'].median())}")
    k3.metric("Tổng Vốn lưu động WC (VP)", vnd(vp_sub["WC"].sum()), f"Tổng tồn y*: {vp_sub['total_y'].sum():,.0f} chai")
if len(vp_sub) and len(dec_sub):
    diff_ex = float(dec_sub["TEC_ex_fulfill"].sum() - vp_sub["TEC_ex_fulfill"].sum())
    k4.metric("Tiết kiệm TEC_ex_fulfill (VP so với Dec)", vnd(diff_ex), f"Phân vùng: {target_split.upper()}")

tab_res, tab_place, tab_cost, tab_rq2, tab_po = st.tabs([
    "📋 1. Bảng Kết quả Chi tiết (Train & OOS)",
    "🏗️ 2. Vị trí Đặt Tồn kho (y*) & Độ tập trung (CL / SA)",
    "💰 3. Cấu trúc Chi phí (Holding / Penalty / Fulfillment)",
    "⚖️ 4. Đối đầu Trực tiếp Lịch sử vs Dự báo (RQ2 Live)",
    "📦 5. Kế hoạch Đặt hàng NCC & Chia Kênh (PO & MOQ 600)",
])

with tab_res:
    split_filter = st.radio("Lọc theo phân vùng:", ["Tất cả (Train + OOS)", "Chỉ OOS (Kiểm định T9–T12)", "Chỉ Train (Huấn luyện T1–T8)"], horizontal=True)
    view_rows = rows.copy()
    if split_filter.startswith("Chỉ OOS"):
        view_rows = view_rows[view_rows["split"] == "oos"]
    elif split_filter.startswith("Chỉ Train"):
        view_rows = view_rows[view_rows["split"] == "train"]

    cols_order = [
        c for c in [
            "sku", "item_name", "ABC", "policy", "split", "status",
            "beta_target", "beta_used", "fill_rate",
            "TEC", "TEC_ex_fulfill", "cost_per_unit",
            "E_holding", "E_penalty", "E_fulfill",
            "WC", "total_y", "CL_stock", "CL_open", "SA", "note"
        ] if c in view_rows.columns
    ]
    st.dataframe(
        view_rows[cols_order],
        use_container_width=True,
        hide_index=True,
        column_config={
            "sku": "Mã SKU",
            "item_name": "Tên sản phẩm",
            "ABC": "Lớp",
            "policy": "Chính sách",
            "split": "Phân vùng",
            "status": "Trạng thái LP",
            "beta_target": st.column_config.NumberColumn("β chuẩn lớp", format="%.2f"),
            "beta_used": st.column_config.NumberColumn("β đã dùng", format="%.2f"),
            "fill_rate": st.column_config.NumberColumn("Fill Rate thực tế", format="%.3f"),
            "TEC": st.column_config.NumberColumn("TEC (VND)", format="%,.0f"),
            "TEC_ex_fulfill": st.column_config.NumberColumn("TEC trừ cước (VND)", format="%,.0f"),
            "cost_per_unit": st.column_config.NumberColumn("Cost/Unit (VND/chai)", format="%,.0f"),
            "E_holding": st.column_config.NumberColumn("CP Lưu kho (VND)", format="%,.0f"),
            "E_penalty": st.column_config.NumberColumn("CP Phạt thiếu (VND)", format="%,.0f"),
            "E_fulfill": st.column_config.NumberColumn("CP Vận chuyển (VND)", format="%,.0f"),
            "WC": st.column_config.NumberColumn("Vốn lưu động WC (VND)", format="%,.0f"),
            "total_y": st.column_config.NumberColumn("Tổng Base-stock y* (chai)", format="%,.1f"),
            "CL_stock": st.column_config.NumberColumn("CL_stock (Tỷ trọng CIW)", format="%.3f"),
            "CL_open": st.column_config.NumberColumn("CL_open", format="%.3f"),
            "SA": st.column_config.NumberColumn("SA (Tỷ lệ xuất từ CIW)", format="%.3f"),
        },
    )

with tab_place:
    train_rows = rows[rows["split"] == "train"].copy()
    if "y_star" in train_rows.columns and len(train_rows):
        place_records = []
        for r in train_rows.itertuples(index=False):
            y_dict = r.y_star if isinstance(r.y_star, dict) else {}
            if r.policy == "Decentralized":
                for ch_k, val in y_dict.items():
                    place_records.append({
                        "sku": r.sku,
                        "item_name": getattr(r, "item_name", r.sku),
                        "policy": r.policy,
                        "location": f"Kênh {ch_k}",
                        "y": float(val or 0.0),
                    })
            else:
                for loc in ("CIW", "Hub_Retail", "Hub_B2B"):
                    place_records.append({
                        "sku": r.sku,
                        "item_name": getattr(r, "item_name", r.sku),
                        "policy": r.policy,
                        "location": loc,
                        "y": float(y_dict.get(loc, 0.0) or 0.0),
                    })
        place_df = pd.DataFrame(place_records)
        dual_place = place_df[place_df["policy"] != "Decentralized"]
        if len(dual_place):
            st.markdown("#### Phân bổ Định mức Tồn kho Mục tiêu $y^*$ tại 3 Nút Dual-Hub (`CIW`, `Hub_Retail`, `Hub_B2B`)")
            p_chart = (
                alt.Chart(dual_place)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("sku:N", title="Mã SKU"),
                    xOffset=alt.XOffset("policy:N"),
                    y=alt.Y("y:Q", title="Lượng tồn kho mục tiêu y* (chai)"),
                    color=alt.Color("location:N", title="Vị trí nút kho (Hub)"),
                    tooltip=[
                        alt.Tooltip("sku:N", title="SKU"),
                        alt.Tooltip("item_name:N", title="Tên rượu"),
                        alt.Tooltip("policy:N", title="Chính sách"),
                        alt.Tooltip("location:N", title="Vị trí kho"),
                        alt.Tooltip("y:Q", title="Base-stock y* (chai)", format=",.1f"),
                    ],
                )
                .properties(height=350)
            )
            st.altair_chart(p_chart, use_container_width=True)

        st.markdown("#### Bảng Chỉ số Tập trung Tồn kho (`CL_stock`, `CL_open`) & Phân bổ Luồng hàng (`SA`)")
        cl_cols = [c for c in ["sku", "item_name", "ABC", "policy", "split", "total_y", "CL_stock", "CL_open", "SA", "WC"] if c in rows.columns]
        st.dataframe(rows[rows["policy"] != "Decentralized"][cl_cols], use_container_width=True, hide_index=True)

with tab_cost:
    st.markdown(f"#### Cơ cấu 3 Thành phần Chi phí trên tập `{target_split.upper()}` (`E_holding` + `E_penalty` + `E_fulfill`)")
    cost_melt = sub_kpi.melt(
        id_vars=["sku", "policy"],
        value_vars=[c for c in ["E_holding", "E_penalty", "E_fulfill"] if c in sub_kpi.columns],
        var_name="Thành phần chi phí",
        value_name="Chi phí (VND)",
    )
    cost_melt["Thành phần chi phí"] = cost_melt["Thành phần chi phí"].map({
        "E_holding": "1. Chi phí lưu kho (E_holding)",
        "E_penalty": "2. Chi phí phạt thiếu hàng (E_penalty)",
        "E_fulfill": "3. Chi phí vận chuyển nội địa (E_fulfill)",
    })
    cost_melt["sku_policy"] = cost_melt["sku"] + " • " + cost_melt["policy"].str.replace("Kieu-DDA-", "")
    c_stack = (
        alt.Chart(cost_melt)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("sku_policy:N", title="SKU • Chính sách", sort=None),
            y=alt.Y("Chi phí (VND):Q", title="Tổng chi phí kỳ vọng TEC (VND)"),
            color=alt.Color(
                "Thành phần chi phí:N",
                scale=alt.Scale(range=["#0284c7", "#ef4444", "#f59e0b"]),
            ),
            tooltip=["sku:N", "policy:N", "Thành phần chi phí:N", alt.Tooltip("Chi phí (VND):Q", format=",.0f")],
        )
        .properties(height=360)
    )
    st.altair_chart(c_stack, use_container_width=True)

with tab_rq2:
    st.markdown("#### So sánh Trực tiếp Kịch bản Lịch sử (`hist`) vs Kịch bản Dự báo (`fcst`) trên Cùng Tập OOS")
    if st.button("⚖️ Chạy Đối đầu Lịch sử vs Dự báo (Live RQ2)", type="primary"):
        if not fcst_run:
            st.error("Cần chọn hoặc chạy một `forecast_run_id` ở Trang 3 trước khi so sánh Hist vs Fcst.")
        else:
            with st.spinner("Đang giải song song nhánh Lịch sử và nhánh Dự báo trên cùng tập SKU…"):
                mu, resid = load_forecast_tables(fcst_run, int(L) if coverage else None)
                cmp = compare_hist_fcst(
                    sku_list=skus,
                    coverage=coverage,
                    L=int(L) if coverage else None,
                    policies=policies,
                    baseline=baseline,
                    network="legacy_6buffer" if legacy else "dual_hub",
                    mu=mu,
                    residuals=resid,
                )
            if not cmp["oos_available"]:
                st.warning(cmp.get("oos_note") or "no window")
            st.caption(cmp["caption"])
            st.dataframe(pd.DataFrame(cmp["rows"]), use_container_width=True, hide_index=True)

with tab_po:
    st.markdown("#### 📦 Kế Hoạch Đặt Hàng Nhà Cung Cấp Về Kho Trung Tâm (CIW) & Phân Bổ 5 Kênh Tiêu Thụ")
    st.markdown(
        "Ràng buộc nghiệp vụ thực tế: **1 thùng = 6 chai** (không xé lẻ thùng) • **MOQ Nhà cung cấp = 600 chai/chuyến hàng** (tương đương 100 thùng nguyên)."
    )

    po_pol_opts = sorted(list(set(rows["policy"].dropna())))
    def_pol = "Kieu-DDA-VP" if "Kieu-DDA-VP" in po_pol_opts else (po_pol_opts[0] if po_pol_opts else "Kieu-DDA-VP")

    po_c1, po_c2, po_c3, po_c4 = st.columns([1.4, 1.0, 1.0, 1.6])
    with po_c1:
        chosen_po_pol = st.selectbox(
            "Cơ sở chính sách tồn kho:",
            options=po_pol_opts,
            index=po_pol_opts.index(def_pol) if def_pol in po_pol_opts else 0,
            format_func=lambda p: POLICY_LABEL.get(p, p),
        )
    with po_c2:
        po_split = st.selectbox("Phân vùng định mức:", options=["train", "oos"], format_func=lambda s: "Train (T1–T8)" if s == "train" else "OOS (T9–T12)")
    with po_c3:
        moq_input = st.number_input("MOQ Nhà cung cấp (chai):", min_value=0, max_value=20000, value=600, step=60)
    with po_c4:
        topup_strat = st.selectbox(
            "Chiến lược xử lý khi thiếu MOQ (< 600 chai):",
            options=["velocity", "proportional", "none"],
            format_func=lambda x: {
                "velocity": "🚀 Ưu tiên hàng bán chạy (Lớp AX/AY & Vòng quay)",
                "proportional": "📊 Phân bổ theo tỷ trọng nhu cầu (Proportional)",
                "none": "⚠️ Giữ nguyên & Báo thiếu MOQ",
            }[x],
        )

    # Filter rows by policy and split
    sel_rows = rows[(rows["policy"] == chosen_po_pol) & (rows["split"] == po_split)].copy()
    if sel_rows.empty:
        # Fallback to train if oos not available
        sel_rows = rows[rows["policy"] == chosen_po_pol].copy()

    # Optional On-hand stock input
    with st.expander("📝 Điều chỉnh Tồn kho Khả dụng Hiện có (On-hand Stock) — Mặc định 0 chai cho đợt đặt hàng mới"):
        st.caption("Nhập số lượng chai hiện có trong kho nếu muốn khấu trừ trước khi đặt hàng mới:")
        stock_cols = st.columns(min(len(skus), 4) or 1)
        on_hand_stocks: dict[str, float] = {}
        for idx, s in enumerate(skus):
            c = stock_cols[idx % len(stock_cols)]
            on_hand_stocks[s] = c.number_input(f"Tồn kho {s} (chai):", min_value=0.0, value=0.0, step=6.0, key=f"on_hand_{s}")

    # Calculate plan
    plan = calculate_procurement_plan(
        sel_rows.to_dict(orient="records"),
        moq=int(moq_input),
        pack_size=6,
        topup_strategy=topup_strat,
        current_stocks=on_hand_stocks,
        baseline=baseline,
    )

    summary = plan["summary"]
    items = plan["items"]

    # Metric Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MOQ Nhà cung cấp", f"{summary['moq_bottles']:,} chai", f"{summary['target_cartons']} thùng")
    m2.metric("Nhu cầu cơ sở", f"{summary['base_bottles']:,} chai", f"{summary['base_cartons']} thùng")
    m3.metric("Bù thiếu MOQ (Top-up)", f"+{summary['topup_cartons'] * 6:,} chai", f"+{summary['topup_cartons']} thùng")
    m4.metric("Tổng Đặt về CIW", f"{summary['final_bottles']:,} chai", f"{summary['final_cartons']} thùng")
    m5.metric("Tổng Giá trị Đơn hàng (PO)", vnd(summary["total_po_amount_vnd"]), f"{len(items)} SKU")

    # Progress bar and status
    prog_ratio = min(1.0, float(summary["final_bottles"]) / max(1, summary["moq_bottles"]))
    st.progress(prog_ratio)

    if summary["moq_met"]:
        st.success(
            f"✅ **Đạt chỉ tiêu MOQ Nhà cung cấp!** Tổng lượng đặt: **{summary['final_bottles']:,} chai** "
            f"(= **{summary['final_cartons']} thùng nguyên**), vượt ngưỡng tối thiểu {summary['moq_bottles']:,} chai. Đủ điều kiện phát lệnh PO."
        )
    else:
        st.warning(
            f"⚠️ **Chưa đủ MOQ Nhà cung cấp!** Hiện có **{summary['final_bottles']:,} chai** ({summary['final_cartons']} thùng), "
            f"còn thiếu **{summary['deficit_bottles']:,} chai** ({summary['deficit_cartons']} thùng) để đủ chuyến hàng {summary['moq_bottles']:,} chai. "
            f"Hãy bật chiến lược *Ưu tiên hàng bán chạy* ở trên hoặc chọn thêm SKU vào giỏ."
        )

    # Table 1: PO to Supplier
    st.markdown("##### 📑 1. Chi Tiết Đơn Đặt Hàng Nhà Cung Cấp Về Kho Trung Tâm (CIW)")
    po_records = []
    for it in items:
        po_records.append({
            "Mã SKU": it["sku"],
            "Tên sản phẩm": it["item_name"],
            "Lớp": it["ABC"],
            "Định mức y* (chai)": round(it["target_y"], 1),
            "Tồn hiện có (chai)": int(it["current_stock"]),
            "Nhu cầu ròng (chai)": round(it["net_requirement"], 1),
            "Thùng cơ sở": it["base_cartons"],
            "Thùng bù MOQ": it["topup_cartons"],
            "Tổng thùng đặt (Cartons)": it["final_cartons"],
            "Tổng chai đặt (Bottles)": it["final_bottles"],
            "Đơn giá COGS (VND)": it["unit_cost_vnd"],
            "Thành tiền PO (VND)": it["po_amount_vnd"],
        })
    df_po = pd.DataFrame(po_records)
    st.dataframe(
        df_po,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Định mức y* (chai)": st.column_config.NumberColumn(format="%.1f"),
            "Nhu cầu ròng (chai)": st.column_config.NumberColumn(format="%.1f"),
            "Thùng cơ sở": st.column_config.NumberColumn(format="%d thùng"),
            "Thùng bù MOQ": st.column_config.NumberColumn(format="+%d thùng"),
            "Tổng thùng đặt (Cartons)": st.column_config.NumberColumn(format="%d thùng"),
            "Tổng chai đặt (Bottles)": st.column_config.NumberColumn(format="%,d chai"),
            "Đơn giá COGS (VND)": st.column_config.NumberColumn(format="%,.0f đ"),
            "Thành tiền PO (VND)": st.column_config.NumberColumn(format="%,.0f đ"),
        },
    )

    po_csv = df_po.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Tải File Đơn Đặt Hàng PO (CSV)",
        data=po_csv,
        file_name="TWV_Purchase_Order_MOQ600.csv",
        mime="text/csv",
    )

    # Table 2: Channel Allocation
    st.markdown("##### 🚚 2. Kế Hoạch Phân Bổ Điều Phối Về 5 Kênh Tiêu Thụ (Dual-Hub Allocation)")
    st.caption("Nguyên tắc: Mỗi kênh nhận số thùng nguyên (bội số 6 chai), không xé lẻ, tổng số thùng các kênh bằng đúng tổng số thùng nhập về.")

    ch_records = []
    for it in items:
        rec = {
            "Mã SKU": it["sku"],
            "Tên sản phẩm": it["item_name"],
            "Lớp": it["ABC"],
            "Tổng thùng": it["final_cartons"],
            "Tổng chai": it["final_bottles"],
        }
        for ch in CHANNELS:
            c_k = it["channel_cartons"].get(ch, 0)
            c_b = it["channel_bottles"].get(ch, 0)
            rec[f"{ch} (Thùng)"] = c_k
            rec[f"{ch} (Chai)"] = c_b
        ch_records.append(rec)

    # Add Total summary row
    tot_rec = {
        "Mã SKU": "TỔNG CỘNG",
        "Tên sản phẩm": f"Toàn bộ {len(items)} SKU",
        "Lớp": "—",
        "Tổng thùng": summary["final_cartons"],
        "Tổng chai": summary["final_bottles"],
    }
    for ch in CHANNELS:
        tot_rec[f"{ch} (Thùng)"] = summary["channel_totals_cartons"].get(ch, 0)
        tot_rec[f"{ch} (Chai)"] = summary["channel_totals_bottles"].get(ch, 0)
    ch_records.append(tot_rec)

    df_ch = pd.DataFrame(ch_records)
    st.dataframe(
        df_ch,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Tổng thùng": st.column_config.NumberColumn(format="%d thùng"),
            "Tổng chai": st.column_config.NumberColumn(format="%,d chai"),
            "RT (Thùng)": st.column_config.NumberColumn(format="%d thùng"),
            "RT (Chai)": st.column_config.NumberColumn(format="%,d chai"),
            "HO (Thùng)": st.column_config.NumberColumn(format="%d thùng"),
            "HO (Chai)": st.column_config.NumberColumn(format="%,d chai"),
            "HC (Thùng)": st.column_config.NumberColumn(format="%d thùng"),
            "HC (Chai)": st.column_config.NumberColumn(format="%,d chai"),
            "GT (Thùng)": st.column_config.NumberColumn(format="%d thùng"),
            "GT (Chai)": st.column_config.NumberColumn(format="%,d chai"),
            "MT (Thùng)": st.column_config.NumberColumn(format="%d thùng"),
            "MT (Chai)": st.column_config.NumberColumn(format="%,d chai"),
        },
    )

    ch_csv = df_ch.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 Tải Bảng Phân Bổ 5 Kênh (CSV)",
        data=ch_csv,
        file_name="TWV_Channel_Allocation_Plan.csv",
        mime="text/csv",
    )

caption(out.get("period_label", ""), out.get("omega_type", ""), out.get("L"), "2024-08", out.get("n_sku", 0))
