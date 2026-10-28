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
importlib.reload(_fl_mod)
from api.services.data_store import history_monthly, sku_abc_map, sku_name
from api.services.explainability import (
    FEATURE_DEFINITIONS,
    build_feature_table,
    calculate_acf_pacf,
    calculate_hierarchical_shares,
    calculate_train_correlation,
    compute_shap_values,
    load_lgb_artifact,
)
from api.services.forecast_lab import run_forecast
from api.services.paths import CHANNELS, S4
from api.services.replay import replay_frame
from api.services.runs import list_runs, run_dir
from bootstrap import caption, format_sku, gate, guardrail_banner, page_header, sku_options

gate("TWV — 3. Lab Dự báo Phân tầng (M4)")

page_header(
    "3. Lab Dự báo Nhu cầu Phân tầng theo ABC–XYZ (Module M4)",
    "Huấn luyện đa mô hình (SARIMA, Holt-Winters TES, LightGBM Tweedie, MA_4, Hierarchical) cắt chống rò rỉ tại 2024-08 và trích xuất vector phần dư cho SAA.",
    badge="MODULE 03 • TIERED FORECASTING ENGINE & RESIDUAL EXTRACTION",
)

guardrail_banner(
    "<strong>Quy tắc Champion Phase 3 (Optimized_Best):</strong> Lớp <code>AX</code> → Hybrid (0,5 SARIMA + 0,5 LightGBM); "
    "Lớp <code>AY</code> → Holt-Winters TES; Lớp <code>B</code> → Ensemble LightGBM; Lớp <code>C</code> → MA_4; "
    "Lớp <code>CZ</code> → Hierarchical Top-Down. Không trộn 30% Seasonal Naïve; tuyệt đối không dùng dữ liệu sau tháng 8/2024 khi huấn luyện."
)

baseline = st.session_state.get("baseline", "preclean")
tab_live, tab_replay, tab_explain = st.tabs([
    "⚡ Chạy Dự báo Live & Khám phá Phiên (M4 Engine)",
    "📚 Replay Kết quả Dự báo Chuẩn Luận văn (03c WAPE & 03d Series)",
    "🔬 Giải thích Mô hình & Đặc trưng (Feature Engineering & Explainability)",
])

with tab_live:
    opts = sku_options(baseline)
    default_skus = [s for s in (st.session_state.get("sku_list") or S4) if s in opts] or list(S4)

    c_cfg1, c_cfg2, c_cfg3 = st.columns([2.2, 0.8, 1.0])
    with c_cfg1:
        skus = st.multiselect(
            "Danh sách SKU cần huấn luyện dự báo (Tối đa 40 SKU):",
            options=opts,
            default=default_skus,
            format_func=lambda s: format_sku(s, baseline),
        )
    with c_cfg2:
        cutoff = st.text_input("Mốc cắt chống rò rỉ (Cutoff)", "2024-08")
    with c_cfg3:
        st.write("")
        run_btn = st.button("🚀 Chạy Động cơ Dự báo M4", type="primary", use_container_width=True)

    if run_btn:
        if not skus:
            st.error("Vui lòng chọn ít nhất 1 SKU.")
        else:
            with st.spinner(f"Đang huấn luyện SARIMA, TES, LightGBM Tweedie và Hierarchical cho {len(skus)} SKU × 5 kênh…"):
                out = run_forecast(skus, baseline, cutoff)
            st.session_state["fcst_run"] = out["run_id"]
            st.success(f"Hoàn tất huấn luyện dự báo! Mã phiên: `{out['run_id']}`")

    saved_fcst_runs = list_runs(kind="forecast")
    run_ids = [r["run_id"] for r in saved_fcst_runs]
    cur_run = st.session_state.get("fcst_run")
    if cur_run and cur_run not in run_ids:
        run_ids.insert(0, cur_run)

    if not run_ids:
        st.info("Chưa có phiên chạy dự báo Live nào được lưu. Bấm **Chạy Động cơ Dự báo M4** ở trên hoặc chuyển sang tab **Replay Kết quả Dự báo Chuẩn Luận văn**.")
    else:
        picked_run = st.selectbox(
            "Chọn phiên dự báo (`forecast_run_id`) để phân tích & truyền sang Lab Tồn kho:",
            options=run_ids,
            index=0,
        )
        st.session_state["fcst_run"] = picked_run

        fc_path = run_dir(picked_run) / "forecast.csv"
        wape_path = run_dir(picked_run) / "wape.csv"
        resid_path = run_dir(picked_run) / "residuals.csv"

        if fc_path.exists():
            fc = pd.read_csv(fc_path)
            wape_df = pd.read_csv(wape_path) if wape_path.exists() else pd.DataFrame()
            resid_df = pd.read_csv(resid_path) if resid_path.exists() else pd.DataFrame()

            avail_skus = list(fc["sku"].astype(str).unique())
            avail_chs = sorted(fc["channel"].astype(str).unique())

            sel1, sel2, sel3 = st.columns([1.4, 0.8, 1.4])
            with sel1:
                sku_view = st.selectbox("Xem chuỗi SKU:", avail_skus, format_func=lambda s: format_sku(s, baseline))
            with sel2:
                ch_view = st.selectbox("Kênh phân phối:", avail_chs)
            with sel3:
                models_avail = sorted(fc["model"].unique().tolist())
                default_models = [m for m in ["Optimized_Best", "LightGBM", "SARIMA", "TES", "MA_4"] if m in models_avail]
                models_view = st.multiselect("Mô hình so sánh trên đồ thị:", models_avail, default=default_models or ["Optimized_Best"])

            sub_fc = fc[(fc["sku"] == sku_view) & (fc["channel"] == ch_view)].sort_values(["year", "month"]).copy()
            sub_fc["period"] = sub_fc["year"].astype(str) + "-" + sub_fc["month"].astype(str).str.zfill(2)

            # Build long dataframe for Altair plot (Actual + selected models)
            actual_series = (
                sub_fc[sub_fc["model"] == "Optimized_Best"][["period", "actual", "split"]]
                .rename(columns={"actual": "value"})
                .assign(series="1. Thực tế (Actual)")
            )
            pred_series = (
                sub_fc[sub_fc["model"].isin(models_view)][["period", "yhat", "model", "split"]]
                .dropna(subset=["yhat"])
                .rename(columns={"yhat": "value", "model": "series"})
            )
            plot_df = pd.concat([actual_series, pred_series], ignore_index=True)

            line_chart = (
                alt.Chart(plot_df)
                .mark_line(point=True, strokeWidth=2.4)
                .encode(
                    x=alt.X("period:N", title="Kỳ (Năm - Tháng)", sort=None),
                    y=alt.Y("value:Q", title="Sản lượng (chai)"),
                    color=alt.Color("series:N", title="Chuỗi"),
                    strokeDash=alt.condition(
                        alt.datum.series == "1. Thực tế (Actual)",
                        alt.value([1, 0]),
                        alt.value([5, 3]),
                    ),
                    tooltip=[
                        alt.Tooltip("period:N", title="Kỳ"),
                        alt.Tooltip("series:N", title="Mô hình"),
                        alt.Tooltip("value:Q", title="Sản lượng (chai)", format=",.1f"),
                        alt.Tooltip("split:N", title="Phân vùng"),
                    ],
                )
            )
            cutoff_rule = (
                alt.Chart(pd.DataFrame({"period": [cutoff]}))
                .mark_rule(color="#ef4444", strokeWidth=2, strokeDash=[4, 4])
                .encode(x="period:N")
            )
            st.altair_chart((line_chart + cutoff_rule).properties(height=360), use_container_width=True)
            st.caption(f"Đường dọc đỏ đứt nét: Mốc cắt `{cutoff}`. Bên phải là vùng kiểm định độc lập Out-of-Sample (`oos_9_12`).")

            col_tags, col_res = st.columns([1.15, 1.0])
            with col_tags:
                st.markdown("#### Quy tắc gán mô hình `Optimized_Best` theo SKU × Kênh")
                best_rows = fc[fc["model"] == "Optimized_Best"]
                tag_cols = [c for c in ["sku", "item_name", "ABC", "channel", "rule_tag", "sarima_ok", "lgb_ok"] if c in best_rows.columns]
                st.dataframe(best_rows[tag_cols].drop_duplicates(), use_container_width=True, hide_index=True)
            with col_res:
                st.markdown(r"#### Phần dư In-Sample (`residuals.csv`) dùng để dựng kịch bản $\Omega$ SAA")
                st.dataframe(resid_df.head(20), use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Tải file Residual SAA (CSV)",
                    resid_df.to_csv(index=False),
                    file_name=f"{picked_run}_residuals.csv",
                    mime="text/csv",
                )

            if len(wape_df):
                st.markdown("#### Bảng Đánh giá Sai số Dự báo (WAPE · MAPE · RMSE · Bias)")
                st.dataframe(
                    wape_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "wape": st.column_config.NumberColumn("WAPE", format="%.3f"),
                        "mape": st.column_config.NumberColumn("MAPE", format="%.3f"),
                        "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
                        "bias": st.column_config.NumberColumn("Bias", format="%+.2f"),
                    },
                )
            caption("tháng", f"run {picked_run} (residual sẵn sàng cho M5)", None, cutoff, int(fc["sku"].nunique()))

with tab_replay:
    st.subheader("📊 Replay 03c — Bảng Sai số WAPE theo Mô hình & Phân khúc (`03c_wape_sku.csv`)")
    try:
        w03c = replay_frame("03c", "03c_wape_sku.csv").copy()
        w03c["item_name"] = w03c["sku"].map(sku_name)
        s4_only = st.checkbox("Chỉ lọc 4 mã đại diện S4", value=True, key="_w03c_s4")
        if s4_only:
            w03c = w03c[w03c["sku"].isin(S4)]

        oos_wape = w03c[(w03c["split"] == "test_9_12") & (w03c["channel"] == "ALL")].copy()
        if len(oos_wape):
            w_chart = (
                alt.Chart(oos_wape)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("sku:N", title="Mã SKU (Kênh ALL — Tập kiểm định T9–T12/2024)"),
                    xOffset=alt.XOffset("model:N"),
                    y=alt.Y("wape:Q", title="WAPE (Weighted Absolute Percentage Error)"),
                    color=alt.Color("model:N", title="Mô hình"),
                    tooltip=[
                        alt.Tooltip("sku:N", title="SKU"),
                        alt.Tooltip("item_name:N", title="Tên rượu"),
                        alt.Tooltip("ABC:N", title="Lớp"),
                        alt.Tooltip("model:N", title="Mô hình"),
                        alt.Tooltip("wape:Q", title="WAPE", format=".3f"),
                        alt.Tooltip("rmse:Q", title="RMSE", format=".2f"),
                        alt.Tooltip("bias:Q", title="Bias", format="+.2f"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(w_chart, use_container_width=True)

        st.dataframe(w03c, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(f"Không đọc được 03c_wape_sku.csv: {exc}")

    st.subheader("📈 Replay 03d — Chuỗi Dự báo Thực tế vs Dự đoán (`03d_yhat_source.csv`)")
    try:
        y03d = replay_frame("03d", "03d_yhat_source.csv").copy()
        r_sku = st.selectbox("Chọn SKU trong 03d:", sorted(y03d["sku"].unique()), format_func=lambda s: format_sku(s, baseline))
        r_ch = st.selectbox("Chọn Kênh trong 03d:", sorted(y03d["channel"].unique()))
        sub_d = y03d[(y03d["sku"] == r_sku) & (y03d["channel"] == r_ch)].sort_values(["year", "month"]).copy()
        sub_d["period"] = sub_d["year"].astype(str) + "-" + sub_d["month"].astype(str).str.zfill(2)
        m_d = sub_d.melt(
            id_vars=["period"],
            value_vars=[c for c in ["actual", "yhat_best", "yhat_lgb", "yhat_sn"] if c in sub_d.columns],
            var_name="Chuỗi",
            value_name="Sản lượng",
        )
        c_d = (
            alt.Chart(m_d.dropna(subset=["Sản lượng"]))
            .mark_line(point=True, strokeWidth=2.2)
            .encode(
                x=alt.X("period:N", title="Kỳ (2024)", sort=None),
                y=alt.Y("Sản lượng:Q", title="Sản lượng (chai)"),
                color=alt.Color("Chuỗi:N"),
                tooltip=["period:N", "Chuỗi:N", alt.Tooltip("Sản lượng:Q", format=",.1f")],
            )
            .properties(height=320)
        )
        st.altair_chart(c_d, use_container_width=True)
    except Exception as exc:
        st.info(f"Không đọc được 03d_yhat_source.csv: {exc}")

with tab_explain:
    st.subheader("🔬 Giải thích Kỹ thuật Đặc trưng & Cơ chế Mô hình Dự báo (M4 Engine)")

    st.markdown(
        """
        > **Kiến trúc Kỹ thuật Đặc trưng Chuẩn Luận văn (Phase 3):**
        > - **Đặc trưng Tháng Thực tế (Live LightGBM):** Bao gồm đúng 6 đặc trưng chuỗi thời gian: `lag1`, `lag2`, `lag3`, `lag12`, `roll4`, và `month`.
        > - **Chống Rò rỉ Dữ liệu Tuyệt đối (No OOS Leakage):** Mọi tính toán đặc trưng trượt, độ tương quan Pearson, ACF/PACF, tỷ trọng kênh và huấn luyện mô hình đều được cố định tại mốc cắt **`cutoff`** (mặc định: `2024-08`). Vùng `2024-09` đến `2024-12` là tập kiểm định ngoài mẫu (Out-of-Sample / `oos_9_12`).
        > - *(Lưu ý: Module `features.py` theo tuần thuộc một luồng nghiên cứu thử nghiệm riêng và không được đưa vào pipeline mô hình Live này).*
        """
    )

    exp_c1, exp_c2, exp_c3, exp_c4 = st.columns([1.3, 1.3, 0.8, 1.6])

    # Available runs
    saved_runs = list_runs(kind="forecast")
    exp_run_options = [r["run_id"] for r in saved_runs]
    active_fcst_run = st.session_state.get("fcst_run")
    if active_fcst_run and active_fcst_run not in exp_run_options:
        exp_run_options.insert(0, active_fcst_run)

    with exp_c1:
        exp_run = st.selectbox(
            "Phiên chạy dự báo (Run):",
            options=exp_run_options if exp_run_options else ["(Chưa có run đã lưu - Dùng dữ liệu lịch sử)"],
            index=0 if exp_run_options else 0,
            key="_exp_run_sel",
        )

    # Selected run directory / files
    exp_fc_df = None
    exp_resid_df = None
    run_has_artifacts = False

    if exp_run_options and exp_run != "(Chưa có run đã lưu - Dùng dữ liệu lịch sử)":
        r_dir = run_dir(exp_run)
        if (r_dir / "forecast.csv").exists():
            try:
                exp_fc_df = pd.read_csv(r_dir / "forecast.csv")
            except Exception:
                pass
        if (r_dir / "residuals.csv").exists():
            try:
                exp_resid_df = pd.read_csv(r_dir / "residuals.csv")
            except Exception:
                pass
        if (r_dir / "lgb_artifacts.json").exists():
            run_has_artifacts = True

    # SKU options
    if exp_fc_df is not None and "sku" in exp_fc_df.columns:
        exp_sku_list = list(exp_fc_df["sku"].astype(str).unique())
    else:
        exp_sku_list = sku_options(baseline)

    with exp_c2:
        exp_sku = st.selectbox(
            "Mã SKU cần giải thích:",
            options=exp_sku_list,
            format_func=lambda s: format_sku(s, baseline),
            key="_exp_sku_sel",
        )

    with exp_c3:
        exp_ch = st.selectbox(
            "Kênh bán lẻ:",
            options=CHANNELS,
            index=0,
            key="_exp_ch_sel",
        )

    with exp_c4:
        exp_model_choice = st.selectbox(
            "Mô hình / Phân hệ cần chẩn đoán:",
            options=[
                "LightGBM Tweedie (Đặc trưng, Tương quan, Trọng số & SHAP)",
                "SARIMA (ACF, PACF & Phân phối Phần dư)",
                "Holt-Winters TES (San bằng mũ 3 tham số)",
                "MA_4 (Cơ chế trượt 4 kỳ)",
                "Hierarchical Top-Down (Tỷ trọng kênh lịch sử)",
                "Hybrid (Kết hợp SARIMA + LightGBM)",
            ],
            index=0,
            key="_exp_model_sel",
        )

    exp_cutoff_str = "2024-08"
    exp_cutoff_p = pd.Period(exp_cutoff_str, freq="M")

    # Load historical panel
    hist_raw = history_monthly(baseline, [exp_sku])
    sku_col = "sku" if "sku" in hist_raw.columns else "Item number"
    ch_col = "channel" if "channel" in hist_raw.columns else "Channel"
    sub_hist = hist_raw[(hist_raw[sku_col].astype(str) == str(exp_sku)) & (hist_raw[ch_col].astype(str) == str(exp_ch))]

    full_idx = pd.period_range("2021-01", "2024-12", freq="M")
    y_series = pd.Series(0.0, index=full_idx)
    for r in sub_hist.itertuples(index=False):
        per = pd.Period(f"{int(r.year)}-{int(r.month):02d}", freq="M")
        if per in y_series.index:
            y_series.loc[per] = float(r.Quantity)

    # -------------------------------------------------------------
    # 1. LIGHTGBM TWEEDIE VIEW
    # -------------------------------------------------------------
    if exp_model_choice.startswith("LightGBM"):
        st.markdown("### 1. Khung Kỹ thuật Đặc trưng Tháng (Monthly Feature Engineering)")
        
        col_def, col_info = st.columns([1.6, 1.0])
        with col_def:
            st.markdown("##### 📋 Danh mục 6 Đặc trưng Tháng Chuẩn M4 Engine")
            st.dataframe(pd.DataFrame(FEATURE_DEFINITIONS), use_container_width=True, hide_index=True)
        with col_info:
            st.markdown("##### 📌 Quy tắc Thiết kế Đặc trưng")
            st.info(
                "• **Không rò rỉ (Zero Leakage):** Cửa sổ trượt `roll4` bắt đầu từ `t-1` lùi về trước, không chứa giá trị tháng hiện tại `t`.\n\n"
                "• **Độ trễ mùa vụ `lag12`:** Bắt nhịp chu kỳ tiêu thụ rượu vang tăng vọt vào dịp Tết Nguyên đán và Giáng sinh.\n\n"
                "• **Chỉ số `month` (1..12):** Cung cấp biến định lượng giúp bộ cây phân nhánh theo mùa."
            )

        feat_df = build_feature_table(y_series, exp_cutoff_p)

        # Plot actual vs lag1, lag12, roll4
        st.markdown("##### 📈 Đối sánh Chuỗi Sản lượng Thực tế vs Các Đặc trưng Trễ / Trượt")
        plot_features = feat_df.melt(
            id_vars=["period_str", "split"],
            value_vars=["actual_y", "lag1", "lag12", "roll4"],
            var_name="Đặc trưng",
            value_name="Sản lượng (chai)",
        ).dropna(subset=["Sản lượng (chai)"])

        feat_chart = (
            alt.Chart(plot_features)
            .mark_line(point=True, strokeWidth=2.2)
            .encode(
                x=alt.X("period_str:N", title="Kỳ (Năm - Tháng)", sort=None),
                y=alt.Y("Sản lượng (chai):Q", title="Sản lượng (chai)"),
                color=alt.Color("Đặc trưng:N", title="Chuỗi Đặc trưng"),
                strokeDash=alt.condition(
                    alt.datum["Đặc trưng"] == "actual_y",
                    alt.value([1, 0]),
                    alt.value([4, 4]),
                ),
                tooltip=[
                    alt.Tooltip("period_str:N", title="Kỳ"),
                    alt.Tooltip("Đặc trưng:N", title="Đặc trưng"),
                    alt.Tooltip("Sản lượng (chai):Q", format=",.1f"),
                    alt.Tooltip("split:N", title="Tập"),
                ],
            )
        )
        cutoff_line = (
            alt.Chart(pd.DataFrame({"cutoff": [exp_cutoff_str]}))
            .mark_rule(color="#ef4444", strokeWidth=2, strokeDash=[4, 4])
            .encode(x="cutoff:N")
        )
        st.altair_chart((feat_chart + cutoff_line).properties(height=340), use_container_width=True)
        st.caption(f"Đường đỏ nét đứt: Mốc cắt `{exp_cutoff_str}`. Bên trái là vùng Train dùng để tính toán tương quan và huấn luyện cây.")

        with st.expander("🔍 Xem Bảng Số liệu Toàn bộ Đặc trưng Theo Từng Tháng (2021-01 đến 2024-12)"):
            disp_cols = ["period_str", "actual_y", "lag1", "lag2", "lag3", "lag12", "roll4", "month", "split"]
            st.dataframe(feat_df[disp_cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 2. Ma trận Tương quan Pearson trên Tập Train (Zero OOS Leakage)")
        corr_matrix = calculate_train_correlation(y_series, exp_cutoff_p)
        if len(corr_matrix) > 0:
            corr_actual = corr_matrix["actual_y"].drop("actual_y").reset_index()
            corr_actual.columns = ["Đặc trưng", "Hệ số tương quan (r)"]
            corr_actual["Độ mạnh |r|"] = corr_actual["Hệ số tương quan (r)"].abs()
            corr_actual = corr_actual.sort_values("Độ mạnh |r|", ascending=False)

            c_c1, c_c2 = st.columns([1.2, 1.2])
            with c_c1:
                corr_bar = (
                    alt.Chart(corr_actual)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Đặc trưng:N", sort="-y", title="Đặc trưng"),
                        y=alt.Y("Hệ số tương quan (r):Q", title="Hệ số Pearson (r) với Nhu cầu Thực tế y"),
                        color=alt.condition(
                            alt.datum["Hệ số tương quan (r)"] >= 0,
                            alt.value("#2563eb"),
                            alt.value("#dc2626"),
                        ),
                        tooltip=["Đặc trưng", alt.Tooltip("Hệ số tương quan (r):Q", format=".3f")],
                    )
                    .properties(height=260)
                )
                st.altair_chart(corr_bar, use_container_width=True)
            with c_c2:
                st.markdown("##### Bảng Tương quan Chi tiết")
                st.dataframe(
                    corr_actual.drop(columns=["Độ mạnh |r|"]),
                    use_container_width=True,
                    hide_index=True,
                    column_config={"Hệ số tương quan (r)": st.column_config.NumberColumn(format="%+.3f")},
                )
                st.caption("Ghi chú: Giá trị dương thể hiện mối quan hệ đồng biến; giá trị âm thể hiện nghịch biến.")

            with st.expander("🔍 Xem Ma trận Tương quan Đầy đủ (7 × 7) trên Tập Train"):
                st.dataframe(corr_matrix.style.format("{:+.3f}"), use_container_width=True)
        else:
            st.info("Chuỗi nhu cầu trên tập Train có quá ít dữ liệu hoặc phương sai bằng 0 để tính toán ma trận tương quan.")

        st.markdown("---")
        st.markdown("### 3. Trọng số Đặc trưng LightGBM (Feature Importance: Split vs Gain)")

        art_data = None
        if exp_run and exp_run != "(Chưa có run đã lưu - Dùng dữ liệu lịch sử)":
            art_data = load_lgb_artifact(exp_run, exp_sku, exp_ch)

        if not art_data:
            st.info(
                f"ℹ️ **Chưa có dữ liệu giải thích cho run này (`{exp_run}`).**\n\n"
                f"Run này được tạo trước khi hệ thống lưu artifact mô hình LightGBM, hoặc cặp SKU **`{exp_sku}`** × Kênh **`{exp_ch}`** "
                f"được gán mô hình khác (như SARIMA hoặc TES) theo luật phân tầng ABC–XYZ. "
                f"Bạn có thể nhấn nút **🚀 Chạy Động cơ Dự báo M4** ở Tab 1 để sinh phiên mới kèm đầy đủ artifact giải thích."
            )
        else:
            imp_df = pd.DataFrame({
                "Feature": art_data["features"],
                "Split": art_data["importance_split"],
                "Gain": art_data["importance_gain"],
            })
            tot_split = max(1, sum(art_data["importance_split"]))
            tot_gain = max(1e-6, sum(art_data["importance_gain"]))
            imp_df["Tỷ trọng Split (%)"] = (imp_df["Split"] / tot_split) * 100.0
            imp_df["Tỷ trọng Gain (%)"] = (imp_df["Gain"] / tot_gain) * 100.0

            c_sp, c_gn = st.columns(2)
            with c_sp:
                st.markdown("##### 🌿 Tần suất Phân nhánh (Split Importance)")
                sp_chart = (
                    alt.Chart(imp_df)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#3b82f6")
                    .encode(
                        y=alt.Y("Feature:N", sort="-x", title="Đặc trưng"),
                        x=alt.X("Split:Q", title="Số lần phân nhánh trong Ensemble"),
                        tooltip=["Feature", "Split", alt.Tooltip("Tỷ trọng Split (%):Q", format=".1f%")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(sp_chart, use_container_width=True)
                st.caption("Số lần đặc trưng được chọn để chia nhánh ở mọi cây quyết định.")

            with c_gn:
                st.markdown("##### ⚡ Mức Giảm Mất mát (Gain Importance — Tweedie Deviance)")
                gn_chart = (
                    alt.Chart(imp_df)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#10b981")
                    .encode(
                        y=alt.Y("Feature:N", sort="-x", title="Đặc trưng"),
                        x=alt.X("Gain:Q", title="Tổng mức giảm sai số Tweedie"),
                        tooltip=["Feature", alt.Tooltip("Gain:Q", format=",.2f"), alt.Tooltip("Tỷ trọng Gain (%):Q", format=".1f%")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(gn_chart, use_container_width=True)
                st.caption("Đóng góp thực chất vào việc thu hẹp sai số mục tiêu Tweedie.")

        st.markdown("---")
        st.markdown("### 4. Phân rã Giá trị SHAP (TreeExplainer — Tính toán theo Nhu cầu)")
        st.caption("Phân rã chính xác đóng góp cận biên của từng biến vào đầu ra dự báo theo lý thuyết trò chơi hợp tác. Quá trình tính toán tốn tài nguyên nên chỉ kích hoạt khi bạn yêu cầu.")

        shap_key = f"shap_{exp_run}_{exp_sku}_{exp_ch}"
        c_btn, _ = st.columns([1.5, 2.5])
        with c_btn:
            run_shap = st.button("⚡ Phân tích Tác động SHAP (TreeExplainer)", key=f"btn_{shap_key}")

        if run_shap:
            with st.spinner("Đang khởi tạo SHAP TreeExplainer trên mô hình LightGBM đã fit…"):
                res_shap = compute_shap_values(exp_run, exp_sku, exp_ch, baseline, exp_cutoff_str)
                if res_shap is not None:
                    st.session_state[shap_key] = res_shap
                else:
                    st.warning("Không tìm thấy file mô hình booster đã lưu hoặc dữ liệu không đủ mẫu để tính SHAP.")

        if shap_key in st.session_state:
            shap_data = st.session_state[shap_key]
            c_sh1, c_sh2 = st.columns([1.4, 1.0])
            with c_sh1:
                sh_chart = (
                    alt.Chart(shap_data)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#8b5cf6")
                    .encode(
                        y=alt.Y("Feature:N", sort="-x", title="Đặc trưng"),
                        x=alt.X("Mean_Abs_SHAP:Q", title="Tác động cận biên trung bình (|SHAP value| - chai)"),
                        tooltip=["Feature", alt.Tooltip("Mean_Abs_SHAP:Q", format=".3f"), alt.Tooltip("Impact_Share_Pct:Q", format=".1f%")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(sh_chart, use_container_width=True)
            with c_sh2:
                st.markdown("##### Bảng Tỷ trọng Tác động SHAP")
                st.dataframe(
                    shap_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Mean_Abs_SHAP": st.column_config.NumberColumn("Tác động (|SHAP|)", format="%.3f chai"),
                        "Impact_Share_Pct": st.column_config.NumberColumn("Tỷ trọng (%)", format="%.1f%%"),
                    },
                )
            st.caption("Giá trị trung bình |SHAP| thể hiện mức độ dịch chuyển sản lượng dự báo (chai) gây ra bởi từng biến so với kỳ vọng nền.")

    # -------------------------------------------------------------
    # 2. SARIMA VIEW
    # -------------------------------------------------------------
    elif exp_model_choice.startswith("SARIMA"):
        st.markdown("### Mô hình Chuỗi Thời gian Tham số: SARIMA")
        st.markdown(
            r"""
            **Công thức Tổng quát:**
            $$\Phi_P(B^s)\phi_p(B)(1 - B)^d(1 - B^s)^D y_t = \Theta_Q(B^s)\theta_q(B)\epsilon_t$$
            - **Chu kỳ mùa vụ:** $s = 12$ tháng.
            - **Không gian tham số Auto-ARIMA trên tập Train:** $p \in [0, 2]$, $d \in [0, 1]$, $q \in [0, 2]$, $P \in [0, 1]$, $D \in [0, 1]$, $Q \in [0, 1]$.
            - **Tiêu chuẩn chọn:** Tối thiểu hóa chỉ số thông tin hiệu chỉnh Akaike (AICc) độc lập trên tập Train.
            """
        )

        st.markdown("#### 📊 Hàm Tự Tương Quan (ACF) & Tự Tương Quan Từng Phần (PACF) trên Tập Train")
        acf_df = calculate_acf_pacf(y_series, exp_cutoff_p)

        if len(acf_df) > 0:
            c_acf, c_pacf = st.columns(2)
            with c_acf:
                st.markdown("##### Tự tương quan toàn phần (ACF)")
                acf_chart = (
                    alt.Chart(acf_df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#2563eb")
                    .encode(
                        x=alt.X("Lag:O", title="Độ trễ (Lag k)"),
                        y=alt.Y("ACF:Q", title="Hệ số ACF"),
                        tooltip=["Lag", alt.Tooltip("ACF:Q", format=".3f")],
                    )
                    .properties(height=240)
                )
                sig_bound = float(acf_df["Sig_Bound"].iloc[0])
                bound_df = pd.DataFrame({"y": [sig_bound, -sig_bound]})
                sig_lines = alt.Chart(bound_df).mark_rule(color="#ef4444", strokeDash=[4, 4]).encode(y="y:Q")
                st.altair_chart(acf_chart + sig_lines, use_container_width=True)
                st.caption(f"Đường đỏ nét đứt: Ngưỡng ý nghĩa thống kê 95% (±1.96 / √N = ±{sig_bound:.3f}).")

            with c_pacf:
                st.markdown("##### Tự tương quan từng phần (PACF)")
                pacf_chart = (
                    alt.Chart(acf_df)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#059669")
                    .encode(
                        x=alt.X("Lag:O", title="Độ trễ (Lag k)"),
                        y=alt.Y("PACF:Q", title="Hệ số PACF"),
                        tooltip=["Lag", alt.Tooltip("PACF:Q", format=".3f")],
                    )
                    .properties(height=240)
                )
                st.altair_chart(pacf_chart + sig_lines, use_container_width=True)
                st.caption("PACF giúp xác định bậc tự hồi quy p: nếu PACF cắt cụt sau lag p thì chuỗi tuân theo mô hình AR(p).")

            with st.expander("🔍 Xem Bảng Số liệu ACF & PACF Chi tiết"):
                st.dataframe(acf_df[["Lag", "ACF", "PACF", "Sig_Bound"]], use_container_width=True, hide_index=True)
        else:
            st.info("Chuỗi dữ liệu trên tập Train quá ngắn hoặc phương sai bằng 0 để tính toán ACF/PACF.")

        if exp_resid_df is not None and len(exp_resid_df) > 0:
            st.markdown(r"#### 🎯 Phân tích Phần dư In-Sample (Tập kịch bản $\Omega$ cho SAA)")
            r_sub = exp_resid_df[(exp_resid_df["sku"].astype(str) == str(exp_sku)) & (exp_resid_df["channel"].astype(str) == str(exp_ch))].copy()
            if len(r_sub) > 0 and "residual" in r_sub.columns:
                r_sub["period"] = r_sub["year"].astype(str) + "-" + r_sub["month"].astype(str).str.zfill(2)
                r_chart = (
                    alt.Chart(r_sub)
                    .mark_bar()
                    .encode(
                        x=alt.X("period:N", title="Kỳ"),
                        y=alt.Y("residual:Q", title="Phần dư e_t = y_t - yhat_t (chai)"),
                        color=alt.condition(alt.datum.residual >= 0, alt.value("#3b82f6"), alt.value("#f59e0b")),
                        tooltip=["period", alt.Tooltip("residual:Q", format=",.1f")],
                    )
                    .properties(height=220)
                )
                st.altair_chart(r_chart, use_container_width=True)
                st.caption("Các vector phần dư này được đưa trực tiếp vào ma trận kịch bản SAA (Module M5) để tối ưu hóa chính sách tồn kho không cần giả định phân phối chuẩn.")

    # -------------------------------------------------------------
    # 3. HOLT-WINTERS TES VIEW
    # -------------------------------------------------------------
    elif exp_model_choice.startswith("Holt-Winters"):
        st.markdown("### Mô hình San bằng Mũ Ba Tham số: Holt-Winters TES")
        st.markdown(
            r"""
            **Công thức San bằng Cộng tính (Additive Triple Exponential Smoothing):**
            - **Mức nền (Level):** $L_t = \alpha (y_t - S_{t-m}) + (1 - \alpha)(L_{t-1} + T_{t-1})$
            - **Xu hướng (Trend):** $T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}$
            - **Mùa vụ (Seasonality):** $S_t = \gamma (y_t - L_{t-1} - T_{t-1}) + (1 - \gamma) S_{t-m}$
            - **Dự báo $h$ bước tới:** $\hat{y}_{t+h} = L_t + h \cdot T_t + S_{t+h-m}$
            """
        )
        st.markdown(
            """
            > **Phân khúc Champion:** **Lớp `AY`** (Sản lượng cao, Mùa vụ mạnh).
            >
            > **Giải thích nghiệp vụ:** Các dòng rượu vang nhóm AY tiêu thụ rất mạnh vào mùa Giáng sinh và Tết Nguyên Đán, lặp lại chu kỳ hàng năm ($m = 12$).
            > Cơ chế san bằng mũ 3 tham số lọc bỏ nhiễu ngẫu nhiên đồng thời nắm bắt trọn vẹn đỉnh tiêu thụ mùa vụ mà không cần cấu hình quá nhiều siêu tham số.
            """
        )

    # -------------------------------------------------------------
    # 4. MA_4 VIEW
    # -------------------------------------------------------------
    elif exp_model_choice.startswith("MA_4"):
        st.markdown("### Mô hình Trung bình Trượt 4 Kỳ: Moving Average (MA_4)")
        st.markdown(
            r"""
            **Công thức:**
            $$\hat{y}_t = \frac{1}{4} \sum_{k=1}^4 y_{t-k}$$
            """
        )
        st.markdown(
            """
            > **Phân khúc Champion:** **Lớp `C`** (Sản lượng thấp, nhu cầu thưa thớt/ngắt quãng).
            >
            > **Giải thích nghiệp vụ:** Đối với các dòng rượu phân khúc C có nhiều tháng bán bằng 0 hoặc sản lượng rất nhỏ,
            > các mô hình phức tạp (như LightGBM hay SARIMA) thường bị hiện tượng quá khớp (overfitting) vào nhiễu ngẫu nhiên.
            > Phương pháp MA_4 tạo ra mức dự báo phẳng, an toàn, ổn định và tránh thổi phồng lượng tồn kho không cần thiết.
            """
        )

    # -------------------------------------------------------------
    # 5. HIERARCHICAL TOP-DOWN VIEW
    # -------------------------------------------------------------
    elif exp_model_choice.startswith("Hierarchical"):
        st.markdown("### Mô hình Phân tầng Top-Down: Hierarchical")
        st.markdown(
            r"""
            **Công thức Phân bổ:**
            $$\hat{y}_{sku, ch, t} = \hat{Y}_{sku, ALL, t} \times w_{ch}^{hist}$$
            Trong đó:
            - $\hat{Y}_{sku, ALL, t}$: Sản lượng dự báo gộp toàn công ty của SKU tại thời điểm $t$.
            - $w_{ch}^{hist} = \frac{\sum_{\tau \le cutoff} y_{sku, ch, \tau}}{\sum_{\tau \le cutoff} Y_{sku, ALL, \tau}}$: Tỷ trọng tiêu thụ lịch sử của kênh trên tập Train.
            """
        )
        st.markdown(
            """
            > **Phân khúc Champion:** **Lớp `CZ`** (Sản lượng thấp, biến động cao, kênh ngắt quãng).
            >
            > **Giải thích nghiệp vụ:** Nhu cầu ở từng kênh lẻ quá ngắt quãng để mô hình hóa riêng biệt.
            > Bằng cách gom chung nhu cầu toàn bộ các kênh, tín hiệu tiêu thụ vĩ mô trở nên liên tục và dễ dự báo hơn,
            > sau đó phân bổ ngược lại theo tỷ lệ lịch sử đã được kiểm chứng.
            """
        )

        st.markdown("##### 🍰 Tỷ trọng Tiêu thụ Lịch sử giữa các Kênh trên Tập Train")
        shares = calculate_hierarchical_shares(baseline, exp_sku, exp_cutoff_str)
        c_sh1, c_sh2 = st.columns([1.4, 1.0])
        with c_sh1:
            sh_bar = (
                alt.Chart(shares)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#0ea5e9")
                .encode(
                    y=alt.Y("Channel:N", sort="-x", title="Kênh"),
                    x=alt.X("Share_Pct:Q", title="Tỷ trọng Tiêu thụ (%)"),
                    tooltip=["Channel", alt.Tooltip("Train_Quantity:Q", format=",.0f"), alt.Tooltip("Share_Pct:Q", format=".1f%")],
                )
                .properties(height=220)
            )
            st.altair_chart(sh_bar, use_container_width=True)
        with c_sh2:
            st.dataframe(
                shares,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Train_Quantity": st.column_config.NumberColumn("Sản lượng Train", format="%,.0f chai"),
                    "Share_Pct": st.column_config.NumberColumn("Tỷ trọng (%)", format="%.2f%%"),
                },
            )

    # -------------------------------------------------------------
    # 6. HYBRID VIEW
    # -------------------------------------------------------------
    elif exp_model_choice.startswith("Hybrid"):
        st.markdown("### Mô hình Lai Kết hợp: Hybrid Ensemble")
        st.markdown(
            r"""
            **Công thức:**
            $$\hat{y}_{Hybrid, t} = 0{,}5 \cdot \hat{y}_{SARIMA, t} + 0{,}5 \cdot \hat{y}_{LightGBM, t}$$
            """
        )
        st.markdown(
            """
            > **Phân khúc Champion:** **Lớp `AX`** (Sản lượng lớn nhất, phương sai thấp, dòng sản phẩm huyết mạch).
            >
            > **Giải thích nghiệp vụ:** Nhóm AX đòi hỏi độ chính xác cao nhất vì bất kỳ sai lệch nào cũng ảnh hưởng lớn đến chi phí tồn kho và doanh thu.
            > Sự kết hợp bình đẳng giữa một mô hình thống kê tuyến tính có cấu trúc (SARIMA) và một mô hình máy học phi tuyến mạnh mẽ (LightGBM)
            > tạo ra hiệu ứng triệt tiêu phương sai sai số, giúp chuỗi dự báo ổn định và đạt WAPE ngoài mẫu tối ưu.
            """
        )

        if exp_fc_df is not None:
            sub_hy = exp_fc_df[(exp_fc_df["sku"].astype(str) == str(exp_sku)) & (exp_fc_df["channel"].astype(str) == str(exp_ch))].copy()
            hy_models = [m for m in ["SARIMA", "LightGBM", "Optimized_Best"] if m in sub_hy["model"].unique()]
            if len(hy_models) > 0:
                sub_hy["period"] = sub_hy["year"].astype(str) + "-" + sub_hy["month"].astype(str).str.zfill(2)
                act_df = sub_hy[sub_hy["model"] == "Optimized_Best"][["period", "actual"]].rename(columns={"actual": "value"}).assign(series="Thực tế")
                pred_df = sub_hy[sub_hy["model"].isin(hy_models)][["period", "yhat", "model"]].rename(columns={"yhat": "value", "model": "series"})
                all_df = pd.concat([act_df, pred_df], ignore_index=True)

                hy_chart = (
                    alt.Chart(all_df.dropna(subset=["value"]))
                    .mark_line(point=True, strokeWidth=2.2)
                    .encode(
                        x=alt.X("period:N", title="Kỳ"),
                        y=alt.Y("value:Q", title="Sản lượng (chai)"),
                        color=alt.Color("series:N", title="Thành phần Mô hình"),
                        tooltip=["period", "series", alt.Tooltip("value:Q", format=",.1f")],
                    )
                    .properties(height=320)
                )
                cutoff_hy = (
                    alt.Chart(pd.DataFrame({"cutoff": [exp_cutoff_str]}))
                    .mark_rule(color="#ef4444", strokeWidth=2, strokeDash=[4, 4])
                    .encode(x="cutoff:N")
                )
                st.altair_chart((hy_chart + cutoff_hy), use_container_width=True)
                st.caption("Đối sánh chuỗi thành phần SARIMA, LightGBM và mô hình kết hợp Hybrid (`Optimized_Best`) so với thực tế.")
