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
from api.services.data_store import sku_name
from api.services.forecast_lab import run_forecast
from api.services.paths import S4
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
tab_live, tab_replay = st.tabs([
    "⚡ Chạy Dự báo Live & Khám phá Phiên (M4 Engine)",
    "📚 Replay Kết quả Dự báo Chuẩn Luận văn (03c WAPE & 03d Series)",
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
