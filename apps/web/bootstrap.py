"""Shared Streamlit design system, sidebar state manager, and formatting helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "apps"), str(ROOT / "apps" / "web")]

from api.services.data_store import available_baselines, sku_abc_map, sku_label, top_skus_2024
from api.services.paths import APP_VERSION, CUTOFF, S4, S4_CLASS
from api.services.runs import list_runs

POLICY_COLOR = {
    "Kieu-DDA-VP": "#0072B2",
    "Kieu-DDA-FP": "#E69F00",
    "Decentralized": "#009E73",
}
POLICY_LABEL = {
    "Kieu-DDA-VP": "Virtual Pooling (VP — LP)",
    "Kieu-DDA-FP": "Fixed Pooling (FP — MILP)",
    "Decentralized": "Decentralized (Dec — Newsvendor)",
}
BASELINE_LABEL = {
    "preclean": "Pre-clean Baseline (Đã tách Top 3 & 10 B2B Whales)",
    "active": "Active Lifecycle (Chỉ SKU đang hoạt động ổn định)",
    "upload": "Uploaded Parquet (Tập dữ liệu tải lên)",
}

CUSTOM_CSS = """
<style>
    /* Typography & Layout */
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }
    /* Hero / Page Header */
    .twv-hero {
        background: linear-gradient(135deg, #1e1b2e 0%, #3b1d38 55%, #5b1e31 100%);
        color: #ffffff;
        padding: 1.35rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 20px rgba(30, 27, 46, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .twv-hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        color: #ffd166;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.45rem;
    }
    .twv-hero h1 {
        color: #ffffff !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.3rem 0 !important;
        padding: 0 !important;
        line-height: 1.25 !important;
    }
    .twv-hero p {
        color: rgba(255, 255, 255, 0.86) !important;
        font-size: 0.94rem !important;
        margin: 0 !important;
        line-height: 1.45 !important;
    }
    /* Guardrail Banner */
    .twv-guardrail {
        background: #f8fafc;
        border-left: 4px solid #0072B2;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        padding: 0.75rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #1e293b;
    }
    /* Card Container */
    .twv-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .twv-card h4 {
        margin: 0 0 0.35rem 0;
        font-size: 1.0rem;
        color: #0f172a;
    }
    .twv-pill {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.35rem;
    }
    .pill-vp { background: #e0f2fe; color: #0369a1; }
    .pill-fp { background: #fef3c7; color: #b45309; }
    .pill-dec { background: #d1fae5; color: #047857; }
    .pill-abc { background: #f3e8ff; color: #6b21a8; }
    /* Metric styling */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
    }
</style>
"""


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🍷 TWV Planning Desk")
        st.caption(f"Kieu-DDA Optimization Suite • v{APP_VERSION}")
        st.divider()

        baselines = available_baselines()
        cur_base = st.session_state.get("baseline", "preclean")
        if cur_base not in baselines:
            cur_base = "preclean"
        picked_base = st.selectbox(
            "Tập dữ liệu nguồn (Baseline)",
            options=baselines,
            index=baselines.index(cur_base),
            format_func=lambda b: BASELINE_LABEL.get(b, b),
            key="_sidebar_baseline_select",
        )
        st.session_state["baseline"] = picked_base

        cur_skus = st.session_state.get("sku_list") or list(S4)
        st.markdown(f"**Giỏ SKU đang chọn:** `{len(cur_skus)}` mã")
        col_a, col_b = st.columns(2)
        if col_a.button("Chốt 4 mã S4", use_container_width=True, help="Đặt lại giỏ SKU về 4 mã đại diện S4 chuẩn luận văn"):
            st.session_state["sku_list"] = list(S4)
            st.rerun()
        if col_b.button("Top 12 SKU", use_container_width=True, help="Chọn 12 SKU sản lượng cao nhất năm 2024"):
            try:
                st.session_state["sku_list"] = top_skus_2024(picked_base, 12)[:12]
                st.rerun()
            except Exception:
                pass

        fcst_runs = list_runs(kind="forecast")
        inv_runs = list_runs(kind="inventory")
        if not st.session_state.get("fcst_run") and fcst_runs:
            st.session_state["fcst_run"] = fcst_runs[0]["run_id"]

        st.divider()
        st.markdown("**Trạng thái phiên (Session)**")
        st.caption(
            f"• **Cutoff:** `{CUTOFF}` (Train 1–8 / OOS 9–12)\n"
            f"• **Forecast Run:** `{st.session_state.get('fcst_run') or 'Chưa chọn'}`\n"
            f"• **Runs đã lưu:** `{len(fcst_runs)}` dự báo, `{len(inv_runs)}` tồn kho"
        )
        st.markdown(
            "<span class='twv-pill pill-vp'>VP: Xanh dương</span>"
            "<span class='twv-pill pill-fp'>FP: Cam</span>"
            "<span class='twv-pill pill-dec'>Dec: Xanh lá</span>",
            unsafe_allow_html=True,
        )


def gate(page_title: str = "TWV — Hoạch định Tồn kho Đa điểm") -> None:
    st.set_page_config(
        page_title=page_title,
        page_icon="🍷",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        st.session_state["auth"] = True
    if not st.session_state.get("auth"):
        st.title("🔒 Đăng nhập TWV Planning Desk")
        pwd = st.text_input("Mật khẩu truy cập", type="password")
        if st.button("Vào hệ thống", type="primary") and pwd == expected:
            st.session_state["auth"] = True
            st.rerun()
        st.stop()
    if "sku_list" not in st.session_state:
        st.session_state["sku_list"] = list(S4)
    if "baseline" not in st.session_state:
        st.session_state["baseline"] = "preclean"
    _render_sidebar()


def page_header(title: str, subtitle: str, badge: str = "THE WAREHOUSE VIETNAM • KIEU-DDA WORKBENCH") -> None:
    st.markdown(
        f"""
        <div class="twv-hero">
            <div class="twv-hero-badge">{badge}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def guardrail_banner(text: str) -> None:
    st.markdown(f'<div class="twv-guardrail">ℹ️ {text}</div>', unsafe_allow_html=True)


def vnd(value) -> str:
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
        return f"{float(value):,.0f} VND"
    except (TypeError, ValueError):
        return "—"


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    try:
        if value != value:
            return "—"
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def sku_options(baseline: str = "preclean", extra: list[str] | None = None) -> list[str]:
    """Return a safe deduplicated list of SKUs containing S4, session basket, and Top 40."""
    seen: list[str] = []
    for s in list(S4) + list(st.session_state.get("sku_list") or []) + list(extra or []):
        s_str = str(s)
        if s_str and s_str not in seen:
            seen.append(s_str)
    try:
        for s in top_skus_2024(baseline, 40):
            if s not in seen:
                seen.append(s)
    except Exception:
        pass
    return seen


def format_sku(sku: str, baseline: str = "preclean") -> str:
    try:
        abc_map = sku_abc_map(baseline)
    except Exception:
        abc_map = S4_CLASS
    return sku_label(sku, abc_map)


def caption(grain: str, omega: str, L, cutoff: str, n: int) -> None:
    Ltxt = "không dùng (1 kỳ)" if L in (None, "") else f"{L} tháng"
    st.caption(f"📌 **Ngữ cảnh thực nghiệm** — Grain: `{grain}` | Kịch bản Ω: `{omega}` | Cửa sổ phủ L: `{Ltxt}` | Cutoff: `{cutoff}` | Số lượng SKU: `{n}`.")
