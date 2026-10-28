"""Project paths. Thesis CSVs are read-only."""
from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = APP_ROOT.parent
PACKAGES = APP_ROOT / "packages"
DATA = APP_ROOT / "data"
RESULTS = DATA / "results"
APP_RUNS = RESULTS / "app_runs"
PARQUET = DATA / "parquet"

# Resolve 05_Full_project whether located in workspace root (../05_Full_project) or inside APP_ROOT
_THESIS_CANDIDATES = [
    WORKSPACE_ROOT / "05_Full_project" / "3. Database and result file",
    APP_ROOT / "05_Full_project" / "3. Database and result file",
]
THESIS = next((p for p in _THESIS_CANDIDATES if p.exists()), _THESIS_CANDIDATES[0])
THESIS_RESULT = THESIS / "result"

_FIGURES_CANDIDATES = [
    DATA / "figures",
    WORKSPACE_ROOT / "06" / "figures",
    THESIS / "plot",
]
FIGURES_DIR = next((p for p in _FIGURES_CANDIDATES if p.exists()), _FIGURES_CANDIDATES[0])

S4 = ["W100133", "W103028", "W131012", "W100627"]
S4_CLASS = {"W100133": "AX", "W103028": "AY", "W131012": "BY", "W100627": "CZ"}
CHANNELS = ["RT", "GT", "HC", "HO", "MT"]
CHANNEL_LABELS = {
    "RT": "Retail (Cửa hàng bán lẻ)",
    "HO": "Home / E-Commerce (Giao tận nơi)",
    "HC": "HoReCa (Nhà hàng - Khách sạn)",
    "GT": "General Trade (Đại lý truyền thống)",
    "MT": "Modern Trade (Siêu thị / Chuỗi)",
}
CUTOFF = "2024-08"
RANDOM_SEED = 42
APP_VERSION = "1.1.0"
MAX_SKUS = 40

BASELINES = {
    "preclean": PARQUET / "sales_data_clean_baseline.parquet",
    "active": PARQUET / "sales_data_active_baseline.parquet",
    "upload": PARQUET / "sales_upload.parquet",
}
COGS_PATH = PARQUET / "COGS.parquet"

REPLAY_FILES = {
    "03a": ["03a_align_train.csv", "03a_align_oos.csv", "03a_align_pareto.csv"],
    "03b": ["03b_leadtime_L.csv"],
    "03c": ["03c_wape_sku.csv", "03c_hist_vs_fcst.csv"],
    "03d": ["03d_hist_vs_fcst.csv", "03d_fcst_oos.csv", "03d_yhat_source.csv"],
    "03e": ["03e_sa_param.csv", "03e_mc.csv", "03e_jackknife.csv"],
}

OMEGA_TYPES = ("hist", "hist_coverage", "fcst_joint", "fcst_coverage")
POLICIES = ("Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized")


def result_csv(name: str) -> Path:
    linked = RESULTS / name
    if linked.exists():
        return linked
    thesis = THESIS_RESULT / name
    if thesis.exists():
        return thesis
    raise FileNotFoundError(name)


def password_ok(header: str | None) -> bool:
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected:
        return True
    return header == expected
