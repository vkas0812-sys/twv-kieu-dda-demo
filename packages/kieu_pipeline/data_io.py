"""Data path resolution and file ensure helpers (Wave 1)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

DEFAULT_FILE_IDS: Dict[str, str] = {
    "sales_data_clean_baseline.parquet": "1PlGJjIH91FtUG2XP51gsArODC0OzZHm9",
    "exogenous_data_2021_2024.parquet": "1n8ttUR6lYvjEupG7909g5mBj39PvsNN_",
    "forecasting_master_features.parquet": "1Qpc4Q3K6ofXCv_uyEbXAQ79ie9v-PG3S",
    "promotions.csv": "1puMqLd2Y9iO8rS41-PZTRJqoX_SxoPsP",
    "COGS.parquet": "1YxcEwEoJlHe-xadlu_NwkYzqky3f61jk",
}


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_data_dir(
    extra_candidates: Optional[list] = None,
    sales_name: str = "sales_data_clean_baseline.parquet",
) -> Path:
    """Find a directory that already contains sales parquet, else ./data."""
    candidates = [
        Path("./data"),
        Path("/content/drive/MyDrive/01_Forecasting_data_lan_2/data"),
    ]
    if extra_candidates:
        candidates = list(extra_candidates) + candidates
    for p in candidates:
        p = Path(p)
        if (p / sales_name).exists():
            print(f"Found data at: {p}")
            return p
    p = Path("./data")
    p.mkdir(exist_ok=True, parents=True)
    print(f"Data folder not found on Drive mount → will use/download into: {p}")
    return p


def ensure_file(
    name: str,
    data_dir: Path,
    file_ids: Optional[Dict[str, str]] = None,
) -> Path:
    """Return path to file; download via gdown if missing and File ID known."""
    data_dir = Path(data_dir)
    target = data_dir / name
    if target.exists():
        return target
    ids = file_ids or DEFAULT_FILE_IDS
    file_id = ids.get(name)
    if not file_id:
        raise FileNotFoundError(f"No File ID for {name}")
    print(f"Downloading {name} via gdown ...")
    try:
        import gdown
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        import gdown
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(target), quiet=False)
    return target
