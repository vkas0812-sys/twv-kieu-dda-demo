"""Read-only replay of thesis result CSVs. Column names stay as on disk."""
from __future__ import annotations

import pandas as pd

from api.services.paths import REPLAY_FILES, result_csv


def replay_frame(layer: str, filename: str | None = None) -> pd.DataFrame:
    if layer not in REPLAY_FILES:
        raise ValueError(f"unknown layer {layer}")
    files = REPLAY_FILES[layer]
    name = filename or files[0]
    if name not in files:
        raise ValueError(f"{name} is not a {layer} replay file")
    return pd.read_csv(result_csv(name))


def replay_payload(layer: str, filename: str | None = None, limit: int | None = None) -> dict:
    df = replay_frame(layer, filename)
    view = df if limit is None else df.head(limit)
    return {
        "layer": layer,
        "file": filename or REPLAY_FILES[layer][0],
        "n": int(len(df)),
        "columns": list(df.columns),
        "rows": view.where(view.notna(), None).to_dict(orient="records"),
    }
