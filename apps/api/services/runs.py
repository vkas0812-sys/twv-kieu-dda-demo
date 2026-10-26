"""Run manifests under data/results/app_runs. Never writes 03a–03e."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from api.services.jsonutil import scrub
from api.services.paths import APP_RUNS, APP_VERSION


def new_run_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{kind}-{uuid4().hex[:8]}"


def run_dir(run_id: str) -> Path:
    path = APP_RUNS / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(run_id: str, payload: dict[str, Any]) -> Path:
    body = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "app_version": APP_VERSION,
        **payload,
    }
    path = run_dir(run_id) / "manifest.json"
    path.write_text(json.dumps(scrub(body), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_runs(kind: str | None = None) -> list[dict[str, Any]]:
    if not APP_RUNS.exists():
        return []
    rows = []
    for manifest in sorted(APP_RUNS.glob("*/manifest.json"), reverse=True):
        if manifest.name.startswith("._") or manifest.parent.name.startswith("._"):
            continue
        try:
            item = json.loads(manifest.read_text(encoding="utf-8"))
            if kind is None or item.get("kind") == kind:
                rows.append(item)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
    return rows


def read_run(run_id: str) -> dict[str, Any]:
    path = APP_RUNS / run_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(run_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    extra = APP_RUNS / run_id / "results.json"
    if extra.exists():
        data["results"] = json.loads(extra.read_text(encoding="utf-8"))
    return data


def write_results(run_id: str, payload: dict[str, Any]) -> None:
    path = run_dir(run_id) / "results.json"
    path.write_text(json.dumps(scrub(payload), ensure_ascii=False), encoding="utf-8")
