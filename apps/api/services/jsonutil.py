"""JSON-safe conversion for solver dicts (sets, numpy)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def scrub(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        return scrub(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [scrub(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)
