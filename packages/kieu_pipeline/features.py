"""Feature engineering for weekly demand forecasting (Wave 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd

def add_core_features(
    df: pd.DataFrame,
    group_cols: list = None,
    lag_list: list = None,
    roll_windows: list = None,
) -> pd.DataFrame:
    """
    Feature Engineering cho Demand Forecasting (weekly grain).

    Tạo các nhóm feature:
      1. Calendar / Cyclical   – month, week, sin/cos encoding
      2. Lag features          – demand lịch sử (1 → 52 tuần)
      3. Rolling window        – mean / std / max (shift 1 → no leakage)
      4. Sparsity / Pattern    – zero_ratio, is_zero
      5. Event / Exogenous     – is_tet, months_to_tet (nếu có từ M1)
      6. Interaction           – tet × lag, tet × rolling
      7. Entity IDs            – item_id, channel_id (cho LightGBM)

    Parameters
    ----------
    df : DataFrame
        Phải có cột: date, Item number, Channel, Quantity.
        Có thể đã chứa exogenous từ M1 (Is_Tet_Month, Months_To_Tet, ...).
    group_cols : list, optional
        Cột group cho lag/rolling. Mặc định ["Item number", "Channel"].
    lag_list : list, optional
        Các độ trễ (tuần). Mặc định [1, 2, 4, 8, 12, 52].
    roll_windows : list, optional
        Cửa sổ rolling. Mặc định [4, 8, 12].

    Returns
    -------
    DataFrame với đầy đủ feature, sẵn sàng đưa vào LightGBM.
    """
    df = df.copy()
    group_cols = group_cols or ["Item number", "Channel"]
    lag_list = lag_list or [1, 2, 4, 8, 12, 52]
    roll_windows = roll_windows or [4, 8, 12]

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(group_cols + ["date"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Calendar / Cyclical
    # ------------------------------------------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Global lag (trung bình toàn hệ thống tuần trước)
    df["global_lag1"] = df.groupby("date")["Quantity"].transform("mean").shift(1)

    # ------------------------------------------------------------------
    # 2. Lag features (per SKU × Channel)
    # ------------------------------------------------------------------
    g = df.groupby(group_cols)["Quantity"]
    for lag in lag_list:
        df[f"lag_{lag}"] = g.shift(lag)

    # ------------------------------------------------------------------
    # 3. Rolling window (per group, shift 1 → no leakage)
    # ------------------------------------------------------------------
    for w in roll_windows:
        df[f"roll_mean_{w}"] = g.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"roll_std_{w}"] = g.transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).std()
        )
    df["roll_max_12"] = g.transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).max()
    )

    # ------------------------------------------------------------------
    # 4. Sparsity / Demand pattern
    # ------------------------------------------------------------------
    df["zero_ratio_12"] = g.transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).apply(
            lambda x: (x == 0).mean(), raw=True
        )
    )
    df["zero_ratio"] = g.transform(lambda x: (x == 0).mean())
    df["is_zero"] = (df["Quantity"] == 0).astype(int)

    # ------------------------------------------------------------------
    # 5. Event / Exogenous (dùng cột đã merge từ M1 nếu có)
    # ------------------------------------------------------------------
    if "Is_Tet_Month" in df.columns:
        df["is_tet"] = df["Is_Tet_Month"].fillna(0).astype(int)
    else:
        df["is_tet"] = 0

    if "Months_To_Tet" in df.columns:
        df["months_to_tet"] = df["Months_To_Tet"].fillna(6)
    else:
        df["months_to_tet"] = 6

    # ------------------------------------------------------------------
    # 6. Interaction features (rất mạnh với LightGBM)
    # ------------------------------------------------------------------
    df["tet_x_lag1"] = df["is_tet"] * df["lag_1"].fillna(0)
    df["tet_x_roll_mean_4"] = df["is_tet"] * df["roll_mean_4"].fillna(0)

    # ------------------------------------------------------------------
    # 7. Entity IDs (categorical codes cho tree models)
    # ------------------------------------------------------------------
    df["item_id"] = df["Item number"].astype("category").cat.codes
    df["channel_id"] = df["Channel"].astype("category").cat.codes

    n_new = sum(1 for c in df.columns if c not in ["date", "Item number", "Channel", "Quantity"])
    print(f"add_core_features: +{n_new} feature columns ready")
    return df
