"""Kieu integrated pipeline package.

Wave 1: metrics, features, data_io
Wave 2: segmentation, stationarity, scenarios, evaluation
Wave 3: inventory solvers
"""

from kieu_pipeline.metrics import (
    calculate_wape,
    calculate_mape,
    calculate_rmse,
    calculate_bias,
    forecast_metric_summary,
    print_forecast_metrics,
)
from kieu_pipeline.features import add_core_features
from kieu_pipeline.data_io import resolve_data_dir, ensure_file, in_colab, DEFAULT_FILE_IDS
from kieu_pipeline.segmentation import classify_portfolio, flag_outliers, DEFAULT_SERVICE_TARGETS
from kieu_pipeline.stationarity import analyze_stationarity
from kieu_pipeline.scenarios import (
    generate_saa_scenarios,
    saa_from_history,
    saa_from_history_coverage,
    coverage_window_starts,
    saa_from_forecast_joint,
    saa_from_forecast_coverage,
    saa_from_forecast_gaussian,
)
from kieu_pipeline.evaluation import publish_targets, evaluate
from kieu_pipeline.inventory import (
    LOCATIONS, CHANNELS,
    CARRYING_RATE_ANNUAL, BUF_HOLDING_MULT, FULFILL_CIW, FULFILL_BUF,
    default_feasible_arcs, make_holding, make_fulfill,
    solve_kieu_dda_vp, solve_kieu_dda_fp, solve_decentralized_newsvendor,
    DUAL_HUB_LOCATIONS, DUAL_HUB_CHANNELS,
    dual_hub_arcs, dual_hub_holding, dual_hub_fulfill,
    placement_metrics, evaluate_given_y, evaluate_decentralized_given_y,
)

__all__ = [
    "calculate_wape",
    "calculate_mape",
    "calculate_rmse",
    "calculate_bias",
    "forecast_metric_summary",
    "print_forecast_metrics",
    "add_core_features",
    "resolve_data_dir",
    "ensure_file",
    "in_colab",
    "DEFAULT_FILE_IDS",
    "classify_portfolio",
    "flag_outliers",
    "DEFAULT_SERVICE_TARGETS",
    "analyze_stationarity",
    "generate_saa_scenarios",
    "saa_from_history",
    "saa_from_history_coverage",
    "coverage_window_starts",
    "saa_from_forecast_joint",
    "saa_from_forecast_coverage",
    "saa_from_forecast_gaussian",
    "publish_targets",
    "evaluate",
    "LOCATIONS", "CHANNELS",
    "CARRYING_RATE_ANNUAL", "BUF_HOLDING_MULT", "FULFILL_CIW", "FULFILL_BUF",
    "default_feasible_arcs", "make_holding", "make_fulfill",
    "solve_kieu_dda_vp", "solve_kieu_dda_fp", "solve_decentralized_newsvendor",
    "DUAL_HUB_LOCATIONS", "DUAL_HUB_CHANNELS",
    "dual_hub_arcs", "dual_hub_holding", "dual_hub_fulfill",
    "placement_metrics", "evaluate_given_y", "evaluate_decentralized_given_y",
]
