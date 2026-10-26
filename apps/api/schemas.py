"""Pydantic models for the planning API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from api.services.paths import OMEGA_TYPES, POLICIES, S4

Omega = Literal["hist", "hist_coverage", "fcst_joint", "fcst_coverage"]
Policy = Literal["Kieu-DDA-VP", "Kieu-DDA-FP", "Decentralized"]
Network = Literal["dual_hub", "legacy_6buffer"]
Baseline = Literal["preclean", "active", "upload"]


class LoadIn(BaseModel):
    baseline: Baseline = "preclean"


class ForecastIn(BaseModel):
    sku_list: list[str] = Field(default_factory=lambda: list(S4))
    baseline: Baseline = "preclean"
    cutoff: str = "2024-08"


class SolveIn(BaseModel):
    sku_list: list[str] = Field(default_factory=lambda: list(S4))
    omega_type: Omega = "hist"
    L: int | None = None
    policies: list[Policy] = Field(default_factory=lambda: list(POLICIES))
    beta_overrides: dict[str, float] = Field(default_factory=dict)
    h_mult: float = 1.0
    beta_mult: float = 1.0
    budget_on: bool = False
    budget_mult: float = 1.0
    network: Network = "dual_hub"
    baseline: Baseline = "preclean"
    forecast_run_id: str | None = None
    evaluate_oos: bool = True


class EvaluateIn(BaseModel):
    omega_type: Omega = "hist_coverage"
    L: int | None = 6
    run_id: str | None = None


class CompareIn(BaseModel):
    sku_list: list[str] = Field(default_factory=lambda: list(S4))
    L: int | None = None
    coverage: bool = False
    policies: list[Policy] = Field(default_factory=lambda: ["Kieu-DDA-VP", "Decentralized"])
    baseline: Baseline = "preclean"
    forecast_run_id: str | None = None
    network: Network = "dual_hub"


class SaIn(BaseModel):
    sku_list: list[str] = Field(default_factory=lambda: ["W100133"])
    baseline: Baseline = "preclean"


class McIn(BaseModel):
    sku: str = "W100133"
    n: int = 8
    baseline: Baseline = "preclean"


class PublishIn(BaseModel):
    run_id: str
    policies: list[Policy] | None = None


def assert_omega(name: str) -> None:
    if name not in OMEGA_TYPES:
        raise ValueError(name)
