"""FastAPI surface. UI may call services directly; these routes are the contract."""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Query

from kieu_pipeline.inventory import (
    DEMO_FULFILL_VND,
    DEMO_HOLDING_RATES,
    DUAL_HUB_CHANNELS,
    DUAL_HUB_LOCATIONS,
    dual_hub_arcs,
)
from kieu_pipeline.scenarios import coverage_window_starts
from kieu_pipeline.segmentation import DEFAULT_SERVICE_TARGETS

from api.schemas import CompareIn, EvaluateIn, ForecastIn, LoadIn, McIn, PublishIn, SaIn, SolveIn
from api.services.data_store import sku_counts
from api.services.forecast_lab import load_forecast_tables, run_forecast
from api.services.inventory_lab import compare_hist_fcst, solve_protocol
from api.services.paths import APP_VERSION, REPLAY_FILES, S4, password_ok
from api.services.portfolio import build_portfolio
from api.services.publish import publish_run
from api.services.replay import replay_payload
from api.services.runs import list_runs, read_run
from api.services.sensitivity import live_jackknife, live_mc, live_sa, replay_jackknife, replay_mc, replay_sa

app = FastAPI(title="TWV inventory placement", version=APP_VERSION)


def _auth(x_app_password: str | None) -> None:
    if not password_ok(x_app_password):
        raise HTTPException(status_code=401, detail="sai APP_PASSWORD")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": APP_VERSION, "ui": "streamlit"}


@app.get("/meta/network")
def meta_network(x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    arcs = sorted([list(a) for a in dual_hub_arcs()])
    return {
        "locations": DUAL_HUB_LOCATIONS,
        "channels": DUAL_HUB_CHANNELS,
        "arcs": arcs,
        "n_arcs": len(arcs),
        "holding_rates_per_period": DEMO_HOLDING_RATES,
        "fulfill_vnd_per_bottle": DEMO_FULFILL_VND,
        "s4": S4,
        "cutoff": "2024-08",
    }


@app.get("/meta/service-targets")
def meta_targets(x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    return {"beta": DEFAULT_SERVICE_TARGETS, "note": "β là critical ratio, không phải ràng buộc CSL"}


@app.post("/data/load")
def data_load(body: LoadIn, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    try:
        return sku_counts(body.baseline)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/portfolio")
def portfolio(
    baseline: str = "preclean",
    year: int = 2024,
    top_n: int = 40,
    x_app_password: str | None = Header(default=None),
) -> dict:
    _auth(x_app_password)
    return build_portfolio(baseline=baseline, year=year, top_n=top_n)


@app.post("/forecast/run")
def forecast_run(body: ForecastIn, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    if len(body.sku_list) > 40:
        raise HTTPException(400, "tối đa 40 SKU")
    return run_forecast(body.sku_list, body.baseline, body.cutoff)


@app.get("/forecast/{run_id}/wape")
def forecast_wape(run_id: str, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    from api.services.runs import run_dir
    import pandas as pd

    path = run_dir(run_id) / "wape.csv"
    if not path.exists():
        raise HTTPException(404, run_id)
    df = pd.read_csv(path)
    return {"run_id": run_id, "n": int(len(df)), "rows": df.to_dict(orient="records")}


@app.get("/forecast/{run_id}/residuals")
def forecast_resid(run_id: str, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    from api.services.runs import run_dir
    import pandas as pd

    path = run_dir(run_id) / "residuals.csv"
    if not path.exists():
        raise HTTPException(404, run_id)
    df = pd.read_csv(path)
    return {"run_id": run_id, "columns": ["sku", "channel", "month", "residual"], "n": int(len(df)), "rows": df.to_dict(orient="records")}


def _tables(body: SolveIn):
    if not str(body.omega_type).startswith("fcst"):
        return None, None
    if not body.forecast_run_id:
        raise HTTPException(400, "omega dự báo cần forecast_run_id")
    L = body.L if body.omega_type == "fcst_coverage" else None
    try:
        return load_forecast_tables(body.forecast_run_id, L)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/inventory/solve")
def inventory_solve(body: SolveIn, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    mu, resid = _tables(body)
    try:
        return solve_protocol(
            sku_list=body.sku_list,
            omega_type=body.omega_type,
            L=body.L,
            policies=list(body.policies),
            beta_overrides=body.beta_overrides,
            h_mult=body.h_mult,
            beta_mult=body.beta_mult,
            budget_on=body.budget_on,
            budget_mult=body.budget_mult,
            network=body.network,
            baseline=body.baseline,
            mu=mu,
            residuals=resid,
            evaluate_oos=body.evaluate_oos,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/inventory/evaluate")
def inventory_evaluate(body: EvaluateIn, x_app_password: str | None = Header(default=None)) -> dict:
    """OOS window check. L=6 on Sep–Dec returns 200 and oos_available false."""
    _auth(x_app_password)
    if body.omega_type.endswith("coverage"):
        L = int(body.L or 0)
        starts = coverage_window_starts(L, 9, 12)
        if not starts:
            return {
                "oos_available": False,
                "oos_note": f"no window — L={L} không đủ tháng 9–12",
                "L": L,
                "split": "oos",
                "rows": [],
            }
    if body.run_id:
        try:
            data = read_run(body.run_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, body.run_id) from exc
        rows = [r for r in (data.get("results") or {}).get("rows", []) if r.get("split") == "oos"]
        return {"oos_available": True, "rows": rows, "run_id": body.run_id}
    return {"oos_available": True, "oos_note": "có cửa sổ OOS", "L": body.L, "split": "oos"}


@app.post("/inventory/compare")
def inventory_compare(body: CompareIn, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    if not body.forecast_run_id:
        raise HTTPException(400, "cần forecast_run_id để so hist với fcst")
    mu, resid = load_forecast_tables(body.forecast_run_id, body.L if body.coverage else None)
    return compare_hist_fcst(
        sku_list=body.sku_list,
        coverage=body.coverage,
        L=body.L,
        policies=list(body.policies),
        baseline=body.baseline,
        network=body.network,
        mu=mu,
        residuals=resid,
    )


@app.post("/sensitivity/sa")
def sensitivity_sa(body: SaIn, live: bool = Query(False), x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    if live:
        return live_sa(body.sku_list, body.baseline)
    return replay_sa()


@app.post("/sensitivity/mc")
def sensitivity_mc(body: McIn, live: bool = Query(False), x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    if live:
        return live_mc(body.sku, body.n, body.baseline)
    payload = replay_mc()
    payload["note"] = "Replay 03e. MC i.i.d. là ablation khi chạy live=true."
    return payload


@app.post("/sensitivity/jackknife")
def sensitivity_jk(body: SaIn, live: bool = Query(False), x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    if live:
        return live_jackknife(body.sku_list[0], body.baseline)
    return replay_jackknife()


@app.get("/runs")
def runs(x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    return {"runs": list_runs()}


@app.get("/runs/{run_id}")
def run_one(run_id: str, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    try:
        return read_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, run_id) from exc


@app.post("/publish")
def publish(body: PublishIn, x_app_password: str | None = Header(default=None)) -> dict:
    _auth(x_app_password)
    try:
        return publish_run(body.run_id, list(body.policies) if body.policies else None)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/replay/{layer}")
def replay(
    layer: str,
    file: str | None = None,
    limit: int | None = None,
    x_app_password: str | None = Header(default=None),
) -> dict:
    _auth(x_app_password)
    if layer not in REPLAY_FILES:
        raise HTTPException(404, f"unknown layer {layer}")
    try:
        return replay_payload(layer, file, limit)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc
