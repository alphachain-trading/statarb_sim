#!/usr/bin/env python3
"""
Track B baseline harness.

A fixed, small-universe config run end-to-end through the shipped
run_panel_batch -> run_from_config -> compute_performance path, writing
trades and summary metrics to a committed text file
(docs/refactor/B_baseline.txt).

Reused, not reimplemented (B_spec.md §9): run_panel_batch with max_steps is
the existing small-universe pattern from tests/test_panel_batch_windows.py
and notebook 01; compute_performance + _METRICS_ORDER (via
result.performance, already computed by run_from_config) is the existing
metrics summary. No new runner, no new metrics code.

Config is fixed and small on purpose: two of the smallest committed sector
universes (energy, 11 equities; materials, 13 equities — B_spec.md §9), an
EQ_EXPANDING residual with a low min_history (20) so the run starts at the
very beginning of each sector's history, and explicit hedge/diagnostics
windows (252) that are wider than what's available that early. That is
deliberate: it is exactly the min_obs defect (B_window_admission.md items
4/5) the baseline is meant to capture before it is fixed.

Usage:
    python scripts/b_baseline_harness.py
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

from src.settings import CANDIDATE_PANELS_ROOT, CONFIG_UNIVERSE, DATA_UNIVERSES, PROJECT_ROOT
from src.candidates.panel_batch import PanelBatchConfig, run_panel_batch
from src.candidates.candidate_panel import load_candidate_panel_result
from src.candidates.pair_candidate_panel_creator import PairSpreadConfig
from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode
from src.residuals.series import compute_and_persist_series
from src.simulator.config import (
    SimulatorConfig,
    DataConfig,
    ZScoreConfig,
    PairSpreadTraderConfig,
    SizingConfig,
    VolSizingConfig,
    RiskManagerConfig,
    RunConfig,
    PerformanceConfig,
    PersistenceConfig,
    discover_sector_data_sources,
)
from src.simulator.simulator_factory import run_from_config, _load_umd
from src.simulator.sweep_defaults import get_default_bundle, merge_defaults
from src.simulator.performance.performance_report import _METRICS_ORDER, _fmt_value

SECTORS = ["energy", "materials"]
PANEL_SUBDIR = "refactor_b_harness"
MAX_STEPS = 40
HEDGE_RATIO_LB = 252
MR_DIAG_LB = 252
# No default anywhere (PairSpreadConfig.min_obs is mandatory; PanelBatchConfig
# no longer supplies one either) — the harness states it explicitly, matching
# the requested lookback: a rolling 252-day fit that ran on 40 observations is
# not a 252-day fit. min_obs=hedge_ratio_lb is the fully-binding choice.
MIN_OBS = 252
# Common start for both sectors' panel walk, chosen near materials' own
# history start (2005-08-11) so the two candidate windows land in the same
# calendar span — energy has ample history there (no min_obs defect for it),
# materials has almost none (the defect this baseline is meant to capture).
# A shared, calendar-close window also keeps the simulate stage's trading-day
# span small instead of spanning the multi-year gap between the two sectors'
# own history starts.
PANEL_START_DATE = "2005-08-15"

OUT_PATH = PROJECT_ROOT / "docs" / "refactor" / "B_baseline.txt"

TRADE_COLS = [
    "trade_id", "group_id", "spread_id",
    "entry_date", "exit_date", "days_open", "direction",
    "pair_notional", "entry_z_score", "exit_z_score",
    "realized_pnl_gross", "realized_pnl_net",
]


def _residual_cfg() -> CausalResidualConfig:
    return CausalResidualConfig(
        mode=ResidualMode.EQ_EXPANDING,
        subtract_risk_free=True,
        min_lb_eq_exp=20,
    )


def _build_panels() -> tuple[float, dict]:
    cfg = PanelBatchConfig(
        residual_configs=[_residual_cfg()],
        hedge_ratio_lb=HEDGE_RATIO_LB,
        mr_diag_lb=MR_DIAG_LB,
        selected_sectors=SECTORS,
        frequency="W-FRI",
        start_date=PANEL_START_DATE,
        max_steps=MAX_STEPS,
        pair_cfg=PairSpreadConfig(hedge_ratio_methods=["pca"], min_obs=MIN_OBS),
        persist_result=True,
        persist_residual_params=True,
        persist_dir_template=PANEL_SUBDIR,
    )
    t0 = time.perf_counter()
    results = run_panel_batch(cfg)
    build_time = time.perf_counter() - t0
    return build_time, results


def _persist_series(sim_config: SimulatorConfig, active_sectors: list[str]) -> None:
    """Mirror run_me.py's _persist_series_multi for this harness's panel dir."""
    panel_dir = Path(CANDIDATE_PANELS_ROOT) / PANEL_SUBDIR
    umd = _load_umd(sim_config.data)

    sources = discover_sector_data_sources(
        panel_dir=panel_dir,
        universe_dir=CONFIG_UNIVERSE,
        selected_sectors=active_sectors,
    )

    for src in sources:
        result = load_candidate_panel_result(out_dir=panel_dir, stem=src.candidate_panel_stem)
        panel = result.panel

        if panel.empty:
            # A sector can legitimately produce zero candidate rows (e.g.
            # min_obs rejecting every pair on every date) — nothing to
            # persist series for. Excluded from active_sectors below too,
            # so run_from_config never tries to load it.
            continue

        if src.residual_params_stem is None:
            print(f"[harness] no residual params for {src.candidate_panel_stem}; skipping series")
            continue

        params_path = panel_dir / f"{src.residual_params_stem}_residual_params.pkl"
        with params_path.open("rb") as f:
            raw_params = pickle.load(f)

        residual_params = {
            (str(group_id), ""): raw_params
            for group_id in panel["group_id"].unique()
        }

        compute_and_persist_series(
            panel_dir=panel_dir,
            candidate_panel=panel,
            residual_params=residual_params,
            market_data=umd,
        )


def _build_sim_config(active_sectors: list[str]) -> SimulatorConfig:
    sweep_derived = {
        "data": DataConfig(
            candidate_panel_subdir=PANEL_SUBDIR,
            selected_sectors=active_sectors,
            data_path=str(DATA_UNIVERSES),
        ),
        "z_score": ZScoreConfig(lookback=21, method="ewm", residual_key=_residual_cfg().key),
        "trader": PairSpreadTraderConfig(entry_z=1.75, exit_z=0.0),
        "sizing": SizingConfig(
            base_pair_notional=100_000.0,
            vol_normalize=VolSizingConfig(),
        ),
        "risk_manager": RiskManagerConfig(
            max_gross_exposure=10.0,
            max_ticker_exposure_pct=0.15,
        ),
        "run": RunConfig(progress=False, progress_step=10, start_date=None, end_date=None),
        "spectrum": None,
        "entry_features": None,
    }

    # standard_v1 bundle, with performance/persistence overridden for a fast,
    # committed-file-only run: no quantstats HTML report, no run-dir persistence.
    bundle = dict(get_default_bundle("standard_v1"))
    bundle["performance"] = PerformanceConfig(
        enabled=True, metrics_table=True, report_html=False, per_group_breakdown=False,
    )
    bundle["persistence"] = PersistenceConfig(enabled=False)

    kwargs = merge_defaults(sweep_derived, bundle, "standard_v1")
    return SimulatorConfig(**kwargs)


def main() -> None:
    panel_build_time, panel_results = _build_panels()

    active_sectors = sorted({
        group_id for (group_id, _residual_key), pr in panel_results.items()
        if len(pr.panel) > 0
    })

    sim_time: float | None = None
    trades_df = None
    metrics: dict = {}

    if active_sectors:
        # A sector can legitimately produce zero candidate rows (e.g. min_obs
        # rejecting every pair on every date) — excluded here so
        # run_from_config never tries to load it.
        sim_config = _build_sim_config(active_sectors)
        _persist_series(sim_config, active_sectors)

        t0 = time.perf_counter()
        result = run_from_config(sim_config)
        sim_time = time.perf_counter() - t0

        trades_df = result.closed_trades_df()
        metrics = result.performance.metrics

    lines: list[str] = []
    lines.append("# Track B baseline harness output")
    lines.append(f"# sectors={SECTORS} max_steps={MAX_STEPS} "
                  f"hedge_ratio_lb={HEDGE_RATIO_LB} mr_diag_lb={MR_DIAG_LB} "
                  f"min_obs={MIN_OBS} residual={_residual_cfg().key}")
    lines.append(f"# active_sectors={active_sectors}")
    lines.append(f"# panel build time: {panel_build_time:.2f}s")
    lines.append(f"# simulate time: {'n/a (no active sectors)' if sim_time is None else f'{sim_time:.2f}s'}")
    lines.append("")

    lines.append("## panels")
    for (group_id, residual_key), pr in sorted(panel_results.items()):
        n_rows = len(pr.panel)
        n_valid = int(pr.panel["is_valid"].sum()) if n_rows else 0
        lines.append(f"{group_id} / {residual_key}: {n_valid}/{n_rows} valid candidates")
    lines.append("")

    lines.append("## metrics")
    if not active_sectors:
        lines.append("(no active sectors — simulate stage skipped)")
    for key, label, fmt in _METRICS_ORDER:
        if key not in metrics:
            continue
        lines.append(f"{label:<38} {_fmt_value(metrics[key], fmt):>10}")
    lines.append("")

    n_trades = 0 if trades_df is None else len(trades_df)
    lines.append(f"## trades ({n_trades})")
    if trades_df is not None and not trades_df.empty:
        present_cols = [c for c in TRADE_COLS if c in trades_df.columns]
        df = trades_df[present_cols].sort_values(["entry_date", "trade_id"]).reset_index(drop=True)
        lines.append(df.to_string(index=False))
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[harness] wrote {OUT_PATH} ({n_trades} trades)")


if __name__ == "__main__":
    main()
