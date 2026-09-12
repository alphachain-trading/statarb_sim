#!/usr/bin/env python3
"""
Track B baseline harness.

A fixed, small-universe config run end-to-end through the shipped
generate_candidate_panels_by_group -> run_from_config -> compute_performance
path (F2 C6b: live in-simulator candidate generation, not the offline
run_panel_batch this harness used before), writing trades and summary metrics
to a committed text file (docs/refactor/B_baseline.txt).

Reused, not reimplemented: generate_candidate_panels_by_group with max_steps
is the same small-universe pattern the offline path used
(tests/test_panel_batch_windows.py, notebook 01); compute_performance +
_METRICS_ORDER (via result.performance, already computed by run_from_config)
is the existing metrics summary. No new runner, no new metrics code.

Config is fixed and small on purpose: two of the smallest committed group
universes (energy, 11 equities; materials, 13 equities — B_spec.md §9), an
EQ_EXPANDING residual with a low min_history (20), and explicit hedge/
diagnostics windows (252) stated via min_obs (Track B removed the
PanelBatchConfig default — B_window_admission.md items 4/5). PANEL_START_DATE
is chosen so BOTH groups already clear a full 252-day trailing window on the
first panel date (materials' binding constraint is CF, whose price history
starts 2005-08-11 — see the comment at PANEL_START_DATE below); an earlier
start starves materials of history and produces a 0/0-candidate group,
which is a fixture artifact, not a finding. This baseline's job is to move
when, and only when, a later change should move it — a counting question,
answered by the ## counts section (candidate/trade counts), not by the
## performance metrics section (noise at this trade count).

Usage:
    python scripts/b_baseline_harness.py
    python scripts/b_baseline_harness.py --allow-existing-artifacts

Artifacts guard (F2 E8, found.md "A leftover artifacts directory silently serves
stale values into a before/after diff"): retained from before C6b even though
this harness no longer writes candidate-panel/weights/residual-params files
itself (live generation, F2 C6b) — PANEL_DIR may still hold leftovers from
other tools/sessions that share CANDIDATE_PANELS_ROOT, and this guard is cheap
insurance against ever reading one by accident. At startup this script reports
PANEL_DIR's file count and newest mtime and refuses to proceed if it is
non-empty, unless --allow-existing-artifacts is passed. Guard only — it never
deletes anything; clear PANEL_DIR by hand (or pass the flag once you've
confirmed the leftovers are harmless for what you're about to measure).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.settings import CANDIDATE_PANELS_ROOT, DATA_UNIVERSES, PROJECT_ROOT
from src.candidates.pair_candidate_panel_creator import PairSpreadConfig
from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode
from src.simulator.config import (
    SimulatorConfig,
    CandidateGenerationConfig,
    DataConfig,
    GroupDataSource,
    ZScoreConfig,
    PairSpreadTraderConfig,
    SizingConfig,
    VolSizingConfig,
    RiskManagerConfig,
    RunConfig,
    PerformanceConfig,
    PersistenceConfig,
)
from src.simulator.simulator_factory import run_from_config, generate_candidate_panels_by_group
from src.simulator.sweep_defaults import get_default_bundle, merge_defaults
from src.simulator.performance.performance_report import _METRICS_ORDER, _fmt_value

UNIVERSE_NAME = "sp500_v1"  # config/universes/{UNIVERSE_NAME}/{group_id}.yaml
GROUPS = ["energy", "materials"]
PANEL_SUBDIR = "refactor_b_harness"
MAX_STEPS = 40
HEDGE_RATIO_LB = 252
MR_DIAG_LB = 252
# No default anywhere (PairSpreadConfig.min_obs is mandatory; PanelBatchConfig
# no longer supplies one either) — the harness states it explicitly, matching
# the requested lookback: a rolling 252-day fit that ran on 40 observations is
# not a 252-day fit. min_obs=hedge_ratio_lb is the fully-binding choice.
MIN_OBS = 252
# Common start for both groups' panel walk. materials' binding constraint is
# CF, whose price history starts 2005-08-11 — the 252nd return available from
# CF lands 2006-08-11 (a return needs day n and day n-1, so 252 returns need
# 253 price obs). PANEL_START_DATE sits ~3 weeks past that (2006-09-08), so
# CF already clears a full 252-day trailing window on the first panel date
# rather than at min_obs=252 arriving mid-walk. energy has ample history
# (from 1990) regardless of where this is set. This is a fixture choice, not
# a finding: an earlier start (e.g. 2005-08-15, one of this harness's earlier
# revisions) makes materials' 0/0 an artifact of CF not existing yet, not
# evidence about min_obs itself.
PANEL_START_DATE = "2006-09-08"

OUT_PATH = PROJECT_ROOT / "docs" / "refactor" / "B_baseline.txt"
PANEL_DIR = Path(CANDIDATE_PANELS_ROOT) / PANEL_SUBDIR

TRADE_COLS = [
    "trade_id", "group_id", "spread_id",
    "entry_date", "exit_date", "days_open", "direction",
    "pair_notional", "entry_z_score", "exit_z_score",
    "realized_pnl_gross", "realized_pnl_net",
]


def _check_artifacts_dir(*, allow_existing: bool) -> None:
    """
    Report PANEL_DIR's contents and refuse to proceed if non-empty, unless
    explicitly allowed (F2 E8). Guard only — never deletes anything.

    Since F2 C6b this harness no longer reads or writes PANEL_DIR itself
    (candidates are generated live, in-memory) — kept as cheap insurance
    against ever picking up a leftover file from another tool or session
    that shares CANDIDATE_PANELS_ROOT, not because this harness's own run
    could leave or find anything here any more.
    """
    if not PANEL_DIR.exists():
        print(f"[harness] artifacts dir {PANEL_DIR} does not exist yet — nothing to check.")
        return

    files = [p for p in PANEL_DIR.rglob("*") if p.is_file()]
    print(f"[harness] artifacts dir: {PANEL_DIR}")
    print(f"[harness]   file count: {len(files)}")
    if files:
        newest = max(files, key=lambda p: p.stat().st_mtime)
        newest_mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(newest.stat().st_mtime))
        print(f"[harness]   newest mtime: {newest_mtime} ({newest.name})")

    if files and not allow_existing:
        sys.exit(
            f"[harness] {PANEL_DIR} is non-empty ({len(files)} files) — refusing to "
            "run a before/after comparison against a directory that may contain "
            "stale panel/weights/residual-params files from a previous run "
            "(found.md: \"A leftover artifacts directory silently serves stale "
            "values into a before/after diff\"). Inspect and clear it by hand, or "
            "pass --allow-existing-artifacts once you've confirmed the leftovers "
            "are harmless for what you're about to measure."
        )


def _residual_cfg() -> CausalResidualConfig:
    return CausalResidualConfig(
        mode=ResidualMode.EQ_EXPANDING,
        subtract_risk_free=True,
        min_lb_eq_exp=20,
    )


def _pair_cfg() -> PairSpreadConfig:
    return PairSpreadConfig(
        hedge_ratio_methods=["pca"],
        min_obs=MIN_OBS,
        min_return_std=1e-8,
        min_level_std=1e-8,
        min_kappa=1e-6,
        max_half_life=126.0,
        tiny_weight_threshold=1e-6,
    )


def _build_sim_config(selected_groups: list[str]) -> SimulatorConfig:
    sweep_derived = {
        "data": DataConfig(
            # groups set explicitly (not just selected_groups) so
            # DataConfig.resolved_groups() -- which _load_umd calls -- returns
            # this list directly rather than falling through to
            # discover_group_data_sources's stem-based panel-file discovery,
            # which requires files on disk this live-generation path never
            # writes. candidate_panel_stem is unused here: _load_umd only
            # reads universe_config_name, and generate_candidate_panels_by_group
            # (F2 C6b) reads selected_groups directly, not resolved_groups().
            groups=[
                GroupDataSource(universe_config_name=f"{UNIVERSE_NAME}/{gid}.yaml", candidate_panel_stem="")
                for gid in selected_groups
            ],
            selected_groups=selected_groups,
            universe_name=UNIVERSE_NAME,
            data_path=str(DATA_UNIVERSES),
        ),
        "residual": _residual_cfg(),
        "candidate_generation": CandidateGenerationConfig(
            pair_cfg=_pair_cfg(),
            hedge_ratio_lb=HEDGE_RATIO_LB,
            mr_diag_lb=MR_DIAG_LB,
            frequency="W-FRI",
            start_date=PANEL_START_DATE,
            max_steps=MAX_STEPS,
        ),
        "z_score": ZScoreConfig(lookback=21, ddof=1, method="ewm", residual_key=_residual_cfg().key),
        "trader": PairSpreadTraderConfig(entry_z=1.75, exit_z=0.0),
        "sizing": SizingConfig(
            base_pair_notional=100_000.0,
            vol_normalize=VolSizingConfig(floor_multiplier=0.2, cap_multiplier=5.0),
        ),
        "risk_manager": RiskManagerConfig(
            max_gross_exposure=10.0,
            max_ticker_exposure_pct=0.15,
        ),
        "run": RunConfig(progress=False, progress_step=10, start_date=None, end_date=None),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-existing-artifacts",
        action="store_true",
        help="Proceed even if PANEL_DIR already has files from a previous run (F2 E8).",
    )
    args = parser.parse_args()
    _check_artifacts_dir(allow_existing=args.allow_existing_artifacts)

    # F2 C6b: generate candidates live (generate_candidate_panels_by_group),
    # not via the offline run_panel_batch this harness used before C6b.
    # active_groups must still be known before building the sim_config that
    # goes to run_from_config, so generate once against GROUPS, then narrow.
    t0 = time.perf_counter()
    prelim_sim_config = _build_sim_config(GROUPS)
    panel_results = generate_candidate_panels_by_group(prelim_sim_config)
    panel_build_time = time.perf_counter() - t0

    active_groups = sorted({
        group_id for (group_id, _residual_key), v in panel_results.items()
        if len(v["panel"]) > 0
    })

    sim_time: float | None = None
    trades_df = None
    metrics: dict = {}

    if active_groups:
        # A group can legitimately produce zero candidate rows (e.g. min_obs
        # rejecting every pair on every date) — excluded here so
        # run_from_config's own live generation never tries it again.
        sim_config = (
            prelim_sim_config if active_groups == GROUPS else _build_sim_config(active_groups)
        )

        t0 = time.perf_counter()
        result = run_from_config(sim_config)
        sim_time = time.perf_counter() - t0

        trades_df = result.closed_trades_df()
        metrics = result.performance.metrics

    lines: list[str] = []
    lines.append("# Track B baseline harness output")
    lines.append(f"# groups={GROUPS} max_steps={MAX_STEPS} "
                  f"hedge_ratio_lb={HEDGE_RATIO_LB} mr_diag_lb={MR_DIAG_LB} "
                  f"min_obs={MIN_OBS} residual={_residual_cfg().key}")
    lines.append(f"# active_groups={active_groups}")
    lines.append(f"# panel build time: {panel_build_time:.2f}s")
    lines.append(f"# simulate time: {'n/a (no active groups)' if sim_time is None else f'{sim_time:.2f}s'}")
    lines.append("")

    # Counts first: at a few dozen trades the performance numbers are noise
    # and invite misreading, while candidate/trade counts are directly
    # interpretable. This harness's job for the rest of the refactor is
    # detecting whether a change moved something it should not have — a
    # counting question, not a performance one.
    n_trades = 0 if trades_df is None else len(trades_df)

    lines.append("## counts")
    for (group_id, residual_key), v in sorted(panel_results.items()):
        panel = v["panel"]
        n_rows = len(panel)
        n_valid = int(panel["is_valid"].sum()) if n_rows else 0
        lines.append(f"{group_id} / {residual_key}: {n_valid}/{n_rows} valid candidates")
    # Positions-opened / positions-rejected counts are not tracked anywhere
    # in the simulator today (grepped, NOT FOUND) — n_trades and
    # n_groups_traded are the trade-side counts that are available.
    for key, label, fmt in _METRICS_ORDER:
        if key not in metrics or key not in ("n_trades", "n_groups_traded"):
            continue
        lines.append(f"{label:<38} {_fmt_value(metrics[key], fmt):>10}")
    lines.append("")

    lines.append(f"## trades ({n_trades})")
    if trades_df is not None and not trades_df.empty:
        present_cols = [c for c in TRADE_COLS if c in trades_df.columns]
        df = trades_df[present_cols].sort_values(["entry_date", "trade_id"]).reset_index(drop=True)
        lines.append(df.to_string(index=False))
    lines.append("")

    lines.append("## performance metrics (noise at this trade count — see counts above)")
    if not active_groups:
        lines.append("(no active groups — simulate stage skipped)")
    for key, label, fmt in _METRICS_ORDER:
        if key not in metrics:
            continue
        lines.append(f"{label:<38} {_fmt_value(metrics[key], fmt):>10}")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[harness] wrote {OUT_PATH} ({n_trades} trades)")


if __name__ == "__main__":
    main()
