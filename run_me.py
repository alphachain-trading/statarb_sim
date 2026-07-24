#!/usr/bin/env python3
"""
run_me.py — staged pipeline runner for statarb_sim.

Pipeline stages, in order:

    download  -> fetch/cache market data          (NOT YET IMPLEMENTED)
    residuals -> fit causal residuals + build the
                 candidate panel                   (NOT YET IMPLEMENTED)
    simulate  -> run the pair-spread backtest via
                 src.simulator.simulator_factory.run_from_config

Usage:
    python run_me.py --stage simulate
    python run_me.py --stage all
    python run_me.py --stage simulate --force

Design rules:
- All configuration comes from config/demo_materials.yaml.
- No filesystem paths are hardcoded here: every root is imported from
  settings.py; demo_materials.yaml only names sub-locations (panel subdir, run
  name) and strategy/model parameters.
- Each stage checks for its own output artifacts and skips if they already
  exist, unless --force is passed.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

from src.settings import (
    CONFIG_ROOT,
    CONFIG_UNIVERSE,
    PROJECT_ROOT,
    DATA_UNIVERSES,
    CANDIDATE_PANELS_ROOT,
    SIMULATION_RUNS_ROOT,
)

CONFIG_PATH = CONFIG_ROOT / "demo_materials.yaml"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if not CONFIG_PATH.exists():
        sys.exit(f"[config] Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Stage logging helpers
# ---------------------------------------------------------------------------

def _start(stage: str) -> None:
    print(f"\n=== [{stage}] START ===")


def _skip(stage: str, why: str) -> None:
    print(f"=== [{stage}] SKIP — {why} (use --force to rerun) ===")


def _done(stage: str) -> None:
    print(f"=== [{stage}] DONE ===")


# ---------------------------------------------------------------------------
# Path resolution (everything derived from settings.py + demo_materials.yaml)
# ---------------------------------------------------------------------------

def _panel_dir(cfg: dict) -> Path:
    return CANDIDATE_PANELS_ROOT / cfg["panels"]["subdir"]


def _panel_file(cfg: dict) -> Path:
    return _panel_dir(cfg) / f"{cfg['panels']['stem']}.panel.parquet"


def _selected_sectors(cfg: dict) -> list[str]:
    """Sectors to operate on, from demo_materials.yaml's `selected_sectors` list."""
    return list(cfg["selected_sectors"])


def _universe_yaml_path(sector: str) -> Path:
    """Resolve a sector's universe config yaml under CONFIG_UNIVERSE."""
    return CONFIG_UNIVERSE / f"universe.{sector}_only.v1.yaml"


# ---------------------------------------------------------------------------
# Stage: download
# ---------------------------------------------------------------------------

def stage_download(cfg: dict, force: bool) -> None:
    from src.data.universe_config import UniverseConfig
    from src.data.universe_loader import UniverseDataLoader

    stage = "download"
    started = False

    for sector in _selected_sectors(cfg):
        yaml_path = _universe_yaml_path(sector)
        if not yaml_path.exists():
            sys.exit(f"[download] missing universe config for sector '{sector}': {yaml_path}")

        config = UniverseConfig.from_yaml(yaml_path)

        # Skip-check: the loader's cache is keyed by universe_name; the daily
        # price parquet is the artifact that gates the downstream stages.
        prices_path = DATA_UNIVERSES / config.universe_name / "prices_daily.parquet"
        if prices_path.exists() and not force:
            _skip(stage, f"universe prices exist for '{sector}': {prices_path}")
            continue

        if not started:
            _start(stage)
            started = True

        print(f"[download] sector '{sector}': loading universe {config.universe_name} "
              f"(force_download={force})")
        loader = UniverseDataLoader(config, data_path=DATA_UNIVERSES, progress=True)
        loader.load(force_download=force)

        # Fresh market data invalidates the downstream candidate panel; delete it
        # so the residuals stage rebuilds it from the newly downloaded data. Only
        # applies to single-sector configs, which pin one panel via `stem`;
        # multi-sector configs (no stem) are discovery-based, so skip silently.
        if cfg["panels"].get("stem"):
            panel_file = _panel_file(cfg)
            if panel_file.exists():
                print(f"[download] removing stale candidate panel to force rebuild: {panel_file}")
                panel_file.unlink()

    if started:
        _done(stage)


# ---------------------------------------------------------------------------
# Stage: residuals
# ---------------------------------------------------------------------------

def _extract_panel_stem(results: dict, panel_dir: Path) -> str:
    """
    Recover the file stem run_panel_batch generated for the panel triple.

    run_panel_batch names its outputs with a now()-timestamped stem it builds
    internally (no override hook), so we read it back rather than assume it.
    This pipeline is single-sector/single-timescale, so exactly one panel is
    expected. Prefer the stem embedded in the result metadata; fall back to the
    newest *.panel.parquet on disk.
    """
    if len(results) != 1:
        sys.exit(
            f"[residuals] expected exactly one panel (single sector/timescale), "
            f"got {len(results)}: {sorted(results.keys())}"
        )

    result = next(iter(results.values()))
    params_path = result.metadata.get("residual_params_path")
    if params_path:
        name = Path(params_path).name
        suffix = "_residual_params.pkl"
        if name.endswith(suffix):
            return name[: -len(suffix)]

    panels = sorted(panel_dir.glob("*.panel.parquet"), key=lambda p: p.stat().st_mtime)
    if not panels:
        sys.exit(f"[residuals] run_panel_batch produced no panel under {panel_dir}")
    return panels[-1].name[: -len(".panel.parquet")]


def _update_panel_stem(cfg: dict, stem: str) -> None:
    """
    Adopt `stem` as the pipeline's canonical panel stem.

    Updates the in-memory cfg (so a same-process `--stage all` sees it) and
    rewrites only the `stem:` line in demo_materials.yaml, preserving comments and
    layout. Downstream (_panel_file, _persist_spread_series, simulate bootstrap)
    then resolves the panel just built.
    """
    old = cfg["panels"]["stem"]
    cfg["panels"]["stem"] = stem
    if stem == old:
        return

    text = CONFIG_PATH.read_text()
    new_text, n = re.subn(
        r"(?m)^(\s*)stem:.*$",
        lambda m: f"{m.group(1)}stem: {stem}",
        text,
    )
    if n != 1:
        sys.exit(
            f"[residuals] expected exactly one 'stem:' line in {CONFIG_PATH}, found {n}"
        )
    CONFIG_PATH.write_text(new_text)
    print(f"[residuals] {CONFIG_PATH.name} panels.stem updated: {old} -> {stem}")


def _make_panel_batch_cfg(cfg: dict):
    """Build the PanelBatchConfig shared by the single- and multi-sector paths."""
    from src.candidates.panel_batch import PanelBatchConfig
    from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode, AbsOrMult

    # decay_expanding, hl=504, min_history = 504 * 2 = 1008, subtract RF.
    # Reproduces the historical residual_key exp_hl504_mh1008_rf exactly.
    residual_cfg = CausalResidualConfig(
        mode=ResidualMode.DECAY_EXPANDING,
        subtract_risk_free=True,
        hl=504,
        min_lb_type_dec_exp=AbsOrMult.MULTIPLIER,
        min_lb_dec_exp=2,
    )
    return PanelBatchConfig(
        residual_configs=[residual_cfg],
        hedge_ratio_lb=252,
        mr_diag_lb=252,
        selected_sectors=_selected_sectors(cfg),
        universe_dir=CONFIG_UNIVERSE,
        data_path=DATA_UNIVERSES,
        persist_dir_template=cfg["panels"]["subdir"],
        frequency="W-FRI",
    )


def _all_sector_panels_exist(cfg: dict) -> bool:
    """True when a candidate panel already exists for every selected sector.

    Multi-sector skip-check: there is no single canonical stem, so gate on the
    per-sector panel files in the subdir (one ``{abbrev}_pairs_*`` each).
    """
    from src.simulator.config import sector_abbrev, sector_resolve

    panel_dir = _panel_dir(cfg)
    if not panel_dir.exists():
        return False
    for sector in _selected_sectors(cfg):
        abbrev = sector_abbrev(sector_resolve(sector))
        if not list(panel_dir.glob(f"{abbrev}_pairs_*.panel.parquet")):
            return False
    return True


def _persist_series_multi(cfg: dict, sim_config) -> None:
    """
    Precompute + persist series for every sector panel in the subdir.

    Discovery-based (via discover_sector_data_sources) so it scales to any number
    of sectors: one panel triple per sector, each keyed by its own group_id. The
    UMD is loaded once across all selected sectors; existing series files are
    skipped, so reruns are no-ops.
    """
    import pickle

    from src.candidates.candidate_panel import load_candidate_panel_result
    from src.residuals.series import compute_and_persist_series
    from src.simulator.simulator_factory import _load_umd
    from src.simulator.config import discover_sector_data_sources

    panel_dir = _panel_dir(cfg)
    umd = _load_umd(sim_config.data)

    sources = discover_sector_data_sources(
        panel_dir=panel_dir,
        universe_dir=CONFIG_UNIVERSE,
        selected_sectors=_selected_sectors(cfg),
    )

    for src in sources:
        result = load_candidate_panel_result(out_dir=panel_dir, stem=src.candidate_panel_stem)
        panel = result.panel

        if src.residual_params_stem is None:
            print(f"[residuals] no residual params for {src.candidate_panel_stem}; skipping series")
            continue

        params_path = panel_dir / f"{src.residual_params_stem}_residual_params.pkl"
        with params_path.open("rb") as f:
            raw_params = pickle.load(f)

        # Panel carries no residual_key column → series keyed by (group_id, "").
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


def stage_residuals(cfg: dict, force: bool) -> None:
    stage = "residuals"
    sectors = _selected_sectors(cfg)
    panel_dir = _panel_dir(cfg)

    from src.candidates.panel_batch import run_panel_batch

    # ── multi-sector: one panel per sector, no single canonical stem ──────
    if len(sectors) > 1:
        if not force and _all_sector_panels_exist(cfg):
            _skip(stage, f"candidate panels exist for all {len(sectors)} sectors in {panel_dir}")
            return

        _start(stage)
        run_panel_batch(_make_panel_batch_cfg(cfg))

        # Precompute + persist series for every sector panel; the simulator
        # discovers all panels in the subdir at run time.
        sim_config = _build_sim_config(cfg)
        _persist_series_multi(cfg, sim_config)
        _done(stage)
        return

    # ── single-sector: adopt the built panel's stem as the canonical stem ──
    panel_file = _panel_file(cfg)
    if panel_file.exists() and not force:
        _skip(stage, f"candidate panel exists: {panel_file}")
        return

    _start(stage)
    results = run_panel_batch(_make_panel_batch_cfg(cfg))

    # run_panel_batch stamps its own timestamped stem on the panel triple; adopt
    # it as the canonical stem so the skip-check and simulate stage resolve the
    # freshly built panel instead of the committed fixtures.
    new_stem = _extract_panel_stem(results, panel_dir)
    _update_panel_stem(cfg, new_stem)

    # Precompute + persist stock-residual and spread-level series so the simulate
    # stage loads them from disk instead of recomputing residuals each step.
    sim_config = _build_sim_config(cfg)
    _persist_spread_series(cfg, sim_config)

    _done(stage)


# ---------------------------------------------------------------------------
# Stage: simulate
# ---------------------------------------------------------------------------

def _bootstrap_panel_from_fixtures(cfg: dict) -> None:
    """
    Stand-in for the not-yet-implemented residuals stage: ensure the candidate
    panel triple is present under _panel_dir(cfg), copying it from the committed
    fixtures if necessary. No-op once the residuals stage produces real panels.
    """
    panel_dir = _panel_dir(cfg)
    stem = cfg["panels"]["stem"]
    if _panel_file(cfg).exists():
        return

    fixtures_dir = PROJECT_ROOT / cfg["panels"]["fixtures_dir"]
    print(
        f"[simulate] candidate panel not found in {panel_dir}; "
        f"bootstrapping from fixtures {fixtures_dir} "
        f"(stand-in for the not-yet-implemented residuals stage)"
    )
    panel_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".panel.parquet", ".meta.json", "_residual_params.pkl"):
        src = fixtures_dir / f"{stem}{suffix}"
        if not src.exists():
            sys.exit(f"[simulate] missing required fixture: {src}")
        shutil.copy2(src, panel_dir / src.name)


def _build_sim_config(cfg: dict):
    """Assemble a SimulatorConfig from demo_materials.yaml for a single-sector,
    single-timescale pair-spread run."""
    from src.simulator.config import (
        SimulatorConfig,
        DataConfig,
        CapitalConfig,
        ActivationConfig,
        ZScoreConfig,
        MRDiagnosticsConfig,
        PairSpreadTraderConfig,
        RunConfig,
        ExecutionConfig,
        SizingConfig,
        VolSizingConfig,
        RiskManagerConfig,
        PerformanceConfig,
        PersistenceConfig,
    )
    from src.candidates.candidate_selector import CandidateSelectionConfig

    z = cfg["z_score"]
    data_cfg = cfg["data"]
    sel = cfg["candidate_selection"]
    trd = cfg["trader"]
    szg = cfg["sizing"]
    rsk = cfg["risk_manager"]
    run = cfg["run"]

    return SimulatorConfig(
        data=DataConfig(
            candidate_panel_subdir=cfg["panels"]["subdir"],
            selected_sectors=_selected_sectors(cfg),
            data_path=str(DATA_UNIVERSES),
            price_field=data_cfg.get("price_field", "Close"),
            return_method=data_cfg.get("return_method", "log"),
        ),
        capital=CapitalConfig(total_capital=cfg["capital"]["total_capital"]),
        candidate_selection=CandidateSelectionConfig(
            allowed_candidate_subtypes=tuple(sel["allowed_candidate_subtypes"]),
            require_is_valid=sel.get("require_is_valid", True),
            require_success=sel.get("require_success", True),
        ),
        activation=ActivationConfig(
            one_active_per_group=False,
            switch_only_when_flat=False,
        ),
        z_score=ZScoreConfig(
            lookback=z["lookback"],
            method=z["method"],
            residual_key=cfg["residual_key"],
        ),
        diagnostics=MRDiagnosticsConfig(
            lookback=cfg["diagnostics"]["lookback"],
            compute_frequency=cfg["diagnostics"].get("compute_frequency", "off"),
        ),
        trader=PairSpreadTraderConfig(
            entry_z=trd["entry_z"],
            exit_z=trd["exit_z"],
            max_holding_days=trd.get("max_holding_days"),
            exit_rule=trd.get("exit_rule"),
        ),
        sizing=SizingConfig(
            base_pair_notional=szg["base_pair_notional"],
            vol_normalize=VolSizingConfig() if szg.get("vol_normalize", True) else None,
        ),
        risk_manager=RiskManagerConfig(
            max_gross_exposure=rsk["max_gross_exposure"],
            max_ticker_exposure_pct=rsk["max_ticker_exposure_pct"],
        ),
        run=RunConfig(
            start_date=run.get("start_date"),
            end_date=run.get("end_date"),
            progress=run.get("progress", True),
            progress_step=run.get("progress_step", 10),
        ),
        execution=ExecutionConfig(
            allow_fractional_shares=False,
            share_rounding="nearest",
        ),
        performance=PerformanceConfig(enabled=True, metrics_table=True, report_html=True),
        # output_dir is the *parent* under which the persistence layer creates a
        # single run dir named {datetime}_{config_hash[:8]}. Passing a pre-built
        # run dir here double-nests, so hand it the runs root.
        persistence=PersistenceConfig(enabled=True, output_dir=str(SIMULATION_RUNS_ROOT)),
    )


def _persist_spread_series(cfg: dict, sim_config) -> None:
    """
    Precompute and persist stock-residual and spread-level series under the
    candidate-panel dir, so the simulator can load spread levels from disk
    instead of recomputing residuals each step. Individual files that already
    exist are skipped, so this is a no-op on reruns.
    """
    import pickle

    from src.candidates.candidate_panel import load_candidate_panel_result
    from src.residuals.series import compute_and_persist_series
    from src.simulator.simulator_factory import _load_umd

    panel_dir = _panel_dir(cfg)
    stem = cfg["panels"]["stem"]

    result = load_candidate_panel_result(out_dir=panel_dir, stem=stem)
    panel = result.panel

    params_path = panel_dir / f"{stem}_residual_params.pkl"
    with params_path.open("rb") as f:
        raw_params = pickle.load(f)  # {asof_date: FittedCausalResidualModel}

    # Match the simulator's single-timescale keying: residual_key defaults to "".
    residual_params = {
        (str(group_id), ""): raw_params
        for group_id in panel["group_id"].unique()
    }

    umd = _load_umd(sim_config.data)
    compute_and_persist_series(
        panel_dir=panel_dir,
        candidate_panel=panel,
        residual_params=residual_params,
        market_data=umd,
    )


def stage_simulate(cfg: dict, force: bool) -> None:
    # No skip-check: every invocation runs and creates a fresh run dir.
    # --force has no effect here.
    stage = "simulate"
    _start(stage)

    sim_config = _build_sim_config(cfg)

    if len(_selected_sectors(cfg)) > 1:
        # Multi-sector: panels are produced by the residuals stage and discovered
        # at run time; there are no single-sector fixtures to bootstrap from.
        _persist_series_multi(cfg, sim_config)
    else:
        _bootstrap_panel_from_fixtures(cfg)
        _persist_spread_series(cfg, sim_config)

    from src.simulator.simulator_factory import run_from_config
    from src.simulator.simulation_persistence import hash_config

    result = run_from_config(sim_config)

    # The persistence layer names the run dir {datetime}_{config_hash[:8]} under
    # SIMULATION_RUNS_ROOT; recover the one just written (newest matching hash).
    run_hash = hash_config(sim_config)[:8]
    matches = sorted(SIMULATION_RUNS_ROOT.glob(f"*_{run_hash}"))
    run_output_dir = matches[-1] if matches else SIMULATION_RUNS_ROOT

    print(
        f"[simulate] run complete: {len(result.closed_trades)} closed trade(s); "
        f"output under {run_output_dir}"
    )
    _done(stage)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

STAGES = {
    "download": stage_download,
    "residuals": stage_residuals,
    "simulate": stage_simulate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="statarb_sim staged pipeline runner")
    parser.add_argument(
        "--stage",
        choices=["download", "residuals", "simulate", "all"],
        default="all",
        help="Which stage to run (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun a stage even if its output artifacts already exist.",
    )
    args = parser.parse_args()

    cfg = load_config()

    order = ["download", "residuals", "simulate"]
    to_run = order if args.stage == "all" else [args.stage]

    for stage in to_run:
        STAGES[stage](cfg, args.force)


if __name__ == "__main__":
    main()
