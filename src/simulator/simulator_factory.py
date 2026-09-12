from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.settings import CANDIDATE_PANELS_ROOT, CONFIG_UNIVERSE
from src.simulator.candidate_activation import CandidateActivation
from src.simulator.candidate_filter import CandidateFilter
from src.simulator.candidate_signals import CandidateSignalGenerator
from src.simulator.config import (
    CapitalConfig,
    DataConfig,
    PairSpreadTraderConfig,
    PortfolioMeanReversionConfig,
    RiskManagerConfig,
    GroupDataSource,
    SimulatorConfig,
    ZScoreConfig,
    group_resolve,
    discover_group_data_sources,
)
from src.simulator.entry_feature_engine import EntryFeatureEngine
from src.simulator.execution import ExecutionEngine
from src.simulator.position_translator import PositionTranslator
from src.simulator.risk_manager import RiskManager
from src.simulator.sizing_engine import SizingEngine
from src.simulator.simulator import SimulationResult, Simulator
from src.simulator.traders.pair_spread_mean_reversion import PairSpreadMeanReversionTrader
from src.simulator.traders.portfolio_mean_reversion import PortfolioMeanReversionTrader
from src.candidates.candidate_panel import (
    CandidatePanelResult,
    load_candidate_panel_result,
    load_weights,
    weights_lookup_from_df,
)
from src.data.universe_loader import UniverseConfig, UniverseDataLoader
from src.data.universe_marketdata import UniverseMarketData
from src.residuals.causal_residuals import CausalResidualConfig, FittedCausalResidualModel, load_residual_params


def create_simulator(
    config: SimulatorConfig,
    *,
    umd: UniverseMarketData | None = None,
    residual_configs: dict[str, CausalResidualConfig] | None = None,
    precomputed_residual_params: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] | None = None,
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]] | None = None,
) -> Simulator:
    """
    Build a fully configured Simulator from a SimulatorConfig.

    Parameters
    ----------
    config
        Complete simulation configuration.
    umd
        Pre-loaded UniverseMarketData. If None, loaded from config.data.
    residual_configs
        Dict of residual_key → CausalResidualConfig.
        If None, must be resolved from panel metadata via run_from_config().
    precomputed_residual_params
        Optional {(group_id, residual_key): {date: FittedCausalResidualModel}}.
        Skips expensive model fitting in the signal generator.
    weights_lookup
        {(spread_id, asof_date): {ticker: weight}}, loaded from weights.parquet.
        Required for the pair-spread trader to activate any candidate.
    """
    if umd is None:
        umd = _load_umd(config.data)

    if residual_configs is None:
        raise ValueError("residual_configs must be provided.")

    z_configs = config.resolved_z_score_configs()

    z_score_by_timescale_label: dict[str, ZScoreConfig] = {
        zc.timescale_label: zc for zc in z_configs
    }

    signal_generator = CandidateSignalGenerator(
        z_score_configs=z_score_by_timescale_label,
        diagnostics_config=config.diagnostics,
        residual_configs=residual_configs,
        umd=umd,
        price_field=config.data.price_field,
        return_method=config.data.return_method,
        precomputed_residual_params=precomputed_residual_params or {},
    )

    if isinstance(config.trader, PortfolioMeanReversionConfig):
        trader = PortfolioMeanReversionTrader(
            config=config.trader, default_abs_target_group_exposure=1.0,
        )
    elif isinstance(config.trader, PairSpreadTraderConfig):
        trader = PairSpreadMeanReversionTrader(config=config.trader)
    else:
        raise ValueError(f"Unsupported trader config type: {type(config.trader).__name__}")

    sizing_engine = SizingEngine(config=config.sizing)

    entry_feature_engine = None
    if config.entry_features is not None:
        entry_feature_engine = EntryFeatureEngine(
            config=config.entry_features,
            signal_generator=signal_generator,
        )

    risk_manager = None
    if config.risk_manager is not None:
        risk_manager = RiskManager(
            config=config.risk_manager,
            total_capital=config.capital.total_capital,
        )

    return Simulator(
        config=config,
        umd=umd,
        candidate_filter=CandidateFilter(
            config=config.candidate_selection,
            z_score_configs_by_rkey=config.z_score_configs_by_rkey(),
        ),
        candidate_activation=CandidateActivation(config=config.activation),
        signal_generator=signal_generator,
        trader=trader,
        position_translator=PositionTranslator(),
        execution_engine=ExecutionEngine(config=config.execution),
        sizing_engine=sizing_engine,
        entry_feature_engine=entry_feature_engine,
        risk_manager=risk_manager,
        weights_lookup=weights_lookup or {},
    )


def run_from_config(config: SimulatorConfig) -> SimulationResult:
    """
    One-call entry point: load data, run simulation.

    Supports single-group, multi-group, and multi-timescale configurations.
    """
    import time as _time

    print("[run] Loading universe market data...")
    t0 = _time.time()
    umd = _load_umd(config.data)
    print(f"[run] UMD loaded in {_time.time() - t0:.1f}s")

    z_configs = config.resolved_z_score_configs()

    run_id: str | None = None
    if config.candidate_generation is not None:
        # F2 C6b: generate live, inside the simulator, instead of loading a
        # panel/weights/residual_params built earlier by the offline
        # run_panel_batch path. Opt-in — every other caller is unaffected.
        #
        # F2 C6c: run_id/run_dir are resolved here, before generation, and
        # threaded through to sim.run() below (run_id=run_id) so persistence
        # writes candidate/weights/residual-param artifacts into the SAME
        # run_dir the simulation's own trades/config/performance end up in,
        # rather than Simulator.run() resolving a second run_id afterwards
        # (make_run_id includes a minute-resolution timestamp -- resolving
        # twice risks two different, mismatched directories if a minute
        # boundary is crossed in between). None/no-op when persistence is
        # disabled, matching Simulator.run()'s own guard.
        run_dir: Path | None = None
        if config.persistence.enabled:
            from src.simulator.simulation_persistence import make_run_id, _resolve_run_dir
            run_id = make_run_id(config)
            run_dir = _resolve_run_dir(config.persistence, run_id)
            run_dir.mkdir(parents=True, exist_ok=True)

        print("[run] Generating candidates live...")
        t0 = _time.time()
        panel, residual_configs, precomputed_residual_params, weights_lookup = (
            _generate_candidates_live(config, run_dir=run_dir)
        )
        print(f"[run] Candidates generated in {_time.time() - t0:.1f}s")
    else:
        print("[run] Loading candidate panels...")
        t0 = _time.time()
        panel, metadata_by_key = _load_panels(config.data, z_configs)
        print(f"[run] Panels loaded in {_time.time() - t0:.1f}s")

        _assert_fit_input_tickers_available(umd, metadata_by_key)

        residual_configs = _resolve_residual_configs(config, metadata_by_key)

        print("[run] Loading residual params...")
        t0 = _time.time()
        precomputed_residual_params = _load_residual_params(config.data, z_configs)
        print(f"[run] Residual params loaded in {_time.time() - t0:.1f}s")

        print("[run] Loading weights...")
        t0 = _time.time()
        weights_lookup = _load_weights(config.data, z_configs)
        print(f"[run] Weights loaded in {_time.time() - t0:.1f}s")

    print("[run] Creating simulator...")
    t0 = _time.time()
    sim = create_simulator(
        config=config,
        umd=umd,
        residual_configs=residual_configs,
        precomputed_residual_params=precomputed_residual_params,
        weights_lookup=weights_lookup,
    )
    print(f"[run] Simulator created in {_time.time() - t0:.1f}s")

    print("[run] Starting simulation...")
    return sim.run(panel, run_id=run_id)


# ---------------------------------------------------------------------------
# F2 C6b — live candidate generation (opt-in, alongside the offline load path)
# ---------------------------------------------------------------------------

def generate_candidate_panels_by_group(
    config: SimulatorConfig,
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Run the in-simulator candidate generator (candidate_generation.py) once
    per selected group, live — no offline run_panel_batch, no disk.

    Returns {(group_id, residual_key): {"panel": <full scored panel df,
    including invalid rows>, "weight_rows": [...], "fitted_params": {...}}}.

    Used both by run_from_config's live-generation branch
    (config.candidate_generation is not None) and directly by callers that
    need the per-group breakdown before running a simulation — e.g.
    b_baseline_harness.py's ## counts section, which needs the full scored
    panel (valid + invalid rows) per group, not just what
    SimulationResult.selected_panel keeps after CandidateFilter.

    Loads its own UniverseMarketData per group, under
    config.candidate_generation's own force_download/check_for_corruptions/
    start_after_nan flags — NOT the umd run_from_config loads under
    config.data's flags for simulation. The offline pipeline this replaces
    used two separately-flagged UMD loads (PanelBatchConfig for scoring,
    DataConfig for simulation; found.md's "four combinations" entry), and
    every committed B_baseline.txt bakes in that split. Reusing
    run_from_config's single merged umd for both would silently unify the
    two and move every trade's pair_notional by rounding-level amounts —
    caught empirically while building this function. CandidateGenerationConfig's
    own flags default to PanelBatchConfig's (False/False/True), preserving
    the split; unifying it for real is Track I's job, not F2's.
    """
    cg = config.candidate_generation
    if cg is None:
        raise ValueError("config.candidate_generation must be set to generate candidates live.")
    if config.residual is None:
        raise ValueError(
            "config.residual must be set explicitly to generate candidates live "
            "— there is no persisted panel metadata to resolve it from."
        )
    if not config.data.selected_groups:
        raise ValueError(
            "config.data.selected_groups must be a non-empty list of group_id "
            "strings to generate candidates live."
        )

    from src.data.returns import build_group_return_bundle
    from src.simulator.candidate_generation import generate_candidates

    residual_cfg = config.residual
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for group_id in config.data.selected_groups:
        yaml_path = Path(CONFIG_UNIVERSE) / config.data.universe_name / f"{group_id}.yaml"
        loader = UniverseDataLoader(
            UniverseConfig.from_yaml(yaml_path),
            data_path=config.data.data_path,
            progress=False,
        )
        umd = loader.load(
            force_download=cg.force_download,
            check_for_corruptions=cg.check_for_corruptions,
            start_after_nan=cg.start_after_nan,
        )
        bundle = build_group_return_bundle(
            umd=umd,
            group_id=group_id,
            field=config.data.price_field,
            return_method=config.data.return_method,
            dropna="any",
        )
        rows, weight_rows, fitted_params = generate_candidates(
            bundle=bundle,
            residual_cfg=residual_cfg,
            pair_cfg=cg.pair_cfg,
            hedge_ratio_lb=cg.hedge_ratio_lb,
            mr_diag_lb=cg.mr_diag_lb,
            frequency=cg.frequency,
            start_date=cg.start_date,
            end_date=cg.end_date,
            max_steps=cg.max_steps,
        )
        panel = pd.DataFrame(rows)
        # Neither create_pair_candidate_panel nor generate_candidates itself
        # adds this column -- only the offline disk-load path does
        # (_load_panels, both branches below), deriving it from the
        # discovered GroupDataSource.residual_key. generate_candidates'
        # own equivalence test (C6a) never caught its absence because it
        # compared against create_pair_candidate_panel directly, not against
        # what _load_panels actually hands the simulator — the real
        # consumer needs this column to route candidates by residual_key at
        # all (CandidateFilter, ZScoreConfig resolution).
        if not panel.empty:
            panel["residual_key"] = residual_cfg.key
        out[(group_id, residual_cfg.key)] = {
            "panel": panel,
            "weight_rows": weight_rows,
            "fitted_params": fitted_params,
        }
    return out


def _generate_candidates_live(
    config: SimulatorConfig,
    *,
    run_dir: Path | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, CausalResidualConfig],
    dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]],
    dict[tuple[str, pd.Timestamp], dict[str, float]],
]:
    """Combines generate_candidate_panels_by_group's per-group breakdown into
    the same (panel, residual_configs, precomputed_residual_params,
    weights_lookup) shape run_from_config's offline-load branch produces.

    run_dir: when given (persistence enabled), persists each group's panel/
    weights/residual_params into run_dir/"candidates" (F2 C6c) — see
    persist_live_candidate_artifacts."""
    by_group = generate_candidate_panels_by_group(config)

    if run_dir is not None:
        persist_live_candidate_artifacts(by_group, run_dir)

    panels = [v["panel"] for v in by_group.values() if not v["panel"].empty]
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()

    all_weight_rows: list[dict[str, Any]] = []
    precomputed_residual_params: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] = {}
    for key, v in by_group.items():
        all_weight_rows.extend(v["weight_rows"])
        precomputed_residual_params[key] = v["fitted_params"]

    weights_lookup = weights_lookup_from_df(pd.DataFrame(all_weight_rows)) if all_weight_rows else {}
    residual_configs = {config.residual.key: config.residual}

    return panel, residual_configs, precomputed_residual_params, weights_lookup


def persist_live_candidate_artifacts(
    by_group: dict[tuple[str, str], dict[str, Any]],
    run_dir: Path,
) -> None:
    """
    F2 C6c: persist the live generation path's candidates/weights/residual
    params into the run's own run_dir (run_dir/"candidates"/{group_id}_
    {residual_key}...), via candidate_panel.py's/causal_residuals.py's
    existing writers -- the same ones run_panel_batch uses
    (pair_candidate_panel_creator.py:944-968), not a second implementation.

    Schema is identical to the offline path's by construction, not just by
    convention: generate_candidates (candidate_generation.py:102-114) calls
    the exact same _build_pair_candidate_rows_for_date / fit_causal_residual_
    model create_pair_candidate_panel's own walk calls
    (pair_candidate_panel_creator.py:842-869), producing rows/weight_rows/
    fitted_params in the identical shapes save_candidate_panel_result/
    save_weights/save_residual_params already accept — weights.parquet's
    (group_id, residual_key, hedge_key, spread_id, asof_date, ticker) grain
    and residual_params.parquet's (fit_date, ticker, factor) grain
    (F1_artifact_schema.md §3-4) can't drift here; there is no separate
    persistence-side row-building logic to diverge. The only difference from
    the offline artifacts is density: residual_params.parquet here has one
    row per OUTER asof_date, not one per daily grid date, since nothing
    downstream reads fits at non-outer dates (F2_spec.md D5/P3) — same
    columns, fewer rows.
    """
    from src.candidates.candidate_panel import CandidatePanelResult, save_candidate_panel_result, save_weights
    from src.residuals.causal_residuals import save_residual_params

    out_dir = run_dir / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    for (group_id, residual_key), v in by_group.items():
        stem = f"{group_id}_{residual_key}"
        panel = v["panel"]
        if not panel.empty:
            save_candidate_panel_result(
                result=CandidatePanelResult(
                    panel=panel,
                    metadata={"group_id": group_id, "residual_key": residual_key},
                ),
                out_dir=out_dir,
                stem=stem,
            )
        if v["weight_rows"]:
            save_weights(v["weight_rows"], str(out_dir / f"{stem}_weights.parquet"))
        if v["fitted_params"]:
            save_residual_params(v["fitted_params"], str(out_dir / f"{stem}_residual_params.parquet"))


# ---------------------------------------------------------------------------
# Internal helpers — data loading
# ---------------------------------------------------------------------------

def _load_umd(data_cfg: DataConfig) -> UniverseMarketData:
    """
    Load and merge UniverseMarketData from all group sources.

    Deduplicates by universe_config_name — same universe loaded once even
    if multiple timescales reference it.
    """
    groups = data_cfg.resolved_groups()

    seen_universes: dict[str, UniverseMarketData] = {}
    unique_groups: list[GroupDataSource] = []
    for src in groups:
        if src.universe_config_name not in seen_universes:
            unique_groups.append(src)
            seen_universes[src.universe_config_name] = None

    print(f"[factory] Loading {len(unique_groups)} universe(s)...")

    umds: list[UniverseMarketData] = []
    for src in unique_groups:
        universe_path = Path(CONFIG_UNIVERSE) / src.universe_config_name
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe config not found: {universe_path}")

        loader = UniverseDataLoader(
            config=UniverseConfig.from_yaml(universe_path),
            data_path=data_cfg.data_path,
        )
        umd = loader.load(
            force_download=data_cfg.force_download,
            check_for_corruptions=data_cfg.check_for_corruptions,
            start_after_nan=data_cfg.start_after_nan,
        )
        umds.append(umd)

    if len(umds) == 1:
        return umds[0]

    return _merge_umds(umds, validate_overlap=False)


def _merge_umds(umds: list[UniverseMarketData], validate_overlap: bool = False) -> UniverseMarketData:
    """
    Merge multiple single-group UniverseMarketData into one.

    - prices: outer join on dates, overlapping tickers validated for consistency
    - ticker_info: concat + deduplicate (first occurrence wins)
    - group_info: concat + deduplicate (root groups like us_equities shared)
    - membership: concat + deduplicate
    """
    import time as _time
    t0 = _time.time()

    if validate_overlap:
        seen_tickers: dict[str, int] = {}
        tickers_to_validate: dict[str, tuple[int, int]] = {}

        for i, u in enumerate(umds):
            for ticker in u.tickers():
                if ticker in seen_tickers and ticker not in tickers_to_validate:
                    tickers_to_validate[ticker] = (seen_tickers[ticker], i)
                elif ticker not in seen_tickers:
                    seen_tickers[ticker] = i

        if tickers_to_validate:
            print(f"[merge] Validating {len(tickers_to_validate)} overlapping tickers...")
            for ticker, (i, j) in tickers_to_validate.items():
                try:
                    p1 = umds[i].prices[(ticker, "Close")]
                    p2 = umds[j].prices[(ticker, "Close")]
                except KeyError:
                    continue
                common_idx = p1.index.intersection(p2.index)
                if not common_idx.empty:
                    diff = (p1.loc[common_idx] - p2.loc[common_idx]).abs()
                    max_diff = diff.max()
                    if max_diff > 1e-6:
                        print(
                            f"[merge] Warning: ticker {ticker} Close price mismatch "
                            f"(max diff={max_diff:.6f}). Using first occurrence."
                        )

    print(f"[merge] Joining price matrices from {len(umds)} UMDs...")
    merged_prices = umds[0].prices.copy()
    for k, u in enumerate(umds[1:], start=2):
        new_cols = [c for c in u.prices.columns if c not in merged_prices.columns]
        if new_cols:
            merged_prices = merged_prices.join(u.prices[new_cols], how="outer")
        print(f"[merge]   joined {k}/{len(umds)} ({len(new_cols)} new columns)")

    merged_ticker_info = pd.concat([u.ticker_info for u in umds], axis=0)
    merged_ticker_info = merged_ticker_info[~merged_ticker_info.index.duplicated(keep="first")]

    merged_group_info = pd.concat([u.group_info for u in umds], axis=0)
    merged_group_info = merged_group_info[~merged_group_info.index.duplicated(keep="first")]

    merged_membership = pd.concat([u.membership for u in umds], axis=0, ignore_index=True)
    merged_membership = merged_membership.drop_duplicates()

    groups_per_ticker = merged_membership.groupby("ticker")["group_id"].nunique()
    multi_group_tickers = groups_per_ticker[groups_per_ticker > 1]
    if not multi_group_tickers.empty:
        examples = {
            ticker: sorted(
                merged_membership.loc[merged_membership["ticker"] == ticker, "group_id"].unique()
            )
            for ticker in multi_group_tickers.index[:5]
        }
        raise ValueError(
            f"{len(multi_group_tickers)} ticker(s) belong to more than one group: {examples}. "
            "A ticker must belong to exactly one group: residual fits are group-scoped, so a "
            "ticker in two groups would get two different residual series, and cross-group "
            "analysis would have to pick one or double-count."
        )

    merged = UniverseMarketData(
        prices=merged_prices,
        ticker_info=merged_ticker_info,
        group_info=merged_group_info,
        membership=merged_membership,
    )

    elapsed = _time.time() - t0
    print(
        f"[merge] Merged {len(umds)} UMDs in {elapsed:.1f}s: "
        f"{len(merged.tickers())} tickers, "
        f"{len(merged_group_info)} groups, "
        f"{len(merged_prices)} dates"
    )
    return merged


def _load_panels(
    data_cfg: DataConfig,
    z_score_configs: list[ZScoreConfig],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """
    Load candidate panels, optionally filtered by residual_key.

    Returns (merged_panel, metadata_by_residual_key).
    """
    groups = data_cfg.resolved_groups()
    requested_keys = {zc.residual_key for zc in z_score_configs}
    is_multi = any(k != "" for k in requested_keys)

    panel_dir = Path(CANDIDATE_PANELS_ROOT)
    if data_cfg.candidate_panel_subdir:
        panel_dir = panel_dir / data_cfg.candidate_panel_subdir

    panels: list[pd.DataFrame] = []
    metadata_by_key: dict[str, dict[str, Any]] = {}

    if is_multi:
        for src in groups:
            if src.residual_key not in requested_keys:
                continue

            result = load_candidate_panel_result(out_dir=panel_dir, stem=src.candidate_panel_stem)
            df = result.panel.copy()
            df["residual_key"] = src.residual_key

            panels.append(df)

            if src.residual_key not in metadata_by_key:
                metadata_by_key[src.residual_key] = result.metadata

        loaded_keys = set(metadata_by_key.keys())
        missing_keys = requested_keys - loaded_keys
        if missing_keys:
            available = sorted(
                {src.residual_key for src in groups if src.residual_key}
            )
            raise FileNotFoundError(
                f"No panels found for residual_keys: {sorted(missing_keys)}. "
                f"Available residual_keys in {panel_dir}: {available}"
            )
    else:
        unique_keys = {src.residual_key for src in groups if src.residual_key}
        if len(unique_keys) > 1:
            raise ValueError(
                f"Multiple residual timescales found in panels: {sorted(unique_keys)}. "
                f"Set residual_key explicitly on each ZScoreConfig to select which to use."
            )

        for src in groups:
            result = load_candidate_panel_result(out_dir=panel_dir, stem=src.candidate_panel_stem)
            df = result.panel
            df["residual_key"] = src.residual_key or ""
            panels.append(df)

            if not metadata_by_key:
                metadata_by_key[""] = result.metadata

    if not panels:
        raise FileNotFoundError(
            f"No candidate panels loaded from {panel_dir}. "
            f"Groups: {[s.candidate_panel_stem for s in groups]}, "
            f"requested_keys: {sorted(requested_keys)}"
        )

    merged = pd.concat(panels, ignore_index=True) if len(panels) > 1 else panels[0]

    weight_models_by_key = merged.groupby(["spread_id", "residual_key"])["weight_model"].nunique()
    contending = weight_models_by_key[weight_models_by_key > 1]
    if not contending.empty:
        examples = contending.index.tolist()[:5]
        raise ValueError(
            "Multiple weight_model (hedge config) values found under the same "
            f"(spread_id, residual_key), e.g. {examples}. The trader's "
            "open-position dedup key is (spread_id, residual_key), which does "
            "not include weight_model, so these sleeves would silently contend "
            "for the same position slot — arrival order would decide which one "
            "trades. Run them as separate runs for now."
        )

    group_counts = merged.groupby(["group_id", "residual_key"]).size()
    total = len(merged)
    n_panels = len(panels)

    if is_multi:
        print(f"[panels] Loaded {n_panels} panels across {len(requested_keys)} timescales, {total} total rows:")
        for (gid, rkey), cnt in group_counts.items():
            print(f"  {gid} / {rkey}: {cnt} rows")
    else:
        print(f"[panels] Loaded {n_panels} panels, {total} total rows:")
        for (gid, _), cnt in group_counts.items():
            print(f"  {gid}: {cnt} rows")

    return merged, metadata_by_key


def _assert_fit_input_tickers_available(
    umd: UniverseMarketData,
    metadata_by_key: dict[str, dict[str, Any]],
) -> None:
    """
    Load-time "different universe" guard (F1 commit 6).

    fit_input_tickers (panel metadata, the union of a residual model's
    realized input tickers across every fit date) names what reconstruction
    must load. Missing tickers here means the UMD backing this run is a
    different universe than the one the panel was built against -- fail
    loud now, not partway through a residual fit. Metadata from a panel
    built before this commit has no fit_input_tickers key; skipped, not an
    error (nothing to check against).

    Deliberately narrower than Track C's snapshot content-hash: this only
    checks "is every required ticker present," not "does its price content
    match what the panel was built from" -- that is what the snapshot hash
    covers, per a run that is actually bound to one (not yet wired, see
    DataConfig.snapshot_id).
    """
    available = set(umd.tickers())
    for rkey, metadata in metadata_by_key.items():
        fit_input_tickers = metadata.get("fit_input_tickers")
        if not fit_input_tickers:
            continue
        missing = sorted(set(fit_input_tickers) - available)
        if missing:
            raise ValueError(
                f"UniverseMarketData is missing tickers required by the candidate "
                f"panel for residual_key={rkey!r}: {missing}. This UMD is a "
                f"different universe than the one this panel was built against."
            )


def _load_residual_params(
    data_cfg: DataConfig,
    z_score_configs: list[ZScoreConfig],
) -> dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] | None:
    """
    Load precomputed residual params.

    Returns {(group_id, residual_key): {date: FittedCausalResidualModel}} or None.
    """
    groups = data_cfg.resolved_groups()
    requested_keys = {zc.residual_key for zc in z_score_configs}
    is_multi = any(k != "" for k in requested_keys)

    has_any = any(s.residual_params_stem is not None for s in groups)
    if not has_any:
        return None

    panel_dir = Path(CANDIDATE_PANELS_ROOT)
    if data_cfg.candidate_panel_subdir:
        panel_dir = panel_dir / data_cfg.candidate_panel_subdir

    merged: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] = {}

    for src in groups:
        if src.residual_params_stem is None:
            continue
        if is_multi and src.residual_key not in requested_keys:
            continue

        raw_id = src.residual_params_stem.split("_pairs_")[0]
        try:
            group_id = group_resolve(raw_id)
        except KeyError:
            group_id = raw_id

        rkey = src.residual_key or ""
        composite_key = (group_id, rkey)

        if composite_key in merged:
            continue

        params_path = panel_dir / f"{src.residual_params_stem}_residual_params.parquet"
        if not params_path.exists():
            raise FileNotFoundError(f"Residual params not found: {params_path}")

        params: dict[pd.Timestamp, FittedCausalResidualModel] = load_residual_params(str(params_path))

        if not params:
            continue

        merged[composite_key] = params
        print(f"[factory] Loaded residual params for {group_id}/{rkey}: {len(params)} dates")

    return merged if merged else None


def _load_weights(
    data_cfg: DataConfig,
    z_score_configs: list[ZScoreConfig],
) -> dict[tuple[str, pd.Timestamp], dict[str, float]] | None:
    """
    Load precomputed leg weights (weights.parquet), keyed for
    CandidateFilter.build_candidate_refs' lookup.

    Returns {(spread_id, asof_date): {ticker: weight}} or None if no group
    source carries a weights_stem (e.g. an older panel built before F1
    commit 4). Full float64 precision -- replaces the live path's old
    per-candidate weights JSON column, which lost precision through
    to_json(double_precision=12).
    """
    groups = data_cfg.resolved_groups()
    requested_keys = {zc.residual_key for zc in z_score_configs}
    is_multi = any(k != "" for k in requested_keys)

    has_any = any(s.weights_stem is not None for s in groups)
    if not has_any:
        return None

    panel_dir = Path(CANDIDATE_PANELS_ROOT)
    if data_cfg.candidate_panel_subdir:
        panel_dir = panel_dir / data_cfg.candidate_panel_subdir

    seen_stems: set[str] = set()
    frames: list[pd.DataFrame] = []

    for src in groups:
        if src.weights_stem is None or src.weights_stem in seen_stems:
            continue
        if is_multi and src.residual_key not in requested_keys:
            continue
        seen_stems.add(src.weights_stem)

        weights_path = panel_dir / f"{src.weights_stem}_weights.parquet"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        frames.append(load_weights(str(weights_path)))

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    lookup = weights_lookup_from_df(df)

    print(f"[factory] Loaded weights for {len(lookup)} (spread_id, asof_date) keys")
    return lookup


# ---------------------------------------------------------------------------
# Internal helpers — config resolution
# ---------------------------------------------------------------------------

def _resolve_residual_configs(
    config: SimulatorConfig,
    metadata_by_key: dict[str, dict[str, Any]],
) -> dict[str, CausalResidualConfig]:
    """Resolve CausalResidualConfig per residual_key from panel metadata."""
    if config.residual is not None and not config.is_multi_timescale():
        return {"": config.residual}

    resolved: dict[str, CausalResidualConfig] = {}
    for rkey, meta in metadata_by_key.items():
        panel_cfg_raw = meta.get("residual_cfg")
        if panel_cfg_raw is None:
            raise ValueError(f"No residual_cfg in metadata for residual_key={rkey!r}")
        cfg = _normalize_legacy_residual_cfg(panel_cfg_raw)
        print(
            f"[config] Residual config for {rkey or 'default'}: "
            f"window_mode={cfg.window_mode}, half_life={cfg.half_life}, "
            f"min_history={cfg.min_history}"
        )
        resolved[rkey] = cfg

    return resolved


def _normalize_legacy_residual_cfg(raw: dict[str, Any]) -> CausalResidualConfig:
    return CausalResidualConfig.from_dict(raw)