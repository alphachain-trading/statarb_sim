from __future__ import annotations

import json
import pickle
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
    SectorDataSource,
    SimulatorConfig,
    ZScoreConfig,
    sector_resolve,
    discover_sector_data_sources,
)
from src.simulator.entry_feature_engine import EntryFeatureEngine
from src.simulator.execution import ExecutionEngine
from src.simulator.position_translator import PositionTranslator
from src.simulator.risk_manager import RiskManager
from src.simulator.sizing_engine import SizingEngine
from src.simulator.simulator import SimulationResult, Simulator
from src.simulator.traders.pair_spread_mean_reversion import PairSpreadMeanReversionTrader
from src.simulator.traders.portfolio_mean_reversion import PortfolioMeanReversionTrader
from src.candidates.candidate_panel import CandidatePanelResult, load_candidate_panel_result
from src.data.universe_loader import UniverseConfig, UniverseDataLoader
from src.data.universe_marketdata import UniverseMarketData
from src.residuals.causal_residuals import CausalResidualConfig, FittedCausalResidualModel


def create_simulator(
    config: SimulatorConfig,
    *,
    umd: UniverseMarketData | None = None,
    residual_configs: dict[str, CausalResidualConfig] | None = None,
    precomputed_residual_params: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] | None = None,
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
    """
    if umd is None:
        umd = _load_umd(config.data)

    if residual_configs is None:
        raise ValueError("residual_configs must be provided.")

    z_configs = config.resolved_z_score_configs()

    z_score_by_timescale_label: dict[str, ZScoreConfig] = {
        zc.timescale_label: zc for zc in z_configs
    }

    panel_dir: Path | None = None
    if config.data.candidate_panel_subdir:
        panel_dir = Path(CANDIDATE_PANELS_ROOT) / config.data.candidate_panel_subdir

    signal_generator = CandidateSignalGenerator(
        z_score_configs=z_score_by_timescale_label,
        diagnostics_config=config.diagnostics,
        residual_configs=residual_configs,
        umd=umd,
        price_field=config.data.price_field,
        return_method=config.data.return_method,
        precomputed_residual_params=precomputed_residual_params or {},
        panel_dir=panel_dir,
    )

    if isinstance(config.trader, PortfolioMeanReversionConfig):
        trader = PortfolioMeanReversionTrader(config=config.trader)
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
    )


def run_from_config(config: SimulatorConfig) -> SimulationResult:
    """
    One-call entry point: load data, run simulation.

    Supports single-sector, multi-sector, and multi-timescale configurations.
    """
    import time as _time

    print("[run] Loading universe market data...")
    t0 = _time.time()
    umd = _load_umd(config.data)
    print(f"[run] UMD loaded in {_time.time() - t0:.1f}s")

    z_configs = config.resolved_z_score_configs()

    print("[run] Loading candidate panels...")
    t0 = _time.time()
    panel, metadata_by_key = _load_panels(config.data, z_configs)
    print(f"[run] Panels loaded in {_time.time() - t0:.1f}s")

    residual_configs = _resolve_residual_configs(config, metadata_by_key)

    print("[run] Loading residual params...")
    t0 = _time.time()
    precomputed_residual_params = _load_residual_params(config.data, z_configs)
    print(f"[run] Residual params loaded in {_time.time() - t0:.1f}s")

    print("[run] Creating simulator...")
    t0 = _time.time()
    sim = create_simulator(
        config=config,
        umd=umd,
        residual_configs=residual_configs,
        precomputed_residual_params=precomputed_residual_params,
    )
    print(f"[run] Simulator created in {_time.time() - t0:.1f}s")

    print("[run] Starting simulation...")
    return sim.run(panel)


# ---------------------------------------------------------------------------
# Internal helpers — data loading
# ---------------------------------------------------------------------------

def _load_umd(data_cfg: DataConfig) -> UniverseMarketData:
    """
    Load and merge UniverseMarketData from all sector sources.

    Deduplicates by universe_config_name — same universe loaded once even
    if multiple timescales reference it.
    """
    sectors = data_cfg.resolved_sectors()

    seen_universes: dict[str, UniverseMarketData] = {}
    unique_sectors: list[SectorDataSource] = []
    for src in sectors:
        if src.universe_config_name not in seen_universes:
            unique_sectors.append(src)
            seen_universes[src.universe_config_name] = None

    print(f"[factory] Loading {len(unique_sectors)} universe(s)...")

    umds: list[UniverseMarketData] = []
    for src in unique_sectors:
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
    Merge multiple single-sector UniverseMarketData into one.

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
    sectors = data_cfg.resolved_sectors()
    requested_keys = {zc.residual_key for zc in z_score_configs}
    is_multi = any(k != "" for k in requested_keys)

    panel_dir = Path(CANDIDATE_PANELS_ROOT)
    if data_cfg.candidate_panel_subdir:
        panel_dir = panel_dir / data_cfg.candidate_panel_subdir

    panels: list[pd.DataFrame] = []
    metadata_by_key: dict[str, dict[str, Any]] = {}

    if is_multi:
        for src in sectors:
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
                {src.residual_key for src in sectors if src.residual_key}
            )
            raise FileNotFoundError(
                f"No panels found for residual_keys: {sorted(missing_keys)}. "
                f"Available residual_keys in {panel_dir}: {available}"
            )
    else:
        unique_keys = {src.residual_key for src in sectors if src.residual_key}
        if len(unique_keys) > 1:
            raise ValueError(
                f"Multiple residual timescales found in panels: {sorted(unique_keys)}. "
                f"Set residual_key explicitly on each ZScoreConfig to select which to use."
            )

        for src in sectors:
            result = load_candidate_panel_result(out_dir=panel_dir, stem=src.candidate_panel_stem)
            df = result.panel
            df["residual_key"] = src.residual_key or ""
            panels.append(df)

            if not metadata_by_key:
                metadata_by_key[""] = result.metadata

    if not panels:
        raise FileNotFoundError(
            f"No candidate panels loaded from {panel_dir}. "
            f"Sectors: {[s.candidate_panel_stem for s in sectors]}, "
            f"requested_keys: {sorted(requested_keys)}"
        )

    merged = pd.concat(panels, ignore_index=True) if len(panels) > 1 else panels[0]

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


def _load_residual_params(
    data_cfg: DataConfig,
    z_score_configs: list[ZScoreConfig],
) -> dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] | None:
    """
    Load precomputed residual params.

    Returns {(group_id, residual_key): {date: FittedCausalResidualModel}} or None.
    """
    sectors = data_cfg.resolved_sectors()
    requested_keys = {zc.residual_key for zc in z_score_configs}
    is_multi = any(k != "" for k in requested_keys)

    has_any = any(s.residual_params_stem is not None for s in sectors)
    if not has_any:
        return None

    panel_dir = Path(CANDIDATE_PANELS_ROOT)
    if data_cfg.candidate_panel_subdir:
        panel_dir = panel_dir / data_cfg.candidate_panel_subdir

    merged: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] = {}

    for src in sectors:
        if src.residual_params_stem is None:
            continue
        if is_multi and src.residual_key not in requested_keys:
            continue

        raw_id = src.residual_params_stem.split("_pairs_")[0]
        try:
            group_id = sector_resolve(raw_id)
        except KeyError:
            group_id = raw_id

        rkey = src.residual_key or ""
        composite_key = (group_id, rkey)

        if composite_key in merged:
            continue

        params_path = panel_dir / f"{src.residual_params_stem}_residual_params.pkl"
        if not params_path.exists():
            raise FileNotFoundError(f"Residual params not found: {params_path}")

        with open(params_path, "rb") as f:
            params: dict[pd.Timestamp, FittedCausalResidualModel] = pickle.load(f)

        if not params:
            continue

        merged[composite_key] = params
        print(f"[factory] Loaded residual params for {group_id}/{rkey}: {len(params)} dates")

    return merged if merged else None


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