"""
Sweep runner — orchestrate parameter sweeps and persist results.

Usage from Jupyter:
    from sweep_runner import SweepConfig, run_sweep, load_sweep_results, view_*

    # Single-timescale (backward compatible)
    RUNS = [
        SweepConfig(z_method="ewm", z_lookback=21),
        SweepConfig(z_method="ewm", z_lookback=42),
    ]
    run_sweep(RUNS, candidate_panel_subdir="V2.pair")

    # Multi-timescale
    from src.simulator.config import ZScoreConfig
    RUNS = [
        SweepConfig(z_score_overrides=[
            ZScoreConfig(lookback=10, method="ewm", residual_key="exp_hl63_mh126"),
            ZScoreConfig(lookback=21, method="ewm", residual_key="exp_hl126_mh252"),
            ZScoreConfig(lookback=42, method="ewm", residual_key="exp_hl252_mh504"),
        ]),
    ]
    run_sweep(RUNS, candidate_panel_subdir="V3.pair")

    # Multi-timescale with cross-ts gate and timescale risk
    from src.simulator.config import ZScoreConfig, CrossTimescaleEntryConfig, TimescaleRiskConfig
    RUNS = [
        SweepConfig(
            z_score_overrides=[...],
            cross_ts=CrossTimescaleEntryConfig(same_sign_required=True, min_abs_z_all=0.5),
            timescale_risk=TimescaleRiskConfig(max_timescales_per_spread=1, selection="max_abs_z"),
        ),
    ]
    run_sweep(RUNS, candidate_panel_subdir="V3.pair")

    df = load_sweep_results()
    view_signal(df)
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, fields, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from settings import CANDIDATE_PANELS_ROOT, CONFIG_UNIVERSE, DATA_UNIVERSES, SIMULATION_RUNS_ROOT
from src.simulator.config import (
    DataConfig, CapitalConfig, SimulatorConfig,
    ActivationConfig, RunConfig, ExecutionConfig,
    ZScoreConfig, MRDiagnosticsConfig,
    PairSpreadTraderConfig, RiskManagerConfig,
    SizingConfig, VolSizingConfig, KellyConfig,
    EntryFeatureConfig, FeatureSpec,
    FeatureIntervalSpec, IntervalScoringConfig,
    CrossTimescaleEntryConfig, TimescaleRiskConfig,
    PerformanceConfig, PersistenceConfig,
    SpectrumConfig,
    discover_sector_data_sources, sector_abbrev,
)
from src.simulator.simulator import SimulationResult
from src.simulator.simulator_factory import run_from_config
from src.simulator.simulation_persistence import hash_config
from src.candidates.candidate_selector import CandidateSelectionConfig


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """One row of the parameter sweep. Defaults = current best baseline."""

    # Signal (single-timescale)
    entry_z: float = 2.0
    exit_z: float = 0.0
    z_lookback: int | list[int] = 21
    z_method: str = "ewm"

    # Multi-timescale override
    z_score_overrides: list[ZScoreConfig] | None = None

    # Cross-timescale entry gate (Mode B). None = independent per-rkey (Mode A).
    cross_ts: CrossTimescaleEntryConfig | None = None

    # Time stop
    max_holding_days: str | int | dict[str, int] | None = None

    # 2D exit rule
    exit_rule: str | dict[str, str] | None = None

    # Sizing
    base_pair_notional: float = 100_000.0
    vol_normalize: bool = True             # True = VolSizingConfig with defaults
    kelly: KellyConfig | None = None
    entry_features: EntryFeatureConfig | None = None
    interval_scoring: IntervalScoringConfig | None = None

    # Risk constraints
    max_ticker_exposure_pct: float = 0.15
    max_gross_exposure: float = 10.0
    timescale_risk: TimescaleRiskConfig | None = None

    # Spectrum capture
    spectrum: SpectrumConfig | None = None

    # Universe
    excluded_sectors: list[str] | None = None

    # Date range
    start_date: str | None = "2010-01-01"
    end_date: str | None = "2025-12-31"

    @property
    def is_multi_timescale(self) -> bool:
        return self.z_score_overrides is not None and len(self.z_score_overrides) > 0

    @property
    def alias(self) -> str:
        """Always-complete alias — every parameter is written out."""
        parts = [
            f"z{self.entry_z:.1f}",
            f"xz{self.exit_z:.1f}",
        ]

        if self.is_multi_timescale:
            ts_count = len(self.z_score_overrides)
            methods = set(zc.method for zc in self.z_score_overrides)
            method_str = "+".join(sorted(methods)) if len(methods) > 1 else next(iter(methods))
            zhls = "+".join(str(zc.resolved_lookbacks()[0]) for zc in self.z_score_overrides)
            rhls = "+".join(
                zc.residual_key.split("_hl")[1].split("_")[0]
                if "_hl" in zc.residual_key else "?"
                for zc in self.z_score_overrides
            )
            parts.append(f"mt{ts_count}_{method_str}_rhl{rhls}_zhl{zhls}")
        else:
            parts.append(self.z_method)
            if isinstance(self.z_lookback, list):
                parts.append("lb" + "-".join(str(lb) for lb in self.z_lookback))
            else:
                parts.append(f"lb{self.z_lookback}")

        if self.cross_ts is not None:
            cts_parts = []
            if self.cross_ts.same_sign_required:
                cts_parts.append("sign")
            if self.cross_ts.min_abs_z_all is not None:
                cts_parts.append(f"minz{self.cross_ts.min_abs_z_all:.1f}")
            if self.cross_ts.mean_abs_z_min is not None:
                cts_parts.append(f"avgz{self.cross_ts.mean_abs_z_min:.1f}")
            if cts_parts:
                parts.append("cts-" + "+".join(cts_parts))

        if self.max_holding_days is not None:
            if isinstance(self.max_holding_days, str):
                parts.append(f"mhd-{self.max_holding_days.replace('*', 'x')}")
            elif isinstance(self.max_holding_days, int):
                parts.append(f"mhd{self.max_holding_days}")
            else:
                vals = "+".join(str(v) for v in self.max_holding_days.values())
                parts.append(f"mhd{vals}")

        if self.exit_rule is not None:
            if isinstance(self.exit_rule, str):
                alias = self.exit_rule.replace(" AND ", "+").replace(" OR ", "|")
                alias = alias.replace(" ", "").replace("*", "x").replace("<", "lt").replace(">", "gt")
                parts.append(f"xr-{alias}")
            else:
                parts.append(f"xr-dict{hash(repr(sorted(self.exit_rule.items()))) % 10000:04d}")

        parts.append(f"bpn{int(self.base_pair_notional / 1000)}k")
        parts.append("voln" if self.vol_normalize else "flat")
        parts.append(f"tick{self.max_ticker_exposure_pct:.2f}")
        parts.append(f"gx{self.max_gross_exposure:.1f}")

        if self.timescale_risk is not None:
            tr = self.timescale_risk
            tr_parts = []
            if tr.max_timescales_per_spread is not None:
                tr_parts.append(f"mts{tr.max_timescales_per_spread}")
            tr_parts.append(tr.selection)
            if tr.max_pct_single_timescale is not None:
                tr_parts.append(f"pct{tr.max_pct_single_timescale:.0%}")
            parts.append("tsr-" + "+".join(tr_parts))

        if self.kelly is not None:
            k = self.kelly
            k_parts = [f"f{k.fraction:.1f}"]
            if k.half_life is not None:
                k_parts.append(f"hl{k.half_life}")
            if k.per_sector:
                k_parts.append("sec")
            else:
                k_parts.append("glob")
            parts.append("kelly-" + "+".join(k_parts))

        if self.entry_features is not None:
            feat_names = "+".join(f.feature for f in self.entry_features.features)
            parts.append(f"ef-{feat_names}")

        if self.interval_scoring is not None:
            sc = self.interval_scoring
            sc_parts = [f.feature for f in sc.feature_specs]
            parts.append(f"is-{'+'.join(sc_parts)}")

        if self.spectrum is not None:
            parts.append("spec")

        if self.excluded_sectors:
            parts.append("ex-" + sector_abbrev(sectors=self.excluded_sectors, to_string=True))
        else:
            parts.append("all")

        if self.start_date:
            parts.append(f"S{pd.Timestamp(self.start_date).strftime('%Y%m%d')}")
        if self.end_date:
            parts.append(f"E{pd.Timestamp(self.end_date).strftime('%Y%m%d')}")

        return "_".join(parts)


def _extract_run_id(result) -> str:
    if result.run_id is not None:
        return result.run_id
    return f"unknown_{datetime.now().strftime('%Y%m%d_%H%M')}"


def _build_sim_config(
    sweep: SweepConfig,
    candidate_panel_subdir: str,
) -> SimulatorConfig:
    """Convert SweepConfig → SimulatorConfig."""

    if sweep.is_multi_timescale:
        z_score = sweep.z_score_overrides
    else:
        z_score = ZScoreConfig(lookback=sweep.z_lookback, method=sweep.z_method)

    sizing = SizingConfig(
        base_pair_notional=sweep.base_pair_notional,
        vol_normalize=VolSizingConfig() if sweep.vol_normalize else None,
        kelly=sweep.kelly,
        interval_scoring=sweep.interval_scoring,
    )

    risk_manager = RiskManagerConfig(
        max_gross_exposure=sweep.max_gross_exposure,
        max_ticker_exposure_pct=sweep.max_ticker_exposure_pct,
        timescale_risk=sweep.timescale_risk,
    )

    return SimulatorConfig(
        data=DataConfig(
            candidate_panel_subdir=candidate_panel_subdir,
            excluded_sectors=sweep.excluded_sectors,
            data_path=str(DATA_UNIVERSES),
        ),
        capital=CapitalConfig(total_capital=1_000_000.0),
        candidate_selection=CandidateSelectionConfig(
            allowed_candidate_subtypes=("pca",), require_is_valid=True,
        ),
        activation=ActivationConfig(one_active_per_group=False, switch_only_when_flat=False),
        z_score=z_score,
        diagnostics=MRDiagnosticsConfig(lookback=252, compute_frequency="off"),
        trader=PairSpreadTraderConfig(
            entry_z=sweep.entry_z,
            exit_z=sweep.exit_z,
            cross_ts=sweep.cross_ts,
            max_holding_days=sweep.max_holding_days,
            exit_rule=sweep.exit_rule,
        ),
        sizing=sizing,
        risk_manager=risk_manager,
        run=RunConfig(
            progress=True,
            progress_step=10,
            start_date=sweep.start_date,
            end_date=sweep.end_date,
        ),
        execution=ExecutionConfig(allow_fractional_shares=False, share_rounding="nearest"),
        performance=PerformanceConfig(
            enabled=True,
            metrics_table=True,
            report_html=True,
            benchmark_ticker=None,
            annualization_factor=252,
            per_group_breakdown=True,
        ),
        persistence=PersistenceConfig(enabled=True),
        spectrum=sweep.spectrum,
        entry_features=sweep.entry_features,
    )


def dedup_key(sweep: SweepConfig, candidate_panel_subdir: str) -> str:
    """
    Identity of a sweep run, for skip_existing decisions.

    Hashes the fully-built SimulatorConfig, so every parameter that affects
    the run is covered — including candidate_panel_subdir and nested fields
    like interval_scoring weights. SweepConfig.alias is a lossy display
    label (a subset of fields) and must not be used for dedup.
    """
    return hash_config(_build_sim_config(sweep, candidate_panel_subdir))


def _existing_keys(existing: pd.DataFrame) -> set[str]:
    """
    Dedup keys already present in the results index.

    Rows predating the config_hash column contribute no key, so they are
    re-run rather than wrongly skipped.
    """
    if existing.empty or "config_hash" not in existing.columns:
        return set()
    return set(existing["config_hash"].dropna().tolist())


# ---------------------------------------------------------------------------
# Sweep results I/O
# ---------------------------------------------------------------------------

SWEEP_RESULTS_FILENAME = "sweep_results.pkl"


def _sweep_results_path() -> Path:
    return Path(SIMULATION_RUNS_ROOT) / SWEEP_RESULTS_FILENAME


def load_sweep_results(*, rebuild: bool = False) -> pd.DataFrame:
    p = _sweep_results_path()
    if not rebuild and p.exists():
        return pd.read_pickle(p)

    from src.simulator.sweep_rebuild import rebuild_sweep_results
    df = rebuild_sweep_results()
    if not df.empty:
        _save_sweep_results(df)
    return df


def _save_sweep_results(df: pd.DataFrame) -> None:
    p = _sweep_results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p)


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------

_CONFIG_COLS = [
    "alias", "z_method", "z_lookback", "entry_z", "exit_z",
    "base_pair_notional", "vol_normalize", "max_ticker_exposure_pct", "max_gross_exposure",
    "excluded_sectors", "z_score_overrides", "cross_ts", "timescale_risk",
]

_RETURN_COLS = [
    "alias", "sharpe_net", "sortino_net", "calmar_net",
    "annual_return_net", "total_return_net",
    "max_drawdown_net", "max_drawdown_gross",
]

_TRADE_COLS = [
    "alias", "n_trades", "avg_holding_days",
    "win_rate_net", "profit_factor_net",
    "avg_concurrent_positions",
    "mean_gain_per_win_net", "mean_loss_per_loss_net",
]

_COST_COLS = [
    "alias", "total_costs", "total_transaction_costs", "total_borrow_costs",
    "cost_drag_bps", "avg_cost_per_trade", "avg_daily_borrow_cost",
]

_SIGNAL_COLS = [
    "alias", "z_method", "z_lookback", "entry_z", "exit_z",
    "sharpe_net", "annual_return_net", "max_drawdown_net",
    "n_trades", "avg_holding_days", "profit_factor_net",
    "avg_abs_entry_z_score", "avg_entry_z_score", "avg_exit_z_score",
]

_RISK_COLS = [
    "alias", "max_ticker_exposure_pct", "max_gross_exposure", "base_pair_notional",
    "vol_normalize", "sharpe_net", "annual_return_net", "max_drawdown_net", "calmar_net",
    "avg_concurrent_positions", "n_trades",
]

_UNIVERSE_COLS = [
    "alias", "excluded_sectors",
    "sharpe_net", "annual_return_net", "max_drawdown_net", "calmar_net",
    "n_trades", "n_groups_traded",
]


def _view(df: pd.DataFrame, cols: list[str], sort_by: str = "sharpe_net") -> pd.DataFrame:
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()
    if sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=False)
    return out


def view_returns(df: pd.DataFrame) -> pd.DataFrame:
    return _view(df, _RETURN_COLS, "sharpe_net")

def view_trades(df: pd.DataFrame) -> pd.DataFrame:
    return _view(df, _TRADE_COLS, "profit_factor_net")

def view_costs(df: pd.DataFrame) -> pd.DataFrame:
    return _view(df, _COST_COLS, "cost_drag_bps")

def view_signal(df: pd.DataFrame) -> pd.DataFrame:
    return _view(df, _SIGNAL_COLS, "sharpe_net")

def view_risk(df: pd.DataFrame) -> pd.DataFrame:
    return _view(df, _RISK_COLS, "calmar_net")

def view_universe(df: pd.DataFrame) -> pd.DataFrame:
    return _view(df, _UNIVERSE_COLS, "sharpe_net")

def view_config(df: pd.DataFrame) -> pd.DataFrame:
    return _view(df, _CONFIG_COLS, "alias")

def view_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['start_date', 'end_date',
        "alias", "sharpe_net", "calmar_net", "annual_return_net",
        "max_drawdown_net", "n_trades", "profit_factor_net",
        "avg_holding_days", "run_duration_s",
    ]
    return _view(df, cols, "sharpe_net")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_sweep(
    runs: list[SweepConfig],
    candidate_panel_subdir: str,
    *,
    skip_existing: bool = True,
) -> (pd.DataFrame, list[SimulationResult]):
    existing = load_sweep_results()
    results = existing.reset_index().to_dict("records") if not existing.empty else []
    existing_keys = _existing_keys(existing)

    n_total = len(runs)
    n_skipped = 0
    all_sweep_results = []

    for i, sweep in enumerate(runs):
        sim_config = _build_sim_config(sweep, candidate_panel_subdir)
        key = hash_config(sim_config)

        if skip_existing and key in existing_keys:
            print(f"[sweep {i+1}/{n_total}] Skipping {sweep.alias} [{key[:8]}] — already in results")
            n_skipped += 1
            continue

        print("=" * 100)
        print(f"[sweep {i+1}/{n_total}] {sweep.alias} [{key[:8]}]")
        print("=" * 100)

        t0 = _time.perf_counter()
        try:
            result = run_from_config(sim_config)
            all_sweep_results.append(result)
            duration_s = _time.perf_counter() - t0

            run_id = _extract_run_id(result)

            sweep_fields = {}
            for f in fields(sweep):
                val = getattr(sweep, f.name)
                if f.name == "z_score_overrides" and val is not None:
                    sweep_fields[f.name] = str([
                        f"{zc.residual_key}:zhl{zc.resolved_lookbacks()[0]}"
                        for zc in val
                    ])
                elif f.name == "cross_ts" and val is not None:
                    sweep_fields[f.name] = str(asdict(val) if hasattr(val, '__dataclass_fields__') else val)
                elif f.name == "timescale_risk" and val is not None:
                    sweep_fields[f.name] = str(asdict(val) if hasattr(val, '__dataclass_fields__') else val)
                elif f.name == "kelly" and val is not None:
                    sweep_fields[f.name] = str(asdict(val) if hasattr(val, '__dataclass_fields__') else val)
                elif f.name == "spectrum" and val is not None:
                    sweep_fields[f.name] = (
                        f"rhl={val.residual_lookbacks or 'all'} "
                        f"zlb={val.zlb_values or 'auto'} "
                        f"exit={val.record_exit}"
                    )
                else:
                    sweep_fields[f.name] = val

            row = {
                "run_id": run_id,
                "alias": sweep.alias,
                "run_timestamp": datetime.now().isoformat(),
                "run_duration_s": round(duration_s, 1),
                **sweep_fields,
                **result.performance.metrics,
                # After the splats: sweep_fields carries no panel/hash, and
                # these are the real dedup identity of the run.
                "config_hash": key,
                "candidate_panel_subdir": candidate_panel_subdir,
            }
        except Exception as e:
            duration_s = _time.perf_counter() - t0
            print(f"[sweep] FAILED after {duration_s:.0f}s: {e}")
            row = {
                "run_id": f"FAILED_{datetime.now().strftime('%Y%m%d_%H%M')}",
                "alias": sweep.alias,
                "run_timestamp": datetime.now().isoformat(),
                "run_duration_s": round(duration_s, 1),
                "error": str(e),
                **{f.name: getattr(sweep, f.name) for f in fields(sweep)},
                "config_hash": key,
                "candidate_panel_subdir": candidate_panel_subdir,
            }

        results.append(row)
        existing_keys.add(key)

        df = pd.DataFrame(results)
        df = df.set_index("run_id")
        _save_sweep_results(df)
        print(f"[sweep] Saved {len(results)} results ({n_skipped} skipped)")

    df = pd.DataFrame(results).set_index("run_id")
    return df, all_sweep_results


# ---------------------------------------------------------------------------
# Parallel sweep
# ---------------------------------------------------------------------------

def _run_one_sweep(args: tuple) -> dict:
    """Worker function for run_sweep_parallel."""
    sweep, candidate_panel_subdir, i, n_total = args
    import time as _t
    from src.simulator.sweep_runner import _build_sim_config, _extract_run_id
    from src.simulator.simulator_factory import run_from_config
    from src.simulator.simulation_persistence import hash_config
    from dataclasses import fields
    from datetime import datetime

    print(f"[worker] START [{i+1}/{n_total}] {sweep.alias}", flush=True)
    t0 = _t.perf_counter()
    sim_config = _build_sim_config(sweep, candidate_panel_subdir)
    key = hash_config(sim_config)
    try:
        result = run_from_config(sim_config)
        duration_s = _t.perf_counter() - t0
        run_id = _extract_run_id(result)

        sweep_fields = {}
        for f in fields(sweep):
            val = getattr(sweep, f.name)
            if f.name == "z_score_overrides" and val is not None:
                sweep_fields[f.name] = str([
                    f"{zc.residual_key}:zhl{zc.resolved_lookbacks()[0]}"
                    for zc in val
                ])
            else:
                sweep_fields[f.name] = val

        row = {
            "run_id": run_id,
            "alias": sweep.alias,
            "run_timestamp": datetime.now().isoformat(),
            "run_duration_s": round(duration_s, 1),
            **sweep_fields,
            **result.performance.metrics,
            "config_hash": key,
            "candidate_panel_subdir": candidate_panel_subdir,
        }
        print(f"[worker] DONE  [{i+1}/{n_total}] {sweep.alias} [{key[:8]}] in {duration_s/60:.1f}min", flush=True)
    except Exception as e:
        duration_s = _t.perf_counter() - t0
        print(f"[worker] FAIL  [{i+1}/{n_total}] {sweep.alias}: {e}", flush=True)
        row = {
            "run_id": f"FAILED_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "alias": sweep.alias,
            "run_timestamp": datetime.now().isoformat(),
            "run_duration_s": round(duration_s, 1),
            "error": str(e),
            **{f.name: getattr(sweep, f.name) for f in fields(sweep)},
            "config_hash": key,
            "candidate_panel_subdir": candidate_panel_subdir,
        }
    return row


def run_sweep_parallel(
    runs: list[SweepConfig],
    candidate_panel_subdir: str,
    *,
    n_workers: int = 8,
    skip_existing: bool = True,
) -> pd.DataFrame:
    import multiprocessing as mp

    existing = load_sweep_results()
    existing_rows = existing.reset_index().to_dict("records") if not existing.empty else []
    existing_keys = _existing_keys(existing)

    pending = [
        s for s in runs
        if not (skip_existing and dedup_key(s, candidate_panel_subdir) in existing_keys)
    ]
    skipped = len(runs) - len(pending)
    print(f"[sweep_parallel] {len(pending)} runs to execute, {skipped} skipped. Workers: {n_workers}")

    if not pending:
        return existing

    args = [(sweep, candidate_panel_subdir, i, len(pending)) for i, sweep in enumerate(pending)]

    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=n_workers) as pool:
        new_rows = pool.map(_run_one_sweep, args)

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows).set_index("run_id")
    _save_sweep_results(df)
    print(f"[sweep_parallel] Saved {len(all_rows)} results ({skipped} skipped)")
    return df
