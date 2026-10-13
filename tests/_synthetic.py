"""
Deterministic synthetic market data for hermetic end-to-end tests.

Replaces the gitignored ``data/`` directory: seeded random-walk universes with a
few built-in mean-reverting pairs (member B = member A x exp(OU spread)), so the
pair pipeline finds candidates and trades without any download.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.universe_marketdata import UniverseMarketData

FIELDS = ("Open", "High", "Low", "Close", "Volume")
N_DAYS = 900
START = "2018-01-01"


def _ohlcv(close: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    spread = np.abs(rng.normal(0.0, 0.002, len(close))) * close.to_numpy()
    return pd.DataFrame(
        {
            "Open": close.to_numpy(),
            "High": close.to_numpy() + spread,
            "Low": close.to_numpy() - spread,
            "Close": close.to_numpy(),
            "Volume": np.full(len(close), 1_000_000.0),
        },
        index=close.index,
    )


def make_group_prices(
    group_id: str, n_pairs: int, rng: np.random.Generator, index: pd.DatetimeIndex
) -> tuple[dict[str, pd.Series], str, str]:
    """Close series for 2*n_pairs members + proxy + benchmark of one group."""
    n = len(index)
    market = rng.normal(0.0003, 0.008, n)
    sector = rng.normal(0.0, 0.006, n)
    closes: dict[str, pd.Series] = {}
    for p in range(n_pairs):
        base_ret = market + 0.8 * sector + rng.normal(0.0, 0.010, n)
        a = 50.0 * np.exp(np.cumsum(base_ret))
        # OU log-spread: theta pulls it back to 0 (half-life ~ 8 days)
        ou = np.zeros(n)
        for t in range(1, n):
            ou[t] = ou[t - 1] * 0.92 + rng.normal(0.0, 0.02)
        b = a * np.exp(ou)
        closes[f"{group_id[:3].upper()}{p}A"] = pd.Series(a, index=index)
        closes[f"{group_id[:3].upper()}{p}B"] = pd.Series(b, index=index)
    proxy = f"{group_id[:3].upper()}ETF"
    bench = "SPYX"
    closes[proxy] = pd.Series(100.0 * np.exp(np.cumsum(market + 0.8 * sector)), index=index)
    return closes, proxy, bench


def make_umd(
    groups: dict[str, int] | None = None, seed: int = 7, n_days: int = N_DAYS
) -> UniverseMarketData:
    """``groups`` maps group_id -> number of pairs (2 members each)."""
    groups = groups or {"alpha": 3, "beta": 3}
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(START, periods=n_days)
    frames: dict[str, pd.DataFrame] = {}
    membership_rows: list[dict[str, str]] = []
    group_rows: dict[str, dict[str, str]] = {}
    market = rng.normal(0.0003, 0.008, n_days)
    frames["SPYX"] = _ohlcv(pd.Series(400.0 * np.exp(np.cumsum(market)), index=index), rng)
    frames["IRXX"] = _ohlcv(pd.Series(np.full(n_days, 2.0), index=index), rng)
    for gid, n_pairs in groups.items():
        closes, proxy, bench = make_group_prices(gid, n_pairs, rng, index)
        for ticker, close in closes.items():
            frames[ticker] = _ohlcv(close, rng)
            if ticker != proxy:
                membership_rows.append({"group_id": gid, "ticker": ticker})
        group_rows[gid] = {"proxy_etf": proxy, "benchmark": bench, "risk_free": "IRXX"}
    prices = pd.concat(frames, axis=1)
    prices.columns = pd.MultiIndex.from_tuples(
        cast("list[tuple[str, str]]", list(prices.columns)), names=["ticker", "field"]
    )
    membership = pd.DataFrame(membership_rows)
    # only true members belong to groups; proxy/benchmark/risk-free are price-only columns
    ticker_info = pd.DataFrame({"ticker": sorted(frames)}).set_index("ticker")
    group_info = pd.DataFrame.from_dict(group_rows, orient="index")
    group_info.index.name = "group_id"
    return UniverseMarketData(
        prices=prices, ticker_info=ticker_info, group_info=group_info, membership=membership
    )


def build_sim_config(groups: list[str], *, max_steps: int = 30, entry_z: float = 1.5):
    """SimulatorConfig for live generation on the synthetic universe (no disk I/O)."""
    from src.candidates.pair_candidate_panel_creator import PairSpreadConfig
    from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode
    from src.simulator.config import (
        CandidateGenerationConfig,
        DataConfig,
        GroupDataSource,
        PairSpreadTraderConfig,
        PerformanceConfig,
        PersistenceConfig,
        RiskManagerConfig,
        RunConfig,
        SimulatorConfig,
        SizingConfig,
        VolSizingConfig,
        ZScoreConfig,
    )
    from src.simulator.sweep_defaults import get_default_bundle, merge_defaults

    residual = CausalResidualConfig(
        mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=True, min_lb_eq_exp=20
    )
    sweep_derived = {
        "data": DataConfig(
            groups=[
                GroupDataSource(universe_config_name=f"synthetic/{g}.yaml", candidate_panel_stem="")
                for g in groups
            ],
            selected_groups=groups,
            universe_name="synthetic",
            data_path="unused",
        ),
        "residual": residual,
        "candidate_generation": CandidateGenerationConfig(
            pair_cfg=PairSpreadConfig(
                hedge_ratio_methods=["pca"],
                min_obs=252,
                min_return_std=1e-8,
                min_level_std=1e-8,
                min_kappa=1e-6,
                max_half_life=126.0,
                tiny_weight_threshold=1e-6,
            ),
            hedge_ratio_lb=252,
            mr_diag_lb=252,
            frequency="W-FRI",
            start_date="2019-03-01",
            max_steps=max_steps,
        ),
        "z_score": ZScoreConfig(lookback=21, ddof=1, method="ewm", residual_key=residual.key),
        "trader": PairSpreadTraderConfig(entry_z=entry_z, exit_z=0.0),
        "sizing": SizingConfig(
            base_pair_notional=100_000.0,
            vol_normalize=VolSizingConfig(floor_multiplier=0.2, cap_multiplier=5.0),
        ),
        "risk_manager": RiskManagerConfig(max_gross_exposure=10.0, max_ticker_exposure_pct=0.15),
        "run": RunConfig(progress=False, progress_step=10, start_date=None, end_date=None),
        "entry_features": None,
    }
    bundle = dict(get_default_bundle("standard_v1"))
    bundle["performance"] = PerformanceConfig(
        enabled=True, metrics_table=True, report_html=False, per_group_breakdown=False
    )
    bundle["persistence"] = PersistenceConfig(enabled=False)
    return SimulatorConfig(**merge_defaults(sweep_derived, bundle, "standard_v1"))


@contextlib.contextmanager
def patched_universe(umd: UniverseMarketData) -> Iterator[None]:
    """Route every simulator_factory universe load to the in-memory ``umd``."""

    class _FakeLoader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def load(self, **kwargs: Any) -> UniverseMarketData:
            return umd

    mod = "src.simulator.simulator_factory"
    with (
        patch(f"{mod}._load_umd", return_value=umd),
        patch(f"{mod}.UniverseConfig.from_yaml", return_value=object()),
        patch(f"{mod}.UniverseDataLoader", _FakeLoader),
    ):
        yield
