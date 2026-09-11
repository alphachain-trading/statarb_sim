"""
Spread-level and stock-residual series persistence.

Precomputes, once, the residual-return series per ticker and the spread level
series per candidate so that the simulator can *load* levels from disk instead
of recomputing ``_get_residuals @ weights -> cumsum`` on every step.

Layout (under ``panel_dir``):

    series/stock/{group_id}__{ticker}.parquet   # single column "residual_return"
    series/spread/{spread_id}__{asof}.parquet   # single column "level"

Spread files are keyed by ``(spread_id, asof_date)``: the same spread recurs
across many weekly asof-dates with re-estimated weights and a re-fitted residual
model, and each candidate instance freezes both at its own asof-date. The ``|``
separator in a spread_id is replaced with ``_`` so it is filesystem-safe, and
the asof-date is appended as ``YYYYMMDD``.

Because ``apply_causal_residual_model`` is row-wise (the residual at date t is a
function only of that row's returns and the fixed model coefficients), applying
a model frozen at ``asof_date`` over the full return history yields the correct
residual for every date — including dates after ``asof_date`` (out-of-sample) —
so a single persisted series per candidate can be sliced at any later sim date.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.returns import GroupReturnBundle, build_group_return_bundle
from src.data.universe_marketdata import UniverseMarketData
from src.residuals.causal_residuals import (
    FittedCausalResidualModel,
    apply_causal_residual_model,
)

# Bundle-construction defaults — kept in sync with CandidateSignalGenerator.
_FIELD = "Close"
_RETURN_METHOD = "log"
_DROPNA = "any"


def sanitize_spread_id(spread_id: str) -> str:
    """Filesystem-safe spread_id (``|`` -> ``_``)."""
    return spread_id.replace("|", "_")


def spread_series_filename(spread_id: str, asof_date: pd.Timestamp) -> str:
    """Canonical ``{spread_id}__{YYYYMMDD}.parquet`` name for a candidate."""
    asof = pd.Timestamp(asof_date)
    return f"{sanitize_spread_id(spread_id)}__{asof.strftime('%Y%m%d')}.parquet"


def _rkey_series(panel: pd.DataFrame) -> pd.Series:
    """residual_key per row; defaults to "" when the column is absent."""
    if "residual_key" in panel.columns:
        return panel["residual_key"].fillna("").astype(str)
    return pd.Series("", index=panel.index)


def compute_and_persist_series(
    panel_dir: str | Path,
    candidate_panel: pd.DataFrame,
    residual_params: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]],
    market_data: UniverseMarketData,
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]],
) -> None:
    """
    Compute and persist stock-residual and spread-level series under ``panel_dir``.

    Parameters
    ----------
    panel_dir
        Candidate-panel directory; series are written under ``series/``.
    candidate_panel
        Loaded candidate panel (one row per (spread_id, asof_date) candidate).
        Must carry: ``spread_id``, ``group_id``, ``asof_date``.
    residual_params
        ``{(group_id, residual_key): {fit_date: FittedCausalResidualModel}}`` —
        the same in-memory form the simulator factory assembles from the
        ``*_residual_params.parquet`` files. ``fit_date`` is the residual model's
        own (daily) fit date, distinct from a candidate's ``asof_date`` (its
        weights fit date, at outer refit dates); a candidate's ``asof_date``
        is always a valid ``fit_date`` to look up here, since candidates only
        arise on dates the daily residual grid also covers.
    market_data
        Loaded UniverseMarketData used to build per-group return bundles.
    weights_lookup
        ``{(spread_id, asof_date): {ticker: weight}}``, loaded from
        weights.parquet at full float64 precision (F1 commit 4) — the same
        in-memory form CandidateFilter.build_candidate_refs consumes.
        Replaces the old per-row "weights" JSON column on candidate_panel.

    Individual files that already exist are skipped, so the function is
    incremental / resumable across runs.
    """
    panel_dir = Path(panel_dir)
    stock_dir = panel_dir / "series" / "stock"
    spread_dir = panel_dir / "series" / "spread"
    stock_dir.mkdir(parents=True, exist_ok=True)
    spread_dir.mkdir(parents=True, exist_ok=True)

    panel = candidate_panel.copy()
    # Helper columns must not start with "_": itertuples mangles such names.
    panel["series_rkey"] = _rkey_series(panel)
    panel["series_asof"] = pd.to_datetime(panel["asof_date"])

    bundle_cache: dict[str, GroupReturnBundle] = {}

    def get_bundle(group_id: str) -> GroupReturnBundle:
        bundle = bundle_cache.get(group_id)
        if bundle is None:
            bundle = build_group_return_bundle(
                umd=market_data,
                group_id=group_id,
                field=_FIELD,
                return_method=_RETURN_METHOD,
                dropna=_DROPNA,
            )
            bundle_cache[group_id] = bundle
        return bundle

    def full_residuals(
        group_id: str,
        residual_key: str,
        fit_date: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Full-history residual matrix for a (group, key) frozen at fit_date."""
        params = residual_params.get((group_id, residual_key))
        if params is None:
            return None
        model = params.get(pd.Timestamp(fit_date))
        if model is None:
            return None
        bundle = get_bundle(group_id)
        return apply_causal_residual_model(
            model=model,
            aligned_returns=bundle.aligned_returns,
            rf_series=bundle.risk_free_returns if model.subtract_risk_free else None,
        )

    # ── Loop 1: stock residual series (one per unique ticker) ──────────────
    # Use, per (group, key), the model frozen at that group's most recent
    # asof-date in the panel — one representative residual series per ticker.
    group_latest: dict[tuple[str, str], pd.Timestamp] = {}
    for (group_id, rkey), sub in panel.groupby(["group_id", "series_rkey"], sort=True):
        group_latest[(str(group_id), str(rkey))] = pd.Timestamp(sub["series_asof"].max())

    group_resid_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

    def latest_group_resid(group_id: str, rkey: str) -> pd.DataFrame | None:
        key = (group_id, rkey)
        if key not in group_resid_cache:
            group_resid_cache[key] = full_residuals(group_id, rkey, group_latest[key])
        return group_resid_cache[key]

    # Keyed by (group_id, ticker), not ticker alone: residual fits are
    # group-scoped, so a ticker could carry a different residual series per
    # group once the multi-group ticker restriction (simulator_factory
    # ._merge_umds) lifts. Costs nothing today, since that restriction
    # currently guarantees each ticker resolves to exactly one group.
    ticker_group_rkey: dict[tuple[str, str], str] = {}
    for group_id, rkey in group_latest:
        resid = latest_group_resid(group_id, rkey)
        if resid is None:
            continue
        for ticker in resid.columns:
            ticker_group_rkey.setdefault((group_id, str(ticker)), rkey)

    n_stock_written = 0
    for group_id, ticker in tqdm(
        sorted(ticker_group_rkey), desc="[series] stock residuals", unit="ticker"
    ):
        path = stock_dir / f"{group_id}__{ticker}.parquet"
        if path.exists():
            continue
        rkey = ticker_group_rkey[(group_id, ticker)]
        resid = latest_group_resid(group_id, rkey)
        if resid is None or ticker not in resid.columns:
            continue
        col = resid[[ticker]].rename(columns={ticker: "residual_return"})
        col.to_parquet(path)
        n_stock_written += 1

    # ── Loop 2: spread level series (one per candidate = spread_id × asof) ──
    # Process asof-dates in order and hold only the current asof's residual
    # matrix in memory (all spreads on one asof share the same frozen model).
    spread_panel = panel.sort_values(["series_asof", "group_id"], kind="stable")

    resid_cache: "OrderedDict[tuple[str, str, pd.Timestamp], pd.DataFrame | None]" = OrderedDict()

    def asof_resid(
        group_id: str,
        rkey: str,
        asof: pd.Timestamp,
    ) -> pd.DataFrame | None:
        key = (group_id, rkey, asof)
        if key not in resid_cache:
            resid_cache.clear()  # only the current asof is needed at a time
            resid_cache[key] = full_residuals(group_id, rkey, asof)
        return resid_cache[key]

    n_spread_written = 0
    for row in tqdm(
        spread_panel.itertuples(index=False),
        total=len(spread_panel),
        desc="[series] spread levels",
        unit="spread",
    ):
        spread_id = str(row.spread_id)
        group_id = str(row.group_id)
        rkey = str(row.series_rkey)
        asof = pd.Timestamp(row.series_asof)

        path = spread_dir / spread_series_filename(spread_id, asof)
        if path.exists():
            continue

        resid = asof_resid(group_id, rkey, asof)
        if resid is None:
            continue

        weight_dict = weights_lookup.get((spread_id, asof))
        if weight_dict is None:
            raise KeyError(
                f"No weights found for spread_id={spread_id!r} asof_date={asof} "
                f"— weights.parquet is missing this (spread_id, asof_date) key."
            )
        members = list(weight_dict.keys())
        weights = np.asarray([weight_dict[m] for m in members], dtype=float)
        if any(m not in resid.columns for m in members):
            continue

        spread_return = resid.loc[:, members].to_numpy(dtype=float) @ weights
        level = np.cumsum(spread_return)
        df = pd.DataFrame({"level": level}, index=resid.index)
        df.to_parquet(path)
        n_spread_written += 1

    print(
        f"[series] Persisted {n_stock_written} stock and {n_spread_written} spread "
        f"series under {panel_dir / 'series'} "
        f"(existing files skipped)."
    )
