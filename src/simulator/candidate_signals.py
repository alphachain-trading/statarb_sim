from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field as dc_field
import numpy as np
import pandas as pd

from src.analytics.spread_primitives import compute_spread_level
from src.simulator.config import MRDiagnosticsConfig, ZScoreConfig
from src.simulator.types import CandidateAnalyticsState, CandidateRef, ZScoreComponent
from src.data.returns import GroupReturnBundle, build_group_return_bundle
from src.data.universe_marketdata import UniverseMarketData
from src.residuals.causal_residuals import (
    CausalResidualConfig,
    FittedCausalResidualModel,
    apply_causal_residual_model,
)

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover
    adfuller = None

logger = logging.getLogger(__name__)

# Max number of asof-frozen full-history residual matrices to hold in memory
# (LRU). F2 R2's cache is one matrix per (group_id, residual_key, asof_date),
# evicted when no tracked candidate or open position refers to it; an LRU
# bound is used here instead of precise ref-counted eviction, since
# F2_spec.md P4 measured a peak of only 164 concurrently-live keys on a real
# run — well under this bound, so LRU eviction is not expected to ever fire
# in practice at that scale, only to cap memory in a pathological case.
_ASOF_RESIDUAL_CACHE_MAX = 4096


def _rolling_mean_std(
    x: np.ndarray,
    window: int,
    min_periods: int,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized rolling mean and std over a (T, N) array.

    Uses cumsum trick for O(T*N) computation — no per-column loops.
    Returns (mean, std) each of shape (T, N).  Rows with fewer than
    min_periods observations are NaN.
    """
    T, N = x.shape
    mean_out = np.full((T, N), np.nan)
    std_out = np.full((T, N), np.nan)

    cum = np.cumsum(x, axis=0)        # (T, N)
    cum2 = np.cumsum(x ** 2, axis=0)  # (T, N)

    # Prepend a zero row for the subtraction trick
    cum_pad = np.vstack([np.zeros((1, N)), cum])    # (T+1, N)
    cum2_pad = np.vstack([np.zeros((1, N)), cum2])  # (T+1, N)

    for t in range(min_periods - 1, T):
        w = min(t + 1, window)
        s = cum_pad[t + 1] - cum_pad[t + 1 - w]        # (N,)
        s2 = cum2_pad[t + 1] - cum2_pad[t + 1 - w]     # (N,)
        m = s / w
        mean_out[t] = m
        if w > ddof:
            # Var = (sum_x2 - n*mean^2) / (n - ddof)
            var = (s2 - w * m * m) / (w - ddof)
            var = np.maximum(var, 0.0)
            std_out[t] = np.sqrt(var)

    return mean_out, std_out


def _rolling_mean_std_last(
    x: np.ndarray,
    window: int,
    min_periods: int,
    ddof: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling mean and std at the LAST row only.

    x: (T, N).  Returns (mean, std) each shape (N,).
    Rows where T < min_periods return NaN.
    """
    T, N = x.shape
    if T < min_periods:
        return np.full(N, np.nan), np.full(N, np.nan)
    w = min(T, window)
    tail = x[-w:]               # (w, N)
    m = tail.mean(axis=0)       # (N,)
    s = tail.std(axis=0, ddof=ddof)  # (N,)
    return m, s


def _ewm_mean_std_last(
    x: np.ndarray,
    halflife: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    EWM mean and std at the LAST row only, vectorized across columns.

    x: (T, N).  Returns (mean, std) each shape (N,).
    Uses pandas ewm internally to guarantee exact match with pandas semantics
    (adjust=True, bias=False).  ~75ms for (500, 1500) which is acceptable
    for 10 groups per step.  Can be replaced with a pure numpy implementation
    if it becomes a bottleneck.
    """
    T, N = x.shape
    if T < min_periods:
        return np.full(N, np.nan), np.full(N, np.nan)
    df = pd.DataFrame(x)
    ewm = df.ewm(halflife=halflife, min_periods=min_periods)
    m = ewm.mean().iloc[-1].to_numpy()
    s = ewm.std().iloc[-1].to_numpy()
    return m, s


@dataclass(slots=True)
class CandidateSignalGenerator:
    """
    Signal generator supporting single and multi-timescale modes.

    Single-timescale (backward compatible):
        z_score_configs = {"": ZScoreConfig(...)}
        residual_configs = {"": CausalResidualConfig(...)}

    Multi-timescale:
        z_score_configs = {"exp_hl63_mh126": ZScoreConfig(...), "exp_hl126_mh252": ZScoreConfig(...)}
        residual_configs = {"exp_hl63_mh126": CausalResidualConfig(...), ...}

    Residuals are reconstructed once per (group_id, residual_key, asof_date) —
    the candidate's own frozen refit date, never "today" (F2 R1) — and shared
    across all z-score configs that use the same residual_key.
    """
    z_score_configs: dict[str, ZScoreConfig]
    diagnostics_config: MRDiagnosticsConfig
    residual_configs: dict[str, CausalResidualConfig]
    umd: UniverseMarketData

    price_field: str = "Close"
    return_method: str = "log"
    dropna: str = "any"

    # Precomputed fitted residual model params.
    # Key: (group_id, residual_key) → {date: FittedCausalResidualModel}
    # When residual_key="" (single-timescale), legacy key (group_id, "") is used.
    precomputed_residual_params: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] = dc_field(
        default_factory=dict, init=True, repr=False,
    )

    _bundle_cache: dict[str, GroupReturnBundle] = dc_field(default_factory=dict, init=False, repr=False)
    _asof_residual_cache: "OrderedDict[tuple[str, str, pd.Timestamp], pd.DataFrame]" = dc_field(default_factory=OrderedDict, init=False, repr=False)

    def _get_z_score_config(self, timescale_label: str) -> ZScoreConfig:
        """Look up ZScoreConfig by timescale_label (unique per residual_key + zlb)."""
        zc = self.z_score_configs.get(timescale_label)
        if zc is None:
            raise KeyError(
                f"No ZScoreConfig for timescale_label={timescale_label!r}. "
                f"Available: {sorted(self.z_score_configs.keys())}"
            )
        return zc

    def build_signal_frame(
        self,
        *,
        date: pd.Timestamp,
        activation_frame: pd.DataFrame,
        candidate_refs: list[CandidateRef],
        precomputed_analytics: dict[str, CandidateAnalyticsState] | None = None,
    ) -> pd.DataFrame:
        date = pd.Timestamp(date)

        if activation_frame.empty:
            return self._empty_signal_frame()

        if precomputed_analytics is not None:
            analytics_by_id = precomputed_analytics
        else:
            analytics_by_id = self.build_candidate_analytics_states(
                date=date,
                candidate_refs=candidate_refs,
            )

        rows = []
        for candidate_id, act_row in activation_frame.iterrows():
            a = analytics_by_id[candidate_id]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "spread_id": act_row["spread_id"],
                    "group_id": act_row["group_id"],
                    "is_new_arrival": bool(act_row["is_new_arrival"]),
                    "is_active": bool(act_row["is_active"]),
                    "level": a.level,
                    "roll_mean": a.roll_mean,
                    "roll_std": a.roll_std,
                    "z_score": a.z_score,
                    "is_signal_ready": a.is_signal_ready,
                }
            )

        out = pd.DataFrame(rows).set_index("candidate_id")
        return out.sort_values(["group_id", "spread_id"], kind="stable")

    def build_candidate_analytics_states(
        self,
        *,
        date: pd.Timestamp,
        candidate_refs: list[CandidateRef],
        skip_diagnostics: bool = False,
    ) -> dict[str, CandidateAnalyticsState]:
        date = pd.Timestamp(date)

        # Group refs by (group_id, residual_key) for batched processing.
        # All candidates sharing the same (group_id, residual_key) use the
        # same residual matrix and z-score config.
        refs_by_group_ts: dict[tuple[str, str], list[CandidateRef]] = {}
        for ref in candidate_refs:
            # Group by (group_id, residual_key, timescale_label).
            # In multi-zlb sweep mode, refs are pre-fanned-out by CandidateFilter
            # so each ref already carries a unique timescale_label.
            tsl = ref.timescale_label or ref.residual_key
            key = (ref.group_id, ref.residual_key, tsl)
            refs_by_group_ts.setdefault(key, []).append(ref)

        out: dict[str, CandidateAnalyticsState] = {}
        for (group_id, residual_key, timescale_label), group_refs in refs_by_group_ts.items():
            batch = self._batch_analytics_for_group(
                group_id=group_id,
                residual_key=residual_key,
                timescale_label=timescale_label,
                refs=group_refs,
                date=date,
                skip_diagnostics=skip_diagnostics,
            )
            out.update(batch)

        # ── Post-pass: attach cross-timescale z_components ──────────
        # In multi-timescale mode, each CandidateAnalyticsState has its own
        # timescale's z_components (e.g. 1 lookback).  For downstream use
        # (entry/exit analysis), we want each state to carry the full set of
        # primary z-score components across ALL timescales for the same spread.
        #
        # Group by spread_id, collect primary component (index 0) from each
        # timescale sibling, and replace z_components on every sibling.
        if len(refs_by_group_ts) > 1:
            # Multiple (group, rkey) batches → multi-timescale mode
            spread_siblings: dict[str, list[CandidateAnalyticsState]] = {}
            for state in out.values():
                if state.z_components:
                    spread_siblings.setdefault(state.spread_id, []).append(state)

            spread_cross_comps: dict[str, tuple[ZScoreComponent, ...]] = {}
            for sid, siblings in spread_siblings.items():
                sorted_sibs = sorted(siblings, key=lambda s: s.residual_key)
                spread_cross_comps[sid] = tuple(
                    s.z_components[0] for s in sorted_sibs
                )

            # Replace z_components on each state with the cross-timescale tuple
            for cid, state in out.items():
                cross = spread_cross_comps.get(state.spread_id)
                if cross and cross != state.z_components:
                    # Frozen dataclass — rebuild with updated z_components
                    out[cid] = CandidateAnalyticsState(
                        candidate_id=state.candidate_id,
                        group_id=state.group_id,
                        spread_id=state.spread_id,
                        date=state.date,
                        z_score=state.z_score,
                        level=state.level,
                        roll_mean=state.roll_mean,
                        roll_std=state.roll_std,
                        adf_pvalue=state.adf_pvalue,
                        mr_score=state.mr_score,
                        kappa=state.kappa,
                        half_life=state.half_life,
                        is_signal_ready=state.is_signal_ready,
                        z_components=cross,
                        residual_key=state.residual_key,
                    )

        return out

    def _batch_analytics_for_group(
        self,
        *,
        group_id: str,
        residual_key: str,
        timescale_label: str = "",
        refs: list[CandidateRef],
        date: pd.Timestamp,
        skip_diagnostics: bool = False,
    ) -> dict[str, CandidateAnalyticsState]:
        """
        Vectorized z-score computation for all candidates in one (group, timescale).

        Builds a (T, n_candidates) level matrix via each candidate's own
        asof-frozen residual matrix, then computes rolling stats across all
        candidates at once. Falls back to not-ready for candidates with
        missing tickers.
        """
        # Recompute residuals @ weights -> cumsum via the shared
        # spread_primitives.compute_spread_level primitive (F2 R2), always
        # against each candidate's own asof-frozen model (F2 R1) — never
        # "today" (`date`). Refs sharing (group_id, residual_key) can still
        # carry different asof_date values: an open position keeps its own
        # frozen ref while a newer candidate becomes active for the same
        # spread (candidate_activation.py's pair-mode rule), so group by
        # asof_date within the batch and resolve each date's full-history
        # residual matrix once (cached — see _get_asof_residuals).
        valid_refs: list[CandidateRef] = []
        invalid_refs: list[CandidateRef] = []
        level_by_id: dict[str, pd.Series] = {}
        spread_return_by_id: dict[str, pd.Series] = {}

        refs_by_asof: dict[pd.Timestamp, list[CandidateRef]] = {}
        for ref in refs:
            refs_by_asof.setdefault(pd.Timestamp(ref.asof_date), []).append(ref)

        for asof_date, asof_refs in refs_by_asof.items():
            try:
                residuals = self._get_asof_residuals(group_id, residual_key, asof_date)
            except ValueError:
                invalid_refs.extend(asof_refs)
                continue

            for ref in asof_refs:
                if not all(m in residuals.columns for m in ref.members):
                    invalid_refs.append(ref)
                    continue
                weights_by_ticker = dict(zip(ref.members, ref.weights))
                spread_return, level = compute_spread_level(residuals, weights_by_ticker)
                spread_return_by_id[ref.candidate_id] = spread_return.loc[:date]
                level_by_id[ref.candidate_id] = level.loc[:date]
                valid_refs.append(ref)

        if not valid_refs:
            return {
                ref.candidate_id: self._not_ready_analytics(ref=ref, date=date)
                for ref in invalid_refs
            }

        # Every ref in this batch shares one group_id, so every pre-truncation
        # series shares one index (bundle.aligned_returns.index); truncating
        # each to `.loc[:date]` above can only drop trailing (future) rows
        # from that same shared index, so they still align after truncation —
        # no union/reindex needed (unlike the disk path, whose persisted
        # series can have independently-bounded ranges on disk).
        level_index = level_by_id[valid_refs[0].candidate_id].index
        levels = np.column_stack([
            level_by_id[ref.candidate_id].to_numpy(dtype=float) for ref in valid_refs
        ])
        spread_returns = np.column_stack([
            spread_return_by_id[ref.candidate_id].to_numpy(dtype=float) for ref in valid_refs
        ])

        return self._finalize_batch_states(
            valid_refs=valid_refs,
            invalid_refs=invalid_refs,
            levels=levels,
            level_index=level_index,
            spread_returns=spread_returns,
            residual_key=residual_key,
            timescale_label=timescale_label,
            date=date,
            skip_diagnostics=skip_diagnostics,
        )

    def _finalize_batch_states(
        self,
        *,
        valid_refs: list[CandidateRef],
        invalid_refs: list[CandidateRef],
        levels: np.ndarray,          # (T, n_valid)
        level_index: pd.Index,       # DatetimeIndex for the level rows
        spread_returns: np.ndarray,  # (T, n_valid)
        residual_key: str,
        timescale_label: str,
        date: pd.Timestamp,
        skip_diagnostics: bool,
    ) -> dict[str, CandidateAnalyticsState]:
        """
        Shared downstream: rolling z-scores + MR diagnostics from a level matrix.
        """
        out: dict[str, CandidateAnalyticsState] = {
            ref.candidate_id: self._not_ready_analytics(ref=ref, date=date)
            for ref in invalid_refs
        }
        if not valid_refs:
            return out

        n_cands = len(valid_refs)

        # Z-score config for this (group, timescale)
        z_cfg = self._get_z_score_config(timescale_label)
        lookbacks = z_cfg.resolved_lookbacks()
        weights = z_cfg.resolved_weights()
        ddof = z_cfg.ddof
        method = z_cfg.method

        # Compute stats at last row only — O(window*N) or O(T*N) per lookback
        rolling_last: list[tuple[np.ndarray, np.ndarray]] = []
        for lb in lookbacks:
            min_p = z_cfg.resolved_min_periods(lookback=lb)
            if method == "ewm":
                rm, rs = _ewm_mean_std_last(levels, halflife=lb, min_periods=min_p)
            else:
                rm, rs = _rolling_mean_std_last(levels, window=lb, min_periods=min_p, ddof=ddof)
            rolling_last.append((rm, rs))

        # Extract last-row values for all candidates
        level_vals = levels[-1]  # (n_cands,)

        # Compute blended z-scores and components
        z_blended = np.zeros(n_cands, dtype=float)
        all_ready = np.ones(n_cands, dtype=bool)
        components_list: list[list[ZScoreComponent]] = [[] for _ in range(n_cands)]

        for k, (lb, w) in enumerate(zip(lookbacks, weights)):
            rm_last = rolling_last[k][0]  # (n_cands,)
            rs_last = rolling_last[k][1]  # (n_cands,)

            for j in range(n_cands):
                m_val = rm_last[j]
                s_val = rs_last[j]
                if np.isnan(m_val) or np.isnan(s_val) or s_val <= 0.0:
                    components_list[j].append(
                        ZScoreComponent(lookback=lb, z_score=None, roll_mean=None, roll_std=None)
                    )
                    all_ready[j] = False
                else:
                    z_i = (level_vals[j] - m_val) / s_val
                    components_list[j].append(
                        ZScoreComponent(lookback=lb, z_score=float(z_i), roll_mean=float(m_val), roll_std=float(s_val))
                    )
                    z_blended[j] += w * z_i

        # Primary lookback's roll_mean/roll_std for backward compat
        primary_rm_last = rolling_last[0][0]
        primary_rs_last = rolling_last[0][1]

        # Build analytics for each valid candidate
        for j, ref in enumerate(valid_refs):
            if all_ready[j]:
                z_score = float(z_blended[j])
                level = float(level_vals[j])
                rm_val = float(primary_rm_last[j]) if not np.isnan(primary_rm_last[j]) else None
                rs_val = float(primary_rs_last[j]) if not np.isnan(primary_rs_last[j]) else None
                is_signal_ready = True
                z_comps = tuple(components_list[j])
            else:
                z_score = None
                level = float(level_vals[j])
                rm_val = None
                rs_val = None
                is_signal_ready = False
                z_comps = tuple(components_list[j])

            # Build level series once — reused by diagnostics and momentum
            ls = pd.Series(levels[:, j], index=level_index, name="level") if is_signal_ready else None

            # MR diagnostics — expensive, skip when not needed
            if skip_diagnostics or not is_signal_ready:
                adf_pvalue, mr_score, kappa, half_life = None, None, None, None
            else:
                sr = spread_returns[:, j]
                adf_pvalue, mr_score, kappa, half_life = self._compute_mr_diagnostics(
                    spread_return=sr,
                    level_series=ls,
                )


            out[ref.candidate_id] = CandidateAnalyticsState(
                candidate_id=ref.candidate_id,
                group_id=ref.group_id,
                spread_id=ref.spread_id,
                date=date,
                z_score=z_score,
                level=level,
                roll_mean=rm_val,
                roll_std=rs_val,
                adf_pvalue=adf_pvalue,
                mr_score=mr_score,
                kappa=kappa,
                half_life=half_life,
                is_signal_ready=is_signal_ready,
                z_components=z_comps,
                residual_key=residual_key,
            )

        return out

    def compute_analytics_from_weights(
        self,
        *,
        date: pd.Timestamp,
        asof_date: pd.Timestamp,
        group_id: str,
        candidate_id: str,
        spread_id: str,
        weights_by_ticker: dict[str, float],
        residual_key: str = "",
        timescale_label: str = "",
        skip_diagnostics: bool = False,
    ) -> CandidateAnalyticsState:
        """
        Compute analytics using an arbitrary weight dict (e.g. realized weights).

        The residual model is always the one frozen at `asof_date` — the
        candidate's own refit date for a `CandidateRef`-based caller, or a
        position's own `entry_asof_date` for realized-fill / effective-weight
        callers (F2 R1/D2) — never "today" (`date`). `date` is only the
        truncation point for the returned level/z-score/diagnostics; it must
        be >= asof_date for a caller tracking a live position, since a
        position's own asof_date never moves after entry.
        """
        date = pd.Timestamp(date)
        asof_date = pd.Timestamp(asof_date)

        try:
            residuals = self._get_asof_residuals(group_id, residual_key, asof_date)
        except ValueError:
            return self._not_ready_analytics_from_fields(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                date=date,
                residual_key=residual_key,
            )

        tickers = list(weights_by_ticker.keys())
        missing = [t for t in tickers if t not in residuals.columns]
        if missing:
            return self._not_ready_analytics_from_fields(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                date=date,
                residual_key=residual_key,
            )

        spread_return_full, level_full = compute_spread_level(residuals, weights_by_ticker)
        spread_return = spread_return_full.loc[:date]
        level_series = level_full.loc[:date]
        if level_series.empty:
            return self._not_ready_analytics_from_fields(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                date=date,
                residual_key=residual_key,
            )

        # Z-score: fast timing window
        z_score, level, roll_mean, roll_std, is_signal_ready, z_components = self._compute_z_score(
            level_series=level_series,
            residual_key=residual_key,
            timescale_label=timescale_label,
        )

        # MR diagnostics: slower structural window (skippable)
        if skip_diagnostics:
            adf_pvalue, mr_score, kappa, half_life = None, None, None, None
        else:
            adf_pvalue, mr_score, kappa, half_life = self._compute_mr_diagnostics(
                spread_return=spread_return.to_numpy(dtype=float),
                level_series=level_series,
            )

        return CandidateAnalyticsState(
            candidate_id=candidate_id,
            group_id=group_id,
            spread_id=spread_id,
            date=date,
            z_score=z_score,
            level=level,
            roll_mean=roll_mean,
            roll_std=roll_std,
            adf_pvalue=adf_pvalue,
            mr_score=mr_score,
            kappa=kappa,
            half_life=half_life,
            is_signal_ready=is_signal_ready,
            z_components=z_components,
            residual_key=residual_key,
        )

    def _compute_z_score(
        self,
        *,
        level_series: pd.Series,
        residual_key: str = "",
        timescale_label: str = "",
    ) -> tuple[float | None, float, float | None, float | None, bool, tuple[ZScoreComponent, ...]]:
        """
        Compute (blended) z-score for a single candidate.

        This is the scalar fallback used by compute_analytics_from_weights.
        The batched path (_batch_analytics_for_group) uses numpy directly.

        Returns (z_score, level, roll_mean, roll_std, is_signal_ready, z_components).
        """
        z_cfg = self._get_z_score_config(timescale_label)
        lookbacks = z_cfg.resolved_lookbacks()
        weights = z_cfg.resolved_weights()
        ddof = z_cfg.ddof
        method = z_cfg.method
        level = float(level_series.iloc[-1])

        components: list[ZScoreComponent] = []
        weighted_z_sum = 0.0
        all_ready = True

        for lb, w in zip(lookbacks, weights):
            min_p = z_cfg.resolved_min_periods(lookback=lb)
            if method == "ewm":
                ewm = level_series.ewm(halflife=lb, min_periods=min_p)
                rm = ewm.mean()
                rs = ewm.std()
            else:
                rm = level_series.rolling(window=lb, min_periods=min_p).mean()
                rs = level_series.rolling(window=lb, min_periods=min_p).std(ddof=ddof)
            m_val, s_val = rm.iloc[-1], rs.iloc[-1]

            if pd.isna(m_val) or pd.isna(s_val) or float(s_val) <= 0.0:
                components.append(ZScoreComponent(lookback=lb, z_score=None, roll_mean=None, roll_std=None))
                all_ready = False
            else:
                z_i = float((level - float(m_val)) / float(s_val))
                components.append(ZScoreComponent(lookback=lb, z_score=z_i, roll_mean=float(m_val), roll_std=float(s_val)))
                weighted_z_sum += w * z_i

        z_components = tuple(components)

        if not all_ready:
            return None, level, None, None, False, z_components

        primary = components[0]
        return (
            weighted_z_sum,
            level,
            primary.roll_mean,
            primary.roll_std,
            True,
            z_components,
        )

    def _compute_mr_diagnostics(
        self,
        *,
        spread_return: np.ndarray,
        level_series: pd.Series,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        # Truncate to diagnostics lookback window
        diag_lookback = self.diagnostics_config.lookback
        if len(level_series) > diag_lookback:
            level_series = level_series.iloc[-diag_lookback:]
            spread_return = spread_return[-diag_lookback:]

        if len(spread_return) < 11:
            return None, None, None, None

        x_raw = np.asarray(level_series.to_numpy(dtype=float), dtype=float)
        x = x_raw - np.mean(x_raw)
        x_lag = x[:-1]
        dx = np.diff(x)

        if len(dx) < 10:
            return None, None, None, None

        X = np.column_stack([np.ones_like(x_lag), x_lag])
        beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
        intercept, slope = beta

        fitted = X @ beta
        resid = dx - fitted
        residual_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else np.nan
        kappa = float(-slope)

        if not np.isfinite(kappa) or not np.isfinite(residual_std) or residual_std <= 0.0 or kappa <= 0.0:
            return self._safe_adf(x_raw), None, None, None

        half_life = float(np.log(2.0) / kappa)
        mr_score = float(kappa / residual_std)

        return self._safe_adf(x_raw), mr_score, kappa, half_life

    @staticmethod
    @staticmethod
    def _safe_adf(x_raw: np.ndarray) -> float | None:
        if adfuller is None:
            return None
        try:
            return float(adfuller(x_raw, autolag="AIC")[1])
        except Exception:
            return None

    def get_level_series(
        self,
        ref: "CandidateRef",
        date: pd.Timestamp,
    ) -> pd.Series | None:
        """
        Reconstruct the spread level series for a candidate up to date.

        The residual model is always the one frozen at ref.asof_date (F2 R1)
        — never "today" (`date`). Uses the cached asof-keyed residual matrix
        (F2 R2, _get_asof_residuals) so this is cheap after the first call
        for that asof_date.

        Returns None if residuals are unavailable or tickers are missing.
        Used by EntryFeatureEngine to compute entry features.
        """
        try:
            residuals = self._get_asof_residuals(ref.group_id, ref.residual_key, ref.asof_date)
        except ValueError:
            return None

        missing = [m for m in ref.members if m not in residuals.columns]
        if missing:
            return None

        weights_by_ticker = dict(zip(ref.members, ref.weights))
        _, level_series = compute_spread_level(residuals, weights_by_ticker)
        return level_series.loc[:pd.Timestamp(date)]

    def _get_bundle(self, group_id: str) -> GroupReturnBundle:
        bundle = self._bundle_cache.get(group_id)
        if bundle is None:
            bundle = build_group_return_bundle(
                umd=self.umd,
                group_id=group_id,
                field=self.price_field,
                return_method=self.return_method,
                dropna=self.dropna,
            )
            self._bundle_cache[group_id] = bundle
        return bundle

    def _get_asof_residuals(
        self,
        group_id: str,
        residual_key: str,
        asof_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Full-history residual matrix for the model frozen at a candidate's own
        asof_date (F2 R1/R2) — never a model fitted or looked up at "today".

        Cached per (group_id, residual_key, asof_date): candidates sharing a
        refit date within a group share one matrix. LRU-bounded at
        _ASOF_RESIDUAL_CACHE_MAX rather than precisely ref-counted (R2
        describes eviction "when no tracked candidate or open position refers
        to it"; F2_spec.md P4 measured a peak of 164 concurrently-live keys on
        a real run, well under the LRU bound, so this is equivalent in
        practice without wiring cache eviction into the simulator's own
        tracked/open-position bookkeeping).

        Raises ValueError if no model was fitted at exactly this asof_date —
        no live-fit fallback (F2 R2: "a missing asof model raises"). A
        candidate's own asof_date is always a valid fit_date in
        precomputed_residual_params by construction (candidates only arise on
        dates the residual fit grid also covers — series.py's own docstring
        makes the same observation for the disk-cache path); a raise here
        means the wrong residual_params were loaded, not a normal condition.
        """
        asof_date = pd.Timestamp(asof_date)
        cache_key = (group_id, residual_key, asof_date)
        cached = self._asof_residual_cache.get(cache_key)
        if cached is not None:
            self._asof_residual_cache.move_to_end(cache_key)
            return cached

        group_params = self.precomputed_residual_params.get((group_id, residual_key))
        if group_params is None or asof_date not in group_params:
            raise ValueError(
                f"No residual model for group_id={group_id!r} residual_key={residual_key!r} "
                f"asof_date={asof_date} — F2 R2: the asof-frozen model is required at "
                f"panel-build time; there is no live-fit fallback here."
            )
        model = group_params[asof_date]

        bundle = self._get_bundle(group_id)
        residuals = apply_causal_residual_model(
            model=model,
            aligned_returns=bundle.aligned_returns,
            rf_series=bundle.risk_free_returns if model.subtract_risk_free else None,
        )

        self._asof_residual_cache[cache_key] = residuals
        if len(self._asof_residual_cache) > _ASOF_RESIDUAL_CACHE_MAX:
            evicted_key, _ = self._asof_residual_cache.popitem(last=False)
            logger.warning(
                "[signals] asof residual cache evicted %r at size %d — an LRU bound "
                "standing in for R2's ref-counted eviction; this key will be "
                "recomputed if it is still referenced. F2_spec.md P4 measured a peak "
                "of 164 concurrently-live keys on a real run, well under this bound, "
                "so a real eviction here is unexpected and worth investigating.",
                evicted_key, _ASOF_RESIDUAL_CACHE_MAX,
            )

        return residuals

    @staticmethod
    def _empty_signal_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "spread_id",
                "group_id",
                "is_new_arrival",
                "is_active",
                "level",
                "roll_mean",
                "roll_std",
                "z_score",
                "is_signal_ready",
            ],
            index=pd.Index([], name="candidate_id"),
        )

    @staticmethod
    def _not_ready_analytics(
        *,
        ref: CandidateRef,
        date: pd.Timestamp,
    ) -> CandidateAnalyticsState:
        return CandidateAnalyticsState(
            candidate_id=ref.candidate_id,
            group_id=ref.group_id,
            spread_id=ref.spread_id,
            date=date,
            z_score=None,
            level=None,
            roll_mean=None,
            roll_std=None,
            adf_pvalue=None,
            mr_score=None,
            kappa=None,
            half_life=None,
            is_signal_ready=False,
            residual_key=ref.residual_key,
        )

    @staticmethod
    def _not_ready_analytics_from_fields(
        *,
        candidate_id: str,
        group_id: str,
        spread_id: str,
        date: pd.Timestamp,
        residual_key: str = "",
    ) -> CandidateAnalyticsState:
        return CandidateAnalyticsState(
            candidate_id=candidate_id,
            group_id=group_id,
            spread_id=spread_id,
            date=date,
            z_score=None,
            level=None,
            roll_mean=None,
            roll_std=None,
            adf_pvalue=None,
            mr_score=None,
            kappa=None,
            half_life=None,
            is_signal_ready=False,
            residual_key=residual_key,
        )