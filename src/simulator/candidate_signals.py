from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
import numpy as np
import pandas as pd

from src.simulator.config import MRDiagnosticsConfig, ZScoreConfig
from src.simulator.types import CandidateAnalyticsState, CandidateRef, ZScoreComponent
from src.data.returns import GroupReturnBundle, build_group_return_bundle
from src.data.universe_marketdata import UniverseMarketData
from src.residuals.causal_residuals import (
    CausalResidualConfig,
    FittedCausalResidualModel,
    _slice_fit_window,
    apply_causal_residual_model,
    create_causal_residuals,
    fit_causal_residual_model,
)

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover
    adfuller = None

logger = logging.getLogger(__name__)

# Max number of persisted spread level series to hold in memory (LRU).
_SPREAD_SERIES_CACHE_MAX = 4096


def _compute_variance_ratios(
    residuals: np.ndarray,
    n_report: int = 5,
) -> tuple[float, ...]:
    """
    Compute top-k PC variance ratios from a (T, N) residual matrix.

    Cheap: eigendecomposition of an N×N covariance matrix (e.g. 28×28).
    """
    if residuals.shape[0] < 2 or residuals.shape[1] < 2:
        return ()
    cov = np.cov(residuals, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    total = eigvals.sum()
    if total <= 0:
        return ()
    k = min(n_report, len(eigvals))
    return tuple(float(v / total) for v in eigvals[:k])


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

    Residuals are reconstructed once per (group_id, residual_key, date) and shared
    across all z-score configs that use the same residual_key.
    """
    z_score_configs: dict[str, ZScoreConfig]
    diagnostics_config: MRDiagnosticsConfig
    residual_configs: dict[str, CausalResidualConfig]
    umd: UniverseMarketData

    price_field: str = "Close"
    return_method: str = "log"
    dropna: str = "any"

    # Root candidate-panel dir. When set and a persisted spread level series
    # exists under panel_dir/series/spread/, the batch path loads levels from
    # disk instead of recomputing residuals @ weights -> cumsum.
    panel_dir: Path | None = None

    # Precomputed fitted residual model params.
    # Key: (group_id, residual_key) → {date: FittedCausalResidualModel}
    # When residual_key="" (single-timescale), legacy key (group_id, "") is used.
    precomputed_residual_params: dict[tuple[str, str], dict[pd.Timestamp, FittedCausalResidualModel]] = dc_field(
        default_factory=dict, init=True, repr=False,
    )

    _bundle_cache: dict[str, GroupReturnBundle] = dc_field(default_factory=dict, init=False, repr=False)
    _residual_cache: dict[tuple[str, str], tuple[pd.Timestamp, pd.DataFrame]] = dc_field(default_factory=dict, init=False, repr=False)
    _latest_pc_variance_ratios: dict[tuple[str, str], tuple[float, ...]] = dc_field(default_factory=dict, init=False, repr=False)
    _spread_series_cache: "OrderedDict[str, pd.Series]" = dc_field(default_factory=OrderedDict, init=False, repr=False)

    def _get_z_score_config(self, timescale_label: str) -> ZScoreConfig:
        """Look up ZScoreConfig by timescale_label (unique per residual_key + zlb)."""
        zc = self.z_score_configs.get(timescale_label)
        if zc is None:
            raise KeyError(
                f"No ZScoreConfig for timescale_label={timescale_label!r}. "
                f"Available: {sorted(self.z_score_configs.keys())}"
            )
        return zc

    def _get_residual_config(self, residual_key: str) -> CausalResidualConfig:
        rc = self.residual_configs.get(residual_key)
        if rc is None:
            raise KeyError(
                f"No CausalResidualConfig for residual_key={residual_key!r}. "
                f"Available: {sorted(self.residual_configs.keys())}"
            )
        return rc

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

        Builds a (T, n_candidates) level matrix — loaded from persisted spread
        series when available, otherwise via a single residual matmul — then
        computes rolling stats across all candidates at once.
        Falls back to not-ready for candidates with missing tickers.
        """
        # Fast path: load persisted spread level series from disk. Keyed per
        # candidate by (spread_id, asof_date); returns None (→ recompute) if any
        # candidate's series is missing.
        if self.panel_dir is not None:
            disk = self._try_batch_levels_from_disk(refs=refs, date=date)
            if disk is not None:
                levels, level_index, spread_returns = disk
                return self._finalize_batch_states(
                    valid_refs=refs,
                    invalid_refs=[],
                    levels=levels,
                    level_index=level_index,
                    spread_returns=spread_returns,
                    residual_key=residual_key,
                    timescale_label=timescale_label,
                    date=date,
                    skip_diagnostics=skip_diagnostics,
                )

        # Fallback: recompute residuals @ weights -> cumsum.
        try:
            residuals = self._get_residuals(group_id, residual_key, date)
        except ValueError:
            return {
                ref.candidate_id: self._not_ready_analytics(ref=ref, date=date)
                for ref in refs
            }

        R = residuals.to_numpy(dtype=float)    # (T, n_tickers)
        col_list = residuals.columns.tolist()
        col_idx = {c: i for i, c in enumerate(col_list)}

        # Partition refs into valid (all members present) and invalid
        valid_refs: list[CandidateRef] = []
        invalid_refs: list[CandidateRef] = []
        for ref in refs:
            if all(m in col_idx for m in ref.members):
                valid_refs.append(ref)
            else:
                invalid_refs.append(ref)

        if not valid_refs:
            return {
                ref.candidate_id: self._not_ready_analytics(ref=ref, date=date)
                for ref in invalid_refs
            }

        # Build weight matrix: (n_tickers, n_valid_candidates)
        n_cands = len(valid_refs)
        W = np.zeros((len(col_list), n_cands), dtype=float)
        for j, ref in enumerate(valid_refs):
            for member, weight in zip(ref.members, ref.weights):
                W[col_idx[member], j] = weight

        # Vectorized spread levels: (T, n_cands)
        spread_returns = R @ W
        levels = np.cumsum(spread_returns, axis=0)

        return self._finalize_batch_states(
            valid_refs=valid_refs,
            invalid_refs=invalid_refs,
            levels=levels,
            level_index=residuals.index,
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

        Used by both the disk-backed and residual-recompute paths of
        _batch_analytics_for_group.
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

    # ── Persisted spread-level series (disk-backed level loading) ───────────

    def _spread_series_path(self, spread_id: str, asof_date: pd.Timestamp) -> Path:
        """Path to a candidate's persisted level series, keyed by (spread_id, asof_date)."""
        fname = f"{spread_id.replace('|', '_')}__{pd.Timestamp(asof_date).strftime('%Y%m%d')}.parquet"
        return self.panel_dir / "series" / "spread" / fname

    def _load_spread_series(self, path: Path) -> pd.Series:
        """Load a persisted 'level' series (full history), with a small LRU cache."""
        key = str(path)
        cache = self._spread_series_cache
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached
        s = pd.read_parquet(path)["level"]
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        cache[key] = s
        if len(cache) > _SPREAD_SERIES_CACHE_MAX:
            cache.popitem(last=False)
        return s

    def _try_batch_levels_from_disk(
        self,
        *,
        refs: list[CandidateRef],
        date: pd.Timestamp,
    ) -> tuple[np.ndarray, pd.Index, np.ndarray] | None:
        """
        Build (levels, level_index, spread_returns) from persisted spread series.

        Returns None (so the caller recomputes from residuals) if refs is empty
        or any candidate's persisted series is missing. spread_returns are
        reconstructed from the level differences for the MR diagnostics path.
        """
        if not refs:
            return None

        paths = [(ref, self._spread_series_path(ref.spread_id, ref.asof_date)) for ref in refs]
        missing = [ref.spread_id for ref, p in paths if not p.exists()]
        if missing:
            logger.warning(
                "[signals] %d/%d spread level series missing under %s on %s "
                "(e.g. %s); recomputing from residuals.",
                len(missing), len(paths), self.panel_dir,
                pd.Timestamp(date).date(), missing[0],
            )
            return None

        date = pd.Timestamp(date)
        series_list = [self._load_spread_series(p).loc[:date] for _, p in paths]

        # Candidates in one (group, timescale) batch share the group's full
        # aligned-returns index; align defensively on their common index.
        common_index = series_list[0].index
        for s in series_list[1:]:
            if not s.index.equals(common_index):
                common_index = common_index.union(s.index)

        if len(common_index) == 0:
            # date precedes the persisted series — let the caller recompute.
            return None

        levels = np.column_stack([
            s.reindex(common_index).to_numpy(dtype=float) for s in series_list
        ])
        spread_returns = np.diff(levels, axis=0, prepend=0.0)
        return levels, common_index, spread_returns

    def compute_analytics_from_weights(
        self,
        *,
        date: pd.Timestamp,
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

        Reuses the same residualization and OU diagnostic logic as
        _compute_candidate_analytics but accepts weights keyed by ticker
        instead of a CandidateRef.
        """
        date = pd.Timestamp(date)

        try:
            residuals = self._get_residuals(group_id, residual_key, date)
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

        rr = residuals.loc[:, tickers].copy()
        if rr.empty:
            return self._not_ready_analytics_from_fields(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                date=date,
                residual_key=residual_key,
            )

        w = np.asarray([weights_by_ticker[t] for t in tickers], dtype=float)
        spread_return = rr.to_numpy(dtype=float) @ w
        level_series = pd.Series(np.cumsum(spread_return), index=rr.index, name="level")

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
                spread_return=spread_return,
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

    def _compute_candidate_analytics(
        self,
        *,
        ref: CandidateRef,
        date: pd.Timestamp,
        skip_diagnostics: bool = False,
    ) -> CandidateAnalyticsState:
        try:
            residuals = self._get_residuals(ref.group_id, ref.residual_key, date)
        except ValueError:
            return self._not_ready_analytics(ref=ref, date=date)

        members = list(ref.members)
        missing = [m for m in members if m not in residuals.columns]
        if missing:
            return self._not_ready_analytics(ref=ref, date=date)

        rr = residuals.loc[:, members].copy()
        if rr.empty:
            return self._not_ready_analytics(ref=ref, date=date)

        w = np.asarray(ref.weights, dtype=float)
        spread_return = rr.to_numpy(dtype=float) @ w
        level_series = pd.Series(np.cumsum(spread_return), index=rr.index, name="level")

        # Z-score: fast timing window
        z_score, level, roll_mean, roll_std, is_signal_ready, z_components = self._compute_z_score(
            level_series=level_series,
            residual_key=ref.residual_key,
            timescale_label=ref.timescale_label,
        )

        # MR diagnostics: slower structural window (skippable)
        if skip_diagnostics:
            adf_pvalue, mr_score, kappa, half_life = None, None, None, None
        else:
            adf_pvalue, mr_score, kappa, half_life = self._compute_mr_diagnostics(
                spread_return=spread_return,
                level_series=level_series,
            )

        # Spread momentum — cheap, always compute when signal is ready

        return CandidateAnalyticsState(
            candidate_id=ref.candidate_id,
            group_id=ref.group_id,
            spread_id=ref.spread_id,
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
            residual_key=ref.residual_key,
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

        Uses the cached residual matrix (already computed by build_candidate_analytics_states)
        so this is cheap — one matmul over cached data.

        Returns None if residuals are unavailable or tickers are missing.
        Used by EntryFeatureEngine to compute entry features.
        """
        try:
            residuals = self._get_residuals(ref.group_id, ref.residual_key, date)
        except ValueError:
            return None

        missing = [m for m in ref.members if m not in residuals.columns]
        if missing:
            return None

        w = np.asarray(ref.weights, dtype=float)
        rr = residuals.loc[:, list(ref.members)].to_numpy(dtype=float)
        spread_return = rr @ w
        level_series = pd.Series(
            np.cumsum(spread_return),
            index=residuals.index,
            name="level",
        )
        return level_series

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

    def _get_residuals(
        self,
        group_id: str,
        residual_key: str,
        date: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Cached residual matrix per (group_id, residual_key, date).

        All candidates sharing the same (group_id, residual_key) on the same
        date share identical residuals. This avoids re-fitting the causal
        residual model for every candidate.

        When precomputed_residual_params are available, the expensive fit step
        is skipped entirely — only the cheap apply step runs.  Falls back to
        live fitting when no precomputed params exist.

        Also stores the latest fitted model's PC variance ratios for
        regime monitoring.
        """
        cache_key = (group_id, residual_key)
        cached = self._residual_cache.get(cache_key)
        if cached is not None and cached[0] == date:
            return cached[1]

        bundle = self._get_bundle(group_id)
        residual_config = self._get_residual_config(residual_key)

        # Try precomputed params first (cheap path)
        group_params = self.precomputed_residual_params.get(cache_key)
        if group_params is not None and date in group_params:
            model = group_params[date]
        else:
            # Fallback: live fit (expensive path)
            model = fit_causal_residual_model(
                bundle=bundle,
                date=date,
                cfg=residual_config,
            )

        aligned = _slice_fit_window(
            bundle=bundle,
            date=date,
            cfg=residual_config,
        )
        residuals = apply_causal_residual_model(
            model=model,
            aligned_returns=aligned,
            rf_series=bundle.risk_free_returns if model.subtract_risk_free else None,
        )

        self._residual_cache[cache_key] = (date, residuals)

        # Compute PC variance ratios of the CLEANED residuals (post all removals).
        # This is the true regime indicator: if PC1' is high, there's unexplained
        # common structure that the model (market + sector + optional PCs) missed.
        self._latest_pc_variance_ratios[cache_key] = _compute_variance_ratios(
            residuals.to_numpy(dtype=float),
        )

        return residuals

    def get_pc_variance_ratios(
        self,
        group_id: str,
        residual_key: str = "",
    ) -> tuple[float, ...] | None:
        """Return the latest PC variance ratios of cleaned residuals for a (group, timescale)."""
        return self._latest_pc_variance_ratios.get((group_id, residual_key))

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