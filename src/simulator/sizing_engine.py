from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.simulator.actions import OpenCandidateAction
from src.simulator.config import IntervalScoringConfig, KellyConfig, SizingConfig
from src.simulator.types import CandidateAnalyticsState


# ── Kelly tracker ────────────────────────────────────────────────────────────

@dataclass(slots=True)
class _KellyGroupState:
    """EWM accumulator for one group (or global)."""
    n_trades: int = 0
    _w_sum: float = 0.0
    _wx_sum: float = 0.0
    _wxx_sum: float = 0.0


_KELLY_GLOBAL_KEY = "__global__"


@dataclass(slots=True)
class KellyTracker:
    """
    Tracks closed trade outcomes and computes Kelly-derived base notional.

    Uses expanding window with optional EWM half-life.
    Operates per-sector or globally based on config.
    """
    config: KellyConfig
    base_pair_notional: float
    _states: dict[str, _KellyGroupState] = field(default_factory=dict)

    def record_trade(self, group_id: str, pnl_net: float) -> None:
        """Record a closed trade outcome."""
        keys = [group_id, _KELLY_GLOBAL_KEY] if self.config.per_sector else [_KELLY_GLOBAL_KEY]
        for key in keys:
            state = self._states.get(key)
            if state is None:
                state = _KellyGroupState()
                self._states[key] = state
            self._update_state(state, pnl_net)

    def get_base_notional(self, group_id: str) -> float:
        """
        Get Kelly-derived base notional for a group.

        Returns base_pair_notional if Kelly not yet active,
        blended value during transition, or full Kelly value.
        """
        key = group_id if self.config.per_sector else _KELLY_GLOBAL_KEY
        state = self._states.get(key)

        if state is None or state.n_trades < self.config.min_trades:
            return self.base_pair_notional

        kelly_notional = self._compute_kelly_notional(state)

        blend_weight = min(
            (state.n_trades - self.config.min_trades)
            / max(self.config.blend_target - self.config.min_trades, 1),
            1.0,
        )

        return (1.0 - blend_weight) * self.base_pair_notional + blend_weight * kelly_notional

    def _compute_kelly_notional(self, state: _KellyGroupState) -> float:
        if state._w_sum <= 0.0:
            return self.base_pair_notional

        mu = state._wx_sum / state._w_sum
        var = state._wxx_sum / state._w_sum - mu * mu

        if var <= 0.0:
            return self.base_pair_notional

        # Kelly optimal notional as multiple of base:
        #   f* = E[r] / Var[r] = (mu/base) / (var/base²) = mu*base/var
        #   kelly_notional = f* * base = mu * base² / var
        base = self.base_pair_notional
        kelly_raw = mu * base * base / var
        kelly_notional = self.config.fraction * kelly_raw

        floor = self.config.floor_multiplier * base
        cap = self.config.cap_multiplier * base
        return max(floor, min(cap, kelly_notional))

    def _update_state(self, state: _KellyGroupState, pnl: float) -> None:
        state.n_trades += 1

        if self.config.half_life is not None:
            alpha = 1.0 - np.exp(-np.log(2.0) / self.config.half_life)
            decay = 1.0 - alpha
            state._w_sum = decay * state._w_sum + 1.0
            state._wx_sum = decay * state._wx_sum + pnl
            state._wxx_sum = decay * state._wxx_sum + pnl * pnl
        else:
            state._w_sum += 1.0
            state._wx_sum += pnl
            state._wxx_sum += pnl * pnl


# ── SizingEngine ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SizingEngine:
    """
    Computes per-trade notional for approved open actions.

    Pipeline (applied in order):
    1. Base notional: Kelly-derived (if configured) or fixed base_pair_notional.
    2. Vol normalization: scale by (median_roll_std / pair_roll_std), clamped
       to [floor_multiplier, cap_multiplier] (if configured).

    Trades sized to zero (e.g. future feature grid exclusions) are dropped
    before returning, so RiskManager only sees trades with notional > 0.

    Kelly and vol_normalize are composable: Kelly sets the base, vol_normalize
    adjusts it per-pair.
    """
    config: SizingConfig
    _kelly_tracker: KellyTracker | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.kelly is not None:
            self._kelly_tracker = KellyTracker(
                config=self.config.kelly,
                base_pair_notional=self.config.base_pair_notional,
            )

    def record_closed_trade(self, group_id: str, pnl_net: float) -> None:
        """Feed a closed trade outcome to the Kelly tracker (no-op if Kelly disabled)."""
        if self._kelly_tracker is not None:
            self._kelly_tracker.record_trade(group_id, pnl_net)

    def size(
        self,
        proposed_opens: list[OpenCandidateAction],
        analytics_by_id: dict[str, CandidateAnalyticsState] | None = None,
    ) -> list[tuple[OpenCandidateAction, float, dict[str, float], float]]:
        """
        Compute pair_notional for each proposed open.

        Returns list of (action, notional, feature_scores, size_multiplier)
        with notional > 0. Zero-notional trades are dropped before RiskManager.
        """
        if not proposed_opens:
            return []

        median_roll_std = (
            self._compute_median_roll_std(analytics_by_id)
            if self.config.vol_normalize is not None and analytics_by_id is not None
            else None
        )

        result: list[tuple[OpenCandidateAction, float, dict[str, float], float]] = []
        for action in proposed_opens:
            notional, feature_scores, size_multiplier = self._compute_pair_notional(
                action, analytics_by_id, median_roll_std
            )
            if notional > 0.0:
                result.append((action, notional, feature_scores, size_multiplier))

        return result

    def _compute_pair_notional(
        self,
        action: OpenCandidateAction,
        analytics_by_id: dict[str, CandidateAnalyticsState] | None,
        median_roll_std: float | None,
    ) -> tuple[float, dict[str, float], float]:
        """Returns (notional, feature_scores, size_multiplier)."""
        # Step 1: base notional (Kelly or fixed)
        if self._kelly_tracker is not None:
            base = self._kelly_tracker.get_base_notional(action.group_id)
        else:
            base = self.config.base_pair_notional

        # Step 2: vol normalization
        vol_cfg = self.config.vol_normalize
        if vol_cfg is not None and median_roll_std is not None and analytics_by_id is not None:
            a = analytics_by_id.get(action.candidate_id)
            if a is not None and a.roll_std is not None and a.roll_std > 0.0:
                ratio = median_roll_std / a.roll_std
                ratio = max(vol_cfg.floor_multiplier, min(vol_cfg.cap_multiplier, ratio))
                base = base * ratio

        # Step 3: interval scoring multiplier
        scoring_cfg = self.config.interval_scoring
        feature_scores: dict[str, float] = {}
        size_multiplier = 1.0

        if scoring_cfg is not None and analytics_by_id is not None:
            a = analytics_by_id.get(action.candidate_id)
            features = a.features if a is not None else {}
            size_multiplier, feature_scores = self._compute_interval_multiplier(scoring_cfg, features)
            if size_multiplier == 0.0:
                return 0.0, feature_scores, 0.0
            base = base * size_multiplier

        return base, feature_scores, size_multiplier

    @staticmethod
    def _compute_interval_multiplier(
        cfg: IntervalScoringConfig,
        features: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """
        Compute sizing multiplier and per-feature scores.

        Returns (multiplier, feature_scores) where feature_scores maps each
        feature name to its interval weight before aggregation.
        A weight of 0.0 from any feature returns (0.0, scores) — exclusion.
        """
        weighted_sum = 0.0
        total_weight = 0.0
        feature_scores: dict[str, float] = {}

        for spec in cfg.feature_specs:
            if spec.feature not in features:
                raise KeyError(
                    f"IntervalScoringConfig: feature {spec.feature!r} not found in "
                    f"candidate features: {sorted(features.keys())}"
                )
            value = features[spec.feature]
            interval_weight = spec.lookup(value)
            feature_scores[spec.feature] = interval_weight

            if spec.feature not in cfg.feature_weights:
                raise KeyError(
                    f"IntervalScoringConfig: feature {spec.feature!r} not found in "
                    f"feature_weights: {sorted(cfg.feature_weights.keys())}"
                )
            agg_weight = cfg.feature_weights[spec.feature]
            weighted_sum += interval_weight * agg_weight
            total_weight += agg_weight

        if total_weight <= 0.0:
            return 1.0, feature_scores

        raw = weighted_sum / total_weight

        if raw < cfg.floor_multiplier and cfg.floor_mode == "exclude":
            return 0.0, feature_scores

        multiplier = max(cfg.floor_multiplier, min(cfg.cap_multiplier, raw))
        return multiplier, feature_scores

    @staticmethod
    def _compute_median_roll_std(
        analytics_by_id: dict[str, CandidateAnalyticsState] | None,
    ) -> float | None:
        if not analytics_by_id:
            return None
        stds = [
            a.roll_std for a in analytics_by_id.values()
            if a.roll_std is not None and a.roll_std > 0.0
        ]
        if not stds:
            return None
        return float(np.median(stds))
