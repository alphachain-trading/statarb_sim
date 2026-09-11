from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulator.actions import OpenCandidateAction
from src.simulator.config import IntervalScoringConfig, SizingConfig
from src.simulator.types import CandidateAnalyticsState


# ── SizingEngine ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class SizingEngine:
    """
    Computes per-trade notional for approved open actions.

    Pipeline (applied in order):
    1. Base notional: fixed base_pair_notional.
    2. Vol normalization: scale by (median_roll_std / pair_roll_std), clamped
       to [floor_multiplier, cap_multiplier] (if configured).

    Trades sized to zero (e.g. future feature grid exclusions) are dropped
    before returning, so RiskManager only sees trades with notional > 0.
    """
    config: SizingConfig

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
        # Step 1: base notional
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
