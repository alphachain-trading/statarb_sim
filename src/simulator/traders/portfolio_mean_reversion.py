from __future__ import annotations

from dataclasses import dataclass

from src.simulator.actions import CandidateAction, CloseCandidateAction, OpenCandidateAction
from src.simulator.config import PortfolioMeanReversionConfig
from src.simulator.types import CandidateAnalyticsState, CandidateMarketSnapshot, LiveCandidatePosition

import numpy as np


@dataclass(slots=True)
class PortfolioMeanReversionTrader:
    """
    Basic portfolio mean-reversion trader for v0.1.

    Rules
    -----
    - only consider active + signal-ready candidates
    - if flat:
        * open long spread when z <= -entry_z
        * open short spread when z >= +entry_z
    - if open:
        * time stop: exit when days_open >= multiplier × entry_half_life
        * deterioration stop: exit when fz_mr_score < mr_deterioration_threshold × entry_mr_score
        * close long when z >= exit_z
        * close short when z <= exit_z
    """

    config: PortfolioMeanReversionConfig
    default_abs_target_group_exposure: float = 1.0

    def generate_actions(
        self,
        snapshot: CandidateMarketSnapshot,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        live_diagnostics_by_candidate_id: dict[str, CandidateAnalyticsState] | None = None,
        target_group_capital_by_group: dict[str, float] | None = None,
    ) -> list[CandidateAction]:
        actions: list[CandidateAction] = []

        if snapshot.candidate_states.empty:
            return actions

        diagnostics = live_diagnostics_by_candidate_id or {}
        capital = target_group_capital_by_group or {}

        for candidate_id, row in snapshot.candidate_states.iterrows():
            if not bool(row["is_active"]):
                continue
            if not bool(row["is_signal_ready"]):
                continue

            z_score = float(row["z_score"])
            momentum_z_raw = row.get("momentum_z")
            momentum_z: float | None = (
                float(momentum_z_raw)
                if momentum_z_raw is not None and not (isinstance(momentum_z_raw, float) and np.isnan(momentum_z_raw))
                else None
            )
            live_pos = live_positions_by_candidate_id.get(candidate_id)

            if live_pos is None:
                action = self._maybe_open(
                    candidate_id=str(candidate_id),
                    group_id=str(row["group_id"]),
                    spread_id=str(row["spread_id"]),
                    z_score=z_score,
                    momentum_z=momentum_z,
                )
            else:
                fz_a = diagnostics.get(str(candidate_id))
                group_capital = capital.get(live_pos.group_id)
                action = self._maybe_close(
                    live_pos=live_pos,
                    z_score=z_score,
                    fz_analytics=fz_a,
                    target_group_capital=group_capital,
                )

            if action is not None:
                actions.append(action)

        return actions

    def _maybe_open(
        self,
        *,
        candidate_id: str,
        group_id: str,
        spread_id: str,
        z_score: float,
        momentum_z: float | None,
    ) -> OpenCandidateAction | None:
        mom_cfg = self.config.spread_momentum

        if self.config.allow_long and z_score <= -self.config.entry_z:
            if mom_cfg is not None:
                if momentum_z is None:
                    return None
                if momentum_z <= mom_cfg.entry_threshold:
                    return None
            return OpenCandidateAction(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                target_group_exposure=+self.default_abs_target_group_exposure,
                reason="entry_long",
                z_score=z_score,
                momentum_z=momentum_z,
            )

        if self.config.allow_short and z_score >= self.config.entry_z:
            if mom_cfg is not None:
                if momentum_z is None:
                    return None
                if momentum_z >= -mom_cfg.entry_threshold:
                    return None
            return OpenCandidateAction(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                target_group_exposure=-self.default_abs_target_group_exposure,
                reason="entry_short",
                z_score=z_score,
                momentum_z=momentum_z,
            )

        return None

    def _maybe_close(
        self,
        *,
        live_pos: LiveCandidatePosition,
        z_score: float,
        fz_analytics: CandidateAnalyticsState | None,
        target_group_capital: float | None,
    ) -> CloseCandidateAction | None:
        # Time stop: exit if open longer than multiplier × entry_half_life
        if (
            self.config.time_stop_half_life_multiplier is not None
            and live_pos.entry_half_life is not None
            and live_pos.days_open >= self.config.time_stop_half_life_multiplier * live_pos.entry_half_life
        ):
            return CloseCandidateAction(
                candidate_id=live_pos.candidate_id,
                group_id=live_pos.group_id,
                spread_id=live_pos.spread_id,
                reason="time_stop",
                z_score=z_score,
            )

        # Deterioration stop: exit if realized-weight MR score drops below threshold × entry
        if (
            self.config.mr_deterioration_threshold is not None
            and fz_analytics is not None
            and fz_analytics.mr_score is not None
            and live_pos.entry_mr_score is not None
            and live_pos.entry_mr_score > 0.0
            and fz_analytics.mr_score < self.config.mr_deterioration_threshold * live_pos.entry_mr_score
        ):
            return CloseCandidateAction(
                candidate_id=live_pos.candidate_id,
                group_id=live_pos.group_id,
                spread_id=live_pos.spread_id,
                reason="mr_deterioration_stop",
                z_score=z_score,
            )

        # PnL stop: exit if unrealized loss exceeds fraction of group capital
        if (
            self.config.pnl_stop_fraction is not None
            and target_group_capital is not None
            and live_pos.unrealized_pnl < -(self.config.pnl_stop_fraction * target_group_capital)
        ):
            return CloseCandidateAction(
                candidate_id=live_pos.candidate_id,
                group_id=live_pos.group_id,
                spread_id=live_pos.spread_id,
                reason="pnl_stop",
                z_score=z_score,
            )

        if live_pos.direction > 0 and z_score >= self.config.exit_z:
            return CloseCandidateAction(
                candidate_id=live_pos.candidate_id,
                group_id=live_pos.group_id,
                spread_id=live_pos.spread_id,
                reason="exit_long_zero_cross",
                z_score=z_score,
            )

        if live_pos.direction < 0 and z_score <= self.config.exit_z:
            return CloseCandidateAction(
                candidate_id=live_pos.candidate_id,
                group_id=live_pos.group_id,
                spread_id=live_pos.spread_id,
                reason="exit_short_zero_cross",
                z_score=z_score,
            )

        return None