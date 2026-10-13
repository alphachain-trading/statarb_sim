"""
Portfolio mean-reversion trader (v0.1).

Idea
----
Trade spread candidates on a plain z-score rule: open long when z <= -entry_z,
open short when z >= +entry_z, close on the zero-cross (exit_z). Open positions
are additionally guarded by three optional stops: a time stop (multiples of the
entry half-life), an MR-deterioration stop (live mr_score vs. entry mr_score)
and a PnL stop (fraction of target group capital).

Difference to PairSpreadMeanReversionTrader
-------------------------------------------
The pair trader is the live trader: it works on multi-timescale z-score
components, applies signal-quality gates, and resolves exit rules and max-hold
per residual_key. This trader has none of that; it reads the single `z_score`
column and only gates on `is_active` / `is_signal_ready`. It never sets
residual_key or z_components on its actions, and it proposes direction only:
position size comes from SizingEngine (`pair_notional`).

Status: dormant since May 2026 (kept for later revival). It is constructed in
simulator_factory when a PortfolioMeanReversionConfig is given, but no baseline
or sweep does.

Sizing note: originally the strategy set the position size itself via
`target_group_exposure` (magnitude). Since bb69f08 it only sets direction;
size comes from SizingEngine / pair_notional. Revisit the sizing concept on
revival. `default_abs_target_group_exposure` is kept as a placeholder for it:
it is still mandatory but not read.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.simulator.actions import CandidateAction, CloseCandidateAction, OpenCandidateAction
from src.simulator.config import PortfolioMeanReversionConfig
from src.simulator.types import (
    CandidateAnalyticsState,
    CandidateMarketSnapshot,
    LiveCandidatePosition,
)


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
        * time stop: exit when days_open >= multiplier x entry_half_life
        * deterioration stop: exit when fz_mr_score < mr_deterioration_threshold x entry_mr_score
        * close long when z >= exit_z
        * close short when z <= exit_z
    """

    config: PortfolioMeanReversionConfig
    # Mandatory — no default (Track D). Not part of PortfolioMeanReversionConfig
    # (adding a new config field is out of scope for D) but the same
    # anti-pattern: a numeric that affects results, sitting one layer below
    # the config object as a second, ungated dataclass default.
    default_abs_target_group_exposure: float

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
            live_pos = live_positions_by_candidate_id.get(str(candidate_id))
            action: CandidateAction | None

            if live_pos is None:
                action = self._maybe_open(
                    candidate_id=str(candidate_id),
                    group_id=str(row["group_id"]),
                    spread_id=str(row["spread_id"]),
                    z_score=z_score,
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
    ) -> OpenCandidateAction | None:
        if self.config.allow_long and z_score <= -self.config.entry_z:
            return OpenCandidateAction(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                direction=+1.0,
                reason="entry_long",
                z_score=z_score,
            )

        if self.config.allow_short and z_score >= self.config.entry_z:
            return OpenCandidateAction(
                candidate_id=candidate_id,
                group_id=group_id,
                spread_id=spread_id,
                direction=-1.0,
                reason="entry_short",
                z_score=z_score,
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
        # Time stop: exit if open longer than multiplier x entry_half_life
        if (
            self.config.time_stop_half_life_multiplier is not None
            and live_pos.entry_half_life is not None
            and live_pos.days_open
            >= self.config.time_stop_half_life_multiplier * live_pos.entry_half_life
        ):
            return CloseCandidateAction(
                candidate_id=live_pos.candidate_id,
                group_id=live_pos.group_id,
                spread_id=live_pos.spread_id,
                reason="time_stop",
                z_score=z_score,
            )

        # Deterioration stop: exit if realized-weight MR score drops below threshold x entry
        if (
            self.config.mr_deterioration_threshold is not None
            and fz_analytics is not None
            and fz_analytics.mr_score is not None
            and live_pos.entry_mr_score is not None
            and live_pos.entry_mr_score > 0.0
            and fz_analytics.mr_score
            < self.config.mr_deterioration_threshold * live_pos.entry_mr_score
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
