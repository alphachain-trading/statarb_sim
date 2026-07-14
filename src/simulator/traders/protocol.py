from __future__ import annotations

from typing import Protocol

from src.simulator.actions import CandidateAction
from src.simulator.types import CandidateAnalyticsState, CandidateMarketSnapshot, LiveCandidatePosition


class Trader(Protocol):
    """
    Interface shared by all trader implementations.

    The simulator calls generate_actions() once per date. The trader
    proposes opens and closes based on signal logic only — portfolio-level
    constraints are enforced separately by the RiskManager.
    """

    def generate_actions(
        self,
        snapshot: CandidateMarketSnapshot,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        live_diagnostics_by_candidate_id: dict[str, CandidateAnalyticsState] | None = None,
        target_group_capital_by_group: dict[str, float] | None = None,
    ) -> list[CandidateAction]: ...
    