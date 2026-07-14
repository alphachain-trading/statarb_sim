from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ActionKind = Literal[
    "open_candidate",
    "close_candidate",
    "scale_candidate",
    "flip_candidate",
    "substitute_candidate",
]


@dataclass(slots=True, frozen=True)
class CandidateAction:
    candidate_id: str
    group_id: str
    spread_id: str
    reason: str | None = None
    z_score: float | None = None

    @property
    def action_kind(self) -> str:
        raise NotImplementedError


@dataclass(slots=True, frozen=True)
class OpenCandidateAction(CandidateAction):
    """
    Open a new candidate position.

    direction is +1.0 (long spread) or -1.0 (short spread).
    pair_notional is the dollar notional set by SizingEngine after the trader
    proposes the trade. It is None when the action is first created by the
    trader and populated by SizingEngine before execution.
    """
    direction: float = 0.0                  # +1.0 or -1.0; set by trader
    pair_notional: float | None = None      # set by SizingEngine; None until sized
    residual_key: str = ""
    z_components: tuple = ()

    def __post_init__(self) -> None:
        if self.direction not in (1.0, -1.0):
            raise ValueError(
                f"OpenCandidateAction.direction must be +1.0 or -1.0, got {self.direction}."
            )

    @property
    def action_kind(self) -> str:
        return "open_candidate"

    def with_notional(self, pair_notional: float) -> "OpenCandidateAction":
        """Return a copy of this action with pair_notional set."""
        return OpenCandidateAction(
            candidate_id=self.candidate_id,
            group_id=self.group_id,
            spread_id=self.spread_id,
            reason=self.reason,
            z_score=self.z_score,
            direction=self.direction,
            pair_notional=pair_notional,
            residual_key=self.residual_key,
            z_components=self.z_components,
        )


@dataclass(slots=True, frozen=True)
class CloseCandidateAction(CandidateAction):
    """
    Close an existing candidate position fully.
    """
    z_components: tuple = ()

    @property
    def action_kind(self) -> str:
        return "close_candidate"


@dataclass(slots=True, frozen=True)
class ScaleCandidateAction(CandidateAction):
    """
    Change exposure of an already open candidate without changing candidate_id.
    """
    direction: float = 0.0
    pair_notional: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in (1.0, -1.0):
            raise ValueError(
                f"ScaleCandidateAction.direction must be +1.0 or -1.0, got {self.direction}."
            )

    @property
    def action_kind(self) -> str:
        return "scale_candidate"


@dataclass(slots=True, frozen=True)
class FlipCandidateAction(CandidateAction):
    """
    Reverse direction of an already open candidate.
    """
    direction: float = 0.0
    pair_notional: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in (1.0, -1.0):
            raise ValueError(
                f"FlipCandidateAction.direction must be +1.0 or -1.0, got {self.direction}."
            )

    @property
    def action_kind(self) -> str:
        return "flip_candidate"


@dataclass(slots=True, frozen=True)
class SubstituteCandidateAction:
    """
    Replace one open candidate by a newer candidate in the same group.
    """
    old_candidate_id: str
    old_spread_id: str
    new_candidate_id: str
    new_spread_id: str
    group_id: str
    direction: float = 0.0
    pair_notional: float | None = None
    reason: str | None = None
    z_score_old: float | None = None
    z_score_new: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in (1.0, -1.0):
            raise ValueError(
                f"SubstituteCandidateAction.direction must be +1.0 or -1.0, got {self.direction}."
            )
        if self.old_candidate_id == self.new_candidate_id:
            raise ValueError("old_candidate_id and new_candidate_id must differ.")

    @property
    def action_kind(self) -> str:
        return "substitute_candidate"
