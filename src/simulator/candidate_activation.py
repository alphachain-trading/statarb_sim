from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.simulator.config import ActivationConfig
from src.simulator.types import CandidateRef


@dataclass(slots=True)
class CandidateActivation:
    """
    Candidate activation tracking.

    In single-timescale mode, active_by_group is keyed on group_id (str).
    In multi-timescale mode, active_by_group is keyed on (group_id, timescale_label)
    so each (residual_key, zlb) combination has independent activation within a group.

    The key type is always str — in multi-timescale mode, the key is
    "{group_id}|{timescale_label}" to keep the dict flat and hashable.
    """
    config: ActivationConfig
    active_by_group: dict[str, list[CandidateRef]] = field(default_factory=dict)

    @staticmethod
    def _activation_key(group_id: str, timescale_label: str = "") -> str:
        """Build the activation group key, incorporating timescale_label when present."""
        if timescale_label:
            return f"{group_id}|{timescale_label}"
        return group_id

    def process_new_arrivals(
        self,
        selected_refs: list[CandidateRef],
        *,
        is_flat_by_candidate_id: dict[str, bool],
        open_spread_ids: set[str] | set[tuple[str, str]] | None = None,
    ) -> None:
        for ref in selected_refs:
            if self.config.one_active_per_group:
                self._activate_portfolio_candidate(
                    ref=ref,
                    is_flat_by_candidate_id=is_flat_by_candidate_id,
                )
            else:
                self._activate_pair_candidate(
                    ref=ref,
                    open_spread_ids=open_spread_ids or set(),
                )

    def get_active_candidate(
        self,
        group_id: str,
        timescale_label: str = "",
    ) -> CandidateRef | None:
        """Convenience for portfolio path. Returns the single active candidate."""
        key = self._activation_key(group_id, timescale_label)
        refs = self.active_by_group.get(key, [])
        return refs[0] if refs else None

    def get_active_candidates(self, group_id: str | None = None) -> list[CandidateRef]:
        """
        Return active candidates.

        If group_id is provided, return active candidates for that group
        (across all timescales — matches keys starting with group_id).
        If group_id is None, return all active candidates across all groups.
        """
        if group_id is not None:
            out: list[CandidateRef] = []
            for key, refs in self.active_by_group.items():
                # Key is either "group_id" or "group_id|residual_key"
                key_group = key.split("|", 1)[0]
                if key_group == group_id:
                    out.extend(refs)
            return out
        return [
            ref
            for refs in self.active_by_group.values()
            for ref in refs
        ]

    def build_activation_frame(
        self,
        new_arrivals: list[CandidateRef],
    ) -> pd.DataFrame:
        """
        Build per-date activation view for snapshot construction.

        Tracked candidates are:
        - currently active candidates (across all groups)
        - plus new arrivals on the current date
        """
        tracked_by_id: dict[str, CandidateRef] = {}
        for refs in self.active_by_group.values():
            for ref in refs:
                tracked_by_id[ref.candidate_id] = ref

        new_ids = {ref.candidate_id for ref in new_arrivals}

        for ref in new_arrivals:
            tracked_by_id[ref.candidate_id] = ref

        active_ids = {
            ref.candidate_id
            for refs in self.active_by_group.values()
            for ref in refs
        }

        rows = []
        for ref in tracked_by_id.values():
            rows.append(
                {
                    "candidate_id": ref.candidate_id,
                    "spread_id": ref.spread_id,
                    "group_id": ref.group_id,
                    "is_new_arrival": ref.candidate_id in new_ids,
                    "is_active": ref.candidate_id in active_ids,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "spread_id",
                    "group_id",
                    "is_new_arrival",
                    "is_active",
                ],
                index=pd.Index([], name="candidate_id"),
            )

        out = pd.DataFrame(rows).set_index("candidate_id")
        return out.sort_values(["group_id", "spread_id"], kind="stable")

    # ── Portfolio path: one active per (group, timescale) ────────────

    def _activate_portfolio_candidate(
        self,
        ref: CandidateRef,
        *,
        is_flat_by_candidate_id: dict[str, bool],
    ) -> None:
        key = self._activation_key(ref.group_id, ref.timescale_label)
        current_list = self.active_by_group.get(key, [])
        current = current_list[0] if current_list else None

        if current is None:
            self.active_by_group[key] = [ref]
            return

        if current.candidate_id == ref.candidate_id:
            return

        if not self.config.switch_only_when_flat:
            self.active_by_group[key] = [ref]
            return

        current_is_flat = is_flat_by_candidate_id.get(current.candidate_id, True)
        if current_is_flat:
            self.active_by_group[key] = [ref]

    # ── Pair path: multiple active per (group, timescale) ────────────

    def _activate_pair_candidate(
        self,
        ref: CandidateRef,
        *,
        open_spread_ids: set[str] | set[tuple[str, str]],
    ) -> None:
        """
        Pair-mode activation.

        Rules:
        - New candidate always becomes active.
        - If an older candidate exists for the same (spread_id, residual_key)
          with no open position, it is replaced by the new one.
        - If an older candidate exists for the same (spread_id, residual_key)
          with an open position, the old one is kept alongside the new one
          (the open position must continue tracking its frozen weights).
          The new candidate cannot open a second position because the trader
          enforces one-position-per-(spread_id, residual_key).

        open_spread_ids can be set[str] (legacy) or set[tuple[str, str]]
        for multi-timescale (keyed on (spread_id, residual_key)).
        """
        key = self._activation_key(ref.group_id, ref.timescale_label)
        current_list = self.active_by_group.get(key, [])

        # Check whether a spread has an open position.
        # Handles both legacy set[str] and multi-timescale set[tuple[str, str]].
        def _is_open(r: CandidateRef) -> bool:
            if open_spread_ids and isinstance(next(iter(open_spread_ids)), tuple):
                return (r.spread_id, r.timescale_label) in open_spread_ids
            return r.spread_id in open_spread_ids

        # Remove stale candidates for the same spread_id (within this timescale)
        # that have no open position
        updated = [
            r for r in current_list
            if r.spread_id != ref.spread_id or _is_open(r)
        ]
        updated.append(ref)
        self.active_by_group[key] = updated