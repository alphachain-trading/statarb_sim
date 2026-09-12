"""
F2 C5: reconstruct a spread's level and z-score from persisted artifacts.

Every quantity here goes through CandidateSignalGenerator.compute_analytics_from_weights,
which itself calls the single shared spread_primitives.compute_spread_level primitive
(F2 R2) -- no separate implementation exists here, per the "single code path" rule
(F2_loop_and_fidelity.md, "Single code path").

Two views, both fidelity-tested (F2_loop_and_fidelity.md, "Spread level and
reconstruction"):
- candidate view: the latest refit at or before `date` -- what a fresh observer of
  this spread would see today.
- position view: frozen at a specific `asof_date` (a live position's own entry
  refit) -- what that position has been tracking since it opened, regardless of
  any newer candidate having since become active for the same spread.

Both reduce to the same underlying lookup once `asof_date` is fixed: resolve that
candidate's own frozen weights from weights_lookup, and its own asof-frozen
residual model via CandidateSignalGenerator. The two functions below differ only
in how they pick `asof_date` for a given (spread_id, date) query.
"""
from __future__ import annotations

import pandas as pd

from src.simulator.candidate_signals import CandidateSignalGenerator
from src.simulator.types import CandidateAnalyticsState


def reconstruct_level_and_zscore(
    *,
    signal_generator: CandidateSignalGenerator,
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]],
    group_id: str,
    residual_key: str,
    spread_id: str,
    asof_date: pd.Timestamp,
    date: pd.Timestamp,
    timescale_label: str = "",
    candidate_id: str | None = None,
) -> CandidateAnalyticsState:
    """
    Reconstruct one candidate's level/z-score, given which asof-frozen model
    and weights to use. The candidate-view and position-view functions below
    are the only two ways a caller should arrive at `asof_date`.

    Raises NotImplementedError if the resolved model used PC removal
    (remove_residual_pcs > 0): PC sign is defined only up to a factor of -1,
    and near-degenerate eigenvalues can rotate the basis on refit, so this
    path is unverified and not covered by the fidelity test
    (F2_loop_and_fidelity.md, "PC removal").
    """
    asof_date = pd.Timestamp(asof_date)
    date = pd.Timestamp(date)

    group_params = signal_generator.precomputed_residual_params.get((group_id, residual_key))
    model = None if group_params is None else group_params.get(asof_date)
    if model is not None and model.pc_components is not None:
        raise NotImplementedError(
            f"Reconstruction does not support remove_residual_pcs > 0 "
            f"(group_id={group_id!r}, residual_key={residual_key!r}, asof_date={asof_date}): "
            "PC removal's sign is defined only up to a factor of -1 and near-degenerate "
            "eigenvalues can rotate the basis on refit, so this path is unverified and not "
            "covered by the fidelity test."
        )

    weights_by_ticker = weights_lookup.get((spread_id, asof_date))
    if weights_by_ticker is None:
        raise KeyError(
            f"No weights for (spread_id={spread_id!r}, asof_date={asof_date}) in weights_lookup."
        )

    return signal_generator.compute_analytics_from_weights(
        date=date,
        asof_date=asof_date,
        group_id=group_id,
        candidate_id=candidate_id or f"{spread_id}@{asof_date.date()}",
        spread_id=spread_id,
        weights_by_ticker=weights_by_ticker,
        residual_key=residual_key,
        timescale_label=timescale_label,
        skip_diagnostics=True,
    )


def reconstruct_candidate_view(
    *,
    signal_generator: CandidateSignalGenerator,
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]],
    selected_panel: pd.DataFrame,
    group_id: str,
    residual_key: str,
    spread_id: str,
    date: pd.Timestamp,
    timescale_label: str = "",
) -> CandidateAnalyticsState | None:
    """
    Candidate view: the latest refit at or before `date` for this spread.

    `selected_panel` is the run's own selected_panel (candidate_panel.parquet
    after CandidateFilter, or the equivalent loaded from disk) -- must carry
    spread_id, group_id, residual_key, asof_date. Returns None if no candidate
    for this spread exists at or before `date`.
    """
    date = pd.Timestamp(date)
    mask = (
        (selected_panel["spread_id"] == spread_id)
        & (selected_panel["group_id"] == group_id)
        & (selected_panel["residual_key"] == residual_key)
        & (pd.to_datetime(selected_panel["asof_date"]) <= date)
    )
    candidates = selected_panel.loc[mask]
    if candidates.empty:
        return None

    asof_date = pd.Timestamp(pd.to_datetime(candidates["asof_date"]).max())
    return reconstruct_level_and_zscore(
        signal_generator=signal_generator,
        weights_lookup=weights_lookup,
        group_id=group_id,
        residual_key=residual_key,
        spread_id=spread_id,
        asof_date=asof_date,
        date=date,
        timescale_label=timescale_label,
    )


def reconstruct_position_view(
    *,
    signal_generator: CandidateSignalGenerator,
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]],
    group_id: str,
    residual_key: str,
    spread_id: str,
    entry_asof_date: pd.Timestamp,
    date: pd.Timestamp,
    timescale_label: str = "",
    candidate_id: str | None = None,
) -> CandidateAnalyticsState:
    """
    Position view: frozen at `entry_asof_date`, regardless of any newer
    candidate having since become active for the same spread.
    """
    return reconstruct_level_and_zscore(
        signal_generator=signal_generator,
        weights_lookup=weights_lookup,
        group_id=group_id,
        residual_key=residual_key,
        spread_id=spread_id,
        asof_date=pd.Timestamp(entry_asof_date),
        date=date,
        timescale_label=timescale_label,
        candidate_id=candidate_id,
    )
