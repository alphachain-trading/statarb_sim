from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulator.actions import CloseCandidateAction, OpenCandidateAction
from src.simulator.types import CandidateRef, LiveCandidatePosition, TickerDelta


@dataclass(slots=True)
class PositionTranslator:
    """
    Translate high-level candidate actions into ticker-unit deltas.

    OpenCandidateAction must arrive with pair_notional set (by SizingEngine).
    The translator uses direction * pair_notional to compute signed dollar
    targets per ticker leg, then divides by price to get units.
    """

    def translate_open(
        self,
        *,
        action: OpenCandidateAction,
        candidate_ref: CandidateRef,
        current_prices: pd.Series,
    ) -> list[TickerDelta]:
        if action.candidate_id != candidate_ref.candidate_id:
            raise ValueError("action.candidate_id and candidate_ref.candidate_id do not match.")
        if action.pair_notional is None or action.pair_notional <= 0.0:
            raise ValueError(
                f"OpenCandidateAction.pair_notional must be positive before translation, "
                f"got {action.pair_notional}. Ensure SizingEngine.size() ran before execution."
            )

        tickers = list(candidate_ref.members)
        weights = np.asarray(candidate_ref.weights, dtype=float)

        gross_norm = float(np.sum(np.abs(weights)))
        if gross_norm <= 0.0:
            raise ValueError("Candidate weights must have positive gross norm.")

        missing = [t for t in tickers if t not in current_prices.index]
        if missing:
            raise ValueError(f"Missing current prices for tickers: {missing}")

        px = current_prices.loc[tickers].astype(float)
        if (px <= 0.0).any():
            bad = px[px <= 0.0]
            raise ValueError(f"Non-positive prices encountered: {bad.to_dict()}")

        # direction * pair_notional * normalized_weight → signed dollar target per leg
        signed_dollar_targets = (
            action.direction * action.pair_notional * (weights / gross_norm)
        )
        target_units = signed_dollar_targets / px.to_numpy(dtype=float)

        return [
            TickerDelta(ticker=ticker, delta_units=float(delta_units))
            for ticker, delta_units in zip(tickers, target_units, strict=True)
        ]

    def translate_close(
        self,
        *,
        action: CloseCandidateAction,
        live_position: LiveCandidatePosition,
    ) -> list[TickerDelta]:
        if action.candidate_id != live_position.candidate_id:
            raise ValueError("action.candidate_id and live_position.candidate_id do not match.")

        return [
            TickerDelta(ticker=ticker, delta_units=-float(units))
            for ticker, units in live_position.units_by_ticker.items()
        ]
