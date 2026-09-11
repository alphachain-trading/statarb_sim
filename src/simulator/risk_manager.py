from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.simulator.actions import OpenCandidateAction
from src.simulator.config import RiskManagerConfig
from src.simulator.types import CandidateAnalyticsState, LiveCandidatePosition


@dataclass(slots=True)
class RiskManager:
    """
    Filters proposed open actions against portfolio constraints.

    Receives pre-sized (action, pair_notional) pairs from SizingEngine.
    Zero-notional trades have already been dropped by SizingEngine.

    Pipeline (in order):
    1. Max concurrent positions cap
    2. Gross exposure cap
    3. Per-ticker concentration cap
    """

    config: RiskManagerConfig
    total_capital: float

    def approve(
        self,
        sized_opens: list[tuple[OpenCandidateAction, float, dict, float]],
        live_positions: dict[str, LiveCandidatePosition],
        current_prices: pd.Series,
    ) -> list[tuple[OpenCandidateAction, float, dict, float]]:
        """
        Filter sized opens against portfolio constraints.

        Parameters
        ----------
        sized_opens
            (action, pair_notional) pairs from SizingEngine. All have notional > 0.
        live_positions
            Currently open positions.
        current_prices
            Latest prices for gross notional and ticker exposure computation.

        Returns
        -------
        Approved (action, pair_notional) pairs.
        """
        if not sized_opens:
            return []
        if self.total_capital <= 0.0:
            return []

        # ── Portfolio constraint checks ───────────────────────────────
        current_gross_notional = self._compute_gross_notional(live_positions, current_prices)
        ticker_net_notional = self._compute_ticker_net_notional(live_positions, current_prices)
        n_positions = len(live_positions)

        # Rank by signal strength
        ranked = sorted(sized_opens, key=lambda x: abs(x[0].z_score) if x[0].z_score is not None else 0.0, reverse=True)

        approved: list[tuple[OpenCandidateAction, float, dict, float]] = []

        for action, pair_notional, feature_scores, size_multiplier in ranked:
            # Position count cap
            if self.config.max_concurrent_positions is not None:
                if n_positions >= self.config.max_concurrent_positions:
                    break

            # Gross exposure check
            if current_gross_notional + pair_notional > self.config.max_gross_exposure * self.total_capital:
                continue

            # Ticker exposure check
            if not self._check_ticker_exposure(
                action=action,
                pair_notional=pair_notional,
                ticker_net_notional=ticker_net_notional,
            ):
                continue

            approved.append((action, pair_notional, feature_scores, size_multiplier))
            n_positions += 1
            current_gross_notional += pair_notional
            self._update_ticker_net_notional(
                ticker_net_notional=ticker_net_notional,
                action=action,
                pair_notional=pair_notional,
            )

        return approved

    @staticmethod
    def _compute_gross_notional(
        live_positions: dict[str, LiveCandidatePosition],
        current_prices: pd.Series,
    ) -> float:
        total = 0.0
        for pos in live_positions.values():
            for ticker, units in pos.units_by_ticker.items():
                if ticker in current_prices.index:
                    total += abs(int(units) * float(current_prices.loc[ticker]))
        return total

    @staticmethod
    def _compute_ticker_net_notional(
        live_positions: dict[str, LiveCandidatePosition],
        current_prices: pd.Series,
    ) -> dict[str, float]:
        net: dict[str, float] = {}
        for pos in live_positions.values():
            for ticker, units in pos.units_by_ticker.items():
                if ticker in current_prices.index:
                    notional = int(units) * float(current_prices.loc[ticker])
                    net[ticker] = net.get(ticker, 0.0) + notional
        return net

    def _check_ticker_exposure(
        self,
        *,
        action: OpenCandidateAction,
        pair_notional: float,
        ticker_net_notional: dict[str, float],
    ) -> bool:
        max_ticker_notional = self.config.max_ticker_exposure_pct * self.total_capital
        tickers = action.spread_id.split("|")

        for i, ticker in enumerate(tickers):
            leg_sign = action.direction if i == 0 else -action.direction
            leg_notional = leg_sign * (pair_notional / len(tickers))
            new_net = ticker_net_notional.get(ticker, 0.0) + leg_notional
            if abs(new_net) > max_ticker_notional:
                return False

        return True

    @staticmethod
    def _update_ticker_net_notional(
        *,
        ticker_net_notional: dict[str, float],
        action: OpenCandidateAction,
        pair_notional: float,
    ) -> None:
        tickers = action.spread_id.split("|")
        for i, ticker in enumerate(tickers):
            leg_sign = action.direction if i == 0 else -action.direction
            leg_notional = leg_sign * (pair_notional / len(tickers))
            ticker_net_notional[ticker] = ticker_net_notional.get(ticker, 0.0) + leg_notional