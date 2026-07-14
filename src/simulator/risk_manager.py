from __future__ import annotations

from dataclasses import dataclass
from typing import Counter

import numpy as np
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
    1. Timescale selection — filter/select across rkeys per spread
    2. Max concurrent positions cap
    3. Timescale concentration cap
    4. Gross exposure cap
    5. Per-ticker concentration cap
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

        # ── Step 1: Timescale selection ──────────────────────────────
        filtered = self._apply_timescale_policy(
            sized_opens=sized_opens,
            live_positions=live_positions,
        )

        # ── Steps 2-5: Portfolio constraint checks ───────────────────
        current_gross_notional = self._compute_gross_notional(live_positions, current_prices)
        ticker_net_notional = self._compute_ticker_net_notional(live_positions, current_prices)
        n_positions = len(live_positions)

        # Rank by signal strength
        ranked = sorted(filtered, key=lambda x: abs(x[0].z_score) if x[0].z_score is not None else 0.0, reverse=True)

        approved: list[tuple[OpenCandidateAction, float, dict, float]] = []

        ts_counts: Counter[str] = Counter()
        for pos in live_positions.values():
            ts_counts[pos.residual_key] += 1

        for action, pair_notional, feature_scores, size_multiplier in ranked:
            # Position count cap
            if self.config.max_concurrent_positions is not None:
                if n_positions >= self.config.max_concurrent_positions:
                    break

            # Timescale concentration cap
            ts_cfg = self.config.timescale_risk
            if ts_cfg is not None and ts_cfg.max_pct_single_timescale is not None:
                total_after = n_positions + 1
                rkey_count_after = ts_counts[action.residual_key] + 1
                if total_after > 0 and rkey_count_after / total_after > ts_cfg.max_pct_single_timescale:
                    continue

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
            ts_counts[action.residual_key] += 1
            current_gross_notional += pair_notional
            self._update_ticker_net_notional(
                ticker_net_notional=ticker_net_notional,
                action=action,
                pair_notional=pair_notional,
            )

        return approved

    def _apply_timescale_policy(
        self,
        *,
        sized_opens: list[tuple[OpenCandidateAction, float, dict, float]],
        live_positions: dict[str, LiveCandidatePosition],
    ) -> list[tuple[OpenCandidateAction, float, dict, float]]:
        ts_cfg = self.config.timescale_risk
        if ts_cfg is None or ts_cfg.max_timescales_per_spread is None:
            return sized_opens

        max_ts = ts_cfg.max_timescales_per_spread

        live_ts_per_spread: dict[str, set[str]] = {}
        for pos in live_positions.values():
            live_ts_per_spread.setdefault(pos.spread_id, set()).add(pos.residual_key)

        opens_by_spread: dict[str, list[tuple[OpenCandidateAction, float, dict, float]]] = {}
        for item in sized_opens:
            opens_by_spread.setdefault(item[0].spread_id, []).append(item)

        result: list[tuple[OpenCandidateAction, float, dict, float]] = []

        for spread_id, items in opens_by_spread.items():
            live_rkeys = live_ts_per_spread.get(spread_id, set())
            slots_available = max_ts - len(live_rkeys)

            if slots_available <= 0:
                continue

            if len(items) <= slots_available:
                result.extend(items)
                continue

            selected = self._select_timescales(items, slots_available, ts_cfg.selection)
            result.extend(selected)

        return result

    @staticmethod
    def _select_timescales(
        items: list[tuple[OpenCandidateAction, float, dict, float]],
        n: int,
        policy: str,
    ) -> list[tuple[OpenCandidateAction, float, dict, float]]:
        if policy == "max_abs_z":
            ranked = sorted(items, key=lambda x: abs(x[0].z_score) if x[0].z_score is not None else 0.0, reverse=True)
            return ranked[:n]

        def _extract_hl(rkey: str) -> int:
            try:
                for part in rkey.split("_"):
                    if part.startswith("hl") and part[2:].isdigit():
                        return int(part[2:])
            except (ValueError, IndexError):
                pass
            return 999999

        if policy == "min_hl":
            return sorted(items, key=lambda x: _extract_hl(x[0].residual_key))[:n]

        if policy == "max_hl":
            return sorted(items, key=lambda x: _extract_hl(x[0].residual_key), reverse=True)[:n]

        if policy == "balanced":
            ranked = sorted(items, key=lambda x: _extract_hl(x[0].residual_key))
            if n >= len(ranked):
                return ranked
            indices = np.linspace(0, len(ranked) - 1, n, dtype=int)
            return [ranked[i] for i in indices]

        # Fallback: max_abs_z
        return sorted(items, key=lambda x: abs(x[0].z_score) if x[0].z_score is not None else 0.0, reverse=True)[:n]

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