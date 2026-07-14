from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd

from src.simulator.config import ExecutionConfig
from src.simulator.types import ExecutionFill, ExecutionResult, TickerDelta


@dataclass(slots=True)
class ExecutionEngine:
    config: ExecutionConfig

    def execute_deltas(
        self,
        *,
        deltas: list[TickerDelta],
        current_prices: pd.Series,
    ) -> ExecutionResult:
        fills: list[ExecutionFill] = []

        for delta in deltas:
            if delta.ticker not in current_prices.index:
                raise ValueError(f"Missing current price for ticker={delta.ticker!r}")

            fill_price = float(current_prices.loc[delta.ticker])
            filled_units = self._round_units(delta.delta_units)

            if filled_units == 0:
                continue

            traded_notional = abs(float(filled_units) * fill_price)
            commission = self._compute_commission(
                n_shares=abs(float(filled_units)),
                traded_notional=traded_notional,
            )

            fills.append(
                ExecutionFill(
                    ticker=delta.ticker,
                    requested_delta_units=float(delta.delta_units),
                    filled_delta_units=float(filled_units),
                    fill_price=fill_price,
                    traded_notional=traded_notional,
                    commission=commission,
                )
            )

        return ExecutionResult(fills=tuple(fills))

    def _round_units(
        self,
        units: float,
    ) -> float | int:
        if self.config.allow_fractional_shares:
            if abs(units) < self.config.min_abs_units:
                return 0.0
            return float(units)

        if self.config.share_rounding == "nearest":
            rounded = int(round(units))
        elif self.config.share_rounding == "floor":
            rounded = math.floor(units) if units >= 0 else math.ceil(units)
        elif self.config.share_rounding == "ceil":
            rounded = math.ceil(units) if units >= 0 else math.floor(units)
        else:
            raise ValueError(f"Unsupported share_rounding={self.config.share_rounding!r}")

        return 0 if abs(rounded) < 1 else int(rounded)

    def _compute_commission(
        self,
        *,
        n_shares: float,
        traded_notional: float,
    ) -> float:
        """
        IBKR Pro Fixed model:
        raw = commission_per_order + n_shares × commission_per_share
        cap = min(max_commission_per_order, max_commission_pct_of_trade × notional)
        If cap < min_commission, cap wins (IBKR rule for penny stocks).
        Otherwise clamp raw between min and cap.
        """
        cfg = self.config
        raw = cfg.commission_per_order + n_shares * cfg.commission_per_share
        pct_cap = cfg.max_commission_pct_of_trade * traded_notional
        cap = min(cfg.max_commission_per_order, pct_cap)

        if cap < cfg.min_commission_per_order:
            return cap

        return max(cfg.min_commission_per_order, min(raw, cap))