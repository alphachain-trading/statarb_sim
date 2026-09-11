"""
Tests for reconstruct_daily_portfolio_state (Track F1 commit 8a).

Hand-calculated scenario: one pair trade, entry and exit dates known,
weights and prices chosen so every intermediate quantity (unrealized PnL,
gross value, borrow cost, commission) can be checked by arithmetic, not
just "it ran." This is the correctness argument for the reconstruction
this commit stands up to eventually replace DailyPortfolioStateLogEntry's
day-by-day accumulation (commit 8b/8c).
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.config import ExecutionConfig
from src.simulator.execution import ExecutionEngine
from src.simulator.position_translator import PositionTranslator
from src.simulator.performance.daily_state_reconstruction import reconstruct_daily_portfolio_state
from src.simulator.types import LiveCandidatePosition


DATES = pd.bdate_range("2020-01-01", periods=6)  # d0..d5


def _price_matrix():
    # AAA rises, BBB falls -- a long-AAA/short-BBB spread (direction=+1,
    # weights AAA=+1, BBB=-1) profits from both legs.
    aaa = [100.0, 100.0, 102.0, 104.0, 106.0, 108.0]
    bbb = [50.0, 50.0, 49.0, 48.0, 47.0, 46.0]
    return pd.DataFrame({"AAA": aaa, "BBB": bbb}, index=DATES)


def _zero_cost_execution_engine():
    return ExecutionEngine(config=ExecutionConfig(
        allow_fractional_shares=True,
        min_abs_units=0.0,
        commission_per_share=0.0,
        commission_per_order=0.0,
        min_commission_per_order=0.0,
        max_commission_per_order=0.0,
        max_commission_pct_of_trade=0.0,
        short_borrow_rate_annual_bps=0.0,
    ))


class TestReconstructDailyPortfolioStateClosedTradeOnly(unittest.TestCase):
    """One closed trade, zero costs -- isolates the equity-curve arithmetic."""

    def setUp(self):
        self.price_matrix = _price_matrix()
        self.position_translator = PositionTranslator()
        self.execution_engine = _zero_cost_execution_engine()

        # entry on d1 (fit at d0), exit on d3.
        # direction=+1, pair_notional=10_000, weights AAA=1.0, BBB=-1.0 (gross_norm=2.0)
        # => signed_dollar_targets = [+5000, -5000] => units = [5000/100, -5000/50] = [50, -100]
        self.units = {"AAA": 50.0, "BBB": -100.0}
        entry_prices = {"AAA": 100.0, "BBB": 50.0}
        # realized_pnl_gross at exit (d3): AAA 104 vs 100 (+4*50=200), BBB 48 vs 50 (-2*-100=+200) => 400
        exit_prices = {"AAA": 104.0, "BBB": 48.0}
        realized_pnl_gross = sum(
            self.units[t] * (exit_prices[t] - entry_prices[t]) for t in self.units
        )
        self.closed_trades_df = pd.DataFrame([{
            "trade_id": "t1",
            "candidate_id": "t1",
            "group_id": "g1",
            "spread_id": "AAA|BBB",
            "entry_date": DATES[1],
            "entry_asof_date": DATES[0],
            "exit_date": DATES[3],
            "direction": 1.0,
            "pair_notional": 10_000.0,
            "realized_pnl_gross": realized_pnl_gross,
            "transaction_costs": 0.0,
        }])
        self.weights_lookup = {("AAA|BBB", DATES[0]): {"AAA": 1.0, "BBB": -1.0}}

    def _run(self):
        return reconstruct_daily_portfolio_state(
            dates=DATES,
            price_matrix=self.price_matrix,
            closed_trades_df=self.closed_trades_df,
            final_live_positions_by_candidate_id={},
            weights_lookup=self.weights_lookup,
            position_translator=self.position_translator,
            execution_engine=self.execution_engine,
            total_capital=1_000_000.0,
            short_borrow_rate_annual_bps=0.0,
        )

    def test_units_reconstructed_correctly(self):
        """Sanity check on the fixture's own hand-derived units before trusting downstream numbers."""
        from src.simulator.performance.daily_state_reconstruction import _reconstruct_entry_fill
        units, entry_prices, commission = _reconstruct_entry_fill(
            trade_row=self.closed_trades_df.iloc[0],
            weights_lookup=self.weights_lookup,
            price_matrix=self.price_matrix,
            position_translator=self.position_translator,
            execution_engine=self.execution_engine,
        )
        self.assertEqual(units, self.units)
        self.assertEqual(entry_prices, {"AAA": 100.0, "BBB": 50.0})
        self.assertEqual(commission, 0.0)

    def test_n_live_candidate_positions_window(self):
        df = self._run().set_index("date")
        # Live on [entry_date, exit_date - 1] = [d1, d2]; not on d0 (before entry),
        # not on d3 (removed before the snapshot on its own exit day).
        self.assertEqual(df.loc[DATES[0], "n_live_candidate_positions"], 0)
        self.assertEqual(df.loc[DATES[1], "n_live_candidate_positions"], 1)
        self.assertEqual(df.loc[DATES[2], "n_live_candidate_positions"], 1)
        self.assertEqual(df.loc[DATES[3], "n_live_candidate_positions"], 0)
        self.assertEqual(df.loc[DATES[4], "n_live_candidate_positions"], 0)

    def test_equity_curve_matches_hand_calculation(self):
        df = self._run().set_index("date")
        capital = 1_000_000.0

        # d0: no position yet.
        self.assertEqual(df.loc[DATES[0], "total_equity_gross"], capital)
        # d1 (entry day): unrealized_pnl=0 (just opened at today's own price).
        self.assertEqual(df.loc[DATES[1], "total_equity_gross"], capital)
        # d2: MTM at d2's prices (AAA=102, BBB=49) vs entry (AAA=100, BBB=50):
        #   unrealized = 50*(102-100) + (-100)*(49-50) = 100 + 100 = 200
        self.assertAlmostEqual(df.loc[DATES[2], "total_equity_gross"], capital + 200.0)
        # d3 (exit day): position removed before the snapshot; realized_pnl_gross=400
        # credited on d3 via cumulative_realized_pnl.
        self.assertAlmostEqual(df.loc[DATES[3], "total_equity_gross"], capital + 400.0)
        # d4 and onward: still capital + realized, nothing live.
        self.assertAlmostEqual(df.loc[DATES[4], "total_equity_gross"], capital + 400.0)

        # Zero costs in this fixture => equity_net == equity_gross throughout.
        pd.testing.assert_series_equal(
            df["total_equity_net"], df["total_equity_gross"],
            check_names=False,
        )

    def test_no_borrow_cost_when_rate_is_zero(self):
        df = self._run()
        self.assertTrue((df["daily_borrow_cost"] == 0.0).all())
        self.assertTrue((df["cumulative_borrow_costs"] == 0.0).all())


class TestReconstructDailyPortfolioStateBorrowAndCommission(unittest.TestCase):
    """A short position (direction=-1) with a nonzero borrow rate and commissions."""

    def setUp(self):
        self.price_matrix = _price_matrix()
        self.position_translator = PositionTranslator()
        # direction=-1 (short spread): AAA rising + BBB falling now HURTS the position,
        # since spread direction=-1 flips the sign of both legs' targets.
        self.execution_engine = ExecutionEngine(config=ExecutionConfig(
            allow_fractional_shares=True,
            min_abs_units=0.0,
            commission_per_share=0.01,
            commission_per_order=1.0,
            min_commission_per_order=0.0,
            max_commission_per_order=1_000_000.0,
            max_commission_pct_of_trade=1.0,
            short_borrow_rate_annual_bps=252 * 100,  # 1% per day exactly, for round numbers
        ))

    def test_borrow_cost_window_is_shifted_by_one_day_from_equity_window(self):
        """
        Borrow accrues on [entry_date+1, exit_date] -- entry day itself pays
        no borrow (added after the accrual step), but the exit day still
        pays it (removed only after the accrual step). This is the opposite
        window from the equity snapshot's [entry_date, exit_date-1].
        """
        # direction=-1, pair_notional=10_000, weights AAA=1.0, BBB=-1.0 (gross_norm=2.0)
        # signed_dollar_targets = -1 * 10000 * [0.5, -0.5] = [-5000, +5000]
        # units = [-5000/100, 5000/50] = [-50, 100]
        closed_trades_df = pd.DataFrame([{
            "trade_id": "t1", "candidate_id": "t1", "group_id": "g1",
            "spread_id": "AAA|BBB", "entry_date": DATES[1], "entry_asof_date": DATES[0],
            "exit_date": DATES[3], "direction": -1.0, "pair_notional": 10_000.0,
            "realized_pnl_gross": 0.0,  # irrelevant to this test
            "transaction_costs": 999.0,  # irrelevant to this test (only entry/exit split matters elsewhere)
        }])
        weights_lookup = {("AAA|BBB", DATES[0]): {"AAA": 1.0, "BBB": -1.0}}

        df = reconstruct_daily_portfolio_state(
            dates=DATES,
            price_matrix=self.price_matrix,
            closed_trades_df=closed_trades_df,
            final_live_positions_by_candidate_id={},
            weights_lookup=weights_lookup,
            position_translator=self.position_translator,
            execution_engine=self.execution_engine,
            total_capital=1_000_000.0,
            short_borrow_rate_annual_bps=252 * 100,
        ).set_index("date")

        # d1 (entry day): no borrow yet.
        self.assertEqual(df.loc[DATES[1], "daily_borrow_cost"], 0.0)
        # d2: gross_value at d2 prices: |-50*102| + |100*49| = 5100 + 4900 = 10000
        # rate = 100*100/10000/252 = 1.0/252 per day *252*100bps... use the same
        # formula the reconstruction uses: rate_annual_bps/10000/252
        rate = (252 * 100) / 10_000.0 / 252.0  # = 0.01 exactly
        self.assertAlmostEqual(df.loc[DATES[2], "daily_borrow_cost"], 10_000.0 * rate)
        # d3 (exit day): still charged (removed only after accrual) -- gross_value
        # at d3 prices: |-50*104| + |100*48| = 5200 + 4800 = 10000
        self.assertAlmostEqual(df.loc[DATES[3], "daily_borrow_cost"], 10_000.0 * rate)
        # d4: position gone, no more borrow.
        self.assertEqual(df.loc[DATES[4], "daily_borrow_cost"], 0.0)

    def test_entry_and_exit_commission_split_reproduces_total(self):
        closed_trades_df = pd.DataFrame([{
            "trade_id": "t1", "candidate_id": "t1", "group_id": "g1",
            "spread_id": "AAA|BBB", "entry_date": DATES[1], "entry_asof_date": DATES[0],
            "exit_date": DATES[3], "direction": 1.0, "pair_notional": 10_000.0,
            "realized_pnl_gross": 400.0,
            # Entry commission at d1 prices, units [50, -100]:
            #   AAA: 1.0 + 50*0.01 = 1.5; BBB: 1.0 + 100*0.01 = 2.0 => entry = 3.5
            # Exit commission at d3 prices, same units:
            #   identical n_shares => exit = 3.5 too (commission formula doesn't use price
            #   directly here since max_commission_pct_of_trade=1.0 never binds and
            #   min/max don't bind either)
            "transaction_costs": 7.0,
        }])
        weights_lookup = {("AAA|BBB", DATES[0]): {"AAA": 1.0, "BBB": -1.0}}

        df = reconstruct_daily_portfolio_state(
            dates=DATES,
            price_matrix=self.price_matrix,
            closed_trades_df=closed_trades_df,
            final_live_positions_by_candidate_id={},
            weights_lookup=weights_lookup,
            position_translator=self.position_translator,
            execution_engine=self.execution_engine,
            total_capital=1_000_000.0,
            short_borrow_rate_annual_bps=0.0,
        ).set_index("date")

        # Entry commission (3.5) charged on d1; exit commission (3.5) charged on d3.
        self.assertAlmostEqual(df.loc[DATES[0], "cumulative_transaction_costs"], 0.0)
        self.assertAlmostEqual(df.loc[DATES[1], "cumulative_transaction_costs"], 3.5)
        self.assertAlmostEqual(df.loc[DATES[2], "cumulative_transaction_costs"], 3.5)
        self.assertAlmostEqual(df.loc[DATES[3], "cumulative_transaction_costs"], 7.0)
        self.assertAlmostEqual(df.loc[DATES[4], "cumulative_transaction_costs"], 7.0)


class TestReconstructDailyPortfolioStateStillOpenPosition(unittest.TestCase):
    """A still-open position at run end: read directly from LiveCandidatePosition, no reconstruction."""

    def test_still_open_position_contributes_through_last_date(self):
        price_matrix = _price_matrix()
        live_pos = LiveCandidatePosition(
            trade_id="t2", candidate_id="t2", group_id="g1", spread_id="AAA|BBB",
            direction=1.0, pair_notional=10_000.0, gross_value=15_000.0,
            entry_date=DATES[1], entry_asof_date=DATES[0], days_open=4,
            units_by_ticker={"AAA": 50, "BBB": -100},
            entry_prices_by_ticker={"AAA": 100.0, "BBB": 50.0},
            accumulated_transaction_cost=3.5,
            accumulated_borrow_cost=0.0,
        )
        df = reconstruct_daily_portfolio_state(
            dates=DATES,
            price_matrix=price_matrix,
            closed_trades_df=pd.DataFrame(columns=[
                "trade_id", "candidate_id", "group_id", "spread_id", "entry_date",
                "entry_asof_date", "exit_date", "direction", "pair_notional",
                "realized_pnl_gross", "transaction_costs",
            ]),
            final_live_positions_by_candidate_id={"t2": live_pos},
            weights_lookup={},
            position_translator=PositionTranslator(),
            execution_engine=_zero_cost_execution_engine(),
            total_capital=1_000_000.0,
            short_borrow_rate_annual_bps=0.0,
        ).set_index("date")

        # Live from entry (d1) through the last simulated date (d5), never closed.
        self.assertEqual(df.loc[DATES[1], "n_live_candidate_positions"], 1)
        self.assertEqual(df.loc[DATES[-1], "n_live_candidate_positions"], 1)
        # d5 unrealized: AAA 108 vs 100 (+8*50=400), BBB 46 vs 50 (-4*-100=+400) => 800.
        # total_equity_gross does not subtract costs -- that's total_equity_net's job.
        self.assertAlmostEqual(df.loc[DATES[-1], "total_equity_gross"], 1_000_000.0 + 800.0)
        self.assertAlmostEqual(df.loc[DATES[-1], "total_equity_net"], 1_000_000.0 + 800.0 - 3.5)
        # entry commission (3.5) charged once, on d1, never an exit commission (still open).
        self.assertAlmostEqual(df.loc[DATES[1], "cumulative_transaction_costs"], 3.5)
        self.assertAlmostEqual(df.loc[DATES[-1], "cumulative_transaction_costs"], 3.5)


if __name__ == "__main__":
    unittest.main()
