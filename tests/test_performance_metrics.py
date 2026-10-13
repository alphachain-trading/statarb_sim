"""
Hand-computed tests for the headline performance numbers (trade metrics, equity-curve
metrics, Sortino). These are the figures a sweep is ranked on, so a silent definitional
slip here changes every conclusion.
"""

import math
import unittest

import numpy as np
import pandas as pd

from src.simulator.config import PerformanceConfig
from src.simulator.performance.performance_basic import (
    _compute_trade_metrics,
    _extract_returns,
    compute_performance,
)
from src.simulator.performance.performance_objective import sortino_ratio


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group_id": ["G1", "G1", "G2", "G2"],
            "realized_pnl_net": [100.0, -50.0, 0.0, 200.0],
            "realized_pnl_gross": [110.0, -40.0, 5.0, 210.0],
            "transaction_costs": [10.0, 10.0, 5.0, 10.0],
            "borrow_costs": [0.0, 0.0, 0.0, 0.0],
            "days_open": [2, 4, 6, 8],
            "entry_z_score": [-2.0, 2.0, -2.5, 3.0],
            "exit_z_score": [0.0, 0.5, 0.0, -0.5],
        }
    )


class TestTradeMetrics(unittest.TestCase):
    def setUp(self):
        self.m = _compute_trade_metrics(trades=_trades(), label="x")

    def test_counts_and_pnl(self):
        self.assertEqual((self.m["n_trades"], self.m["n_groups_traded"]), (4, 2))
        self.assertEqual((self.m["total_gross_pnl"], self.m["total_net_pnl"]), (285.0, 250.0))

    def test_win_loss_split_counts_zero_pnl_as_a_loss(self):
        # net: wins = {100, 200}; losses = {-50, 0}
        self.assertEqual((self.m["win_rate_net"], self.m["loss_rate_net"]), (0.5, 0.5))
        self.assertEqual(self.m["mean_gain_per_win_net"], 150.0)
        self.assertEqual(self.m["mean_loss_per_loss_net"], -25.0)
        # gross: all but -40 are wins
        self.assertEqual(self.m["win_rate_gross"], 0.75)

    def test_profit_factor(self):
        self.assertEqual(self.m["profit_factor_net"], 300.0 / 50.0)

    def test_costs(self):
        self.assertEqual(self.m["total_transaction_costs"], 35.0)
        self.assertEqual(self.m["total_costs"], 35.0)
        self.assertEqual(self.m["avg_cost_per_trade"], 8.75)

    def test_holding_days_and_entry_exit_z(self):
        self.assertEqual(self.m["avg_holding_days"], 5.0)
        self.assertEqual(self.m["avg_holding_days_wins"], 5.0)  # (2 + 8) / 2
        self.assertEqual(self.m["avg_holding_days_losses"], 5.0)  # (4 + 6) / 2
        self.assertEqual(self.m["avg_entry_z_score"], 0.125)
        self.assertEqual(self.m["avg_abs_entry_z_score"], 2.375)
        self.assertEqual(self.m["avg_exit_z_score"], 0.0)

    def test_edge_cases(self):
        wins_only = _trades().assign(realized_pnl_net=[1.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            _compute_trade_metrics(trades=wins_only, label="x")["profit_factor_net"], math.inf
        )
        losses_only = _trades().assign(realized_pnl_net=[-1.0, -2.0, 0.0, -4.0])
        m = _compute_trade_metrics(trades=losses_only, label="x")
        self.assertEqual(m["profit_factor_net"], 0.0)
        self.assertTrue(math.isnan(m["mean_gain_per_win_net"]))
        flat = _trades().assign(realized_pnl_net=0.0)
        self.assertTrue(
            math.isnan(_compute_trade_metrics(trades=flat, label="x")["profit_factor_net"])
        )

    def test_no_trades_gives_nan_metrics_with_zero_counts(self):
        m = _compute_trade_metrics(trades=pd.DataFrame(), label="x")
        self.assertEqual((m["n_trades"], m["n_groups_traded"]), (0, 0))
        self.assertTrue(math.isnan(m["win_rate_net"]))

    def test_nan_entry_z_values_are_ignored_in_averages(self):
        t = _trades().assign(entry_z_score=[-2.0, np.nan, -2.0, 4.0])
        self.assertAlmostEqual(
            _compute_trade_metrics(trades=t, label="x")["avg_entry_z_score"], 0.0
        )


class TestSortino(unittest.TestCase):
    def test_hand_computed(self):
        # r = [0.1, -0.1, 0.1]: mean 1/30 -> *252 = 8.4 ; only the negative return counts
        # in the downside deviation: sqrt(0.01) * sqrt(252) -> 8.4 / (0.1 * sqrt(252))
        got = sortino_ratio(pd.Series([0.1, -0.1, 0.1]))
        self.assertAlmostEqual(got, 8.4 / (0.1 * math.sqrt(252)))

    def test_downside_deviation_averages_over_negative_observations_only(self):
        # Pins current definition. The textbook downside deviation averages the squared
        # shortfalls over ALL observations (here 3), which would give a sqrt(3) higher ratio.
        r = pd.Series([0.1, -0.1, 0.1])
        textbook = 8.4 / (math.sqrt(0.01 / 3) * math.sqrt(252))
        self.assertAlmostEqual(textbook / sortino_ratio(r), math.sqrt(3))

    def test_degenerate_cases(self):
        self.assertEqual(sortino_ratio(pd.Series([0.1])), 0.0)  # < 2 observations
        self.assertEqual(sortino_ratio(pd.Series([0.1, 0.2, 0.3])), math.inf)  # no downside
        self.assertEqual(sortino_ratio(pd.Series([0.0, 0.0, 0.0])), 0.0)
        self.assertEqual(sortino_ratio(pd.Series([-0.1, -0.1, 0.0])) < 0.0, True)

    def test_mar_shifts_the_threshold(self):
        r = pd.Series([0.02, 0.01, 0.03])
        self.assertEqual(sortino_ratio(r), math.inf)
        self.assertTrue(math.isfinite(sortino_ratio(r, mar=0.02)))


def _daily(equity_net: list[float], equity_gross: list[float] | None = None) -> pd.DataFrame:
    n = len(equity_net)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=n),
            "total_equity_net": equity_net,
            "total_equity_gross": equity_gross or equity_net,
            "total_capital": [1000.0] * n,
            "n_live_candidate_positions": [0, 1, 2, 1][:n] if n == 4 else [1] * n,
            "cumulative_transaction_costs": np.linspace(0.0, 4.0, n),
            "cumulative_borrow_costs": np.linspace(0.0, 1.0, n),
            "daily_borrow_cost": [0.0, 0.25, 0.25, 0.5][:n] if n == 4 else [0.0] * n,
        }
    )


class TestEquityMetrics(unittest.TestCase):
    def setUp(self):
        # net equity 1000 -> 1100 -> 990 -> 1089 : returns +10%, -10%, +10%
        self.perf = compute_performance(
            _trades(), _daily([1000.0, 1100.0, 990.0, 1089.0]), PerformanceConfig(enabled=True)
        )
        self.m = self.perf.metrics

    def test_returns_start_at_the_first_active_day(self):
        np.testing.assert_allclose(self.perf.returns_net.to_numpy(), [0.1, -0.1, 0.1])
        self.assertEqual(self.perf.returns_net.name, "returns_net")

    def test_leading_flat_period_is_trimmed(self):
        _g, n = _extract_returns(_daily([1000.0, 1000.0, 1000.0, 1010.0]))
        self.assertEqual(len(n), 1)
        self.assertAlmostEqual(n.iloc[0], 0.01)

    def test_sharpe(self):
        # mean 1/30, sample std sqrt(0.02666../2) = 0.11547 -> * sqrt(252)
        std = math.sqrt(((0.1 - 1 / 30) ** 2 * 2 + (-0.1 - 1 / 30) ** 2) / 2)
        self.assertAlmostEqual(self.m["sharpe_net"], (1 / 30) / std * math.sqrt(252))

    def test_drawdown_and_returns_use_the_trimmed_window(self):
        # window equity 1100, 990, 1089: max drawdown -10%; total return 1089/1100 - 1
        self.assertAlmostEqual(self.m["max_drawdown_net"], -0.10)
        self.assertAlmostEqual(self.m["total_return_net"], 1089 / 1100 - 1)
        self.assertEqual(self.m["n_trading_days"], 3)

    def test_annualised_return_and_calmar(self):
        # NOTE: 3 daily returns are counted (n_days=3) but the total return spans only the
        # 2 intervals after the first active day's close — pinned as implemented.
        annual = (1089 / 1100) ** (252 / 3) - 1
        self.assertAlmostEqual(self.m["annual_return_net"], annual)
        self.assertAlmostEqual(self.m["calmar_net"], annual / 0.10)

    def test_sortino_is_computed_on_net_and_gross(self):
        expected = 8.4 / (0.1 * math.sqrt(252))
        self.assertAlmostEqual(self.m["sortino_net"], expected)
        self.assertAlmostEqual(self.m["sortino_gross"], expected)

    def test_gross_and_net_are_tracked_separately(self):
        gross = [1000.0, 1200.0, 1200.0, 1200.0]
        m = compute_performance(
            _trades(), _daily([1000.0, 1100.0, 990.0, 1089.0], gross), PerformanceConfig()
        ).metrics
        self.assertAlmostEqual(m["total_return_gross"], 0.0)
        self.assertEqual(m["max_drawdown_gross"], 0.0)

    def test_cost_and_position_stats(self):
        self.assertEqual(self.m["cumulative_transaction_costs"], 4.0)
        self.assertEqual(self.m["cumulative_borrow_costs"], 1.0)
        self.assertAlmostEqual(self.m["cost_drag_bps"], 5.0 / 1000.0 * 10_000)
        self.assertAlmostEqual(self.m["avg_concurrent_positions"], np.mean([1, 2, 1]))
        self.assertAlmostEqual(self.m["avg_daily_borrow_cost"], np.mean([0.25, 0.25, 0.5]))

    def test_trade_and_equity_metrics_are_merged(self):
        self.assertEqual(self.m["n_trades"], 4)
        self.assertIn("sharpe_net", self.m)

    def test_per_group_breakdown(self):
        cfg = PerformanceConfig(enabled=True, per_group_breakdown=True)
        res = compute_performance(_trades(), _daily([1000.0, 1100.0, 990.0, 1089.0]), cfg)
        self.assertEqual(sorted(res.group_metrics), ["G1", "G2"])
        self.assertEqual(res.group_metrics["G1"]["n_trades"], 2)
        self.assertEqual(res.group_metrics["G1"]["total_net_pnl"], 50.0)
        self.assertEqual(res.group_metrics["G2"]["total_net_pnl"], 200.0)
        self.assertEqual(self.perf.group_metrics, {})  # off by default

    def test_empty_equity_curve_yields_nan_equity_metrics(self):
        res = compute_performance(_trades(), pd.DataFrame(), PerformanceConfig())
        self.assertEqual(res.metrics["n_trading_days"], 0)
        self.assertTrue(math.isnan(res.metrics["sharpe_net"]))
        self.assertEqual(res.metrics["n_trades"], 4)
        self.assertTrue(res.returns_net.empty)


if __name__ == "__main__":
    unittest.main()
