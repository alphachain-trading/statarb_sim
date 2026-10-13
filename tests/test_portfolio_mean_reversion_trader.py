"""
Track D: PortfolioMeanReversionTrader.default_abs_target_group_exposure lost
its class default (was 1.0, sitting beside PortfolioMeanReversionConfig as a
second, ungated numeric default with no way to override it at the one
construction site, simulator_factory.py). Dormant on B's baseline (which
never constructs a PortfolioMeanReversionConfig/-Trader), so there is no
B_baseline.txt signal for this one — this test is the verification instead.
"""

import unittest

import pandas as pd

from src.simulator.actions import CloseCandidateAction, OpenCandidateAction
from src.simulator.config import PortfolioMeanReversionConfig
from src.simulator.traders.portfolio_mean_reversion import PortfolioMeanReversionTrader
from src.simulator.types import CandidateMarketSnapshot, LiveCandidatePosition


class TestDefaultAbsTargetGroupExposureIsMandatory(unittest.TestCase):
    def test_omitting_it_raises(self):
        with self.assertRaises(TypeError):
            PortfolioMeanReversionTrader(  # type: ignore[call-arg]  # required argument deliberately omitted
                config=PortfolioMeanReversionConfig(entry_z=2.0, exit_z=0.0),
            )

    def test_stating_it_succeeds(self):
        trader = PortfolioMeanReversionTrader(
            config=PortfolioMeanReversionConfig(entry_z=2.0, exit_z=0.0),
            default_abs_target_group_exposure=1.0,
        )
        self.assertEqual(trader.default_abs_target_group_exposure, 1.0)


def _snapshot(z_by_candidate: dict[str, float]) -> CandidateMarketSnapshot:
    states = pd.DataFrame(
        {
            "spread_id": [f"S_{cid}" for cid in z_by_candidate],
            "group_id": "G1",
            "is_new_arrival": False,
            "is_active": True,
            "level": 0.0,
            "roll_mean": 0.0,
            "roll_std": 1.0,
            "z_score": list(z_by_candidate.values()),
            "is_signal_ready": True,
        },
        index=pd.Index(list(z_by_candidate), name="candidate_id"),
    )
    return CandidateMarketSnapshot(
        date=pd.Timestamp("2024-01-02"),
        current_prices=pd.Series(dtype=float),
        candidate_states=states,
    )


class TestGenerateActionsSmoke(unittest.TestCase):
    """Small fixed data: the trader runs through to opens and a close."""

    def setUp(self):
        self.trader = PortfolioMeanReversionTrader(
            config=PortfolioMeanReversionConfig(entry_z=2.0, exit_z=0.0),
            default_abs_target_group_exposure=1.0,
        )

    def test_opens_long_short_and_skips_inside_band(self):
        snap = _snapshot({"c_long": -2.5, "c_short": 2.5, "c_flat": 0.5})
        actions = self.trader.generate_actions(snap, {})
        by_id = {a.candidate_id: a for a in actions}
        self.assertEqual(set(by_id), {"c_long", "c_short"})
        long_a, short_a = by_id["c_long"], by_id["c_short"]
        assert isinstance(long_a, OpenCandidateAction)
        assert isinstance(short_a, OpenCandidateAction)
        self.assertEqual(long_a.direction, 1.0)
        self.assertEqual(long_a.reason, "entry_long")
        self.assertEqual(short_a.direction, -1.0)
        self.assertEqual(short_a.reason, "entry_short")
        self.assertIsNone(long_a.pair_notional)  # SizingEngine populates it

    def test_closes_long_on_zero_cross(self):
        pos = LiveCandidatePosition(
            trade_id="t1",
            candidate_id="c1",
            group_id="G1",
            spread_id="S_c1",
            direction=1.0,
            pair_notional=1000.0,
            gross_value=1000.0,
            entry_date=pd.Timestamp("2024-01-01"),
            entry_asof_date=pd.Timestamp("2023-12-29"),
            days_open=1,
        )
        actions = self.trader.generate_actions(_snapshot({"c1": 0.3}), {"c1": pos})
        self.assertEqual(len(actions), 1)
        self.assertIsInstance(actions[0], CloseCandidateAction)
        self.assertEqual(actions[0].reason, "exit_long_zero_cross")


if __name__ == "__main__":
    unittest.main()
