"""
Track D: PortfolioMeanReversionTrader.default_abs_target_group_exposure lost
its class default (was 1.0, sitting beside PortfolioMeanReversionConfig as a
second, ungated numeric default with no way to override it at the one
construction site, simulator_factory.py). Dormant on B's baseline (which
never constructs a PortfolioMeanReversionConfig/-Trader), so there is no
B_baseline.txt signal for this one — this test is the verification instead.
"""

import unittest
from typing import Any

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


def _live_pos(**kw):
    fields: dict[str, Any] = {
        "trade_id": "t1",
        "candidate_id": "c1",
        "group_id": "G1",
        "spread_id": "S_c1",
        "direction": 1.0,
        "pair_notional": 1000.0,
        "gross_value": 1000.0,
        "entry_date": pd.Timestamp("2024-01-01"),
        "entry_asof_date": pd.Timestamp("2023-12-29"),
        "days_open": 1,
    }
    fields.update(kw)
    return LiveCandidatePosition(**fields)


class TestPortfolioTraderStopsAndGates(unittest.TestCase):
    """Strategy rules beyond the entry/exit smoke test: gates and the three stops."""

    def _trader(self, **cfg):
        return PortfolioMeanReversionTrader(
            config=PortfolioMeanReversionConfig(entry_z=2.0, exit_z=0.0, **cfg),
            default_abs_target_group_exposure=1.0,
        )

    def _close_reason(self, trader, pos, z=-1.0, diag=None, capital=None):
        actions = trader.generate_actions(
            _snapshot({"c1": z}),
            {"c1": pos},
            {"c1": diag} if diag else None,
            capital,
        )
        self.assertLessEqual(len(actions), 1)
        return actions[0].reason if actions else None

    def test_time_stop_uses_multiple_of_entry_half_life(self):
        trader = self._trader(time_stop_half_life_multiplier=2.0)
        old = _live_pos(entry_half_life=5.0, days_open=10)  # 10 >= 2 * 5
        young = _live_pos(entry_half_life=5.0, days_open=9)
        self.assertEqual(self._close_reason(trader, old), "time_stop")
        self.assertIsNone(self._close_reason(trader, young))
        self.assertIsNone(self._close_reason(trader, _live_pos(days_open=99)))  # no half-life

    def test_mr_deterioration_stop(self):
        from tests._synthetic import make_state

        trader = self._trader(mr_deterioration_threshold=0.5)
        pos = _live_pos(entry_mr_score=4.0)
        self.assertEqual(
            self._close_reason(trader, pos, diag=make_state("c1", mr_score=1.9)),
            "mr_deterioration_stop",  # 1.9 < 0.5 * 4.0
        )
        self.assertIsNone(self._close_reason(trader, pos, diag=make_state("c1", mr_score=2.0)))
        self.assertIsNone(self._close_reason(trader, pos, diag=make_state("c1", mr_score=None)))
        self.assertIsNone(
            self._close_reason(
                trader, _live_pos(entry_mr_score=0.0), diag=make_state("c1", mr_score=-1.0)
            )
        )

    def test_pnl_stop_needs_group_capital(self):
        trader = self._trader(pnl_stop_fraction=0.01)
        pos = _live_pos(unrealized_pnl=-150.0)
        self.assertEqual(
            self._close_reason(trader, pos, capital={"G1": 10_000.0}), "pnl_stop"
        )  # -150 < -(0.01 * 10_000)
        self.assertIsNone(self._close_reason(trader, pos, capital={"G1": 20_000.0}))
        self.assertIsNone(self._close_reason(trader, pos))  # no capital info -> no pnl stop

    def test_stops_take_precedence_over_zero_cross(self):
        trader = self._trader(time_stop_half_life_multiplier=1.0)
        pos = _live_pos(entry_half_life=2.0, days_open=3)
        self.assertEqual(self._close_reason(trader, pos, z=0.5), "time_stop")

    def test_short_exit_and_no_exit_before_the_cross(self):
        trader = self._trader()
        short = _live_pos(direction=-1.0)
        self.assertEqual(self._close_reason(trader, short, z=0.0), "exit_short_zero_cross")
        self.assertIsNone(self._close_reason(trader, short, z=0.2))
        self.assertEqual(self._close_reason(trader, _live_pos(), z=0.0), "exit_long_zero_cross")

    def test_entry_flags_and_gates(self):
        trader = self._trader(allow_short=False)
        got = trader.generate_actions(_snapshot({"a": -3.0, "b": 3.0}), {})
        self.assertEqual([a.candidate_id for a in got], ["a"])
        snap = _snapshot({"a": -3.0})
        snap.candidate_states["is_active"] = False
        self.assertEqual(self._trader().generate_actions(snap, {}), [])
        snap.candidate_states["is_active"] = True
        snap.candidate_states["is_signal_ready"] = False
        self.assertEqual(self._trader().generate_actions(snap, {}), [])
        snap.candidate_states = snap.candidate_states.iloc[0:0]
        self.assertEqual(self._trader().generate_actions(snap, {}), [])


if __name__ == "__main__":
    unittest.main()
