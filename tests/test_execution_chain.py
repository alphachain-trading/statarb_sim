"""
Hand-computed tests for the execution chain: ExecutionEngine (rounding, commission),
PositionTranslator, RiskManager and the Simulator's open -> mark-to-market -> borrow
-> close accounting. Every expected number is derived in a comment.
"""

import unittest
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from src.simulator.actions import CloseCandidateAction, OpenCandidateAction
from src.simulator.config import ExecutionConfig, RiskManagerConfig
from src.simulator.execution import ExecutionEngine
from src.simulator.position_translator import PositionTranslator
from src.simulator.risk_manager import RiskManager
from src.simulator.simulator import Simulator
from src.simulator.types import LiveCandidatePosition, TickerDelta
from tests._synthetic import make_ref


def _exec_cfg(**overrides: Any) -> ExecutionConfig:
    fields: dict[str, Any] = {
        "min_abs_units": 0.5,
        "commission_per_share": 0.005,
        "commission_per_order": 0.0,
        "min_commission_per_order": 1.0,
        "max_commission_per_order": 9.79,
        "max_commission_pct_of_trade": 0.01,
        "short_borrow_rate_annual_bps": 30.0,
    }
    fields.update(overrides)
    return ExecutionConfig(**fields)


def _prices(**p: float) -> pd.Series:
    return pd.Series(p, dtype=float)


class TestShareRounding(unittest.TestCase):
    def _round(self, mode: str, units: float) -> float:
        engine = ExecutionEngine(_exec_cfg(share_rounding=mode))
        return engine._round_units(units)

    def test_nearest_uses_banker_rounding_at_halves(self):
        # Python round(): 2.5 -> 2, 3.5 -> 4, -2.5 -> -2. Pins current behaviour.
        got = [self._round("nearest", u) for u in (2.5, 3.5, -2.5, 2.4, 2.6, -2.6)]
        self.assertEqual(got, [2, 4, -2, 2, 3, -3])

    def test_floor_and_ceil_round_toward_and_away_from_zero(self):
        self.assertEqual([self._round("floor", u) for u in (2.9, -2.9)], [2, -2])
        self.assertEqual([self._round("ceil", u) for u in (2.1, -2.1)], [3, -3])

    def test_less_than_one_share_is_no_trade(self):
        self.assertEqual(self._round("nearest", 0.4), 0)
        self.assertEqual(self._round("nearest", -0.4), 0)
        self.assertEqual(self._round("ceil", 0.4), 1)  # ceil(0.4)=1 share

    def test_fractional_mode_respects_min_abs_units(self):
        engine = ExecutionEngine(_exec_cfg(allow_fractional_shares=True, min_abs_units=0.5))
        self.assertEqual(engine._round_units(0.3), 0.0)
        self.assertEqual(engine._round_units(-0.3), 0.0)
        self.assertEqual(engine._round_units(0.75), 0.75)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self._round("banana", 1.5)


class TestCommission(unittest.TestCase):
    def setUp(self):
        self.engine = ExecutionEngine(_exec_cfg())

    def _c(self, shares: float, notional: float) -> float:
        return self.engine._compute_commission(n_shares=shares, traded_notional=notional)

    def test_minimum_applies_to_small_orders(self):
        # raw = 100 * 0.005 = 0.5 -> clamped up to the 1.0 minimum
        self.assertEqual(self._c(100, 5000.0), 1.0)

    def test_regular_order_is_per_share(self):
        # raw = 1000 * 0.005 = 5.0 ; cap = min(9.79, 1% * 50_000 = 500) = 9.79
        self.assertEqual(self._c(1000, 50_000.0), 5.0)

    def test_max_per_order_caps_large_orders(self):
        # raw = 5000 * 0.005 = 25 -> capped at 9.79
        self.assertEqual(self._c(5000, 500_000.0), 9.79)

    def test_pct_of_trade_cap_wins_for_penny_stocks(self):
        # 10 shares at $0.50 = $5 notional ; pct cap = 0.05 < min 1.0 -> cap wins
        self.assertAlmostEqual(self._c(10, 5.0), 0.05)

    def test_per_order_component_is_added(self):
        engine = ExecutionEngine(_exec_cfg(commission_per_order=2.0))
        # raw = 2 + 1000*0.005 = 7.0
        self.assertEqual(engine._compute_commission(n_shares=1000, traded_notional=50_000.0), 7.0)


class TestExecuteDeltas(unittest.TestCase):
    def test_fill_fields_and_zero_fills_skipped(self):
        engine = ExecutionEngine(_exec_cfg())
        res = engine.execute_deltas(
            deltas=[
                TickerDelta("A", 100.4),
                TickerDelta("B", -49.6),
                TickerDelta("C", 0.3),  # rounds to 0 shares -> no fill
            ],
            current_prices=_prices(A=50.0, B=100.0, C=10.0),
        )
        self.assertEqual([f.ticker for f in res.fills], ["A", "B"])
        a, b = res.fills
        self.assertEqual((a.filled_delta_units, a.fill_price, a.traded_notional), (100, 50.0, 5000))
        self.assertEqual((b.filled_delta_units, b.traded_notional), (-50, 5000.0))
        self.assertEqual((a.requested_delta_units, b.requested_delta_units), (100.4, -49.6))
        self.assertEqual((a.commission, b.commission), (1.0, 1.0))

    def test_missing_price_raises(self):
        with self.assertRaises(ValueError):
            ExecutionEngine(_exec_cfg()).execute_deltas(
                deltas=[TickerDelta("Z", 10.0)], current_prices=_prices(A=1.0)
            )


class TestPositionTranslator(unittest.TestCase):
    def setUp(self):
        self.tr = PositionTranslator()
        self.px = _prices(A=50.0, B=100.0)

    def _open(self, direction: float, notional: float | None) -> OpenCandidateAction:
        return OpenCandidateAction(
            candidate_id="c1",
            group_id="G1",
            spread_id="A|B",
            direction=direction,
            pair_notional=notional,
        )

    def _units(self, action: OpenCandidateAction, weights=(1.0, -1.0)) -> dict[str, float]:
        ref = make_ref(weights=weights)
        deltas = self.tr.translate_open(action=action, candidate_ref=ref, current_prices=self.px)
        return {d.ticker: d.delta_units for d in deltas}

    def test_equal_weights_split_notional_half_half(self):
        # gross_norm 2 -> each leg 0.5 * 10_000 = 5_000 ; A: +5000/50, B: -5000/100
        self.assertEqual(self._units(self._open(+1.0, 10_000.0)), {"A": 100.0, "B": -50.0})

    def test_short_direction_flips_every_leg(self):
        self.assertEqual(self._units(self._open(-1.0, 10_000.0)), {"A": -100.0, "B": 50.0})

    def test_unequal_weights_are_proportional_to_abs_weights(self):
        # weights (1, -3): gross 4 -> A gets 25%, B 75% of the notional
        got = self._units(self._open(+1.0, 10_000.0), weights=(1.0, -3.0))
        self.assertEqual(got, {"A": 2500.0 / 50.0, "B": -7500.0 / 100.0})

    def test_validation(self):
        ref = make_ref()
        with self.assertRaises(ValueError):  # unsized action
            self.tr.translate_open(
                action=self._open(1.0, None), candidate_ref=ref, current_prices=self.px
            )
        with self.assertRaises(ValueError):  # id mismatch
            self.tr.translate_open(
                action=self._open(1.0, 1.0), candidate_ref=make_ref("other"), current_prices=self.px
            )
        with self.assertRaises(ValueError):  # missing price
            self.tr.translate_open(
                action=self._open(1.0, 1.0), candidate_ref=ref, current_prices=_prices(A=1.0)
            )
        with self.assertRaises(ValueError):  # non-positive price
            self.tr.translate_open(
                action=self._open(1.0, 1.0),
                candidate_ref=ref,
                current_prices=_prices(A=1.0, B=0.0),
            )
        with self.assertRaises(ValueError):  # all-zero weights
            self.tr.translate_open(
                action=self._open(1.0, 1.0),
                candidate_ref=make_ref(weights=(0.0, 0.0)),
                current_prices=self.px,
            )

    def test_close_negates_held_units(self):
        pos = _live(direction=1.0, units={"A": 100, "B": -50}, entry={"A": 50.0, "B": 100.0})
        close = CloseCandidateAction(candidate_id="c1", group_id="G1", spread_id="A|B")
        deltas = self.tr.translate_close(action=close, live_position=pos)
        self.assertEqual({d.ticker: d.delta_units for d in deltas}, {"A": -100.0, "B": 50.0})
        with self.assertRaises(ValueError):
            self.tr.translate_close(
                action=CloseCandidateAction(candidate_id="x", group_id="G1", spread_id="A|B"),
                live_position=pos,
            )


def _live(
    *,
    direction: float,
    units: dict[str, int],
    entry: dict[str, float],
    candidate_id: str = "c1",
    spread_id: str = "A|B",
) -> LiveCandidatePosition:
    return LiveCandidatePosition(
        trade_id=f"20240102:{candidate_id}",
        candidate_id=candidate_id,
        group_id="G1",
        spread_id=spread_id,
        direction=direction,
        pair_notional=10_000.0,
        gross_value=sum(abs(u * entry[t]) for t, u in units.items()),
        entry_date=pd.Timestamp("2024-01-02"),
        entry_asof_date=pd.Timestamp("2024-01-01"),
        days_open=0,
        units_by_ticker=dict(units),
        entry_prices_by_ticker=dict(entry),
    )


def _sized(cid: str, spread: str, direction: float, notional: float, z: float | None):
    action = OpenCandidateAction(
        candidate_id=cid, group_id="G1", spread_id=spread, direction=direction, z_score=z
    )
    return (action, notional, {}, 1.0)


class TestRiskManager(unittest.TestCase):
    PX = _prices(A=50.0, B=100.0, C=20.0, D=10.0)

    def _rm(self, **kw: Any) -> RiskManager:
        cfg = RiskManagerConfig(
            max_gross_exposure=kw.get("gross", 10.0),
            max_ticker_exposure_pct=kw.get("ticker", 1.0),
            max_concurrent_positions=kw.get("max_pos"),
        )
        return RiskManager(config=cfg, total_capital=kw.get("capital", 100_000.0))

    def _ids(self, approved) -> list[str]:
        return [a.candidate_id for a, *_ in approved]

    def test_ranked_by_abs_z_and_gross_cap_applied_in_that_order(self):
        # capital 10_000, gross cap 1.0 -> 10_000 ; each trade 6_000 -> only the strongest fits
        sized = [
            _sized("weak", "A|B", 1.0, 6000.0, 1.5),
            _sized("strong", "C|D", 1.0, 6000.0, -3.0),
        ]
        got = self._rm(gross=1.0, capital=10_000.0).approve(sized, {}, self.PX)
        self.assertEqual(self._ids(got), ["strong"])

    def test_missing_z_ranks_last(self):
        sized = [_sized("noz", "A|B", 1.0, 6000.0, None), _sized("z", "C|D", 1.0, 6000.0, 0.1)]
        got = self._rm(gross=1.0, capital=10_000.0).approve(sized, {}, self.PX)
        self.assertEqual(self._ids(got), ["z"])

    def test_existing_gross_counts_against_the_cap(self):
        live = {"x": _live(direction=1.0, units={"A": 100, "B": -50}, entry={"A": 50, "B": 100})}
        # live gross at current prices = 100*50 + 50*100 = 10_000 ; cap 12_000 -> 2_000 room
        rm = self._rm(gross=1.2, capital=10_000.0)
        self.assertEqual(rm.approve([_sized("n", "C|D", 1.0, 3000.0, 2.0)], live, self.PX), [])
        self.assertEqual(
            self._ids(rm.approve([_sized("n", "C|D", 1.0, 1900.0, 2.0)], live, self.PX)), ["n"]
        )

    def test_position_cap_stops_after_limit(self):
        sized = [_sized(f"c{i}", "A|B", 1.0, 100.0, 3.0 - i) for i in range(3)]
        got = self._rm(max_pos=2).approve(sized, {}, self.PX)
        self.assertEqual(self._ids(got), ["c0", "c1"])

    def test_ticker_cap_uses_net_exposure_across_approved_trades(self):
        # capital 10_000, ticker cap 20% = 2_000 ; each pair 3_000 -> 1_500 per leg.
        # Same direction on A|B twice: A net 1_500 then 3_000 > 2_000 -> second rejected.
        sized = [_sized("a1", "A|B", 1.0, 3000.0, 3.0), _sized("a2", "A|B", 1.0, 3000.0, 2.0)]
        got = self._rm(ticker=0.2, capital=10_000.0).approve(sized, {}, self.PX)
        self.assertEqual(self._ids(got), ["a1"])
        # opposite direction nets out and is accepted
        sized = [_sized("a1", "A|B", 1.0, 3000.0, 3.0), _sized("a2", "A|B", -1.0, 3000.0, 2.0)]
        got = self._rm(ticker=0.2, capital=10_000.0).approve(sized, {}, self.PX)
        self.assertEqual(self._ids(got), ["a1", "a2"])

    def test_degenerate_inputs(self):
        self.assertEqual(self._rm().approve([], {}, self.PX), [])
        self.assertEqual(
            self._rm(capital=0.0).approve([_sized("a", "A|B", 1.0, 1.0, 1.0)], {}, self.PX), []
        )

    @unittest.expectedFailure
    def test_ticker_check_should_use_the_actual_leg_weights(self):
        # KNOWN GAP (reported, not fixed): the ticker-exposure check assumes an equal
        # split (+-pair_notional/n_legs, alternating signs) while PositionTranslator splits
        # by |weight|. For weights (1, -3) the real B leg is 75% of the notional (7_500),
        # which would breach a 6_000 per-ticker cap, but the risk check sees only 5_000.
        sized = [_sized("c", "A|B", 1.0, 10_000.0, 2.0)]
        rm = self._rm(ticker=0.06, capital=100_000.0)  # cap 6_000 per ticker
        self.assertEqual(rm.approve(sized, {}, self.PX), [])


class _FakeSignalGenerator:
    def compute_analytics_from_weights(self, **kwargs: Any) -> Any:
        return SimpleNamespace(mr_score=1.5, half_life=7.0, adf_pvalue=0.01)


def _simulator(**exec_overrides: Any) -> Simulator:
    exec_cfg = _exec_cfg(**exec_overrides)
    sim = object.__new__(Simulator)
    sim.config = cast("Any", SimpleNamespace(execution=exec_cfg))
    sim.position_translator = PositionTranslator()
    sim.execution_engine = ExecutionEngine(exec_cfg)
    sim.signal_generator = cast("Any", _FakeSignalGenerator())
    return sim


class TestSimulatorAccounting(unittest.TestCase):
    """Short spread A|B, 10_000 notional: open 01-02, mark 01-03, close 01-04."""

    def test_short_spread_round_trip_pnl_costs_and_borrow(self):
        sim = _simulator()
        live: dict[str, LiveCandidatePosition] = {}
        closed: list = []
        log: list = []
        ref = make_ref()
        open_action = OpenCandidateAction(
            candidate_id="c1",
            group_id="G1",
            spread_id="A|B",
            direction=-1.0,
            pair_notional=10_000.0,
            z_score=2.1,
        )
        pnl, cost = sim._apply_action(
            date=pd.Timestamp("2024-01-02"),
            action=open_action,
            tracked_ref_by_id={"c1": ref},
            analytics_by_id={},
            current_prices=_prices(A=50.0, B=100.0),
            live_positions_by_candidate_id=live,
            closed_trades=closed,
            ticker_trade_log=log,
        )
        # short spread: A -5000/50 = -100 units, B +5000/100 = +50 units.
        # commission per fill: raw 0.5 / 0.25 -> min 1.0 each -> 2.0 total
        pos = live["c1"]
        self.assertEqual((pnl, cost), (0.0, 2.0))
        self.assertEqual(pos.units_by_ticker, {"A": -100, "B": 50})
        self.assertEqual(pos.entry_prices_by_ticker, {"A": 50.0, "B": 100.0})
        self.assertEqual(pos.gross_value, 10_000.0)
        # realised weights = signed notional / gross: A -5000/10000, B +5000/10000
        self.assertEqual(pos.realized_weights_by_ticker, {"A": -0.5, "B": 0.5})
        self.assertEqual((pos.entry_mr_score, pos.entry_half_life), (1.5, 7.0))
        self.assertEqual(pos.trade_id, "20240102:c1")
        self.assertEqual(len(log), 2)

        # mark to market on 01-03: A=55, B=96
        sim._mark_to_market_live_positions(
            live_positions_by_candidate_id=live, current_prices=_prices(A=55.0, B=96.0)
        )
        # unrealised: -100*(55-50) + 50*(96-100) = -500 - 200 = -700 ; gross 5500+4800
        self.assertEqual(pos.unrealized_pnl, -700.0)
        self.assertEqual(pos.gross_value, 10_300.0)
        self.assertEqual(pos.min_unrealized_pnl, -700.0)

        # one day of borrow on the short spread: 10_300 * 30bps / 252
        borrow = sim._accrue_borrow_costs(live_positions_by_candidate_id=live)
        expected_borrow = 10_300.0 * 0.003 / 252
        self.assertAlmostEqual(borrow, expected_borrow)
        self.assertAlmostEqual(pos.accumulated_borrow_cost, expected_borrow)

        # close on 01-04: A=45, B=104
        close = CloseCandidateAction(
            candidate_id="c1", group_id="G1", spread_id="A|B", z_score=-0.1, reason="z_cross"
        )
        pnl, cost = sim._apply_action(
            date=pd.Timestamp("2024-01-04"),
            action=close,
            tracked_ref_by_id={"c1": ref},
            analytics_by_id={},
            current_prices=_prices(A=45.0, B=104.0),
            live_positions_by_candidate_id=live,
            closed_trades=closed,
            ticker_trade_log=log,
        )
        # gross: -100*(45-50) + 50*(104-100) = 500 + 200 = 700 ; close commission 2.0
        self.assertEqual((pnl, cost), (700.0, 2.0))
        self.assertEqual(live, {})
        (trade,) = closed
        self.assertEqual(trade.realized_pnl_gross, 700.0)
        self.assertEqual(trade.transaction_costs, 4.0)  # 2.0 open + 2.0 close
        self.assertAlmostEqual(trade.borrow_costs, expected_borrow)
        self.assertAlmostEqual(trade.realized_pnl_net, 700.0 - 4.0 - expected_borrow)
        self.assertEqual((trade.entry_z_score, trade.exit_z_score), (2.1, -0.1))
        self.assertEqual(len(log), 4)

    def test_long_spread_accrues_no_borrow(self):
        sim = _simulator()
        pos = _live(direction=1.0, units={"A": 100, "B": -50}, entry={"A": 50.0, "B": 100.0})
        self.assertEqual(sim._accrue_borrow_costs(live_positions_by_candidate_id={"c1": pos}), 0.0)

    def test_zero_borrow_rate_accrues_nothing(self):
        sim = _simulator(short_borrow_rate_annual_bps=0.0)
        pos = _live(direction=-1.0, units={"A": -100, "B": 50}, entry={"A": 50.0, "B": 100.0})
        self.assertEqual(sim._accrue_borrow_costs(live_positions_by_candidate_id={"c1": pos}), 0.0)

    def test_open_without_fills_creates_no_position(self):
        sim = _simulator()
        live: dict[str, LiveCandidatePosition] = {}
        tiny = OpenCandidateAction(
            candidate_id="c1", group_id="G1", spread_id="A|B", direction=1.0, pair_notional=10.0
        )
        out = sim._apply_action(
            date=pd.Timestamp("2024-01-02"),
            action=tiny,
            tracked_ref_by_id={"c1": make_ref()},
            analytics_by_id={},
            current_prices=_prices(A=50.0, B=100.0),  # 5/50 and 5/100 units -> round to 0
            live_positions_by_candidate_id=live,
            closed_trades=[],
            ticker_trade_log=[],
        )
        self.assertEqual((out, live), ((0.0, 0.0), {}))

    def test_close_of_unknown_position_is_a_no_op(self):
        sim = _simulator()
        out = sim._apply_action(
            date=pd.Timestamp("2024-01-02"),
            action=CloseCandidateAction(candidate_id="ghost", group_id="G1", spread_id="A|B"),
            tracked_ref_by_id={},
            analytics_by_id={},
            current_prices=_prices(A=50.0, B=100.0),
            live_positions_by_candidate_id={},
            closed_trades=[],
            ticker_trade_log=[],
        )
        self.assertEqual(out, (0.0, 0.0))

    def test_open_without_tracked_ref_raises(self):
        sim = _simulator()
        with self.assertRaises(ValueError):
            sim._apply_action(
                date=pd.Timestamp("2024-01-02"),
                action=OpenCandidateAction(
                    candidate_id="c1",
                    group_id="G1",
                    spread_id="A|B",
                    direction=1.0,
                    pair_notional=1000.0,
                ),
                tracked_ref_by_id={},
                analytics_by_id={},
                current_prices=_prices(A=50.0, B=100.0),
                live_positions_by_candidate_id={},
                closed_trades=[],
                ticker_trade_log=[],
            )

    def test_weight_drift_between_entry_and_current_weights(self):
        # entry (-0.5, 0.5) vs effective (-0.4, 0.6): l2 = sqrt(0.01+0.01)
        l2, cos = Simulator._compute_weight_drift(
            entry_weights={"A": -0.5, "B": 0.5}, effective_weights={"A": -0.4, "B": 0.6}
        )
        assert l2 is not None and cos is not None
        self.assertAlmostEqual(l2, 0.02**0.5)
        self.assertGreater(cos, 0.0)
        self.assertEqual(
            Simulator._compute_weight_drift(entry_weights={}, effective_weights={}), (None, None)
        )

    def test_effective_weights_use_current_prices_and_skip_unpriced(self):
        got = Simulator._compute_effective_weights(
            units_by_ticker={"A": -100, "B": 50, "Z": 5}, current_prices=_prices(A=55.0, B=96.0)
        )
        gross = 5500.0 + 4800.0
        self.assertEqual(got, {"A": -5500.0 / gross, "B": 4800.0 / gross})

    def test_trade_id_includes_time_only_when_present(self):
        self.assertEqual(
            Simulator._make_trade_id("abcdefghij", pd.Timestamp("2024-01-02")), "20240102:abcdefgh"
        )
        self.assertEqual(
            Simulator._make_trade_id("abcdefghij", pd.Timestamp("2024-01-02 09:30")),
            "20240102T0930:abcdefgh",
        )


if __name__ == "__main__":
    unittest.main()
