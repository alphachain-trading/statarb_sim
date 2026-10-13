"""
Unit tests for PairSpreadMeanReversionTrader: entry thresholds and gating, z-cross
exits, exit rules and time stops (including their eval-based string forms).
"""

import unittest
from typing import Any

import pandas as pd

from src.simulator.actions import CloseCandidateAction, OpenCandidateAction
from src.simulator.config import PairSpreadTraderConfig
from src.simulator.traders.pair_spread_mean_reversion import PairSpreadMeanReversionTrader
from src.simulator.types import CandidateMarketSnapshot, LiveCandidatePosition
from tests._synthetic import make_state


def _snapshot(rows: list[dict[str, Any]], with_rkey: bool = True) -> CandidateMarketSnapshot:
    base = {
        "group_id": "G1",
        "is_new_arrival": False,
        "is_active": True,
        "level": 0.0,
        "roll_mean": 0.0,
        "roll_std": 1.0,
        "is_signal_ready": True,
        "residual_key": "exp_hl5_mh20",
    }
    records = []
    for r in rows:
        rec = {**base, **r}
        rec.setdefault("spread_id", f"S_{rec['candidate_id']}")
        if not with_rkey:
            rec.pop("residual_key")
        records.append(rec)
    df = pd.DataFrame(records).set_index("candidate_id")
    return CandidateMarketSnapshot(
        date=pd.Timestamp("2024-01-02"), current_prices=pd.Series(dtype=float), candidate_states=df
    )


def _live(cid: str = "c1", **kw: Any) -> LiveCandidatePosition:
    fields: dict[str, Any] = {
        "trade_id": f"t_{cid}",
        "candidate_id": cid,
        "group_id": "G1",
        "spread_id": f"S_{cid}",
        "direction": 1.0,
        "pair_notional": 10_000.0,
        "gross_value": 10_000.0,
        "entry_date": pd.Timestamp("2024-01-01"),
        "entry_asof_date": pd.Timestamp("2023-12-29"),
        "days_open": 3,
        "residual_key": "exp_hl5_mh20",
    }
    fields.update(kw)
    return LiveCandidatePosition(**fields)


def _trader(**kw: Any) -> PairSpreadMeanReversionTrader:
    cfg: dict[str, Any] = {"entry_z": 2.0, "exit_z": 0.0}
    cfg.update(kw)
    return PairSpreadMeanReversionTrader(config=PairSpreadTraderConfig(**cfg))


def _by_id(actions) -> dict[str, Any]:
    return {a.candidate_id: a for a in actions}


class TestEntries(unittest.TestCase):
    def test_thresholds_are_inclusive_and_direction_follows_sign(self):
        snap = _snapshot(
            [
                {"candidate_id": "long_edge", "z_score": -2.0},
                {"candidate_id": "short_edge", "z_score": 2.0},
                {"candidate_id": "inside", "z_score": 1.99},
                {"candidate_id": "inside_neg", "z_score": -1.99},
            ]
        )
        got = _by_id(_trader().generate_actions(snap, {}))
        self.assertEqual(set(got), {"long_edge", "short_edge"})
        self.assertEqual((got["long_edge"].direction, got["long_edge"].reason), (1.0, "entry_long"))
        self.assertEqual(
            (got["short_edge"].direction, got["short_edge"].reason), (-1.0, "entry_short")
        )
        self.assertIsNone(got["long_edge"].pair_notional)  # SizingEngine's job
        self.assertEqual(got["long_edge"].z_score, -2.0)
        self.assertEqual(got["long_edge"].residual_key, "exp_hl5_mh20")

    def test_allow_flags(self):
        snap = _snapshot(
            [{"candidate_id": "l", "z_score": -3.0}, {"candidate_id": "s", "z_score": 3.0}]
        )
        self.assertEqual(set(_by_id(_trader(allow_short=False).generate_actions(snap, {}))), {"l"})
        self.assertEqual(set(_by_id(_trader(allow_long=False).generate_actions(snap, {}))), {"s"})
        self.assertEqual(
            _trader(allow_long=False, allow_short=False).generate_actions(snap, {}), []
        )

    def test_inactive_or_not_ready_candidates_are_skipped(self):
        snap = _snapshot(
            [
                {"candidate_id": "off", "z_score": -3.0, "is_active": False},
                {"candidate_id": "cold", "z_score": -3.0, "is_signal_ready": False},
                {"candidate_id": "ok", "z_score": -3.0},
            ]
        )
        self.assertEqual(set(_by_id(_trader().generate_actions(snap, {}))), {"ok"})

    def test_nan_z_never_triggers_an_entry(self):
        snap = _snapshot([{"candidate_id": "n", "z_score": float("nan")}])
        self.assertEqual(_trader().generate_actions(snap, {}), [])

    def test_held_candidate_and_occupied_spread_key_are_not_reopened(self):
        snap = _snapshot(
            [
                {"candidate_id": "held", "z_score": -3.0, "spread_id": "A|B"},
                # different candidate, same (spread_id, residual_key) as the live position
                {"candidate_id": "twin", "z_score": -3.0, "spread_id": "A|B"},
                # same spread but another residual key is a different slot
                {
                    "candidate_id": "other_rkey",
                    "z_score": -3.0,
                    "spread_id": "A|B",
                    "residual_key": "exp_hl9_mh20",
                },
                {"candidate_id": "free", "z_score": -3.0, "spread_id": "C|D"},
            ]
        )
        live = {"held": _live("held", spread_id="A|B", residual_key="exp_hl5_mh20")}
        got = _by_id(_trader().generate_actions(snap, live))
        self.assertEqual(set(got), {"other_rkey", "free"})

    def test_residual_key_is_parsed_from_candidate_id_when_column_missing(self):
        snap = _snapshot([{"candidate_id": "abc__exp_hl5_mh20", "z_score": -3.0}], with_rkey=False)
        (action,) = _trader().generate_actions(snap, {})
        assert isinstance(action, OpenCandidateAction)
        self.assertEqual(action.residual_key, "exp_hl5_mh20")

    def test_z_components_come_from_live_diagnostics(self):
        snap = _snapshot([{"candidate_id": "c1", "z_score": -3.0}])
        comps = make_state("c1").z_components
        diag = make_state("c1")
        object.__setattr__(diag, "z_components", ("marker",))
        (action,) = _trader().generate_actions(snap, {}, {"c1": diag})
        (plain,) = _trader().generate_actions(snap, {}, {})
        assert isinstance(action, OpenCandidateAction) and isinstance(plain, OpenCandidateAction)
        self.assertEqual(action.z_components, ("marker",))
        self.assertEqual(plain.z_components, comps)

    def test_empty_snapshot(self):
        empty = _snapshot([{"candidate_id": "x", "z_score": 0.0}])
        empty.candidate_states.drop(index="x", inplace=True)
        self.assertEqual(_trader().generate_actions(empty, {}), [])


class TestZCrossExits(unittest.TestCase):
    def _closes(self, z: float, direction: float, **cfg: Any) -> list[Any]:
        snap = _snapshot([{"candidate_id": "c1", "z_score": z}])
        pos = _live("c1", direction=direction)
        return [
            a
            for a in _trader(**cfg).generate_actions(snap, {"c1": pos})
            if isinstance(a, CloseCandidateAction)
        ]

    def test_long_exits_when_z_reaches_minus_exit_z(self):
        self.assertEqual(len(self._closes(0.0, +1.0)), 1)  # exit_z=0 -> z >= 0
        self.assertEqual(self._closes(-0.01, +1.0), [])
        self.assertEqual(len(self._closes(-0.5, +1.0, exit_z=0.5)), 1)  # z >= -0.5
        self.assertEqual(self._closes(-0.6, +1.0, exit_z=0.5), [])

    def test_short_exits_when_z_reaches_exit_z(self):
        self.assertEqual(len(self._closes(0.0, -1.0)), 1)
        self.assertEqual(self._closes(0.01, -1.0), [])
        self.assertEqual(len(self._closes(0.5, -1.0, exit_z=0.5)), 1)
        self.assertEqual(self._closes(0.6, -1.0, exit_z=0.5), [])

    def test_close_carries_reason_and_current_z(self):
        (c,) = self._closes(1.0, +1.0)
        self.assertEqual((c.reason, c.z_score, c.candidate_id), ("z_cross", 1.0, "c1"))

    def test_not_ready_or_missing_candidates_are_not_closed_by_z(self):
        snap = _snapshot([{"candidate_id": "c1", "z_score": 5.0, "is_signal_ready": False}])
        pos = {"c1": _live("c1"), "ghost": _live("ghost")}
        self.assertEqual(_trader().generate_actions(snap, pos), [])

    def test_closes_come_before_opens(self):
        snap = _snapshot(
            [{"candidate_id": "c1", "z_score": 1.0}, {"candidate_id": "new", "z_score": -3.0}]
        )
        actions = _trader().generate_actions(snap, {"c1": _live("c1")})
        self.assertEqual([type(a) for a in actions], [CloseCandidateAction, OpenCandidateAction])


class TestExitRulesAndTimeStops(unittest.TestCase):
    def _act(self, pos: LiveCandidatePosition, z: float = -3.0, **cfg: Any) -> list[Any]:
        snap = _snapshot([{"candidate_id": pos.candidate_id, "z_score": z}])
        return [
            a
            for a in _trader(**cfg).generate_actions(snap, {pos.candidate_id: pos})
            if isinstance(a, CloseCandidateAction)
        ]

    def test_config_rejects_both_exit_rule_and_max_holding_days(self):
        with self.assertRaises(ValueError):
            _trader(exit_rule="days_open > 1", max_holding_days=5)

    def test_exit_rule_fires_with_reason_and_z_even_when_z_far_from_exit(self):
        (c,) = self._act(_live(days_open=6), exit_rule="days_open > 5")
        self.assertEqual((c.reason, c.z_score), ("exit_rule", -3.0))
        self.assertEqual(self._act(_live(days_open=5), z=-3.0, exit_rule="days_open > 5"), [])

    def test_exit_rule_uses_pnl_percent_of_notional(self):
        pos = _live(unrealized_pnl=-300.0, min_unrealized_pnl=-450.0)  # -3% now, -4.5% worst
        self.assertEqual(len(self._act(pos, exit_rule="pnl_pct < -2.5")), 1)
        self.assertEqual(len(self._act(pos, exit_rule="min_pnl_pct <= -4.5")), 1)
        self.assertEqual(self._act(pos, exit_rule="pnl_pct < -3.5"), [])

    def test_and_or_keywords_are_supported(self):
        pos = _live(days_open=6, unrealized_pnl=-300.0)
        self.assertEqual(len(self._act(pos, exit_rule="days_open > 5 AND pnl_pct < -2")), 1)
        self.assertEqual(self._act(pos, exit_rule="days_open > 50 AND pnl_pct < -2"), [])
        self.assertEqual(len(self._act(pos, exit_rule="days_open > 50 OR pnl_pct < -2")), 1)

    def test_rhl_is_taken_from_the_residual_key(self):
        pos = _live(days_open=11, residual_key="exp_hl5_mh20")
        self.assertEqual(len(self._act(pos, exit_rule="days_open > 2*rhl")), 1)  # 11 > 10
        self.assertEqual(self._act(_live(days_open=10), exit_rule="days_open > 2*rhl"), [])

    def test_exit_rule_per_residual_key_dict(self):
        pos = _live(days_open=6, residual_key="exp_hl5_mh20")
        self.assertEqual(len(self._act(pos, exit_rule={"exp_hl5_mh20": "days_open > 5"})), 1)
        # a residual key without an entry has no rule -> falls back to z-cross only
        self.assertEqual(self._act(pos, z=-3.0, exit_rule={"other": "days_open > 5"}), [])

    def test_broken_rules_never_fire_silently(self):
        # Pins current behaviour: a rule that raises (typo, unknown name, forbidden call)
        # is swallowed and evaluates to False — the position then only exits via z-cross.
        pos = _live(days_open=99)
        for rule in ("days_open >", "unknown_name > 1", "__import__('os').getcwd()"):
            with self.subTest(rule=rule):
                self.assertEqual(self._act(pos, exit_rule=rule), [])

    def test_exit_rule_ignored_for_non_positive_notional(self):
        # LiveCandidatePosition validates notional > 0 at construction; the trader's own
        # guard is defensive, so break the invariant after construction.
        pos = _live(days_open=99)
        pos.pair_notional = 0.0
        self.assertFalse(_trader()._eval_exit_rule("days_open > 1", pos))

    def test_max_holding_days_int(self):
        (c,) = self._act(_live(days_open=7), max_holding_days=7)
        self.assertEqual(c.reason, "time_stop_7d")
        self.assertEqual(self._act(_live(days_open=6), max_holding_days=7), [])

    def test_max_holding_days_dict_and_expression(self):
        pos = _live(days_open=10, residual_key="exp_hl5_mh20")
        (c,) = self._act(pos, max_holding_days="2*rhl")  # round(2*5) = 10
        self.assertEqual(c.reason, "time_stop_10d")
        self.assertEqual(self._act(_live(days_open=9), max_holding_days="2*rhl"), [])
        self.assertEqual(len(self._act(pos, max_holding_days={"exp_hl5_mh20": 10})), 1)
        self.assertEqual(self._act(pos, max_holding_days={"other": 10}), [])

    def test_max_holding_expression_without_rhl_or_with_error_means_no_time_stop(self):
        # Pins current behaviour: unparsable expression / residual key without "hl<N>"
        # silently disables the time stop.
        self.assertEqual(
            self._act(_live(days_open=99, residual_key="rol_lb21"), max_holding_days="2*rhl"), []
        )
        self.assertEqual(self._act(_live(days_open=99), max_holding_days="2*"), [])

    def test_time_stop_takes_precedence_over_z_cross_without_duplicates(self):
        pos = _live(days_open=7, direction=+1.0)
        actions = self._act(pos, z=1.0, max_holding_days=7)  # z also satisfies the z-cross
        self.assertEqual([a.reason for a in actions], ["time_stop_7d"])

    def test_exit_rule_close_for_position_missing_from_snapshot_has_no_z(self):
        pos = _live("ghost", days_open=9)
        snap = _snapshot([{"candidate_id": "other", "z_score": 0.0}])
        (c,) = [
            a
            for a in _trader(exit_rule="days_open > 1").generate_actions(snap, {"ghost": pos})
            if isinstance(a, CloseCandidateAction)
        ]
        self.assertIsNone(c.z_score)

    def test_extract_rhl(self):
        f = PairSpreadMeanReversionTrader._extract_rhl
        self.assertEqual((f("exp_hl504_mh1008"), f("rol_lb21"), f("hlx")), (504, None, None))


if __name__ == "__main__":
    unittest.main()
