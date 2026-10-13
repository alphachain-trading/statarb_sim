"""
Unit tests for SizingEngine and the interval-scoring maths it relies on.

Hand-computed examples (see comments). Where the implementation differs from its
docstring this file pins the *actual* behaviour and says so.
"""

import unittest

from src.simulator.actions import OpenCandidateAction
from src.simulator.config import (
    FeatureIntervalSpec,
    IntervalScoringConfig,
    SizingConfig,
    VolSizingConfig,
)
from src.simulator.sizing_engine import SizingEngine
from tests._synthetic import make_state

INF = float("inf")


def _open(cid: str) -> OpenCandidateAction:
    return OpenCandidateAction(
        candidate_id=cid, group_id="G1", spread_id=f"S_{cid}", direction=1.0, reason="entry_long"
    )


def _states(**roll_std_by_id: float | None):
    return {cid: make_state(cid, roll_std=rs) for cid, rs in roll_std_by_id.items()}


def _notionals(result) -> dict[str, float]:
    return {a.candidate_id: n for a, n, _scores, _mult in result}


class TestBaseAndVolNormalisation(unittest.TestCase):
    def test_empty_proposals(self):
        engine = SizingEngine(SizingConfig(base_pair_notional=1000.0))
        self.assertEqual(engine.size([]), [])

    def test_plain_base_notional(self):
        engine = SizingEngine(SizingConfig(base_pair_notional=1000.0))
        (action, notional, scores, mult), *rest = engine.size([_open("a")])
        self.assertEqual((notional, scores, mult, rest), (1000.0, {}, 1.0, []))
        self.assertEqual(action.candidate_id, "a")

    def test_vol_normalisation_scales_by_median_over_own_std(self):
        # stds a=0.5, b=1.0, c=2.0 -> median 1.0 -> ratios 2.0, 1.0, 0.5
        cfg = SizingConfig(1000.0, vol_normalize=VolSizingConfig(0.1, 10.0))
        got = SizingEngine(cfg).size(
            [_open("a"), _open("b"), _open("c")], _states(a=0.5, b=1.0, c=2.0)
        )
        self.assertEqual(_notionals(got), {"a": 2000.0, "b": 1000.0, "c": 500.0})

    def test_ratio_is_clamped_to_floor_and_cap(self):
        cfg = SizingConfig(1000.0, vol_normalize=VolSizingConfig(0.8, 1.5))
        got = SizingEngine(cfg).size([_open("a"), _open("c")], _states(a=0.5, b=1.0, c=2.0))
        # a: ratio 2.0 -> cap 1.5 ; c: ratio 0.5 -> floor 0.8
        self.assertEqual(_notionals(got), {"a": 1500.0, "c": 800.0})

    def test_median_is_taken_over_all_known_states_not_just_proposed(self):
        cfg = SizingConfig(1000.0, vol_normalize=VolSizingConfig(0.1, 10.0))
        # only "a" is proposed, but the median (1.0) comes from a, b, c
        got = SizingEngine(cfg).size([_open("a")], _states(a=0.5, b=1.0, c=2.0))
        self.assertEqual(_notionals(got), {"a": 2000.0})

    def test_unusable_stds_are_ignored_for_median_and_left_unscaled(self):
        cfg = SizingConfig(1000.0, vol_normalize=VolSizingConfig(0.1, 10.0))
        # None and 0 are excluded from the median (only b=2.0 remains -> median 2.0)
        states = _states(a=None, b=2.0, c=0.0)
        got = SizingEngine(cfg).size([_open("a"), _open("b"), _open("c")], states)
        self.assertEqual(_notionals(got), {"a": 1000.0, "b": 1000.0, "c": 1000.0})

    def test_without_analytics_vol_normalisation_is_skipped(self):
        cfg = SizingConfig(1000.0, vol_normalize=VolSizingConfig(0.1, 10.0))
        self.assertEqual(_notionals(SizingEngine(cfg).size([_open("a")], None)), {"a": 1000.0})
        self.assertEqual(_notionals(SizingEngine(cfg).size([_open("a")], {})), {"a": 1000.0})

    def test_unknown_candidate_is_left_unscaled(self):
        cfg = SizingConfig(1000.0, vol_normalize=VolSizingConfig(0.1, 10.0))
        got = SizingEngine(cfg).size([_open("zzz")], _states(a=1.0))
        self.assertEqual(_notionals(got), {"zzz": 1000.0})


def _spec(feature: str, limits, weights, missing: float = 1.0) -> FeatureIntervalSpec:
    return FeatureIntervalSpec(feature, limits, weights, missing_weight=missing)


def _scoring(specs, agg, floor=0.0, cap=10.0, mode="clamp") -> IntervalScoringConfig:
    return IntervalScoringConfig(
        feature_specs=tuple(specs),
        feature_weights=agg,
        floor_multiplier=floor,
        cap_multiplier=cap,
        floor_mode=mode,
    )


class TestIntervalLookup(unittest.TestCase):
    def setUp(self):
        self.spec = _spec("f", (-INF, 0.0, 1.0, INF), (0.5, 1.0, 2.0), missing=0.7)

    def test_intervals_are_half_open_lower_inclusive(self):
        self.assertEqual(self.spec.lookup(-5.0), 0.5)
        self.assertEqual(self.spec.lookup(0.0), 1.0)  # a boundary value belongs to the upper bin
        self.assertEqual(self.spec.lookup(0.999), 1.0)
        self.assertEqual(self.spec.lookup(1.0), 2.0)
        self.assertEqual(self.spec.lookup(1e9), 2.0)

    def test_missing_values_use_missing_weight(self):
        self.assertEqual(self.spec.lookup(None), 0.7)
        self.assertEqual(self.spec.lookup(float("nan")), 0.7)

    def test_values_outside_finite_limits_fall_into_the_edge_intervals(self):
        # Pins current behaviour: finite outer limits do not exclude anything.
        spec = _spec("g", (0.0, 1.0, 2.0), (3.0, 4.0))
        self.assertEqual(spec.lookup(-100.0), 3.0)
        self.assertEqual(spec.lookup(100.0), 4.0)


class TestIntervalMultiplier(unittest.TestCase):
    def _size(self, scoring, features, base=1000.0):
        cfg = SizingConfig(base, interval_scoring=scoring)
        state = make_state("a")
        state.features.update(features)
        return SizingEngine(cfg).size([_open("a")], {"a": state})

    def test_single_feature_multiplies_base(self):
        scoring = _scoring([_spec("f", (-INF, 0.0, INF), (0.5, 2.0))], {"f": 1.0})
        (_a, notional, scores, mult), *_ = self._size(scoring, {"f": 3.0})
        self.assertEqual((notional, scores, mult), (2000.0, {"f": 2.0}, 2.0))

    def test_weighted_mean_over_features(self):
        # interval weights a->2.0, b->1.0 ; aggregation weights a:1, b:3
        # raw = (2*1 + 1*3) / 4 = 1.25
        scoring = _scoring(
            [
                _spec("a", (-INF, INF), (2.0,)),
                _spec("b", (-INF, INF), (1.0,)),
            ],
            {"a": 1.0, "b": 3.0},
        )
        (_a, notional, scores, mult), *_ = self._size(scoring, {"a": 0.0, "b": 0.0})
        self.assertAlmostEqual(mult, 1.25)
        self.assertAlmostEqual(notional, 1250.0)
        self.assertEqual(scores, {"a": 2.0, "b": 1.0})

    def test_multiplier_is_clamped(self):
        spec = _spec("f", (-INF, 0.0, INF), (0.1, 9.0))
        low = self._size(_scoring([spec], {"f": 1.0}, floor=0.5, cap=3.0), {"f": -1.0})
        high = self._size(_scoring([spec], {"f": 1.0}, floor=0.5, cap=3.0), {"f": 1.0})
        self.assertEqual((low[0][3], high[0][3]), (0.5, 3.0))
        self.assertEqual((low[0][1], high[0][1]), (500.0, 3000.0))

    def test_exclude_mode_drops_trades_below_floor(self):
        spec = _spec("f", (-INF, 0.0, INF), (0.2, 1.0))
        scoring = _scoring([spec], {"f": 1.0}, floor=0.5, cap=3.0, mode="exclude")
        self.assertEqual(self._size(scoring, {"f": -1.0}), [])  # raw 0.2 < 0.5 -> dropped
        self.assertEqual(len(self._size(scoring, {"f": 1.0})), 1)

    def test_zero_multiplier_drops_the_trade(self):
        spec = _spec("f", (-INF, 0.0, INF), (0.0, 1.0))
        scoring = _scoring([spec], {"f": 1.0}, floor=0.0, cap=3.0)
        self.assertEqual(self._size(scoring, {"f": -1.0}), [])

    def test_zero_interval_weight_alone_does_not_exclude(self):
        # NOTE: IntervalScoringConfig's docstring says a 0.0 weight from any feature
        # excludes the trade. The implementation only averages: here raw = (0 + 2) / 2 = 1.
        scoring = _scoring(
            [_spec("a", (-INF, INF), (0.0,)), _spec("b", (-INF, INF), (2.0,))],
            {"a": 1.0, "b": 1.0},
        )
        result = self._size(scoring, {"a": 0.0, "b": 0.0})
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][3], 1.0)

    def test_all_aggregation_weights_zero_means_neutral_multiplier(self):
        scoring = _scoring([_spec("f", (-INF, INF), (5.0,))], {"f": 0.0})
        (_a, notional, _s, mult), *_ = self._size(scoring, {"f": 0.0})
        self.assertEqual((notional, mult), (1000.0, 1.0))

    def test_missing_candidate_feature_raises_loudly(self):
        scoring = _scoring([_spec("f", (-INF, INF), (1.0,))], {"f": 1.0})
        with self.assertRaises(KeyError):
            self._size(scoring, {})

    def test_missing_feature_value_uses_missing_weight(self):
        spec = _spec("f", (-INF, INF), (2.0,), missing=0.25)
        (_a, notional, _s, mult), *_ = self._size(_scoring([spec], {"f": 1.0}), {"f": float("nan")})
        self.assertEqual((notional, mult), (250.0, 0.25))

    def test_vol_normalisation_and_interval_multiplier_compound(self):
        scoring = _scoring([_spec("f", (-INF, INF), (2.0,))], {"f": 1.0})
        cfg = SizingConfig(
            1000.0, vol_normalize=VolSizingConfig(0.1, 10.0), interval_scoring=scoring
        )
        states = _states(a=0.5, b=1.0, c=2.0)
        for st in states.values():
            st.features["f"] = 0.0
        got = SizingEngine(cfg).size([_open("a")], states)
        self.assertEqual(_notionals(got), {"a": 1000.0 * 2.0 * 2.0})


class TestConfigValidation(unittest.TestCase):
    def test_base_notional_must_be_positive(self):
        for bad in (0.0, -1.0):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                SizingConfig(base_pair_notional=bad)

    def test_interval_spec_shape_validation(self):
        with self.assertRaises(ValueError):
            _spec("f", (0.0,), ())  # needs >= 2 limits
        with self.assertRaises(ValueError):
            _spec("f", (0.0, 1.0, 2.0), (1.0,))  # weights length mismatch
        with self.assertRaises(ValueError):
            _spec("f", (2.0, 1.0), (1.0,))  # limits not increasing

    def test_scoring_config_requires_matching_weights(self):
        spec = _spec("f", (-INF, INF), (1.0,))
        with self.assertRaises(ValueError):
            _scoring([spec], {})  # spec without aggregation weight
        with self.assertRaises(ValueError):
            _scoring([spec], {"f": 1.0, "ghost": 1.0})  # weight without spec
        with self.assertRaises(ValueError):
            _scoring([spec], {"f": 1.0}, mode="bogus")
        with self.assertRaises(ValueError):
            _scoring([spec], {"f": 1.0}, floor=2.0, cap=1.0)


if __name__ == "__main__":
    unittest.main()
