"""
Unit tests for EntryFeatureEngine: feature resolution, parameter injection,
direction correction and the skip / failure paths.

Direction correction multiplies signed features by sign(entry z-score): a long entry
(z < 0) flips the sign, a short entry (z > 0) keeps it.
"""

import unittest
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.spread_primitives import compute_x_area_asymmetry_ewm
from src.simulator.actions import OpenCandidateAction
from src.simulator.config import EntryFeatureConfig, FeatureSpec
from src.simulator.entry_feature_engine import EntryFeatureEngine, _resolve_fn
from tests._synthetic import make_ref, make_state

DATE = pd.Timestamp("2024-01-02")
LEVELS = np.array([0.0, 1.0, 0.5, 1.5, 1.0, 2.0, 1.2, 1.8])
XAA = FeatureSpec("x_area_asymmetry_ewm", {"halflife": 3})


class _FakeGenerator:
    def __init__(self, levels: np.ndarray | None) -> None:
        self.levels = levels
        self.calls: list[tuple[Any, pd.Timestamp]] = []

    def get_level_series(self, ref: Any, date: pd.Timestamp) -> pd.Series | None:
        self.calls.append((ref, date))
        return None if self.levels is None else pd.Series(self.levels)


def _open(cid: str = "c1") -> OpenCandidateAction:
    return OpenCandidateAction(
        candidate_id=cid, group_id="G1", spread_id="A|B", direction=1.0, reason="entry_long"
    )


def _engine(*specs: FeatureSpec, levels: np.ndarray | None = LEVELS) -> EntryFeatureEngine:
    return EntryFeatureEngine(
        config=EntryFeatureConfig(features=tuple(specs)), signal_generator=_FakeGenerator(levels)
    )


def _run(engine: EntryFeatureEngine, state: Any, ref: Any = "default") -> None:
    ref = make_ref() if ref == "default" else ref
    states = {} if state is None else {"c1": state}
    refs = {} if ref is None else {"c1": ref}
    engine.compute([_open()], states, refs, DATE)


class TestResolution(unittest.TestCase):
    def test_unknown_feature_fails_at_construction(self):
        with self.assertRaises(ValueError) as ctx:
            _engine(FeatureSpec("no_such_feature"))
        self.assertIn("no_such_feature", str(ctx.exception))

    def test_all_registered_names_resolve(self):
        for name in (
            "x_area_asymmetry_ewm",
            "bw_gain_ann_norm",
            "mean_ewm%",
            "entry_delta_z_ewm3",
            "entry_delta_z_ewm5",
            "entry_delta_z_ewm10",
            "entry_delta_z_ewm21",
        ):
            self.assertTrue(callable(_resolve_fn(name)), name)

    def test_extract_residual_hl(self):
        f = EntryFeatureEngine._extract_residual_hl
        self.assertEqual(f("exp_hl504_mh1008_rf"), 504)
        self.assertEqual(f("rol_lb21"), None)
        self.assertEqual(f("exp_hlx_mh20"), None)


class TestDirectionCorrection(unittest.TestCase):
    def setUp(self):
        raw = compute_x_area_asymmetry_ewm(LEVELS, halflife=3)
        assert raw is not None
        self.raw: float = raw

    def test_long_entry_flips_signed_feature(self):
        state = make_state(z_score=-2.0)
        _run(_engine(XAA), state)
        self.assertAlmostEqual(state.features["x_area_asymmetry_ewm"], -self.raw)

    def test_short_entry_keeps_sign(self):
        state = make_state(z_score=2.5)
        _run(_engine(XAA), state)
        self.assertAlmostEqual(state.features["x_area_asymmetry_ewm"], self.raw)

    def test_only_the_sign_of_z_matters(self):
        a, b = make_state(z_score=-0.1), make_state(z_score=-9.0)
        _run(_engine(XAA), a)
        _run(_engine(XAA), b)
        self.assertEqual(a.features, b.features)

    def test_missing_or_zero_z_skips_signed_features(self):
        for z in (None, 0.0):
            state = make_state(z_score=z)
            _run(_engine(XAA), state)
            self.assertEqual(state.features, {}, f"z={z}")


class TestSkips(unittest.TestCase):
    def test_nothing_written_when_inputs_missing_or_not_ready(self):
        cases = {
            "no state": (None, "default", LEVELS),
            "no ref": (make_state(), None, LEVELS),
            "not ready": (make_state(is_signal_ready=False), "default", LEVELS),
            "no levels": (make_state(), "default", None),
            "one level": (make_state(), "default", np.array([1.0])),
        }
        for label, (state, ref, levels) in cases.items():
            with self.subTest(label):
                _run(_engine(XAA, levels=levels), state, ref)
                if state is not None:
                    self.assertEqual(state.features, {})

    def test_empty_proposals_or_features_do_not_touch_generator(self):
        engine = _engine(XAA)
        engine.compute([], {}, {}, DATE)
        self.assertEqual(engine.signal_generator.calls, [])
        empty = _engine()
        empty.compute([_open()], {"c1": make_state()}, {"c1": make_ref()}, DATE)
        self.assertEqual(empty.signal_generator.calls, [])

    def test_generator_called_with_ref_and_date(self):
        engine = _engine(XAA)
        ref = make_ref()
        engine.compute([_open()], {"c1": make_state()}, {"c1": ref}, DATE)
        self.assertEqual(engine.signal_generator.calls, [(ref, DATE)])


class TestParameterInjection(unittest.TestCase):
    def _recording_engine(self, spec: FeatureSpec) -> tuple[EntryFeatureEngine, list[dict]]:
        engine = _engine(spec)
        seen: list[dict] = []

        def recorder(levels: np.ndarray, **params: Any) -> float:
            seen.append(params)
            return 0.5

        engine._resolved_fns[spec.feature] = recorder
        return engine, seen

    def test_residual_hl_sentinel_is_replaced(self):
        spec = FeatureSpec("x_area_asymmetry_ewm", {"halflife": "residual_hl"})
        engine, seen = self._recording_engine(spec)
        _run(engine, make_state(residual_key="exp_hl63_mh126"))
        self.assertEqual(seen, [{"halflife": 63}])
        self.assertEqual(spec.params, {"halflife": "residual_hl"})  # config not mutated

    def test_roll_std_sentinel_is_replaced(self):
        spec = FeatureSpec("x_area_asymmetry_ewm", {"halflife": 3, "roll_std": None})
        engine, seen = self._recording_engine(spec)
        _run(engine, make_state(roll_std=0.37))
        self.assertEqual(seen, [{"halflife": 3, "roll_std": 0.37}])

    def test_other_params_pass_through(self):
        spec = FeatureSpec("x_area_asymmetry_ewm", {"alpha": 0.2})
        engine, seen = self._recording_engine(spec)
        _run(engine, make_state())
        self.assertEqual(seen, [{"alpha": 0.2}])


class TestFailureHandling(unittest.TestCase):
    def test_feature_error_warns_and_leaves_feature_absent(self):
        # Pins current behaviour: a failing feature never stops the simulation; it is
        # dropped with a warning, and downstream sizing then treats it as missing.
        engine = _engine(XAA)

        def boom(levels: np.ndarray, **params: Any) -> float:
            raise RuntimeError("kaput")

        engine._resolved_fns[XAA.feature] = boom
        state = make_state()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _run(engine, state)
        self.assertEqual(state.features, {})
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("x_area_asymmetry_ewm" in m and "kaput" in m for m in messages))

    def test_none_value_is_not_written(self):
        engine = _engine(FeatureSpec("x_area_asymmetry_ewm", {"halflife": 3}), levels=np.ones(10))
        state = make_state()
        _run(engine, state)  # constant series -> primitive returns None
        self.assertEqual(state.features, {})


class TestMultipleFeatures(unittest.TestCase):
    def test_each_feature_is_written_under_its_own_name(self):
        specs = (
            FeatureSpec("x_area_asymmetry_ewm", {"halflife": 3}),
            FeatureSpec("entry_delta_z_ewm3", {"halflife": 3}),
        )
        state = make_state(z_score=3.0)
        _run(_engine(*specs, levels=np.random.default_rng(0).normal(size=40).cumsum()), state)
        self.assertEqual(set(state.features), {"x_area_asymmetry_ewm", "entry_delta_z_ewm3"})


if __name__ == "__main__":
    unittest.main()
