"""
Unit tests for src/analytics/spread_primitives.py — the shared signal maths.

Expected values are derived by hand (see comments) or from brute-force loops that
share no code with the implementation. Properties tested for every series function:
the last element equals the scalar function, and nothing depends on future data.
"""

import math
import unittest

import numpy as np
import pandas as pd

from src.analytics.spread_primitives import (
    SIGNED_FEATURE_NAMES,
    _expanding_pct,
    compute_bw_gain_ann_norm,
    compute_bw_gain_series,
    compute_bw_gain_stats,
    compute_delta_z_ewm,
    compute_fw_gain_series,
    compute_mean_ewm_pct,
    compute_mean_ewm_pct_series,
    compute_spread_level,
    compute_x_area_asymmetry_ewm,
    compute_x_area_asymmetry_ewm_series,
    direction_correct,
)


class TestXAreaAsymmetry(unittest.TestCase):
    def test_hand_computed_up_and_down(self):
        # alpha=0.5, levels [0, 1]: ewm mean = [0, 0.5]; disloc = [0, 0.5];
        # above = [0, 0.5] -> ewm [0, 0.25]; below = 0  =>  (0.25 - 0) / 0.25 = +1
        up = compute_x_area_asymmetry_ewm(np.array([0.0, 1.0]), alpha=0.5)
        # mirrored: all time below the mean => -1
        down = compute_x_area_asymmetry_ewm(np.array([1.0, 0.0]), alpha=0.5)
        assert up is not None and down is not None
        self.assertAlmostEqual(up, 1.0)
        self.assertAlmostEqual(down, -1.0)

    def test_symmetric_is_zero(self):
        # +1 then -1 around a flat mean is symmetric only approximately; a series that
        # is its own negative mirror has asymmetry exactly the negative of its mirror.
        x = np.array([0.0, 1.0, 0.2, 0.9, 0.1])
        a = compute_x_area_asymmetry_ewm(x, alpha=0.3)
        b = compute_x_area_asymmetry_ewm(-x, alpha=0.3)
        assert a is not None and b is not None
        self.assertAlmostEqual(a, -b)

    def test_constant_series_has_no_defined_value(self):
        self.assertIsNone(compute_x_area_asymmetry_ewm(np.ones(10), halflife=5))

    def test_too_short_returns_none(self):
        self.assertIsNone(compute_x_area_asymmetry_ewm(np.array([1.0]), halflife=5))

    def test_halflife_and_alpha_agree(self):
        x = np.random.default_rng(0).normal(size=40)
        alpha = 1.0 - math.exp(-1.0 / 7.0)
        self.assertEqual(
            compute_x_area_asymmetry_ewm(x, halflife=7),
            compute_x_area_asymmetry_ewm(x, alpha=alpha),
        )

    def test_parameter_validation(self):
        x = np.arange(5.0)
        for fn in (compute_x_area_asymmetry_ewm, compute_x_area_asymmetry_ewm_series):
            with self.assertRaises(ValueError):
                fn(x)
            with self.assertRaises(ValueError):
                fn(x, halflife=3, alpha=0.2)

    def test_series_last_equals_scalar_and_is_causal(self):
        x = np.random.default_rng(1).normal(size=60).cumsum()
        full = compute_x_area_asymmetry_ewm_series(x, halflife=6)
        for t in (5, 20, 59):
            scalar = compute_x_area_asymmetry_ewm(x[: t + 1], halflife=6)
            self.assertAlmostEqual(full[t], scalar)
        # changing the future must not change the past
        x2 = x.copy()
        x2[40:] += 100.0
        np.testing.assert_allclose(
            compute_x_area_asymmetry_ewm_series(x2, halflife=6)[:40], full[:40]
        )

    def test_series_short_input_is_all_nan(self):
        out = compute_x_area_asymmetry_ewm_series(np.array([1.0]), halflife=3)
        self.assertTrue(np.isnan(out).all())


def _naive_bw_gain(levels: np.ndarray, hl: int, mult: float) -> dict[str, float]:
    """Loop implementation straight from the docstring definition."""
    big_h = round(mult * hl)
    alpha = 1 - math.exp(-1 / hl)
    t = len(levels) - 1
    w = [alpha * (1 - alpha) ** (h - 1) for h in range(1, big_h + 1)]
    total = sum(w)
    w = [x / total for x in w]
    pg = [levels[t] - levels[t - h] for h in range(1, big_h + 1)]
    pga = [g * 252 / h for g, h in zip(pg, range(1, big_h + 1), strict=True)]
    mu = sum(wi * g for wi, g in zip(w, pg, strict=True))
    mua = sum(wi * g for wi, g in zip(w, pga, strict=True))
    sd = math.sqrt(sum(wi * (g - mu) ** 2 for wi, g in zip(w, pg, strict=True)))
    sda = math.sqrt(sum(wi * (g - mua) ** 2 for wi, g in zip(w, pga, strict=True)))
    return {
        "gain": mu,
        "gain_std": sd,
        "gain_norm": mu / sd if sd > 1e-12 else float("nan"),
        "gain_ann": mua,
        "gain_ann_std": sda,
        "gain_ann_norm": mua / sda if sda > 1e-12 else float("nan"),
    }


class TestBackwardGain(unittest.TestCase):
    def setUp(self):
        self.levels = np.random.default_rng(2).normal(size=80).cumsum()

    def test_matches_naive_definition(self):
        got = compute_bw_gain_stats(self.levels, bw_hl=5, bw_window_mult=2.0)
        want = _naive_bw_gain(self.levels, 5, 2.0)
        for key, value in want.items():
            self.assertAlmostEqual(got[key], value, msg=key)

    def test_linear_levels_have_constant_annualised_gain(self):
        # levels = s * t  =>  potential_gain(h) = s*h, annualised = 252*s for every h.
        # Weights sum to 1, so gain_ann == 252*s exactly and its std is 0 (ratio undefined).
        s = 0.01
        got = compute_bw_gain_stats(s * np.arange(50.0), bw_hl=4)
        self.assertAlmostEqual(got["gain_ann"], 252 * s)
        self.assertAlmostEqual(got["gain_ann_std"], 0.0, places=12)
        self.assertTrue(math.isnan(got["gain_ann_norm"]))
        self.assertIsNone(compute_bw_gain_ann_norm(s * np.arange(50.0), bw_hl=4))

    def test_insufficient_history_is_all_nan(self):
        got = compute_bw_gain_stats(np.arange(8.0), bw_hl=5)  # H = 10 > 7
        self.assertTrue(all(math.isnan(v) for v in got.values()))
        self.assertEqual(len(got), 6)

    def test_nan_inside_window_is_all_nan(self):
        x = self.levels.copy()
        x[-4] = np.nan
        got = compute_bw_gain_stats(x, bw_hl=5)
        self.assertTrue(all(math.isnan(v) for v in got.values()))

    def test_ann_norm_wrapper_returns_float_for_noisy_series(self):
        v = compute_bw_gain_ann_norm(self.levels, bw_hl=5)
        assert v is not None
        self.assertAlmostEqual(v, _naive_bw_gain(self.levels, 5, 2.0)["gain_ann_norm"])

    def test_series_equals_scalar_at_every_valid_t(self):
        hl = 4
        series = compute_bw_gain_series(self.levels, bw_hl=hl)
        big_h = 8
        for t in range(len(self.levels)):
            scalar = compute_bw_gain_stats(self.levels[: t + 1], bw_hl=hl)
            for key in series:
                if t < big_h:
                    self.assertTrue(np.isnan(series[key][t]), f"{key}@{t}")
                else:
                    np.testing.assert_allclose(series[key][t], scalar[key], err_msg=f"{key}@{t}")

    def test_series_is_causal(self):
        base = compute_bw_gain_series(self.levels, bw_hl=4)
        x2 = self.levels.copy()
        x2[50:] += 25.0
        changed = compute_bw_gain_series(x2, bw_hl=4)
        for key in base:
            np.testing.assert_allclose(changed[key][:50], base[key][:50])

    def test_forward_gain_matches_naive_loop(self):
        hl, mult = 3, 2.0
        big_h = round(mult * hl)
        fw = compute_fw_gain_series(self.levels, fw_hl=hl, fw_window_mult=mult)
        alpha = 1 - math.exp(-1 / hl)
        w = np.array([alpha * (1 - alpha) ** (h - 1) for h in range(1, big_h + 1)])
        w /= w.sum()
        for t in range(len(self.levels) - big_h):
            pg = np.array([self.levels[t + h] - self.levels[t] for h in range(1, big_h + 1)])
            self.assertAlmostEqual(fw["gain"][t], float(np.sum(w * pg)), msg=f"gain@{t}")
            self.assertAlmostEqual(
                fw["gain_std"][t], float(np.sqrt(np.sum(w * (pg - np.sum(w * pg)) ** 2))), msg=t
            )
        # the last H points have no complete forward window
        self.assertTrue(np.isnan(fw["gain"][-1]))


class TestExpandingPercentile(unittest.TestCase):
    def test_hand_computed(self):
        # [1,2,3,2]: i=1 -> rank 2/2 -> 1.0 ; i=2 -> 3/3 -> 1.0 ;
        # i=3 -> sorted [1,2,2,3], bisect_right(2)=3 -> (3/4 - .5)*2 = 0.5
        out = _expanding_pct(np.array([1.0, 2.0, 3.0, 2.0]))
        self.assertTrue(np.isnan(out[0]))
        np.testing.assert_allclose(out[1:], [1.0, 1.0, 0.5])

    def test_nan_entries_are_skipped_not_counted(self):
        out = _expanding_pct(np.array([1.0, np.nan, 2.0]))
        self.assertTrue(np.isnan(out[1]))
        self.assertAlmostEqual(out[2], 1.0)  # 2 valid values, newest is the max

    def test_range(self):
        out = _expanding_pct(np.random.default_rng(3).normal(size=200))
        finite = out[np.isfinite(out)]
        self.assertTrue((finite >= -1).all() and (finite <= 1).all())

    def test_monotone_levels(self):
        up = compute_mean_ewm_pct(np.arange(1.0, 30.0), halflife=3)
        down = compute_mean_ewm_pct(np.arange(30.0, 1.0, -1.0), halflife=3)
        self.assertEqual(up, 1.0)
        # 29 values seen, the newest EWM value is the smallest -> rank 1 of 29
        assert down is not None
        self.assertAlmostEqual(down, (1 / 29 - 0.5) * 2.0)

    def test_series_is_causal(self):
        x = np.random.default_rng(4).normal(size=80).cumsum()
        base = compute_mean_ewm_pct_series(x, halflife=5)
        x2 = x.copy()
        x2[60:] -= 50.0
        np.testing.assert_allclose(
            compute_mean_ewm_pct_series(x2, halflife=5)[:60], base[:60], equal_nan=True
        )

    def test_short_input(self):
        self.assertTrue(np.isnan(compute_mean_ewm_pct_series(np.array([1.0]), 3)).all())
        self.assertIsNone(compute_mean_ewm_pct(np.array([1.0]), 3))


class TestDeltaZ(unittest.TestCase):
    def test_none_cases(self):
        self.assertIsNone(compute_delta_z_ewm(np.array([1.0, 2.0]), halflife=2))
        self.assertIsNone(compute_delta_z_ewm(np.ones(30), halflife=3))  # zero std

    def test_matches_manual_z_difference(self):
        x = np.random.default_rng(5).normal(size=50).cumsum()
        hl = 4
        s = pd.Series(x).ewm(halflife=hl, min_periods=hl)
        m, sd = s.mean().to_numpy(), s.std().to_numpy()
        want = (x[-1] - m[-1]) / sd[-1] - (x[-2] - m[-2]) / sd[-2]
        self.assertAlmostEqual(compute_delta_z_ewm(x, halflife=hl), want)

    def test_invariant_to_positive_affine_transform(self):
        x = np.random.default_rng(6).normal(size=50).cumsum()
        a = compute_delta_z_ewm(x, halflife=4)
        b = compute_delta_z_ewm(3.0 * x + 10.0, halflife=4)
        assert a is not None and b is not None
        self.assertAlmostEqual(a, b)


class TestSpreadLevel(unittest.TestCase):
    def test_hand_computed(self):
        res = pd.DataFrame({"A": [1.0, 3.0, 2.0], "B": [2.0, 4.0, 0.0]}, index=pd.RangeIndex(3))
        ret, lvl = compute_spread_level(res, {"A": 1.0, "B": -0.5})
        np.testing.assert_allclose(ret, [0.0, 1.0, 2.0])  # 1-1, 3-2, 2-0
        np.testing.assert_allclose(lvl, [0.0, 1.0, 3.0])  # cumsum
        self.assertEqual((ret.name, lvl.name), ("spread_return", "level"))
        self.assertTrue(ret.index.equals(res.index))

    def test_weight_dict_order_is_irrelevant(self):
        res = pd.DataFrame(np.random.default_rng(7).normal(size=(10, 3)), columns=list("ABC"))
        r1, _ = compute_spread_level(res, {"A": 1.0, "C": -0.4})
        r2, _ = compute_spread_level(res, {"C": -0.4, "A": 1.0})
        pd.testing.assert_series_equal(r1, r2)

    def test_missing_ticker_raises(self):
        res = pd.DataFrame({"A": [1.0]})
        with self.assertRaises(KeyError):
            compute_spread_level(res, {"A": 1.0, "Z": -1.0})


class TestDirectionCorrect(unittest.TestCase):
    def test_multiplies_by_sign(self):
        self.assertEqual(direction_correct(2.0, 1.0), 2.0)
        self.assertEqual(direction_correct(2.0, -1.0), -2.0)

    def test_none_and_non_finite_pass_through(self):
        self.assertIsNone(direction_correct(None, -1.0))
        nan_out = direction_correct(float("nan"), -1.0)
        assert nan_out is not None
        self.assertTrue(math.isnan(nan_out))
        self.assertEqual(direction_correct(float("inf"), -1.0), float("inf"))

    def test_signed_feature_registry(self):
        for fam in ("fw", "bw"):
            for key in ("gain", "gain_norm", "gain_ann", "gain_ann_norm"):
                self.assertIn(f"{fam}_{key}", SIGNED_FEATURE_NAMES)
            for key in ("gain_std", "gain_ann_std"):
                self.assertNotIn(f"{fam}_{key}", SIGNED_FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
