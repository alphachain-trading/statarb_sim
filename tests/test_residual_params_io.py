"""
Tests for residual_params.parquet (Track F1 commit 3).

save_residual_params/load_residual_params replace the old per-run
{stem}_residual_params.pkl (a pickled dict[fit_date, FittedCausalResidualModel])
with a long-form parquet: one row per (fit_date, ticker, factor). Fit logic
itself is untouched -- this is a pure storage-format swap, and these tests are
its result-neutrality argument: round-tripping a FittedCausalResidualModel
through save/load must reproduce byte-identical numpy arrays, with and
without optional PC removal, and must reject a malformed model that would
silently collide two rows onto the same (fit_date, ticker, factor) key.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.residuals.causal_residuals import (
    FittedCausalResidualModel,
    apply_causal_residual_model,
    load_residual_params,
    save_residual_params,
)


def _make_model(members, rng, *, pc_components=None, subtract_risk_free=False):
    return FittedCausalResidualModel(
        members=list(members),
        proxy_name="PROXY",
        bench_name="BENCH",
        B_proxy=rng.normal(size=(2, 1)),
        B_stock=rng.normal(size=(3, len(members))),
        pc_components=pc_components,
        subtract_risk_free=subtract_risk_free,
    )


class TestResidualParamsRoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = str(Path(self._tmp.name) / "test_residual_params.parquet")

    def test_round_trip_no_pc_removal(self):
        rng = np.random.default_rng(0)
        members = ["AAA", "BBB", "CCC"]
        dates = pd.bdate_range("2020-01-01", periods=4)
        fitted = {d: _make_model(members, rng, subtract_risk_free=(i % 2 == 0))
                  for i, d in enumerate(dates)}

        save_residual_params(fitted, self.path)
        loaded = load_residual_params(self.path)

        self.assertEqual(set(loaded.keys()), {pd.Timestamp(d) for d in dates})
        for d, model in fitted.items():
            lm = loaded[pd.Timestamp(d)]
            self.assertEqual(lm.members, sorted(model.members))
            self.assertEqual(lm.proxy_name, model.proxy_name)
            self.assertEqual(lm.bench_name, model.bench_name)
            self.assertEqual(lm.subtract_risk_free, model.subtract_risk_free)
            self.assertIsNone(lm.pc_components)
            np.testing.assert_array_equal(lm.B_proxy, model.B_proxy)
            np.testing.assert_array_equal(lm.B_stock, model.B_stock)

    def test_round_trip_with_pc_removal(self):
        rng = np.random.default_rng(1)
        members = ["AAA", "BBB", "CCC", "DDD"]
        d = pd.Timestamp("2021-06-01")
        pc = rng.normal(size=(2, len(members)))
        model = _make_model(members, rng, pc_components=pc)

        save_residual_params({d: model}, self.path)
        loaded = load_residual_params(self.path)[d]

        self.assertEqual(loaded.members, sorted(members))
        np.testing.assert_array_equal(loaded.pc_components, pc)

    def test_round_trip_mixed_pc_removal_across_dates(self):
        """One fit_date with PC removal on, another with it off -- independently reconstructed."""
        rng = np.random.default_rng(2)
        members = ["AAA", "BBB"]
        d_on, d_off = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
        pc = rng.normal(size=(1, len(members)))
        fitted = {
            d_on: _make_model(members, rng, pc_components=pc),
            d_off: _make_model(members, rng, pc_components=None),
        }

        save_residual_params(fitted, self.path)
        loaded = load_residual_params(self.path)

        np.testing.assert_array_equal(loaded[d_on].pc_components, pc)
        self.assertIsNone(loaded[d_off].pc_components)

    def test_round_trip_preserves_apply_output(self):
        """Reconstructed model reproduces the exact same residuals as the original."""
        rng = np.random.default_rng(3)
        members = ["ZZZ", "AAA", "MMM"]  # deliberately unsorted
        d = pd.Timestamp("2020-01-01")
        model = _make_model(members, rng)

        save_residual_params({d: model}, self.path)
        loaded = load_residual_params(self.path)[d]

        idx = pd.bdate_range("2020-01-01", periods=30)
        cols = [model.bench_name, model.proxy_name, *members]
        aligned = pd.DataFrame(rng.normal(size=(30, len(cols))), index=idx, columns=cols)

        original = apply_causal_residual_model(model, aligned)
        reconstructed = apply_causal_residual_model(loaded, aligned)

        pd.testing.assert_frame_equal(
            original[sorted(members)], reconstructed[sorted(members)],
            check_exact=True,
        )

    def test_empty_fitted_params_raises(self):
        with self.assertRaises(ValueError):
            save_residual_params({}, self.path)

    def test_duplicate_ticker_in_members_raises(self):
        """A model whose members list repeats a ticker would collide two rows
        onto the same (fit_date, ticker, factor) key -- must fail loud, not
        silently drop one."""
        rng = np.random.default_rng(4)
        model = _make_model(["AAA", "AAA"], rng)
        with self.assertRaises(ValueError):
            save_residual_params({pd.Timestamp("2020-01-01"): model}, self.path)


if __name__ == "__main__":
    unittest.main()
