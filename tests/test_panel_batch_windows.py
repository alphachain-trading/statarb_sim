"""
End-to-end tests for the ResidualMode redesign, exercising run_panel_batch on
the committed materials universe.

Proves the two behavioural claims the refactor rests on:
  1. EQ_ROLLING actually performs equal-weight fitting — a uniform weight
     vector (half_life=None → all-ones) reaches the WLS solve, not just
     "it didn't crash".
  2. hedge_ratio_lb and mr_diag_lb are applied as two independent candidate
     windows — both lookbacks reach _slice_candidate_window with distinct row
     counts within a single run.

Skips cleanly if the materials market data is not present.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.settings import DATA_UNIVERSES
from src.candidates.panel_batch import PanelBatchConfig, run_panel_batch
from src.simulator.config import ResidualMode, AbsOrMult
import src.residuals.causal_residuals as cr
import src.candidates.pair_candidate_panel_creator as pcpc

_MATERIALS_DATA = Path(DATA_UNIVERSES) / "materials_only_v1"


@unittest.skipUnless(_MATERIALS_DATA.exists(), "materials market data not present")
class TestEqualWeightFitting(unittest.TestCase):
    def test_eq_rolling_uses_uniform_weights_in_the_fit(self):
        captured: list[tuple] = []
        real_make_sqrt_w = cr._make_sqrt_w

        def spy(n, half_life):
            w = real_make_sqrt_w(n=n, half_life=half_life)
            captured.append((half_life, bool((w == w[0]).all())))
            return w

        cfg = PanelBatchConfig(
            residual_mode=ResidualMode.EQ_ROLLING,
            residual_lb=21,
            hedge_ratio_lb=21,
            mr_diag_lb=21,
            subtract_risk_free=True,
            selected_sectors=["materials"],
            max_steps=5,
            persist_result=False,
            persist_residual_params=False,
        )

        with mock.patch.object(cr, "_make_sqrt_w", spy):
            results = run_panel_batch(cfg)

        self.assertTrue(captured, "residual model was never fit")
        # Every fit must have received half_life=None → a uniform weight vector.
        self.assertTrue(all(hl is None for hl, _ in captured),
                        f"non-None half_life reached the fit: {set(hl for hl, _ in captured)}")
        self.assertTrue(all(uniform for _, uniform in captured),
                        "weight vector reaching the WLS solve was not uniform")

        # And the run actually produced candidates.
        (result,) = results.values()
        self.assertGreater(len(result.panel), 0)
        self.assertIsNone(result.metadata["residual_cfg"]["half_life"])


@unittest.skipUnless(_MATERIALS_DATA.exists(), "materials market data not present")
class TestIndependentWindows(unittest.TestCase):
    def test_hedge_and_diag_windows_are_sliced_independently(self):
        hedge_lb, diag_lb = 30, 60
        observed: list[tuple[int, int]] = []
        real_slice = pcpc._slice_candidate_window

        def spy(*, bundle, asof_datetime, lookback):
            out = real_slice(bundle=bundle, asof_datetime=asof_datetime, lookback=lookback)
            observed.append((lookback, len(out)))
            return out

        cfg = PanelBatchConfig(
            residual_mode=ResidualMode.EQ_EXPANDING,
            hedge_ratio_lb=hedge_lb,
            mr_diag_lb=diag_lb,
            residual_min_lb_eq_exp=300,
            subtract_risk_free=True,
            selected_sectors=["materials"],
            max_steps=3,
            persist_result=False,
            persist_residual_params=False,
        )

        with mock.patch.object(pcpc, "_slice_candidate_window", spy):
            run_panel_batch(cfg)

        lbs = {lb for lb, _ in observed}
        self.assertIn(hedge_lb, lbs)
        self.assertIn(diag_lb, lbs)

        # Row counts must honour each window independently.
        hedge_rows = {rows for lb, rows in observed if lb == hedge_lb}
        diag_rows = {rows for lb, rows in observed if lb == diag_lb}
        self.assertTrue(all(r <= hedge_lb for r in hedge_rows), hedge_rows)
        self.assertTrue(all(r <= diag_lb for r in diag_rows), diag_rows)
        # With ample history both windows fill exactly, and differ from each other.
        self.assertEqual(hedge_rows, {hedge_lb})
        self.assertEqual(diag_rows, {diag_lb})
        self.assertNotEqual(hedge_rows, diag_rows)


if __name__ == "__main__":
    unittest.main()
