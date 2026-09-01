"""
Tests for per-pair min_obs (Track B commit 3).

min_obs is the fix for B_window_admission.md items 4/5: the window slicer
only raises on an empty slice, and neither the OLS beta helper nor the PCA
weights helper checks sample size, so a pair could be fit on a handful of
retained observations and returned as if it were a full-lookback estimate.
min_obs is mandatory (no default) on PairSpreadConfig, absolute, and
counted on the pair's own retained rows after its own pairwise dropna
(Track B commit 2) — not on the window's nominal length and not on a
different pair's rows.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.candidates.pair_candidate_panel_creator as pcpc
from src.data.returns import GroupReturnBundle
from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode


def _make_bundle(n: int) -> GroupReturnBundle:
    idx = pd.bdate_range("2020-01-01", periods=n)
    zeros = pd.DataFrame(0.0, index=idx, columns=["AAA", "BBB"])
    proxy = pd.Series(0.0, index=idx, name="PROXY")
    bench = pd.Series(0.0, index=idx, name="BENCH")
    aligned = pd.concat([zeros, proxy, bench], axis=1)
    rf = pd.Series(0.0, index=idx, name="RF")
    return GroupReturnBundle(
        group_id="test",
        member_returns=zeros,
        proxy_returns=proxy,
        benchmark_returns=bench,
        aligned_returns=aligned,
        risk_free_returns=rf,
    )


class TestMinObsIsMandatory(unittest.TestCase):
    def test_omitting_min_obs_raises(self):
        with self.assertRaises(TypeError):
            pcpc.PairSpreadConfig(hedge_ratio_methods=["ols"])

    def test_min_obs_below_two_raises(self):
        with self.assertRaises(ValueError):
            pcpc.PairSpreadConfig(hedge_ratio_methods=["ols"], min_obs=1)


class TestMinObsGate(unittest.TestCase):
    def _build_rows(self, n, n_retained, min_obs):
        bundle = _make_bundle(n=n)
        idx = bundle.aligned_returns.index

        def fake_apply(*, model, aligned_returns, rf_series):
            k = len(aligned_returns)
            rng = np.random.default_rng(2)
            df = pd.DataFrame({
                "AAA": rng.normal(scale=0.01, size=k),
                "BBB": rng.normal(scale=0.01, size=k),
            })
            # BBB is only populated for its first n_retained rows — as if it
            # joined the universe late, or has a long trailing gap. AAA is
            # fully populated throughout.
            df.iloc[n_retained:, df.columns.get_loc("BBB")] = np.nan
            df.index = aligned_returns.index
            return df

        fitted_model = SimpleNamespace(subtract_risk_free=False)
        residual_cfg = CausalResidualConfig(
            mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False, lb=n,
        )
        pair_cfg = pcpc.PairSpreadConfig(
            hedge_ratio_methods=["ols"],
            skip_adf=True,
            min_obs=min_obs,
            min_return_std=0.0,
            min_level_std=0.0,
            min_kappa=-1e9,
            max_half_life=1e9,
        )

        with patch.object(pcpc, "apply_causal_residual_model", side_effect=fake_apply):
            rows = pcpc._build_pair_candidate_rows_for_date(
                bundle=bundle,
                asof_datetime=idx[-1],
                residual_cfg=residual_cfg,
                pair_cfg=pair_cfg,
                hedge_ratio_lb=n,
                mr_diag_lb=n,
                debug=False,
                fitted_model=fitted_model,
                progress=False,
            )
        return {r["spread_id"]: r for r in rows}

    def test_below_min_obs_produces_no_row(self):
        # AAA|BBB retains only 4 rows post pairwise-dropna; min_obs=10 —
        # no hedge ratio is fit, and no panel row is produced for it.
        rows = self._build_rows(n=20, n_retained=4, min_obs=10)
        self.assertNotIn("AAA|BBB", rows)

    def test_below_min_obs_is_reported_via_logging(self):
        # "Reported, not silently fitted": the drop is logged, both per-pair
        # (DEBUG) and as a per-date aggregate count (INFO), matching the
        # brief's "log the count of pairs dropped by min_obs per date".
        with self.assertLogs(pcpc.logger.name, level="DEBUG") as cm:
            self._build_rows(n=20, n_retained=4, min_obs=10)
        joined = "\n".join(cm.output)
        self.assertIn("min_obs_dropped", joined)
        self.assertIn("pairs_dropped=1", joined)

    def test_at_min_obs_is_fitted(self):
        # Exactly min_obs retained rows is enough — the gate is >=, not >.
        rows = self._build_rows(n=20, n_retained=10, min_obs=10)
        self.assertIn("AAA|BBB", rows)
        self.assertTrue(rows["AAA|BBB"]["is_valid"] or rows["AAA|BBB"]["why_invalid"])

    def test_min_obs_is_absolute_not_a_fraction_of_window(self):
        # A wider window (n=100) with the same 4 retained rows for BBB is
        # still dropped at min_obs=10 — the gate counts retained rows, not
        # the nominal window length.
        rows = self._build_rows(n=100, n_retained=4, min_obs=10)
        self.assertNotIn("AAA|BBB", rows)


if __name__ == "__main__":
    unittest.main()
