"""
Tests for pairwise dropna (Track B commit 2).

Before this fix, the hedge/diagnostics windows were cleaned once with a
single shared `.dropna(axis=0, how="any")` over every member column
(pair_candidate_panel_creator.py, pre-fix `rr_hedge_clean` /
`rr_diag_clean`). Because a row-wise dropna leaves the survivors fully
dense, `.dropna(axis=1, how="any")` right after it is a no-op — pandas
never drops a column there (verified directly: dropna(axis=0) always
produces a dense remainder, so a following dropna(axis=1) can never find a
column with a NaN left to drop). The real, verified consequence is
narrower than "a gapped ticker's column disappears": a single NaN in ANY
one member's window silently removes that date from EVERY pair's fit,
including pairs whose own two legs were never affected. A pair's fitted
weights depended on data from tickers it doesn't even trade —
confirmed empirically pre-fix: AAA|BBB's OLS beta changed from -0.12020 to
-0.12702 purely because an unrelated ticker CCC gained a gap, with AAA and
BBB's own values held fixed.

Pairwise dropna (B_window_admission.md items 1 / B_spec.md §6) fixes this
by building each pair's fit input from its own two legs only. These tests
call _build_pair_candidate_rows_for_date directly, mocking
apply_causal_residual_model so the residual window can carry a synthetic
gap without needing a real fit.
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


def _make_bundle(n: int = 15) -> GroupReturnBundle:
    idx = pd.bdate_range("2020-01-01", periods=n)
    zeros = pd.DataFrame(0.0, index=idx, columns=["AAA", "BBB", "CCC"])
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


def _fake_residuals(n: int, gap_row: int | None) -> pd.DataFrame:
    # Fixed seed, freshly reset per call: AAA/BBB are bit-identical across
    # calls regardless of gap_row, so any difference in AAA|BBB's fitted
    # output can only come from CCC's gap leaking into a shared row set.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "AAA": rng.normal(scale=0.01, size=n),
        "BBB": rng.normal(scale=0.01, size=n),
        "CCC": rng.normal(scale=0.01, size=n),
    })
    if gap_row is not None:
        df.loc[gap_row, "CCC"] = np.nan
    return df


class TestPairwiseDropna(unittest.TestCase):
    def _build_rows(self, n=15, gap_row: int | None = None):
        bundle = _make_bundle(n=n)
        idx = bundle.aligned_returns.index

        def fake_apply(*, model, aligned_returns, rf_series):
            residuals = _fake_residuals(len(aligned_returns), gap_row=gap_row)
            residuals.index = aligned_returns.index
            return residuals

        fitted_model = SimpleNamespace(subtract_risk_free=False)
        residual_cfg = CausalResidualConfig(
            mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False, lb=n,
        )
        pair_cfg = pcpc.PairSpreadConfig(
            hedge_ratio_methods=["ols"],
            skip_adf=True,
            min_obs=2,
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

    def test_clean_pair_unaffected_by_third_tickers_gap(self):
        clean = self._build_rows(gap_row=None)
        gapped = self._build_rows(gap_row=5)

        ab_clean = clean["AAA|BBB"]
        ab_gapped = gapped["AAA|BBB"]

        # AAA|BBB touches neither leg CCC gains a gap on — its fit must be
        # bit-identical whether or not CCC has a gap. Pre-fix this failed:
        # the shared dropna dropped date index 5 for every pair, so AAA|BBB
        # was silently refit on 14 rows instead of 15 whenever CCC gapped.
        # np.testing.assert_equal treats NaN == NaN as equal, unlike ==.
        self.assertEqual(ab_clean["weights"], ab_gapped["weights"])
        np.testing.assert_equal(ab_clean["residual_std"], ab_gapped["residual_std"])
        np.testing.assert_equal(ab_clean["kappa"], ab_gapped["kappa"])

    def test_pairs_touching_the_gapped_ticker_still_produce_a_row(self):
        rows = self._build_rows(gap_row=5)
        # CCC's own pairs are still attempted, fit on their own 14 retained
        # rows — not dropped from the candidate universe entirely.
        self.assertIn("AAA|CCC", rows)
        self.assertIn("BBB|CCC", rows)


if __name__ == "__main__":
    unittest.main()
