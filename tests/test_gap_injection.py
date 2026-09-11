"""
Gap-injection test through the real fit + apply pipeline (Track B).

Unlike test_pairwise_dropna.py (which mocks apply_causal_residual_model),
this test injects a single NaN directly into the returns frame the panel
builder receives — GroupReturnBundle.aligned_returns — and runs the real
fit_causal_residual_model / apply_causal_residual_model / pairwise-dropna
path end to end. Not market data: aligned_returns is what
_slice_candidate_window / _slice_fit_window read from; UMD/price data is
untouched (Track C's immutability boundary is one layer below this).

The gap sits inside the (wider) hedge/diagnostics candidate window but
outside the (narrower) residual fit window, both trailing slices ending at
the same asof date — so the fit itself is undisturbed (the gapped ticker's
own regression coefficients are fitted cleanly on dense data) and only
apply_causal_residual_model's row-wise subtraction sees the NaN, confined
to the gapped ticker's column on that one date.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.candidates.pair_candidate_panel_creator as pcpc
from src.data.returns import GroupReturnBundle
from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode

N = 50
FIT_LB = 15          # fit window: last 15 rows -> absolute indices 35..49
CANDIDATE_LB = 40    # candidate window: last 40 rows -> absolute indices 10..49
GAP_ABS_INDEX = 15   # inside the candidate window, outside the fit window


def _make_bundle(seed: int = 7) -> GroupReturnBundle:
    idx = pd.bdate_range("2015-01-01", periods=N)
    rng = np.random.default_rng(seed)
    members = pd.DataFrame(
        {t: rng.normal(scale=0.01, size=N) for t in ("AAA", "BBB", "CCC")},
        index=idx,
    )
    proxy = pd.Series(rng.normal(scale=0.01, size=N), index=idx, name="PROXY")
    bench = pd.Series(rng.normal(scale=0.01, size=N), index=idx, name="BENCH")
    aligned = pd.concat([members, proxy, bench], axis=1)
    rf = pd.Series(0.0, index=idx, name="RF")
    return GroupReturnBundle(
        group_id="test",
        member_returns=members,
        proxy_returns=proxy,
        benchmark_returns=bench,
        aligned_returns=aligned,
        risk_free_returns=rf,
    )


def _punch_hole(bundle: GroupReturnBundle, ticker: str, abs_index: int) -> GroupReturnBundle:
    aligned = bundle.aligned_returns.copy()
    aligned.iloc[abs_index, aligned.columns.get_loc(ticker)] = np.nan
    return GroupReturnBundle(
        group_id=bundle.group_id,
        member_returns=bundle.member_returns,
        proxy_returns=bundle.proxy_returns,
        benchmark_returns=bundle.benchmark_returns,
        aligned_returns=aligned,
        risk_free_returns=bundle.risk_free_returns,
    )


def _residual_cfg() -> CausalResidualConfig:
    return CausalResidualConfig(mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False, lb=FIT_LB)


def _pair_cfg() -> pcpc.PairSpreadConfig:
    return pcpc.PairSpreadConfig(
        hedge_ratio_methods=["ols"],
        skip_adf=True,
        min_obs=2,
        min_return_std=0.0,
        min_level_std=0.0,
        min_kappa=-1e9,
        max_half_life=1e9,
        tiny_weight_threshold=1e-6,
    )


def _build_rows(bundle: GroupReturnBundle) -> tuple[dict[str, dict], dict[str, dict]]:
    idx = bundle.aligned_returns.index
    rows, weight_rows = pcpc._build_pair_candidate_rows_for_date(
        bundle=bundle,
        asof_datetime=idx[-1],
        residual_cfg=_residual_cfg(),
        pair_cfg=_pair_cfg(),
        hedge_ratio_lb=CANDIDATE_LB,
        mr_diag_lb=CANDIDATE_LB,
        debug=True,
        progress=False,
    )
    weights_by_spread: dict[str, dict] = {}
    for wr in weight_rows:
        weights_by_spread.setdefault(wr["spread_id"], {})[wr["ticker"]] = wr["weight"]
    return {r["spread_id"]: r for r in rows}, weights_by_spread


class TestGapInjectionRealPipeline(unittest.TestCase):
    def test_setup_sanity_gap_outside_fit_window_inside_candidate_window(self):
        idx = _make_bundle().aligned_returns.index
        fit_start = idx[N - FIT_LB]
        candidate_start = idx[N - CANDIDATE_LB]
        gap_date = idx[GAP_ABS_INDEX]
        self.assertLess(gap_date, fit_start, "gap must be outside the fit window")
        self.assertGreaterEqual(gap_date, candidate_start, "gap must be inside the candidate window")

    def test_clean_pairs_beta_is_bit_identical_with_and_without_the_gap(self):
        clean_rows, clean_weights = _build_rows(_make_bundle())
        gapped_rows, gapped_weights = _build_rows(_punch_hole(_make_bundle(), "CCC", GAP_ABS_INDEX))

        ab_clean = clean_rows["AAA|BBB"]
        ab_gapped = gapped_rows["AAA|BBB"]

        # check_exact / no tolerance: pairwise construction means CCC's gap
        # never enters AAA|BBB's fit input at all, so the hedge beta must be
        # bit-identical, not merely close.
        self.assertEqual(ab_clean["hedge_beta"], ab_gapped["hedge_beta"])
        self.assertEqual(clean_weights["AAA|BBB"], gapped_weights["AAA|BBB"])
        self.assertEqual(ab_clean["kappa"], ab_gapped["kappa"])
        self.assertEqual(ab_clean["residual_std"], ab_gapped["residual_std"])

    def test_gapped_tickers_own_pairs_lose_exactly_the_gapped_date(self):
        with self.assertLogs(pcpc.logger.name, level="DEBUG") as cm:
            rows, _weights = _build_rows(_punch_hole(_make_bundle(), "CCC", GAP_ABS_INDEX))
        self.assertIn("AAA|CCC", rows)
        self.assertIn("BBB|CCC", rows)

        joined = "\n".join(cm.output)
        for pair in ("AAA|CCC", "BBB|CCC"):
            # Exactly 1 of CANDIDATE_LB rows dropped for each pair touching
            # the gapped ticker — not more (the shared-matrix defect this
            # replaces would have dropped every pair, or every row, instead
            # of exactly the one gapped date for exactly the pairs that
            # touch it).
            needle = f"pair={pair} hedge_dropped=1/{CANDIDATE_LB} diag_dropped=1/{CANDIDATE_LB}"
            self.assertIn(needle, joined, f"expected exactly one dropped row for {pair}")

        # AAA|BBB touches neither leg on the gapped ticker — no drop logged
        # for it at all.
        self.assertNotIn("pair=AAA|BBB hedge_dropped", joined)


if __name__ == "__main__":
    unittest.main()
