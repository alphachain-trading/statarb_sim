"""Unit tests for price QC (corruption detection) and return / bundle construction."""

import math
import unittest

import numpy as np
import pandas as pd

from src.data.qc import check_prices_for_corruption, print_qc_warnings
from src.data.returns import build_group_return_bundle, compute_returns
from src.data.universe_marketdata import UniverseMarketData
from tests._synthetic import make_umd

IDX = pd.bdate_range("2024-01-01", periods=6)


def _prices(**tickers: dict[str, list[float]]) -> pd.DataFrame:
    """Per ticker: field -> list of 6 values. Missing fields default to a clean series."""
    frames = {}
    for t, fields in tickers.items():
        base = {
            "Open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "High": [10.5, 10.6, 10.7, 10.8, 10.9, 11.0],
            "Low": [9.5, 9.6, 9.7, 9.8, 9.9, 10.0],
            "Close": [10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
            "Volume": [1e6] * 6,
        }
        base.update(fields)
        frames[t] = pd.DataFrame(base, index=IDX)
    out = pd.concat(frames, axis=1)
    out.columns.names = ["ticker", "field"]
    return out


def _issues(prices: pd.DataFrame, **kw: float | bool) -> set[tuple[str, str]]:
    return {(w.ticker, w.issue) for w in check_prices_for_corruption(prices, **kw)}  # type: ignore[arg-type]


class TestQC(unittest.TestCase):
    def test_clean_prices_have_no_warnings(self):
        self.assertEqual(check_prices_for_corruption(_prices(AAA={}, BBB={})), [])

    def test_requires_multiindex_columns(self):
        with self.assertRaises(ValueError):
            check_prices_for_corruption(pd.DataFrame({"Close": [1.0]}))

    def test_missing_fields_reported_and_ticker_skipped(self):
        p = _prices(AAA={})
        p = p.drop(columns=[("AAA", "Volume")])
        self.assertEqual(_issues(p), {("AAA", "missing_fields")})

    def test_each_issue_type_is_detected_for_the_right_ticker(self):
        nan_ohlc = [10.2, np.nan, 10.4, 10.5, 10.6, 10.7]
        cases = {
            "missing_ohlc": {"Close": nan_ohlc},
            "missing_volume": {"Volume": [1e6, np.nan, 1e6, 1e6, 1e6, 1e6]},
            "high_below_low": {"High": [10.5, 9.0, 10.7, 10.8, 10.9, 11.0]},
            "open_outside_range": {"Open": [10.0, 12.0, 10.2, 10.3, 10.4, 10.5]},
            "close_outside_range": {"Close": [10.2, 11.5, 10.4, 10.5, 10.6, 10.7]},
            "nonpositive_price": {"Low": [9.5, 0.0, 9.7, 9.8, 9.9, 10.0]},
            "negative_volume": {"Volume": [1e6, -1.0, 1e6, 1e6, 1e6, 1e6]},
            "extreme_return": {"Close": [10.2, 20.0, 10.4, 10.5, 10.6, 10.7]},
        }
        for issue, fields in cases.items():
            with self.subTest(issue):
                found = _issues(_prices(BAD=fields, GOOD={}))
                self.assertIn(("BAD", issue), found)
                self.assertFalse({t for t, _ in found} - {"BAD"}, "clean ticker must stay clean")

    def test_close_slightly_outside_range_is_within_tolerance(self):
        # High 10.5, Close 10.5042 -> 0.04% above: inside the default 0.1% tolerance
        p = _prices(AAA={"Close": [10.2, 10.5042, 10.4, 10.3, 10.3, 10.3], "High": [10.5] * 6})
        self.assertNotIn(("AAA", "close_outside_range"), _issues(p))
        self.assertIn(("AAA", "close_outside_range"), _issues(p, outside_range_tolerance=0.0001))

    def test_extreme_return_threshold_and_details(self):
        p = _prices(AAA={"Close": [10.0, 14.0, 14.1, 14.2, 14.3, 14.4]})  # +40% on day 2
        self.assertNotIn(("AAA", "extreme_return"), _issues(p))  # default threshold 50%
        (w,) = [
            w
            for w in check_prices_for_corruption(p, extreme_return_threshold=0.3)
            if w.issue == "extreme_return"
        ]
        self.assertIn("40.0%", w.details)
        self.assertIn("2024.01.02", w.details)

    def test_pre_inception_nans_are_ignored_by_default(self):
        nan = [np.nan, np.nan, 10.0, 10.1, 10.2, 10.3]
        p = _prices(
            NEW={
                "Open": nan,
                "High": [np.nan, np.nan, 10.5, 10.6, 10.7, 10.8],
                "Low": [np.nan, np.nan, 9.5, 9.6, 9.7, 9.8],
                "Close": nan,
            }
        )
        self.assertEqual(_issues(p), set())
        self.assertIn(("NEW", "missing_ohlc"), _issues(p, ignore_pre_inception=False))

    def test_print_helper(self):
        print_qc_warnings([])  # must not raise on the empty list


class TestComputeReturns(unittest.TestCase):
    def setUp(self):
        self.px = pd.DataFrame(
            {"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3)
        )

    def test_pct_and_log_hand_computed(self):
        pct = compute_returns(self.px, "pct")["A"]
        self.assertTrue(math.isnan(pct.iloc[0]))
        self.assertAlmostEqual(pct.iloc[1], 0.10)
        self.assertAlmostEqual(pct.iloc[2], -0.10)
        lg = compute_returns(self.px, "log")["A"]
        self.assertAlmostEqual(lg.iloc[1], math.log(1.1))
        self.assertAlmostEqual(lg.iloc[2], math.log(0.9))

    def test_unsorted_input_is_sorted_first(self):
        shuffled = self.px.iloc[[2, 0, 1]]
        pd.testing.assert_frame_equal(
            compute_returns(shuffled, "pct"), compute_returns(self.px, "pct"), check_freq=False
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            compute_returns(pd.DataFrame(), "log")
        with self.assertRaises(ValueError):
            compute_returns(self.px, "simple")


def _with_nan(umd: UniverseMarketData, ticker: str, row: int) -> UniverseMarketData:
    prices = umd.prices.copy()
    prices.loc[prices.index[row], (ticker, "Close")] = np.nan
    return UniverseMarketData(prices, umd.ticker_info, umd.group_info, umd.membership)


class TestBuildGroupReturnBundle(unittest.TestCase):
    def setUp(self):
        self.umd = make_umd({"alpha": 1, "beta": 1}, n_days=40)

    def test_structure_and_alignment(self):
        b = build_group_return_bundle(self.umd, "alpha")
        self.assertEqual(b.members, ["ALP0A", "ALP0B"])
        self.assertEqual((b.proxy, b.benchmark, b.risk_free), ("ALPETF", "SPYX", "IRXX"))
        self.assertEqual(list(b.aligned_returns.columns), ["ALP0A", "ALP0B", "ALPETF", "SPYX"])
        self.assertEqual(len(b.aligned_returns), 39)  # first return row is NaN -> dropped
        self.assertTrue(b.aligned_returns.index.equals(b.member_returns.index))
        self.assertEqual(float(b.risk_free_returns.iloc[0]), 2.0)  # raw level, not a return

    def test_log_return_value_matches_prices(self):
        b = build_group_return_bundle(self.umd, "alpha")
        px = self.umd.prices[("ALP0A", "Close")]
        self.assertAlmostEqual(b.member_returns["ALP0A"].iloc[0], math.log(px.iloc[1] / px.iloc[0]))
        pct = build_group_return_bundle(self.umd, "alpha", return_method="pct")
        self.assertAlmostEqual(pct.member_returns["ALP0A"].iloc[0], px.iloc[1] / px.iloc[0] - 1)

    def test_groups_only_see_their_own_members(self):
        self.assertEqual(build_group_return_bundle(self.umd, "beta").members, ["BET0A", "BET0B"])

    def test_dropna_modes(self):
        holed = _with_nan(self.umd, "ALP0A", 10)  # NaN price -> NaN returns on rows 9 and 10

        def n_rows(mode: str) -> int:
            return len(build_group_return_bundle(holed, "alpha", dropna=mode).aligned_returns)

        self.assertEqual(n_rows("any"), 37)  # both affected rows dropped
        self.assertEqual(n_rows("all"), 39)  # only all-NaN rows would go
        self.assertEqual(n_rows("none"), 40)  # keeps the leading NaN row too
        with self.assertRaises(ValueError):
            build_group_return_bundle(self.umd, "alpha", dropna="bogus")

    def test_unknown_group_raises(self):
        with self.assertRaises(KeyError):
            build_group_return_bundle(self.umd, "gamma")


if __name__ == "__main__":
    unittest.main()
