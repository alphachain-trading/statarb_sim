"""Unit tests for src/residuals/spreads.py: ids, weight parsing, spread construction."""

import unittest

import pandas as pd

from src.residuals.spreads import (
    _parse_weights_json,
    add_relative_weights,
    build_spread_returns,
    extract_relative_weights,
    generate_spread_ids,
    ols_beta_no_intercept,
    spread_id,
    spread_members,
)


class TestSpreadIds(unittest.TestCase):
    def test_single_ticker_is_stripped(self):
        self.assertEqual(spread_id("  AAA "), "AAA")

    def test_multiple_tickers_are_sorted_and_joined(self):
        self.assertEqual(spread_id(["ZZZ", "AAA"]), "AAA|ZZZ")
        self.assertEqual(spread_id(["b", "a", "c"]), "a|b|c")

    def test_invalid_inputs(self):
        for bad in ("", "  ", [], ["A", ""], ["A", "A"], ["A", " A "]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                spread_id(bad)

    def test_members_round_trip_in_sorted_order(self):
        self.assertEqual(spread_members(spread_id(["Q", "B"])), ["B", "Q"])

    def test_members_rejects_malformed(self):
        for bad in ("A||B", "A|A", "|A", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                spread_members(bad)

    def test_generate_pairs_follow_input_combination_order_but_canonical_ids(self):
        # combinations of the *input order* (C,A),(C,B),(A,B) -> canonical sorted ids
        self.assertEqual(generate_spread_ids(["C", "A", "B"], 2), ["A|C", "B|C", "A|B"])
        self.assertEqual(generate_spread_ids(["A", "B", "C"], 2), ["A|B", "A|C", "B|C"])
        self.assertEqual(generate_spread_ids(["A", "B"], 1), ["A", "B"])

    def test_generate_validation(self):
        for tickers, n in ((["A", "B"], 0), (["A", "A"], 2), (["A"], 2), (["A", ""], 1)):
            with self.subTest(tickers=tickers, n=n), self.assertRaises(ValueError):
                generate_spread_ids(tickers, n)


class TestOlsBeta(unittest.TestCase):
    def test_hand_computed(self):
        # y = [2, 4], x = [1, 2]: sum(yx)=10, sum(xx)=5 -> beta 2
        self.assertEqual(ols_beta_no_intercept(pd.Series([2.0, 4.0]), pd.Series([1.0, 2.0])), 2.0)

    def test_no_intercept_means_offset_is_not_absorbed(self):
        # y = x + 1 with x = [1, 2]: beta = (1*2 + 2*3) / 5 = 1.6 (not 1.0)
        beta = ols_beta_no_intercept(pd.Series([2.0, 3.0]), pd.Series([1.0, 2.0]))
        self.assertAlmostEqual(beta, 1.6)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ols_beta_no_intercept(pd.Series([1.0]), pd.Series([1.0, 2.0]))
        with self.assertRaises(ValueError):
            ols_beta_no_intercept(pd.Series([1.0, 2.0]), pd.Series([0.0, 0.0]))


class TestWeights(unittest.TestCase):
    def test_parse_accepts_json_list_and_tuple(self):
        self.assertEqual(_parse_weights_json("[1, -2.5]"), [1.0, -2.5])
        self.assertEqual(_parse_weights_json([1, -2]), [1.0, -2.0])
        self.assertEqual(_parse_weights_json((1.0, 2.0, 3.0)), [1.0, 2.0, 3.0])

    def test_parse_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            _parse_weights_json("[1]")  # < 2 weights
        with self.assertRaises(ValueError):
            _parse_weights_json('{"a": 1}')  # not a list
        with self.assertRaises(ValueError):
            _parse_weights_json('[1, "x"]')  # non-numeric
        with self.assertRaises(TypeError):
            _parse_weights_json(5)  # type: ignore[arg-type]  # wrong type on purpose

    def test_relative_weights_are_relative_to_first_leg(self):
        self.assertEqual(extract_relative_weights([2.0, -4.0, 6.0]), [-2.0, 3.0])
        self.assertEqual(extract_relative_weights("[1.0, -1.0]"), [-1.0])

    def test_zero_first_weight_raises(self):
        with self.assertRaises(ValueError):
            extract_relative_weights([0.0, 1.0])

    def test_add_relative_weights_pads_shorter_vectors_with_na(self):
        df = pd.DataFrame({"weights_json": ["[1, -2]", "[2, 4, 6]"]})
        out = add_relative_weights(df)
        self.assertEqual(out["rel_w_2"].tolist(), [-2.0, 2.0])
        self.assertTrue(pd.isna(out["rel_w_3"].iloc[0]))
        self.assertEqual(out["rel_w_3"].iloc[1], 3.0)
        self.assertNotIn("rel_w_2", df.columns)  # input untouched unless inplace=True

    def test_add_relative_weights_inplace_and_missing_column(self):
        df = pd.DataFrame({"weights_json": ["[1, 3]"]})
        add_relative_weights(df, inplace=True)
        self.assertIn("rel_w_2", df.columns)
        with self.assertRaises(ValueError):
            add_relative_weights(pd.DataFrame({"x": [1]}))


class TestBuildSpreadReturns(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2024-01-01", periods=3)
        self.res = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [1.0, 1.0, 1.0], "C": [0.0, 1.0, 0.0]}, index=idx
        )

    def test_unit_spread_is_left_minus_right_with_alphabetical_legs(self):
        out = build_spread_returns(self.res[["B", "A"]], hedge_ratio_method="unit")
        # input column order B, A — canonical id "A|B", leg order alphabetical: A - B
        self.assertEqual(list(out.spread_returns.columns), ["A|B"])
        self.assertEqual(out.spread_returns["A|B"].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(out.spread_metadata.loc["A|B", "weights_json"], "[1.0, -1.0]")

    def test_ols_spread_hand_computed(self):
        # beta = sum(A*B)/sum(B*B) = (1+2+3)/3 = 2 -> spread = A - 2B = [-1, 0, 1]
        out = build_spread_returns(self.res[["A", "B"]], hedge_ratio_method="ols")
        self.assertEqual(out.spread_returns["A|B"].tolist(), [-1.0, 0.0, 1.0])
        meta = out.spread_metadata.loc["A|B"]
        self.assertEqual(meta["weights_json"], "[1.0, -2.0]")
        self.assertEqual(meta["relative_weights"], [-2.0])
        self.assertEqual(meta["hedge_ratio_method"], "ols")

    def test_all_pairs_are_built_and_metadata_sorted(self):
        out = build_spread_returns(self.res)
        self.assertEqual(sorted(out.spread_returns.columns), ["A|B", "A|C", "B|C"])
        self.assertEqual(list(out.spread_metadata.index), ["A|B", "A|C", "B|C"])
        self.assertEqual(out.spread_returns["B|C"].tolist(), [1.0, 0.0, 1.0])

    def test_validation(self):
        idx = self.res.index
        with self.assertRaises(ValueError):
            build_spread_returns(pd.DataFrame())
        with self.assertRaises(ValueError):
            build_spread_returns(pd.DataFrame({"A": [1.0, float("nan"), 2.0], "B": 1.0}, index=idx))
        dup = pd.DataFrame([[1.0, 2.0]] * 3, columns=["A", "A"], index=idx)
        with self.assertRaises(ValueError):
            build_spread_returns(dup)
        with self.assertRaises(NotImplementedError):
            build_spread_returns(self.res, n_legs=3)
        with self.assertRaises(NotImplementedError):
            build_spread_returns(self.res, hedge_ratio_method="pca")

    def test_single_column_cannot_form_a_pair(self):
        with self.assertRaises(ValueError):
            build_spread_returns(self.res[["A"]])


if __name__ == "__main__":
    unittest.main()
