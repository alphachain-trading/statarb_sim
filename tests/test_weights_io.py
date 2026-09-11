"""
Tests for weights.parquet (Track F1 commit 4).

save_weights/load_weights/weights_lookup_from_df replace the live path's old
per-candidate "weights" column, written via
`weights.to_json(double_precision=12)` -- which does not round-trip float64.
weights.parquet stores ALL legs at full float64 precision, one row per leg:
(group_id, residual_key, hedge_key, spread_id, asof_date, ticker) -> weight.

These tests are the artifact's uniqueness/round-trip argument: a write with
a duplicate (group_id, residual_key, hedge_key, spread_id, asof_date,
ticker) key must fail loud, and a round trip through save/load/
weights_lookup_from_df must reproduce exact float64 values, not
12-significant-digit-rounded ones.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.candidates.candidate_panel import load_weights, save_weights, weights_lookup_from_df


def _row(*, group_id="mat", residual_key="exp_hl20_rf", hedge_key="pca",
         spread_id="AAA|BBB", asof_date="2020-01-03", ticker="AAA", weight=1.0):
    return {
        "group_id": group_id,
        "residual_key": residual_key,
        "hedge_key": hedge_key,
        "spread_id": spread_id,
        "asof_date": pd.Timestamp(asof_date),
        "ticker": ticker,
        "weight": weight,
    }


class TestWeightsIO(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = str(Path(self._tmp.name) / "test_weights.parquet")

    def test_empty_rows_raises(self):
        with self.assertRaises(ValueError):
            save_weights([], self.path)

    def test_duplicate_key_raises(self):
        rows = [
            _row(ticker="AAA", weight=1.0),
            _row(ticker="AAA", weight=2.0),  # same (group_id, residual_key, hedge_key, spread_id, asof_date, ticker)
        ]
        with self.assertRaises(ValueError):
            save_weights(rows, self.path)

    def test_round_trip_full_precision(self):
        # A value that does NOT round-trip through to_json(double_precision=12).
        precise_weight = -3.4526955689887654321
        rows = [
            _row(ticker="AAA", weight=1.0),
            _row(ticker="BBB", weight=precise_weight),
        ]
        save_weights(rows, self.path)
        df = load_weights(self.path)

        self.assertEqual(len(df), 2)
        bbb_weight = df.loc[df["ticker"] == "BBB", "weight"].iloc[0]
        self.assertEqual(bbb_weight, precise_weight)

        # The old lossy path would have rounded this and lost the tail digits.
        lossy_json = pd.Series({"BBB": precise_weight}).to_json(double_precision=12)
        lossy = json.loads(lossy_json)["BBB"]
        self.assertNotEqual(lossy, precise_weight)
        self.assertEqual(bbb_weight, precise_weight)

    def test_weights_lookup_from_df_shape(self):
        rows = [
            _row(spread_id="AAA|BBB", asof_date="2020-01-03", ticker="AAA", weight=1.0),
            _row(spread_id="AAA|BBB", asof_date="2020-01-03", ticker="BBB", weight=-0.5),
            _row(spread_id="CCC|DDD", asof_date="2020-01-10", ticker="CCC", weight=1.0),
            _row(spread_id="CCC|DDD", asof_date="2020-01-10", ticker="DDD", weight=-2.0),
        ]
        save_weights(rows, self.path)
        df = load_weights(self.path)
        lookup = weights_lookup_from_df(df)

        key1 = ("AAA|BBB", pd.Timestamp("2020-01-03"))
        key2 = ("CCC|DDD", pd.Timestamp("2020-01-10"))
        self.assertEqual(set(lookup.keys()), {key1, key2})
        self.assertEqual(lookup[key1], {"AAA": 1.0, "BBB": -0.5})
        self.assertEqual(lookup[key2], {"CCC": 1.0, "DDD": -2.0})

    def test_stores_all_legs_regardless_of_magnitude(self):
        """A near-zero weight must still get its own row -- no thresholding on write."""
        rows = [
            _row(ticker="AAA", weight=1.0),
            _row(ticker="BBB", weight=1e-32),
        ]
        save_weights(rows, self.path)
        df = load_weights(self.path)
        self.assertEqual(len(df), 2)
        self.assertIn(1e-32, df["weight"].to_list())


if __name__ == "__main__":
    unittest.main()
