"""
Multi-group ticker assert (Track E).

A ticker must belong to exactly one group: residual fits are group-scoped, so a
ticker in two groups would get two different residual series, and cross-group
analysis would have to pick one or double-count. `_merge_umds` is where
universe composition for a multi-group run happens (merging each group's
`UniverseMarketData.membership`), so that is where the assert lives.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.universe_marketdata import UniverseMarketData
from src.simulator.simulator_factory import _merge_umds


def _make_umd(group_id: str, tickers: list[str]) -> UniverseMarketData:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    cols = pd.MultiIndex.from_tuples([(t, "Close") for t in tickers])
    prices = pd.DataFrame(1.0, index=dates, columns=cols)
    ticker_info = pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    group_info = pd.DataFrame(index=pd.Index([group_id], name="group_id"))
    membership = pd.DataFrame({
        "group_id": [group_id] * len(tickers),
        "ticker": tickers,
    })
    return UniverseMarketData(
        prices=prices,
        ticker_info=ticker_info,
        group_info=group_info,
        membership=membership,
    )


class TestMultiGroupTickerAssert(unittest.TestCase):
    def test_ticker_in_two_groups_raises(self):
        umd_a = _make_umd("g1", ["AAA", "BBB"])
        umd_b = _make_umd("g2", ["AAA", "CCC"])  # AAA also in g1

        with self.assertRaises(ValueError) as cm:
            _merge_umds([umd_a, umd_b])

        msg = str(cm.exception)
        self.assertIn("AAA", msg)
        self.assertIn("group-scoped", msg)
        self.assertIn("double-count", msg)

    def test_disjoint_groups_do_not_raise(self):
        umd_a = _make_umd("g1", ["AAA", "BBB"])
        umd_b = _make_umd("g2", ["CCC", "DDD"])

        merged = _merge_umds([umd_a, umd_b])

        self.assertEqual(sorted(merged.tickers()), ["AAA", "BBB", "CCC", "DDD"])
        self.assertEqual(sorted(merged.membership["ticker"]), ["AAA", "BBB", "CCC", "DDD"])


if __name__ == "__main__":
    unittest.main()
