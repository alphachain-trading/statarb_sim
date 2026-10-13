"""The synthetic universe must be deterministic and structurally valid."""

import unittest

import pandas as pd

from src.data.returns import build_group_return_bundle
from tests._synthetic import make_umd


class TestSyntheticUniverse(unittest.TestCase):
    def test_same_seed_same_prices(self):
        pd.testing.assert_frame_equal(make_umd(seed=3).prices, make_umd(seed=3).prices)

    def test_different_seed_different_prices(self):
        a, b = make_umd(seed=3).prices, make_umd(seed=4).prices
        self.assertFalse(a.equals(b))

    def test_group_structure(self):
        umd = make_umd({"alpha": 2, "beta": 1})
        self.assertEqual(len(umd.members("alpha")), 4)
        self.assertEqual(len(umd.members("beta")), 2)
        bundle = build_group_return_bundle(umd, "alpha")
        self.assertEqual(bundle.member_returns.shape[1], 4)
        self.assertFalse(bundle.aligned_returns.isna().any().any())


if __name__ == "__main__":
    unittest.main()
