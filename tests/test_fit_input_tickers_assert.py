"""
Tests for the fit_input_tickers load-time "different universe" guard
(Track F1 commit 6).

_assert_fit_input_tickers_available compares each loaded panel's
fit_input_tickers metadata (the union, across every daily fit date, of a
residual model's realized input tickers) against the tickers actually
present in the loaded UniverseMarketData -- catching a UMD that is a
different universe than the one the panel was built against, before a
residual fit fails partway through with a confusing KeyError.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.simulator_factory import _assert_fit_input_tickers_available


def _umd(tickers):
    return SimpleNamespace(tickers=lambda: list(tickers))


class TestFitInputTickersAssert(unittest.TestCase):
    def test_passes_when_all_tickers_present(self):
        umd = _umd(["AAA", "BBB", "CCC"])
        metadata_by_key = {"": {"fit_input_tickers": ["AAA", "BBB"]}}
        _assert_fit_input_tickers_available(umd, metadata_by_key)  # no raise

    def test_raises_naming_missing_tickers_and_residual_key(self):
        umd = _umd(["AAA", "CCC"])
        metadata_by_key = {"exp_hl20_rf": {"fit_input_tickers": ["AAA", "BBB", "DDD"]}}
        with self.assertRaises(ValueError) as cm:
            _assert_fit_input_tickers_available(umd, metadata_by_key)
        msg = str(cm.exception)
        self.assertIn("BBB", msg)
        self.assertIn("DDD", msg)
        self.assertNotIn("'AAA'", msg)  # present ticker not reported as missing
        self.assertIn("exp_hl20_rf", msg)

    def test_skips_metadata_without_fit_input_tickers_key(self):
        """Panel built before F1 commit 6 has no fit_input_tickers -- nothing to check, not an error."""
        umd = _umd(["AAA"])
        metadata_by_key = {"": {"group_id": "materials"}}
        _assert_fit_input_tickers_available(umd, metadata_by_key)  # no raise

    def test_skips_empty_fit_input_tickers(self):
        umd = _umd(["AAA"])
        metadata_by_key = {"": {"fit_input_tickers": []}}
        _assert_fit_input_tickers_available(umd, metadata_by_key)  # no raise

    def test_checks_every_residual_key_independently(self):
        umd = _umd(["AAA", "BBB"])
        metadata_by_key = {
            "rkey1": {"fit_input_tickers": ["AAA", "BBB"]},
            "rkey2": {"fit_input_tickers": ["AAA", "ZZZ"]},
        }
        with self.assertRaises(ValueError) as cm:
            _assert_fit_input_tickers_available(umd, metadata_by_key)
        self.assertIn("rkey2", str(cm.exception))
        self.assertIn("ZZZ", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
