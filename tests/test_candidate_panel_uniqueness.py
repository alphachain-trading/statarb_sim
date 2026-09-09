"""
Uniqueness assertion for candidates.parquet (Track F1, per the standing
Acceptance criterion: "Uniqueness asserted on every artifact write, with a
test per artifact"). save_candidate_panel_result must reject a panel with
duplicate candidate_id rows rather than silently persisting them --
candidate_id already encodes the full row identity
(group_id, candidate_type, weight_model, asof_datetime, spread_id) via
make_candidate_id, so a duplicate is a structural bug in the caller, not a
legitimate repeat.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.candidates.candidate_panel import CandidatePanelResult, save_candidate_panel_result


class TestCandidatePanelUniqueness(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_dir = Path(self._tmp.name)

    def test_duplicate_candidate_id_raises(self):
        panel = pd.DataFrame({
            "candidate_id": ["c1", "c1"],
            "spread_id": ["AAA|BBB", "AAA|BBB"],
            "asof_date": [pd.Timestamp("2020-01-01")] * 2,
            "group_id": ["g1", "g1"],
        })
        result = CandidatePanelResult(panel=panel, metadata={})
        with self.assertRaises(ValueError) as cm:
            save_candidate_panel_result(result=result, out_dir=self.out_dir, stem="test")
        self.assertIn("c1", str(cm.exception))
        self.assertFalse((self.out_dir / "test.panel.parquet").exists())

    def test_unique_candidate_ids_write_successfully(self):
        panel = pd.DataFrame({
            "candidate_id": ["c1", "c2"],
            "spread_id": ["AAA|BBB", "CCC|DDD"],
            "asof_date": [pd.Timestamp("2020-01-01")] * 2,
            "group_id": ["g1", "g1"],
        })
        result = CandidatePanelResult(panel=panel, metadata={})
        save_candidate_panel_result(result=result, out_dir=self.out_dir, stem="test")
        self.assertTrue((self.out_dir / "test.panel.parquet").exists())

    def test_empty_panel_does_not_raise(self):
        result = CandidatePanelResult(panel=pd.DataFrame(), metadata={})
        save_candidate_panel_result(result=result, out_dir=self.out_dir, stem="test")
        self.assertTrue((self.out_dir / "test.panel.parquet").exists())


if __name__ == "__main__":
    unittest.main()
