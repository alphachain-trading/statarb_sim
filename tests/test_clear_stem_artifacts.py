"""
Regression test for stem-scoped artifact clearing before a fresh panel write.

create_pair_candidate_panel persists many panels (group x timescale) into one
shared directory, so a fresh write must clear only THIS stem's prior
artifacts — never sibling panels or the shared series/ folder. This guards
against both a reused stem keeping a stale residual_params.pkl and, in the
other direction, a directory-wide wipe destroying a batch's other panels.
"""
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.candidates.pair_candidate_panel_creator import _clear_stem_artifacts


def _write_panel(out_dir: Path, stem: str, content: str) -> None:
    """Mirror the production clear-then-write order for one panel stem."""
    _clear_stem_artifacts(out_dir, stem)
    (out_dir / f"{stem}.panel.parquet").write_text(f"panel:{content}")
    (out_dir / f"{stem}.meta.json").write_text(f"meta:{content}")
    (out_dir / f"{stem}_residual_params.pkl").write_text(f"params:{content}")


def _stem_files(out_dir: Path, stem: str) -> list[Path]:
    return [
        out_dir / f"{stem}.panel.parquet",
        out_dir / f"{stem}.meta.json",
        out_dir / f"{stem}_residual_params.pkl",
    ]


class TestClearStemArtifacts(unittest.TestCase):
    def test_stem_scoped_clear_leaves_siblings_and_series_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            # Two different panels written into the same shared directory.
            _write_panel(out_dir, "A", "v1")
            _write_panel(out_dir, "B", "v1")

            # A separate stage writes the shared series/ tree (not stem-scoped).
            series_stock = out_dir / "series" / "stock" / "AAPL.parquet"
            series_stock.parent.mkdir(parents=True, exist_ok=True)
            series_stock.write_text("series-data")

            # Both coexist untouched.
            for stem in ("A", "B"):
                for p in _stem_files(out_dir, stem):
                    self.assertTrue(p.exists(), f"{p} should exist")

            # Rewrite stem A with different content.
            _write_panel(out_dir, "A", "v2")

            # A's three files replaced with the new content.
            self.assertEqual((out_dir / "A.panel.parquet").read_text(), "panel:v2")
            self.assertEqual((out_dir / "A.meta.json").read_text(), "meta:v2")
            self.assertEqual((out_dir / "A_residual_params.pkl").read_text(), "params:v2")

            # Sibling stem B untouched — still original content.
            self.assertEqual((out_dir / "B.panel.parquet").read_text(), "panel:v1")
            self.assertEqual((out_dir / "B.meta.json").read_text(), "meta:v1")
            self.assertEqual((out_dir / "B_residual_params.pkl").read_text(), "params:v1")

            # Shared series/ content untouched.
            self.assertTrue(series_stock.exists())
            self.assertEqual(series_stock.read_text(), "series-data")

    def test_returns_only_paths_actually_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            # Nothing present yet — first write removes nothing.
            self.assertEqual(_clear_stem_artifacts(out_dir, "A"), [])

            _write_panel(out_dir, "A", "v1")
            # Partial: drop one of the three, confirm only the present ones report.
            (out_dir / "A.meta.json").unlink()
            removed = _clear_stem_artifacts(out_dir, "A")
            names = sorted(p.name for p in removed)
            self.assertEqual(names, ["A.panel.parquet", "A_residual_params.pkl"])


if __name__ == "__main__":
    unittest.main()
