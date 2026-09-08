"""
Tests for the market snapshot layer (Track C).

Downloads are mocked at UniverseDataLoader._download_prices so these tests
never hit the network; everything downstream of that (normalization,
persistence, hashing, manifest, immutability) runs for real.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.market_snapshot import (
    SnapshotExistsError,
    SnapshotHashMismatchError,
    SnapshotResyncGuardError,
    _download_group,
    _finalize_snapshot,
    _group_yaml_path,
    ensure_market_snapshot,
    load_market_snapshot,
)
from src.data.universe_loader import UniverseDataLoader
from src.settings import CONFIG_UNIVERSE


def _fake_download_prices(self, symbols):
    dates = pd.bdate_range("2020-01-01", periods=5)
    frames = []
    for i, sym in enumerate(symbols):
        cols = {
            "Open": 100.0 + i,
            "High": 101.0 + i,
            "Low": 99.0 + i,
            "Close": 100.5 + i,
            "Volume": 1000.0 + i,
        }
        df = pd.DataFrame({f: [v] * len(dates) for f, v in cols.items()}, index=dates)
        df.columns = pd.MultiIndex.from_product([[sym], df.columns])
        frames.append(df)
    return pd.concat(frames, axis=1).sort_index(axis=1)


_MOCK_DOWNLOAD = patch.object(
    UniverseDataLoader, "_download_prices", autospec=True, side_effect=_fake_download_prices
)


class MarketSnapshotTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp(prefix="market_snapshot_test_"))
        self.snapshot_root = self.tmp_root / "snapshots"

    def tearDown(self):
        # Snapshot files are chmod read-only; make writable again before rmtree.
        for path in self.tmp_root.rglob("*"):
            path.chmod(0o777)
        shutil.rmtree(self.tmp_root, ignore_errors=True)


class TestEnsureMarketSnapshot(MarketSnapshotTestBase):
    def test_fresh_download_returns_list_of_one(self):
        with _MOCK_DOWNLOAD:
            ids = ensure_market_snapshot(
                "test_v1", ["materials"], snapshot_root=self.snapshot_root, progress=False
            )
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids[0].startswith("test_v1_"))
        self.assertTrue((self.snapshot_root / ids[0] / "manifest.json").exists())

    def test_repeat_call_same_groups_does_not_redownload(self):
        with _MOCK_DOWNLOAD as mock_dl:
            ids1 = ensure_market_snapshot(
                "test_v1", ["materials"], snapshot_root=self.snapshot_root, progress=False
            )
            calls_after_first = mock_dl.call_count
            ids2 = ensure_market_snapshot(
                "test_v1", ["materials"], snapshot_root=self.snapshot_root, progress=False
            )
        self.assertEqual(ids1, ids2)
        self.assertEqual(len(ids2), 1)
        self.assertEqual(mock_dl.call_count, calls_after_first)  # no new download

    def test_different_group_set_mints_a_second_snapshot(self):
        with _MOCK_DOWNLOAD:
            ensure_market_snapshot(
                "test_v1", ["materials"], snapshot_root=self.snapshot_root, progress=False
            )
            ids = ensure_market_snapshot(
                "test_v1", ["energy"], snapshot_root=self.snapshot_root, progress=False
            )
        self.assertEqual(len(ids), 2)
        # newest first
        self.assertGreater(ids[0].split("_")[-1], ids[1].split("_")[-1])

    def test_force_mints_even_when_group_set_matches(self):
        with _MOCK_DOWNLOAD as mock_dl:
            ensure_market_snapshot(
                "test_v1", ["materials"], snapshot_root=self.snapshot_root, progress=False
            )
            calls_after_first = mock_dl.call_count
            ids = ensure_market_snapshot(
                "test_v1", ["materials"], snapshot_root=self.snapshot_root, force=True, progress=False
            )
        self.assertEqual(len(ids), 2)
        self.assertGreater(mock_dl.call_count, calls_after_first)

    def test_empty_groups_raises(self):
        with self.assertRaises(ValueError):
            ensure_market_snapshot("test_v1", [], snapshot_root=self.snapshot_root)


class TestSnapshotImmutability(MarketSnapshotTestBase):
    def test_write_to_existing_snapshot_directory_raises(self):
        target_dir = self.snapshot_root / "test_v1_20260101T000000"
        target_dir.mkdir(parents=True)
        (target_dir / "sentinel.txt").write_text("pre-existing")

        tmp_dir = Path(tempfile.mkdtemp(dir=self.snapshot_root, prefix=".tmp-"))
        (tmp_dir / "manifest.json").write_text("{}")

        with self.assertRaises(SnapshotExistsError):
            _finalize_snapshot(tmp_dir, target_dir)

        # Target untouched, tmp_dir left for the caller (mint's except-clause
        # is what cleans it up in the real path).
        self.assertEqual((target_dir / "sentinel.txt").read_text(), "pre-existing")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_snapshot_files_are_read_only_after_mint(self):
        with _MOCK_DOWNLOAD:
            ids = ensure_market_snapshot(
                "test_v1", ["materials"], snapshot_root=self.snapshot_root, progress=False
            )
        snapshot_dir = self.snapshot_root / ids[0]
        prices_path = next(snapshot_dir.glob("*/prices_daily.parquet"))
        with self.assertRaises(PermissionError):
            prices_path.write_bytes(b"nope")


class TestResyncGuard(MarketSnapshotTestBase):
    def test_download_group_raises_if_cache_already_present(self):
        yaml_path = _group_yaml_path(CONFIG_UNIVERSE, "materials")
        tmp_dir = Path(tempfile.mkdtemp(dir=self.tmp_root, prefix="group-dl-"))
        with _MOCK_DOWNLOAD:
            # First call populates a real cache under tmp_dir/materials_only_v1.
            _download_group("materials", yaml_path, tmp_dir, progress=False)
            # Second call against the same tmp_dir must refuse rather than
            # silently taking UniverseDataLoader.load's resync branch.
            with self.assertRaises(SnapshotResyncGuardError):
                _download_group("materials", yaml_path, tmp_dir, progress=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestLoadMarketSnapshot(MarketSnapshotTestBase):
    def _mint(self, groups=("materials",)):
        with _MOCK_DOWNLOAD:
            ids = ensure_market_snapshot(
                "test_v1", list(groups), snapshot_root=self.snapshot_root, progress=False
            )
        return ids[0]

    def test_load_round_trips(self):
        snapshot_id = self._mint()
        umd = load_market_snapshot(snapshot_id, snapshot_root=self.snapshot_root)
        self.assertGreater(len(umd.tickers()), 0)

    def test_mutated_prices_file_raises_on_load(self):
        snapshot_id = self._mint()
        snapshot_dir = self.snapshot_root / snapshot_id
        prices_path = next(snapshot_dir.glob("*/prices_daily.parquet"))
        prices_path.chmod(0o644)
        with prices_path.open("ab") as f:
            f.write(b"corruption")

        with self.assertRaises(SnapshotHashMismatchError):
            load_market_snapshot(snapshot_id, snapshot_root=self.snapshot_root)

    def test_mutated_membership_file_raises_on_load(self):
        snapshot_id = self._mint()
        snapshot_dir = self.snapshot_root / snapshot_id
        membership_path = next(snapshot_dir.glob("*/membership.parquet"))
        membership_path.chmod(0o644)
        df = pd.read_parquet(membership_path)
        df = pd.concat([df, pd.DataFrame({"group_id": ["materials"], "ticker": ["ZZZFAKE"]})])
        df.to_parquet(membership_path)

        with self.assertRaises(SnapshotHashMismatchError):
            load_market_snapshot(snapshot_id, snapshot_root=self.snapshot_root)

    def test_never_resolves_a_partial_id(self):
        snapshot_id = self._mint()
        partial = snapshot_id.rsplit("T", 1)[0]  # drop the time-of-day half of dl_ts
        with self.assertRaises(ValueError):
            load_market_snapshot(partial, snapshot_root=self.snapshot_root)
        # sanity: the full id still resolves fine
        load_market_snapshot(snapshot_id, snapshot_root=self.snapshot_root)

    def test_unknown_full_id_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_market_snapshot("test_v1_20260101T000000000000", snapshot_root=self.snapshot_root)


if __name__ == "__main__":
    unittest.main()
