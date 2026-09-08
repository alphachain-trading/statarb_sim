"""
Tests for the market snapshot layer (Track C).

Downloads are mocked at UniverseDataLoader._download_prices so these tests
never hit the network; everything downstream of that (normalization,
persistence, hashing, manifest, immutability, universe-change detection,
missing-ticker aggregation) runs for real. Universe directories are
synthetic fixtures written under a temp dir — never the real
config/universes/sp500_v1 — so tests can freely add/remove/edit groups and
tickers without touching committed config.
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.market_snapshot import (
    IncompleteUniverseDownloadError,
    SnapshotExistsError,
    SnapshotHashMismatchError,
    SnapshotResyncGuardError,
    UniverseChangedError,
    _download_group,
    _finalize_snapshot,
    ensure_market_snapshot,
    load_market_snapshot,
)
from src.data.universe_loader import UniverseDataLoader


# Symbols in this set are dropped from a fake download response, mimicking
# yfinance returning no data for a ticker. Tests set/clear it directly.
_MISSING_SYMBOLS: set[str] = set()


def _fake_download_prices(self, symbols):
    present = [s for s in symbols if s not in _MISSING_SYMBOLS]
    if not present:
        raise RuntimeError("All ticker downloads failed.")

    dates = pd.bdate_range("2020-01-01", periods=5)
    frames = []
    for i, sym in enumerate(present):
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


def _write_universe(universe_root: Path, universe_version: str, groups: dict[str, list[str]]) -> Path:
    """Write a synthetic universe directory: {universe_root}/{universe_version}/{group_id}.yaml."""
    universe_path = universe_root / universe_version
    universe_path.mkdir(parents=True, exist_ok=True)
    for group_id, tickers in groups.items():
        raw = {
            "meta": {"universe_name": f"{group_id}_meta"},
            "loader_defaults": {"start_date": "2020-01-01", "end_date": None},
            "hierarchy": {"root": "root"},
            "groups": {
                "root": {"type": "root", "children": [group_id]},
                group_id: {"type": "sector", "parent": "root", "members": list(tickers)},
            },
            "tickers": {t: {"kind": "equity"} for t in tickers},
        }
        (universe_path / f"{group_id}.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return universe_path


class MarketSnapshotTestBase(unittest.TestCase):
    def setUp(self):
        _MISSING_SYMBOLS.clear()
        self.tmp_root = Path(tempfile.mkdtemp(prefix="market_snapshot_test_"))
        self.universe_root = self.tmp_root / "config_universes"
        self.snapshot_root = self.tmp_root / "snapshots"

    def tearDown(self):
        _MISSING_SYMBOLS.clear()
        # Snapshot files are chmod read-only; make writable again before rmtree.
        for path in self.tmp_root.rglob("*"):
            path.chmod(0o777)
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _mint(self, universe_version="test_v1", groups=None, **kwargs):
        groups = groups if groups is not None else {"g1": ["AAA", "BBB"]}
        _write_universe(self.universe_root, universe_version, groups)
        with _MOCK_DOWNLOAD:
            return ensure_market_snapshot(
                universe_version,
                universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root,
                progress=False,
                **kwargs,
            )


class TestEnsureMarketSnapshot(MarketSnapshotTestBase):
    def test_fresh_download_returns_list_of_one(self):
        ids = self._mint()
        self.assertEqual(len(ids), 1)
        self.assertTrue(ids[0].startswith("test_v1_"))
        self.assertTrue((self.snapshot_root / ids[0] / "manifest.json").exists())

    def test_first_mint_of_new_universe_succeeds_with_no_prior_snapshot(self):
        # No snapshot exists yet for this universe_version, so the
        # immutability check has nothing to compare against and must not
        # block the mint (see module docstring: "the first mint is
        # definitional").
        ids = self._mint(universe_version="brand_new_v1")
        self.assertEqual(len(ids), 1)

    def test_repeat_call_does_not_redownload(self):
        universe_version = "test_v1"
        _write_universe(self.universe_root, universe_version, {"g1": ["AAA", "BBB"]})
        with _MOCK_DOWNLOAD as mock_dl:
            ids1 = ensure_market_snapshot(
                universe_version, universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root, progress=False,
            )
            calls_after_first = mock_dl.call_count
            ids2 = ensure_market_snapshot(
                universe_version, universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root, progress=False,
            )
        self.assertEqual(ids1, ids2)
        self.assertEqual(len(ids2), 1)
        self.assertEqual(mock_dl.call_count, calls_after_first)  # no new download

    def test_force_mints_even_when_snapshot_exists(self):
        universe_version = "test_v1"
        _write_universe(self.universe_root, universe_version, {"g1": ["AAA", "BBB"]})
        with _MOCK_DOWNLOAD as mock_dl:
            ensure_market_snapshot(
                universe_version, universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root, progress=False,
            )
            calls_after_first = mock_dl.call_count
            ids = ensure_market_snapshot(
                universe_version, universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root, force=True, progress=False,
            )
        self.assertEqual(len(ids), 2)
        self.assertGreater(mock_dl.call_count, calls_after_first)

    def test_unknown_universe_version_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ensure_market_snapshot(
                "no_such_universe", universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root, progress=False,
            )

    def test_universe_directory_with_no_yaml_files_raises(self):
        (self.universe_root / "empty_v1").mkdir(parents=True)
        with self.assertRaises(ValueError):
            ensure_market_snapshot(
                "empty_v1", universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root, progress=False,
            )


class TestUniverseImmutability(MarketSnapshotTestBase):
    def test_edited_group_yaml_raises_at_mint_against_existing_snapshot(self):
        universe_version = "test_v1"
        self._mint(universe_version=universe_version, groups={"g1": ["AAA", "BBB"]})

        # Edit the group's yaml after the fact — add a ticker to membership.
        _write_universe(self.universe_root, universe_version, {"g1": ["AAA", "BBB", "CCC"]})

        with _MOCK_DOWNLOAD:
            with self.assertRaises(UniverseChangedError) as cm:
                ensure_market_snapshot(
                    universe_version, universe_dir=self.universe_root,
                    snapshot_root=self.snapshot_root, force=True, progress=False,
                )
        self.assertIn("g1", str(cm.exception))

    def test_added_group_raises_at_mint_against_existing_snapshot(self):
        universe_version = "test_v1"
        self._mint(universe_version=universe_version, groups={"g1": ["AAA", "BBB"]})

        _write_universe(self.universe_root, universe_version, {
            "g1": ["AAA", "BBB"], "g2": ["ZZZ"],
        })

        with _MOCK_DOWNLOAD:
            with self.assertRaises(UniverseChangedError) as cm:
                ensure_market_snapshot(
                    universe_version, universe_dir=self.universe_root,
                    snapshot_root=self.snapshot_root, force=True, progress=False,
                )
        self.assertIn("added", str(cm.exception))

    def test_removed_group_raises_at_mint_against_existing_snapshot(self):
        universe_version = "test_v1"
        self._mint(universe_version=universe_version, groups={
            "g1": ["AAA", "BBB"], "g2": ["ZZZ"],
        })

        shutil.rmtree(self.universe_root / universe_version)
        _write_universe(self.universe_root, universe_version, {"g1": ["AAA", "BBB"]})

        with _MOCK_DOWNLOAD:
            with self.assertRaises(UniverseChangedError) as cm:
                ensure_market_snapshot(
                    universe_version, universe_dir=self.universe_root,
                    snapshot_root=self.snapshot_root, force=True, progress=False,
                )
        self.assertIn("removed", str(cm.exception))

    def test_unchanged_universe_does_not_raise_on_forced_remint(self):
        universe_version = "test_v1"
        self._mint(universe_version=universe_version, groups={"g1": ["AAA", "BBB"]})
        # Same directory content, forced remint: must succeed (new dl_ts,
        # same yaml content — this is the "corporate action re-adjusts
        # history" case the model explicitly allows).
        with _MOCK_DOWNLOAD:
            ids = ensure_market_snapshot(
                universe_version, universe_dir=self.universe_root,
                snapshot_root=self.snapshot_root, force=True, progress=False,
            )
        self.assertEqual(len(ids), 2)


class TestIncompleteDownload(MarketSnapshotTestBase):
    def test_missing_ticker_aborts_mint_and_lists_all(self):
        universe_version = "test_v1"
        _write_universe(self.universe_root, universe_version, {
            "g1": ["AAA", "BBB"],
            "g2": ["CCC", "DDD"],
        })
        _MISSING_SYMBOLS.update({"BBB", "DDD"})

        with _MOCK_DOWNLOAD:
            with self.assertRaises(IncompleteUniverseDownloadError) as cm:
                ensure_market_snapshot(
                    universe_version, universe_dir=self.universe_root,
                    snapshot_root=self.snapshot_root, progress=False,
                )
        msg = str(cm.exception)
        # Every missing ticker across every group must be listed — not just
        # the first group's gap.
        self.assertIn("BBB", msg)
        self.assertIn("DDD", msg)
        self.assertIn("g1", msg)
        self.assertIn("g2", msg)

        # And nothing was written: an aborted mint leaves no snapshot behind.
        self.assertEqual(list(self.snapshot_root.glob(f"{universe_version}_*")), [])

    def test_no_missing_tickers_succeeds(self):
        ids = self._mint(groups={"g1": ["AAA", "BBB"]})
        self.assertEqual(len(ids), 1)


class TestSnapshotImmutability(MarketSnapshotTestBase):
    def test_write_to_existing_snapshot_directory_raises(self):
        target_dir = self.snapshot_root / "test_v1_20260101T000000000000"
        target_dir.mkdir(parents=True)
        (target_dir / "sentinel.txt").write_text("pre-existing")

        tmp_dir = Path(tempfile.mkdtemp(dir=self.snapshot_root, prefix=".tmp-"))
        (tmp_dir / "manifest.json").write_text("{}")

        with self.assertRaises(SnapshotExistsError):
            _finalize_snapshot(tmp_dir, target_dir)

        self.assertEqual((target_dir / "sentinel.txt").read_text(), "pre-existing")
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_snapshot_files_are_read_only_after_mint(self):
        ids = self._mint()
        snapshot_dir = self.snapshot_root / ids[0]
        prices_path = next(snapshot_dir.glob("*/prices_daily.parquet"))
        with self.assertRaises(PermissionError):
            prices_path.write_bytes(b"nope")


class TestResyncGuard(MarketSnapshotTestBase):
    def test_download_group_raises_if_cache_already_present(self):
        universe_path = _write_universe(self.universe_root, "test_v1", {"g1": ["AAA", "BBB"]})
        yaml_path = universe_path / "g1.yaml"
        tmp_dir = Path(tempfile.mkdtemp(dir=self.tmp_root, prefix="group-dl-"))
        with _MOCK_DOWNLOAD:
            _download_group("g1", yaml_path, tmp_dir, progress=False)
            with self.assertRaises(SnapshotResyncGuardError):
                _download_group("g1", yaml_path, tmp_dir, progress=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestLoadMarketSnapshot(MarketSnapshotTestBase):
    def test_load_round_trips(self):
        snapshot_id = self._mint()[0]
        umd = load_market_snapshot(snapshot_id, snapshot_root=self.snapshot_root)
        self.assertGreater(len(umd.tickers()), 0)

    def test_mutated_prices_file_raises_on_load(self):
        snapshot_id = self._mint()[0]
        snapshot_dir = self.snapshot_root / snapshot_id
        prices_path = next(snapshot_dir.glob("*/prices_daily.parquet"))
        prices_path.chmod(0o644)
        with prices_path.open("ab") as f:
            f.write(b"corruption")

        with self.assertRaises(SnapshotHashMismatchError):
            load_market_snapshot(snapshot_id, snapshot_root=self.snapshot_root)

    def test_mutated_membership_file_raises_on_load(self):
        snapshot_id = self._mint()[0]
        snapshot_dir = self.snapshot_root / snapshot_id
        membership_path = next(snapshot_dir.glob("*/membership.parquet"))
        membership_path.chmod(0o644)
        df = pd.read_parquet(membership_path)
        df = pd.concat([df, pd.DataFrame({"group_id": ["g1"], "ticker": ["ZZZFAKE"]})])
        df.to_parquet(membership_path)

        with self.assertRaises(SnapshotHashMismatchError):
            load_market_snapshot(snapshot_id, snapshot_root=self.snapshot_root)

    def test_never_resolves_a_partial_id(self):
        snapshot_id = self._mint()[0]
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
