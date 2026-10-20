"""
Loaders bound to market snapshots (issue #2), always pinned explicitly (#12).

`_load_umd` used to load each group from UniverseDataLoader's live cache and
merge with first-group-wins on shared tickers (e.g. a benchmark like SPY),
so results depended on DataConfig.groups order. Bound to one frozen
snapshot, the result must not depend on that order. The merge's guards
(#13: shared tickers agree across groups; #61: a ticker is a member of one
group only) are tested through load_snapshot_market_data, the real entry
point, against minted snapshots.

No network: downloads are mocked at UniverseDataLoader._download_prices and
the universe/snapshot roots point at a temp dir.
"""

import shutil
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.data.market_snapshot import (
    SHARED_TICKER_RTOL,
    ensure_market_snapshot,
    list_market_snapshots,
    load_snapshot_market_data,
)
from src.data.universe_loader import UniverseDataLoader
from src.simulator.config import DataConfig, GroupDataSource, SimulatorConfig
from src.simulator.simulator_factory import _load_umd, bind_market_snapshot
from tests._synthetic import build_sim_config

UNIVERSE = "bind_v1"
SHARED = "SHR"  # benchmark shared by both groups, like SPY in sp500_v1

# Every download call gets its own price level for group members. SHARED is
# downloaded once per group (like SPY); its copies agree unless a test sets
# shared_drift (relative, per download call after the first) or
# shared_nan_on_call (blanks one row in that call's copy) -- the
# adjustment-vintage drift the merge guard must catch (#13).
_calls: dict[str, float] = {"n": 0, "shared_drift": 0.0, "shared_nan_on_call": 0}


def _fake_download_prices(self, symbols):
    _calls["n"] += 1
    level = 100.0 * _calls["n"]
    dates = pd.bdate_range("2020-01-01", periods=6)
    frames = []
    for i, sym in enumerate(sorted(symbols)):
        if sym == SHARED:
            scale = 1.0 + _calls["shared_drift"] * (_calls["n"] - 1)
            close = [(500.0 + k) * scale for k in range(len(dates))]
        else:
            close = [level + i + k for k in range(len(dates))]
        df = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000.0},
            index=dates,
        )
        if sym == "LATE":
            df.iloc[:2] = float("nan")  # listed later: exercises the per-group NaN trim
        if sym == SHARED and _calls["n"] == _calls["shared_nan_on_call"]:
            df.iloc[3] = float("nan")
        df.columns = pd.MultiIndex.from_product([[sym], df.columns])
        frames.append(df)
    return pd.concat(frames, axis=1).sort_index(axis=1)


def _write_group(universe_path: Path, group_id: str, members: list[str]) -> None:
    raw = {
        "meta": {"universe_name": f"{group_id}_only_v1"},
        "loader_defaults": {"start_date": "2020-01-01", "end_date": None},
        "hierarchy": {"root": "root"},
        "groups": {
            "root": {"type": "root", "children": [group_id], "benchmark": SHARED},
            group_id: {"type": "sector", "parent": "root", "members": members},
        },
        "tickers": {**{t: {"kind": "equity"} for t in members}, SHARED: {"kind": "benchmark_etf"}},
    }
    (universe_path / f"{group_id}.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


class _SnapshotFixture(unittest.TestCase):
    """A two-group temp universe with one minted snapshot; downloads mocked."""

    def setUp(self):
        _calls.update(n=0, shared_drift=0.0, shared_nan_on_call=0)
        self.tmp = Path(tempfile.mkdtemp(prefix="snapshot_binding_test_"))
        self.universe_root = self.tmp / "universes"
        self.snapshot_root = self.tmp / "snapshots"
        universe_path = self.universe_root / UNIVERSE
        universe_path.mkdir(parents=True)
        _write_group(universe_path, "g1", ["AAA", "LATE"])
        _write_group(universe_path, "g2", ["BBB", "CCC"])

        self.stack = ExitStack()
        self.addCleanup(self._cleanup)
        self.stack.enter_context(
            patch.object(
                UniverseDataLoader,
                "_download_prices",
                autospec=True,
                side_effect=_fake_download_prices,
            )
        )
        self.stack.enter_context(
            patch("src.data.market_snapshot.CONFIG_UNIVERSE", self.universe_root)
        )
        self.stack.enter_context(
            patch("src.data.market_snapshot.DATA_MARKET_SNAPSHOTS", self.snapshot_root)
        )
        self.snapshot_id = ensure_market_snapshot(UNIVERSE, progress=False)[0]

    def _cleanup(self):
        self.stack.close()
        for path in self.tmp.rglob("*"):
            path.chmod(0o777)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _data_cfg(self, group_ids: list[str]) -> DataConfig:
        return DataConfig(
            groups=[
                GroupDataSource(
                    universe_config_name=f"{UNIVERSE}/{g}.yaml", candidate_panel_stem=""
                )
                for g in group_ids
            ],
            universe_name=UNIVERSE,
            snapshot_id=self.snapshot_id,
            data_path=str(self.tmp / "live_cache"),
        )


class TestLoadUmdBoundToSnapshot(_SnapshotFixture):
    def test_group_order_does_not_change_market_data(self):
        forward = _load_umd(self._data_cfg(["g1", "g2"]))
        reverse = _load_umd(self._data_cfg(["g2", "g1"]))
        pd.testing.assert_frame_equal(forward.prices, reverse.prices)
        pd.testing.assert_frame_equal(forward.ticker_info, reverse.ticker_info)
        pd.testing.assert_frame_equal(forward.group_info, reverse.group_info)
        pd.testing.assert_frame_equal(forward.membership, reverse.membership)

    def test_start_after_nan_trims_per_group(self):
        umd = _load_umd(self._data_cfg(["g1", "g2"]))
        # g1 starts after LATE's two leading NaN rows; g2 keeps every row, so
        # its members are complete over the full range after the outer join.
        self.assertEqual(len(umd.prices), 6)
        self.assertTrue(umd.prices[("BBB", "Close")].notna().all())
        self.assertEqual(int(umd.prices[("AAA", "Close")].isna().sum()), 2)


class TestSnapshotMergeGuards(_SnapshotFixture):
    """The merge guards, through load_snapshot_market_data (#13, #61)."""

    def _remint(self, **calls: float) -> str:
        _calls.update(n=0, **calls)
        return ensure_market_snapshot(UNIVERSE, force=True, progress=False)[0]

    def _load(self, snapshot_id: str, group_ids: list[str]):
        return load_snapshot_market_data(
            snapshot_id, group_ids, start_after_nan=True, check_for_corruptions=False
        )

    def test_shared_ticker_within_tolerance_loads(self):
        umd = self._load(self._remint(shared_drift=SHARED_TICKER_RTOL / 10), ["g1", "g2"])
        self.assertIn(SHARED, umd.tickers())

    def test_shared_ticker_drift_raises_naming_ticker_and_groups(self):
        snapshot_id = self._remint(shared_drift=SHARED_TICKER_RTOL * 10)
        with self.assertRaisesRegex(ValueError, f"shared ticker '{SHARED}'.*'g1' and 'g2'"):
            self._load(snapshot_id, ["g1", "g2"])

    def test_shared_ticker_nan_pattern_mismatch_raises(self):
        snapshot_id = self._remint(shared_nan_on_call=2)
        with self.assertRaisesRegex(ValueError, "1 NaN-pattern"):
            self._load(snapshot_id, ["g1", "g2"])

    def test_ticker_member_of_two_groups_raises(self):
        _write_group(self.universe_root / UNIVERSE, "g3", ["AAA", "DDD"])  # AAA also in g1
        # A fresh snapshot root: the changed universe may not extend the fixture's.
        self.stack.enter_context(
            patch("src.data.market_snapshot.DATA_MARKET_SNAPSHOTS", self.tmp / "snapshots_g3")
        )
        snapshot_id = self._remint()
        with self.assertRaisesRegex(ValueError, r"more than one group: \{'AAA'"):
            self._load(snapshot_id, ["g1", "g3"])

    def test_disjoint_members_with_shared_benchmark_load(self):
        umd = self._load(self.snapshot_id, ["g1", "g2"])
        self.assertEqual(sorted(umd.membership["ticker"]), ["AAA", "BBB", "CCC", "LATE"])


class TestBindMarketSnapshot(_SnapshotFixture):
    """bind_market_snapshot, against the same temp universe and snapshot root."""

    def _pinned(self, **data_overrides) -> SimulatorConfig:
        cfg: SimulatorConfig = build_sim_config(["g1"])
        return replace(cfg, data=replace(self._data_cfg(["g1"]), **data_overrides))

    def _snapshots(self) -> list[str]:
        return list_market_snapshots(UNIVERSE, snapshot_root=self.snapshot_root)

    def test_pinned_snapshot_is_returned_unchanged(self):
        pinned = self._pinned()
        self.assertIs(bind_market_snapshot(pinned), pinned)

    def test_missing_pinned_snapshot_raises_without_minting(self):
        missing = f"{UNIVERSE}_20000101T000000000000"
        with (
            patch("src.data.market_snapshot._mint_snapshot") as mint,
            self.assertRaises(FileNotFoundError),
        ):
            bind_market_snapshot(self._pinned(snapshot_id=missing))
        mint.assert_not_called()
        self.assertEqual(self._snapshots(), [self.snapshot_id])

    def test_pinned_snapshot_of_another_universe_raises(self):
        other = self.snapshot_id.replace(UNIVERSE, "other_v1", 1)
        with self.assertRaises(ValueError):
            bind_market_snapshot(self._pinned(snapshot_id=other))

    def test_data_force_download_raises_without_minting(self):
        with (
            patch("src.data.market_snapshot._mint_snapshot") as mint,
            self.assertRaisesRegex(ValueError, "never mints"),
        ):
            bind_market_snapshot(self._pinned(force_download=True))
        mint.assert_not_called()
        self.assertEqual(self._snapshots(), [self.snapshot_id])

    def test_candidate_generation_force_download_raises(self):
        cfg = self._pinned()
        assert cfg.candidate_generation is not None
        cg = replace(cfg.candidate_generation, force_download=True)
        with self.assertRaises(ValueError):
            bind_market_snapshot(replace(cfg, candidate_generation=cg))


class TestDataConfigRequiresDataSelection(unittest.TestCase):
    """No implicit universe and no implicit snapshot (#12)."""

    def test_omitting_universe_name_raises(self):
        with self.assertRaisesRegex(TypeError, "universe_name"):
            DataConfig(snapshot_id="bind_v1_20260101T000000000000")  # type: ignore[call-arg]  # the omission under test

    def test_omitting_snapshot_id_raises(self):
        with self.assertRaisesRegex(TypeError, "snapshot_id"):
            DataConfig(universe_name=UNIVERSE)  # type: ignore[call-arg]  # the omission under test


if __name__ == "__main__":
    unittest.main()
