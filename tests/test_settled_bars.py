"""Tests for the unsettled-bar guard (issue #1). No network: yf.download is mocked."""

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from src.data.market_snapshot import ensure_market_snapshot, load_market_snapshot
from src.data.settled_bars import drop_unsettled_bars, require_daily_interval
from src.data.universe_config import UniverseConfig
from src.data.universe_loader import UniverseDataLoader

NY = ZoneInfo("America/New_York")


def _frame(dates: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame({"Close": range(len(idx))}, index=idx, dtype=float)


class TestDropUnsettledBars(unittest.TestCase):
    def test_same_day_bar_dropped_mid_session(self):
        df = _frame(["2024-05-14", "2024-05-15"])
        out = drop_unsettled_bars(df, datetime(2024, 5, 15, 11, 0, tzinfo=NY))
        self.assertEqual(list(out.index), [pd.Timestamp("2024-05-14")])

    def test_same_day_bar_dropped_after_close(self):
        # Simple rule: no buffer, same-day bars are always dropped.
        df = _frame(["2024-05-14", "2024-05-15"])
        out = drop_unsettled_bars(df, datetime(2024, 5, 15, 23, 59, tzinfo=NY))
        self.assertEqual(list(out.index), [pd.Timestamp("2024-05-14")])

    def test_past_bars_unchanged(self):
        df = _frame(["2024-05-13", "2024-05-14"])
        out = drop_unsettled_bars(df, datetime(2024, 5, 15, 0, 1, tzinfo=NY))
        pd.testing.assert_frame_equal(out, df)

    def test_weekend_now_keeps_friday(self):
        df = _frame(["2024-05-16", "2024-05-17"])
        out = drop_unsettled_bars(df, datetime(2024, 5, 18, 12, 0, tzinfo=NY))
        pd.testing.assert_frame_equal(out, df)

    def test_uses_new_york_date_not_utc(self):
        # 2024-05-16 02:00 UTC is still 2024-05-15 22:00 in New York.
        # A UTC-based rule would treat 05-15 as settled.
        df = _frame(["2024-05-14", "2024-05-15"])
        now = datetime(2024, 5, 16, 2, 0, tzinfo=ZoneInfo("UTC"))
        out = drop_unsettled_bars(df, now)
        self.assertEqual(list(out.index), [pd.Timestamp("2024-05-14")])

    def test_naive_now_is_new_york_time(self):
        df = _frame(["2024-05-14", "2024-05-15"])
        out = drop_unsettled_bars(df, datetime(2024, 5, 15, 9, 0))
        self.assertEqual(len(out), 1)

    def test_warns_when_dropping(self):
        df = _frame(["2024-05-15"])
        with patch("builtins.print") as mock_print:
            drop_unsettled_bars(df, datetime(2024, 5, 15, 12, 0, tzinfo=NY), "SPY")
        self.assertIn("SPY", mock_print.call_args[0][0])
        self.assertIn("2024-05-15", mock_print.call_args[0][0])


class TestRequireDailyInterval(unittest.TestCase):
    def test_daily_ok(self):
        require_daily_interval("1d")

    def test_intraday_raises(self):
        with self.assertRaises(ValueError):
            require_daily_interval("1h")


def _today_ny() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(NY).date())


def _fake_yf_download(tickers, **kwargs):
    """Settled bars for the past few days plus a provisional bar dated today."""
    today = _today_ny()
    idx = pd.DatetimeIndex([today - timedelta(days=d) for d in (3, 2, 1, 0)])
    return pd.DataFrame(
        {f: [100.0] * len(idx) for f in ("Open", "High", "Low", "Close", "Volume")}, index=idx
    )


def _write_universe(root: Path, version: str, tickers: list[str], interval: str = "1d") -> None:
    path = root / version
    path.mkdir(parents=True)
    raw = {
        "meta": {"universe_name": "g1_meta"},
        "loader_defaults": {"start_date": "2020-01-01", "end_date": None, "interval": interval},
        "hierarchy": {"root": "root"},
        "groups": {
            "root": {"type": "root", "children": ["g1"]},
            "g1": {"type": "sector", "parent": "root", "members": tickers},
        },
        "tickers": {t: {"kind": "equity"} for t in tickers},
    }
    (path / "g1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


class TestLoaderGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="settled_bars_test_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))

    def _loader(self, interval: str = "1d") -> UniverseDataLoader:
        _write_universe(self.tmp / "cfg", "v1", ["AAA", "BBB"], interval)
        config = UniverseConfig.from_yaml(self.tmp / "cfg" / "v1" / "g1.yaml")
        return UniverseDataLoader(config, data_path=self.tmp / "data")

    @patch("src.data.universe_loader.yf.download", side_effect=_fake_yf_download)
    def test_download_prices_drops_today(self, _):
        raw = self._loader()._download_prices(["AAA", "BBB"])
        self.assertLess(raw.index.max(), _today_ny())
        self.assertEqual(len(raw), 3)

    @patch("src.data.universe_loader.yf.download", side_effect=_fake_yf_download)
    def test_pool_download_drops_today(self, _):
        loader = self._loader()
        loader.load_tickers_to_pool(["AAA"])
        persisted = pd.read_parquet(loader.pool_dir / "AAA.parquet")
        self.assertLess(persisted.index.max(), _today_ny())
        self.assertEqual(len(persisted), 3)

    @patch("src.data.universe_loader.yf.download")
    def test_intraday_interval_raises_before_download(self, mock_dl):
        with self.assertRaises(ValueError):
            self._loader("1h")._download_prices(["AAA"])
        mock_dl.assert_not_called()

    @patch("src.data.universe_loader.yf.download")
    def test_all_bars_unsettled_counts_as_no_data(self, mock_dl):
        today = _today_ny()
        mock_dl.return_value = pd.DataFrame(
            {f: [1.0] for f in ("Open", "High", "Low", "Close", "Volume")},
            index=pd.DatetimeIndex([today]),
        )
        with self.assertRaises(RuntimeError):
            self._loader()._download_prices(["AAA"])


class TestMintGuard(unittest.TestCase):
    def test_minted_snapshot_contains_no_same_day_bar(self):
        tmp = Path(tempfile.mkdtemp(prefix="settled_bars_mint_"))

        def cleanup() -> None:
            # Snapshot files are chmod read-only; make writable before rmtree.
            for path in tmp.rglob("*"):
                path.chmod(0o777)
            shutil.rmtree(tmp, ignore_errors=True)

        self.addCleanup(cleanup)
        _write_universe(tmp / "cfg", "v1", ["AAA", "BBB"])
        with patch("src.data.universe_loader.yf.download", side_effect=_fake_yf_download):
            ids = ensure_market_snapshot(
                "v1", universe_dir=tmp / "cfg", snapshot_root=tmp / "snap", progress=False
            )
        umd = load_market_snapshot(ids[0], snapshot_root=tmp / "snap")
        self.assertLess(umd.prices.index.max(), _today_ny())


if __name__ == "__main__":
    unittest.main()
