"""
Tests for config hashing (the sweep dedup identity) and run persistence round trips.
"""

import dataclasses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.simulator.config import PairSpreadTraderConfig, PersistenceConfig, SizingConfig
from src.simulator.simulation_persistence import (
    _resolve_run_dir,
    _to_serializable,
    flush_yearly_logs,
    hash_config,
    list_simulation_runs,
    load_closed_trades_df,
    load_simulation_run,
    make_run_id,
    save_simulation_run,
)
from tests._synthetic import build_sim_config


class TestHashConfig(unittest.TestCase):
    def setUp(self):
        self.base = build_sim_config(["alpha", "beta"])

    def test_equal_configs_hash_equal_and_have_fixed_shape(self):
        h1, h2 = hash_config(self.base), hash_config(build_sim_config(["alpha", "beta"]))
        self.assertEqual(h1, h2)
        self.assertRegex(h1, r"^[0-9a-f]{32}$")

    def test_every_kind_of_parameter_change_changes_the_hash(self):
        base_hash = hash_config(self.base)
        variants = {
            "entry_z": dataclasses.replace(
                self.base, trader=PairSpreadTraderConfig(entry_z=1.6, exit_z=0.0)
            ),
            "exit_z": dataclasses.replace(
                self.base, trader=PairSpreadTraderConfig(entry_z=1.5, exit_z=0.1)
            ),
            "sizing": dataclasses.replace(self.base, sizing=SizingConfig(base_pair_notional=99.0)),
            "groups": build_sim_config(["alpha"]),
            "max_steps": build_sim_config(["alpha", "beta"], max_steps=31),
            "entry_z_builder": build_sim_config(["alpha", "beta"], entry_z=1.7),
        }
        for name, cfg in variants.items():
            with self.subTest(name):
                self.assertNotEqual(hash_config(cfg), base_hash)
        self.assertEqual(len({hash_config(c) for c in variants.values()}), len(variants))

    def test_hash_is_stable_across_processes_for_this_config(self):
        # Pins the digest so an accidental change to the hashed representation
        # (which would silently re-run or wrongly skip every sweep) is noticed.
        # Adding or renaming a config field changes it on purpose: update the value
        # then, knowing that previously stored config_hash keys no longer match.
        # Updated (f8e5fa2a... -> 1adff499...) when issue #2 bound the synthetic
        # fixture to SYNTHETIC_SNAPSHOT_ID; the representation is unchanged
        # (the same fixture with snapshot_id=None still hashes to f8e5fa2a...).
        self.assertEqual(hash_config(self.base), "1adff4991f9fac87c79ea85d1f59c19e")


class TestRunId(unittest.TestCase):
    def test_format_and_hash_suffix(self):
        cfg = build_sim_config(["alpha"])
        run_id = make_run_id(cfg)
        self.assertRegex(run_id, r"^\d{8}_\d{4}_[0-9a-f]{8}$")
        self.assertTrue(run_id.endswith(hash_config(cfg)[:8]))


class TestToSerializable(unittest.TestCase):
    def test_conversions(self):
        out = _to_serializable(
            {
                "ts": pd.Timestamp("2024-01-02 03:04"),
                "path": Path("/a/b"),
                "tup": (1, (2, 3)),
                "nan": float("nan"),
                "nested": {"x": [pd.Timestamp("2024-01-01")]},
                "plain": 1.5,
            }
        )
        self.assertEqual(
            out,
            {
                "ts": "2024-01-02T03:04:00",
                "path": "/a/b",
                "tup": [1, [2, 3]],
                "nan": None,
                "nested": {"x": ["2024-01-01T00:00:00"]},
                "plain": 1.5,
            },
        )

    def test_nan_and_none_are_indistinguishable(self):
        # Pins current behaviour: a NaN parameter hashes like None.
        self.assertEqual(_to_serializable(float("nan")), _to_serializable(None))


class TestResolveRunDir(unittest.TestCase):
    def test_explicit_output_dir_wins(self):
        self.assertEqual(
            _resolve_run_dir(PersistenceConfig(output_dir="/tmp/x"), "r1"), Path("/tmp/x/r1")
        )

    def test_default_root_is_used_otherwise(self):
        got = _resolve_run_dir(PersistenceConfig(), "r1")
        self.assertEqual(got.name, "r1")


def _fake_result(trades: pd.DataFrame | None = None) -> Any:
    trades = trades if trades is not None else pd.DataFrame({"trade_id": ["t1"], "pnl": [1.0]})
    return SimpleNamespace(
        selected_panel=pd.DataFrame({"spread_id": ["A|B"], "z": [1.0]}),
        closed_trades=[object(), object()],
        closed_trades_df=lambda: trades,
        action_log_df=lambda: pd.DataFrame(),
        ticker_trade_log_df=lambda: pd.DataFrame(),
        diagnostics_log_df=lambda: pd.DataFrame(),
    )


class TestSaveAndLoadRun(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        persistence = PersistenceConfig(
            enabled=True,
            output_dir=str(self.out),
            artifacts=("config", "selected_panel", "performance_metrics"),
        )
        self.cfg = dataclasses.replace(build_sim_config(["alpha"]), persistence=persistence)

    def test_round_trip(self):
        perf = SimpleNamespace(
            metrics={"sharpe_net": 1.25, "n_trades": 2, "other": 9}, group_metrics={"alpha": {}}
        )
        run_dir = save_simulation_run(
            _fake_result(), self.cfg, performance_result=perf, run_id="r1"
        )
        self.assertEqual(run_dir, self.out / "r1")
        loaded = load_simulation_run(run_dir)
        self.assertEqual(loaded["meta"]["run_id"], "r1")
        self.assertEqual(loaded["meta"]["config_hash"], hash_config(self.cfg))
        self.assertEqual(loaded["meta"]["n_closed_trades"], 2)
        self.assertEqual(loaded["meta"]["sharpe_net"], 1.25)
        self.assertNotIn("other", loaded["meta"])  # meta keeps only the headline stats
        self.assertEqual(loaded["performance_metrics"]["n_trades"], 2)
        self.assertEqual(loaded["performance_group_metrics"], {"alpha": {}})
        self.assertEqual(loaded["config"]["trader"]["entry_z"], 1.5)
        pd.testing.assert_frame_equal(loaded["selected_panel"], _fake_result().selected_panel)
        self.assertIsNone(loaded["closed_trades"])

    def test_only_configured_artifacts_are_written(self):
        cfg = dataclasses.replace(
            self.cfg,
            persistence=PersistenceConfig(
                enabled=True, output_dir=str(self.out), artifacts=("config",)
            ),
        )
        run_dir = save_simulation_run(_fake_result(), cfg, run_id="r2")
        names = sorted(p.name for p in run_dir.iterdir())
        self.assertEqual(names, ["config.json", "meta.json"])

    def test_missing_run_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_simulation_run(self.out / "nope")

    def test_list_runs_reads_meta_of_every_run(self):
        save_simulation_run(_fake_result(), self.cfg, run_id="r1")
        save_simulation_run(_fake_result(), self.cfg, run_id="r2")
        (self.out / "not_a_run").mkdir()
        df = list_simulation_runs(self.out)
        self.assertEqual(sorted(df["run_id"]), ["r1", "r2"])
        self.assertTrue(df["run_dir"].str.startswith(str(self.out)).all())
        self.assertTrue(list_simulation_runs(self.out / "missing").empty)


class TestYearlyFlush(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name)

    def test_slices_are_written_per_year_and_concatenated_in_order(self):
        arts = {"closed_trades"}
        for year, tid in ((2021, "b"), (2020, "a")):
            flush_yearly_logs(
                _fake_result(pd.DataFrame({"trade_id": [tid]})), year, self.run_dir, arts
            )
        self.assertTrue((self.run_dir / "closed_trades" / "closed_trades_2020.parquet").exists())
        df = load_closed_trades_df(self.run_dir)
        self.assertEqual(list(df["trade_id"]), ["a", "b"])  # sorted by file name = by year

    def test_empty_frames_are_not_written(self):
        flush_yearly_logs(_fake_result(pd.DataFrame()), 2020, self.run_dir, {"closed_trades"})
        self.assertEqual(list(self.run_dir.iterdir()), [])

    def test_unselected_artifacts_are_skipped(self):
        flush_yearly_logs(_fake_result(), 2020, self.run_dir, {"config"})
        self.assertEqual(list(self.run_dir.iterdir()), [])

    def test_legacy_flat_slices_and_monolithic_fallback(self):
        pd.DataFrame({"trade_id": ["x"]}).to_parquet(self.run_dir / "closed_trades_2019.parquet")
        self.assertEqual(list(load_closed_trades_df(self.run_dir)["trade_id"]), ["x"])
        other = self.run_dir / "mono"
        other.mkdir()
        pd.DataFrame({"trade_id": ["m"]}).to_parquet(other / "closed_trades.parquet")
        self.assertEqual(list(load_closed_trades_df(other)["trade_id"]), ["m"])
        empty = self.run_dir / "empty"
        empty.mkdir()
        self.assertTrue(load_closed_trades_df(empty).empty)

    def test_load_simulation_run_picks_up_yearly_slices(self):
        flush_yearly_logs(_fake_result(), 2020, self.run_dir, {"closed_trades"})
        loaded = load_simulation_run(self.run_dir)
        self.assertEqual(list(loaded["closed_trades"]["trade_id"]), ["t1"])
        self.assertTrue(np.isclose(loaded["closed_trades"]["pnl"].iloc[0], 1.0))
        self.assertIsNone(loaded["diagnostics"])


if __name__ == "__main__":
    unittest.main()
