"""
SweepConfig.alias format and run_sweep bookkeeping (dedup, failure rows, persistence),
with run_from_config replaced by a stub so no simulation runs.
"""

import re
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pandas as pd

from src.simulator.config import SimulatorConfig, ZScoreConfig
from src.simulator.sweep_runner import (
    SweepConfig,
    _existing_keys,
    dedup_key,
    load_sweep_results,
    run_sweep,
)

FAKE_SNAPSHOT_ID = "sp500_v1_20260101T000000000000"
# The sweep-level data selection every run_sweep/dedup_key call states (#12).
SNAP: dict[str, Any] = {"universe_name": "sp500_v1", "snapshot_id": FAKE_SNAPSHOT_ID}


def _fake_bind(config: SimulatorConfig) -> SimulatorConfig:
    """bind_market_snapshot without a snapshot on disk: skips the existence check."""
    return config


def _sweep(**kw: Any) -> SweepConfig:
    kw.setdefault("candidate_panel_subdir", "P1")
    return SweepConfig(**kw)


class TestSweepAlias(unittest.TestCase):
    def test_default_alias(self):
        self.assertEqual(
            _sweep().alias,
            "z2.0_xz0.0_ewm_lb21_bpn100k_voln_tick0.15_gx10.0_all_S20100101_E20251231",
        )

    def test_list_lookback_flat_sizing_and_no_dates(self):
        a = _sweep(z_lookback=[21, 63], vol_normalize=False, start_date=None, end_date=None).alias
        self.assertEqual(a, "z2.0_xz0.0_ewm_lb21-63_bpn100k_flat_tick0.15_gx10.0_all")

    def test_time_stop_forms(self):
        self.assertIn("_mhd7_", _sweep(max_holding_days=7).alias)
        self.assertIn("_mhd-2xrhl_", _sweep(max_holding_days="2*rhl").alias)
        self.assertIn("_mhd1+2_", _sweep(max_holding_days={"a": 1, "b": 2}).alias)

    def test_exit_rule_string_is_made_filename_safe(self):
        alias = _sweep(exit_rule="days_open > 5 AND pnl_pct < -2").alias
        self.assertIn("_xr-days_opengt5+pnl_pctlt-2_", alias)
        self.assertIn("_xr-days_opengt5|x_", _sweep(exit_rule="days_open > 5 OR x").alias)

    def test_exit_rule_dict_alias_has_a_four_digit_tag(self):
        # NOTE: the tag comes from hash(repr(...)), i.e. Python's per-process string
        # hash randomisation: it differs between runs. Display only, but not stable.
        alias = _sweep(exit_rule={"k": "days_open > 5"}).alias
        self.assertRegex(alias, r"_xr-dict\d{4}_")

    def test_excluded_groups_use_abbreviations(self):
        self.assertIn("_ex-nrg,mat_", _sweep(excluded_groups=["energy", "materials"]).alias)

    def test_multi_timescale_alias(self):
        zs = [
            ZScoreConfig(21, ddof=1, method="ewm", residual_key="exp_hl63_mh126"),
            ZScoreConfig(63, ddof=1, method="ewm", residual_key="exp_hl126_mh252"),
        ]
        sweep = _sweep(z_score_overrides=zs)
        self.assertTrue(sweep.is_multi_timescale)
        self.assertIn("_mt2_ewm_rhl63+126_zhl21+63_", sweep.alias)
        self.assertFalse(_sweep(z_score_overrides=[]).is_multi_timescale)

    def test_alias_is_lossy_but_dedup_key_is_not(self):
        a, b = _sweep(candidate_panel_subdir="P1"), _sweep(candidate_panel_subdir="P2")
        self.assertEqual(a.alias, b.alias)
        self.assertNotEqual(dedup_key(a, **SNAP), dedup_key(b, **SNAP))


@contextmanager
def _patched_runs(side_effect: Any) -> Iterator[tuple[mock.MagicMock, Path]]:
    """Redirect every runs-root lookup to a temp dir and stub the simulation itself."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch("src.simulator.sweep_runner.bind_market_snapshot", side_effect=_fake_bind),
        mock.patch("src.simulator.sweep_runner.SIMULATION_RUNS_ROOT", tmp),
        mock.patch("src.simulator.sweep_rebuild.SIMULATION_RUNS_ROOT", tmp),
        mock.patch("src.simulator.sweep_runner.run_from_config", side_effect=side_effect) as runner,
    ):
        yield runner, Path(tmp)


def _fake_result(run_id: str, sharpe: float = 1.0) -> Any:
    return SimpleNamespace(
        run_id=run_id, performance=SimpleNamespace(metrics={"sharpe_net": sharpe})
    )


class TestRunSweep(unittest.TestCase):
    def test_single_run_builds_row_and_persists_results(self):
        sweep = _sweep()
        with _patched_runs([_fake_result("run_a", 1.5)]) as (runner, root):
            df, results = run_sweep([sweep], **SNAP)
            self.assertEqual(runner.call_count, 1)
            self.assertEqual(len(results), 1)
            self.assertEqual(list(df.index), ["run_a"])
            row = df.loc["run_a"]
            self.assertEqual(row["sharpe_net"], 1.5)
            self.assertEqual(row["alias"], sweep.alias)
            self.assertEqual(row["config_hash"], dedup_key(sweep, **SNAP))
            self.assertEqual(row["candidate_panel_subdir"], "P1")
            self.assertEqual(row["entry_z"], 2.0)
            self.assertTrue((root / "sweep_results.pkl").exists())
            pd.testing.assert_frame_equal(load_sweep_results(), df, check_like=True)

    def test_second_invocation_skips_runs_already_in_results(self):
        sweep = _sweep()
        with _patched_runs([_fake_result("run_a")]) as (runner, _root):
            run_sweep([sweep], **SNAP)
            df, results = run_sweep([sweep], **SNAP)
            self.assertEqual(runner.call_count, 1)  # not run again
            self.assertEqual(results, [])
            self.assertEqual(len(df), 1)

    def test_only_new_configs_run_on_a_later_invocation(self):
        with _patched_runs([_fake_result("run_a"), _fake_result("run_b")]) as (runner, _):
            run_sweep([_sweep()], **SNAP)
            df, _ = run_sweep([_sweep(), _sweep(entry_z=2.5)], **SNAP)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(sorted(df.index), ["run_a", "run_b"])
            self.assertEqual(sorted(df["entry_z"]), [2.0, 2.5])

    def test_skip_existing_false_reruns(self):
        with _patched_runs([_fake_result("run_a"), _fake_result("run_b")]) as (runner, _):
            run_sweep([_sweep()], **SNAP)
            df, _ = run_sweep([_sweep()], skip_existing=False, **SNAP)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(sorted(df.index), ["run_a", "run_b"])

    def test_failure_becomes_an_error_row_and_the_sweep_continues(self):
        effects = [RuntimeError("boom"), _fake_result("run_ok", 2.0)]
        with _patched_runs(effects) as (runner, _):
            df, results = run_sweep([_sweep(entry_z=1.0), _sweep(entry_z=3.0)], **SNAP)
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(len(results), 1)
            failed = df[df.index.str.startswith("FAILED_")]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed.iloc[0]["error"], "boom")
            self.assertEqual(df.loc["run_ok", "sharpe_net"], 2.0)

    @unittest.expectedFailure
    def test_failed_runs_should_be_retried_on_the_next_invocation(self):
        # KNOWN GAP (reported, not fixed): the failed row stores config_hash and
        # _existing_keys() does not exclude error rows, so skip_existing=True never
        # retries a run that failed.
        sweep = _sweep()
        with _patched_runs([RuntimeError("boom"), _fake_result("run_ok")]) as (runner, _):
            run_sweep([sweep], **SNAP)
            run_sweep([sweep], **SNAP)
            self.assertEqual(runner.call_count, 2)

    def test_batch_is_validated_before_anything_runs(self):
        with _patched_runs([_fake_result("never")]) as (runner, _):
            with self.assertRaises(ValueError):
                run_sweep([_sweep(), _sweep(candidate_panel_subdir="")], **SNAP)
            runner.assert_not_called()

    def test_multi_timescale_overrides_are_stored_as_a_string(self):
        zs = [ZScoreConfig(21, ddof=1, method="ewm", residual_key="exp_hl63_mh126")]
        with _patched_runs([_fake_result("run_a")]) as (_, _root):
            df, _ = run_sweep([_sweep(z_score_overrides=zs)], **SNAP)
            self.assertEqual(df.loc["run_a", "z_score_overrides"], str(["exp_hl63_mh126:zhl21"]))


class TestExistingKeys(unittest.TestCase):
    def test_keys_come_from_config_hash_column_only(self):
        df = pd.DataFrame({"config_hash": ["a", None, "b"], "alias": ["x", "y", "z"]})
        self.assertEqual(_existing_keys(df), {"a", "b"})

    def test_rows_without_the_column_contribute_nothing(self):
        self.assertEqual(_existing_keys(pd.DataFrame({"alias": ["x"]})), set())
        self.assertEqual(_existing_keys(pd.DataFrame()), set())

    def test_hash_alias_regression_guard(self):
        self.assertTrue(re.fullmatch(r"[0-9a-f]{32}", dedup_key(_sweep(), **SNAP)))


if __name__ == "__main__":
    unittest.main()
