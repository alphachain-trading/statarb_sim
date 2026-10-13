"""
Tests for src/simulator/sweep_rebuild.py.

_build_alias_from_config is a display alias (see its docstring): the first block pins
its exact output format on representative config.json shapes; the regression test
covers the UnboundLocalError that went unnoticed when the function was untested.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulator.sweep_rebuild import (
    _build_alias_from_config,
    _extract_config_fields,
    _flatten_metrics,
    _serialize_if_complex,
    rebuild_sweep_results,
)

SINGLE = {
    "trader": {"entry_z": 1.5, "exit_z": 0.0},
    "z_score": {"method": "ewm", "lookback": 21, "residual_key": "exp_hl504_mh1008_rf"},
    "risk_manager": {"max_ticker_exposure_pct": 0.15, "max_gross_exposure": 10.0},
    "sizing": {"base_pair_notional": 100_000.0},
    "data": {"excluded_groups": ["materials", "energy"]},
    "run": {"start_date": "2010-01-01", "end_date": "2025-12-31"},
}


def _with(base: dict[str, Any], **sections: Any) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(base))
    for key, value in sections.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


class TestAliasFormat(unittest.TestCase):
    def test_single_timescale_full_config(self):
        self.assertEqual(
            _build_alias_from_config(SINGLE),
            "z1.5_xz0.0_mt1_ewm_rhl504_zhl21_tick0.15_bpn100k_gx10.0"
            "_ex-energy,materials_S20100101_E20251231",
        )

    def test_multi_timescale(self):
        cfg = _with(
            SINGLE,
            z_score=[
                {"method": "ewm", "lookback": 21, "residual_key": "exp_hl63_mh126"},
                {"method": "ewm", "lookback": 63, "residual_key": "exp_hl126_mh252"},
            ],
        )
        alias = _build_alias_from_config(cfg)
        self.assertTrue(alias.startswith("z1.5_xz0.0_mt2_ewm_rhl63+126_zhl21+63_"), alias)

    def test_empty_config_yields_minimal_alias(self):
        self.assertEqual(_build_alias_from_config({}), "xz0.0_mt1_ewm_all")

    def test_no_excluded_groups_and_no_dates(self):
        cfg = _with(SINGLE, data={"excluded_groups": []}, run=None)
        alias = _build_alias_from_config(cfg)
        self.assertTrue(alias.endswith("_gx10.0_all"), alias)

    def test_legacy_threshold_keys(self):
        cfg = {"trader": {"entry_z_threshold": 2.0, "exit_z_threshold": 0.5}}
        self.assertEqual(_build_alias_from_config(cfg), "z2.0_xz0.5_mt1_ewm_all")

    def test_kelly_and_timescale_risk_parts(self):
        cfg = _with(
            SINGLE,
            sizing={"kelly": {"fraction": 0.5, "half_life": 63, "per_sector": False}},
            risk_manager={"timescale_risk": {"max_timescales_per_spread": 2, "selection": "x"}},
        )
        alias = _build_alias_from_config(cfg)
        self.assertIn("_kelly-f0.5+hl63+glob_", alias)
        self.assertIn("_tsr-mts2+x_", alias)

    def test_kelly_defaults_when_only_some_keys_are_stored(self):
        alias = _build_alias_from_config(_with(SINGLE, sizing={"kelly": {"half_life": 21}}))
        self.assertIn("_kelly-f0.5+hl21+sec_", alias)

    def test_empty_kelly_dict_is_treated_as_absent(self):
        # Pins current behaviour (`sizing.get("kelly") or rm.get("kelly")`): {} is falsy.
        self.assertNotIn("kelly", _build_alias_from_config(_with(SINGLE, sizing={"kelly": {}})))

    def test_legacy_kelly_and_base_notional_in_risk_manager(self):
        cfg = _with(
            SINGLE,
            sizing=None,
            risk_manager={"base_pair_notional": 50_000.0, "kelly": {"fraction": 1.0}},
        )
        alias = _build_alias_from_config(cfg)
        self.assertIn("_bpn50k_", alias)
        self.assertIn("_kelly-f1.0+sec_", alias)

    def test_timescale_risk_defaults(self):
        alias = _build_alias_from_config(_with(SINGLE, risk_manager={"timescale_risk": {}}))
        self.assertIn("_tsr-max_abs_z_", alias)

    def test_regression_missing_sizing_section_does_not_raise(self):
        # UnboundLocalError: `sizing` was read before it was assigned whenever the
        # config had a risk_manager block but the alias reached the bpn part.
        cfg = {"trader": {"entry_z": 1.0}, "risk_manager": {"max_ticker_exposure_pct": 0.1}}
        self.assertEqual(_build_alias_from_config(cfg), "z1.0_xz0.0_mt1_ewm_tick0.10_all")

    def test_regression_empty_risk_manager_with_sizing_section(self):
        cfg = {"sizing": {"base_pair_notional": 25_000.0}, "risk_manager": None}
        self.assertEqual(_build_alias_from_config(cfg), "xz0.0_mt1_ewm_bpn25k_all")


class TestEntryZFallback(unittest.TestCase):
    def test_zero_entry_z_is_kept_in_alias(self):
        self.assertTrue(
            _build_alias_from_config({"trader": {"entry_z": 0.0}}).startswith("z0.0_xz0.0")
        )

    def test_zero_entry_z_does_not_fall_back_to_the_legacy_key(self):
        cfg = {"trader": {"entry_z": 0.0, "entry_z_threshold": 3.0}}
        self.assertTrue(_build_alias_from_config(cfg).startswith("z0.0_"))
        self.assertEqual(_extract_config_fields(cfg)["entry_z"], 0.0)

    def test_missing_entry_z_is_omitted_and_none_in_fields(self):
        self.assertTrue(_build_alias_from_config({"trader": {}}).startswith("xz"))
        self.assertIsNone(_extract_config_fields({"trader": {}})["entry_z"])

    def test_legacy_key_used_only_when_entry_z_absent_or_none(self):
        self.assertEqual(
            _extract_config_fields({"trader": {"entry_z_threshold": 2.5}})["entry_z"], 2.5
        )
        cfg = {"trader": {"entry_z": None, "entry_z_threshold": 2.5}}
        self.assertEqual(_extract_config_fields(cfg)["entry_z"], 2.5)

    @unittest.expectedFailure
    def test_zero_exit_z_should_be_kept_in_extracted_fields(self):
        # KNOWN BUG (reported, not fixed): `exit_z = trader.get("exit_z") or
        # trader.get("exit_z_threshold")` turns a stored 0.0 into None. exit_z=0.0 is the
        # common value, so the rebuilt results lose it. Same defect class as entry_z.
        self.assertEqual(_extract_config_fields({"trader": {"exit_z": 0.0}})["exit_z"], 0.0)


class TestExtractConfigFields(unittest.TestCase):
    def test_single_timescale(self):
        out = _extract_config_fields(SINGLE)
        self.assertEqual(out["entry_z"], 1.5)
        self.assertEqual((out["z_method"], out["z_lookback"], out["n_timescales"]), ("ewm", 21, 1))
        self.assertIsNone(out["z_score_overrides"])
        self.assertEqual(out["base_pair_notional"], 100_000.0)
        self.assertEqual(out["max_gross_exposure"], 10.0)
        self.assertEqual((out["start_date"], out["end_date"]), ("2010-01-01", "2025-12-31"))
        self.assertEqual(out["excluded_groups"], ["materials", "energy"])

    def test_multi_timescale(self):
        cfg = _with(
            SINGLE,
            z_score=[
                {"method": "ewm", "lookback": 21, "residual_key": "exp_hl63_mh126"},
                {"method": "rolling", "lookback": 63, "residual_key": "exp_hl126_mh252"},
            ],
        )
        out = _extract_config_fields(cfg)
        self.assertEqual(out["n_timescales"], 2)
        self.assertEqual(out["z_lookback"], [21, 63])
        self.assertEqual(out["z_method"], "ewm")  # first entry's method
        self.assertEqual(
            out["z_score_overrides"], str(["exp_hl63_mh126:zhl21", "exp_hl126_mh252:zhl63"])
        )

    def test_sizing_new_vs_legacy_and_vol_normalize_flag(self):
        new = _extract_config_fields(_with(SINGLE, sizing={"vol_normalize": {"floor": 0.2}}))
        self.assertTrue(new["vol_normalize"])
        no_vol = _extract_config_fields(_with(SINGLE, sizing={"vol_normalize": None}))
        self.assertFalse(no_vol["vol_normalize"])
        legacy = _extract_config_fields(
            _with(
                SINGLE, sizing=None, risk_manager={"base_pair_notional": 7.0, "vol_normalize": True}
            )
        )
        self.assertEqual((legacy["base_pair_notional"], legacy["vol_normalize"]), (7.0, True))

    def test_complex_values_are_stringified(self):
        out = _extract_config_fields(
            _with(
                SINGLE,
                trader={"exit_rule": {"a": "days_open > 3"}},
                risk_manager={"timescale_risk": {"k": 1}},
            )
        )
        self.assertEqual(out["exit_rule"], str({"a": "days_open > 3"}))
        self.assertEqual(out["timescale_risk"], str({"k": 1}))
        self.assertEqual(_serialize_if_complex("x"), "x")
        self.assertIsNone(_serialize_if_complex(None))

    def test_capital_new_and_legacy(self):
        self.assertEqual(
            _extract_config_fields({"capital": {"total_capital": 5.0}})["total_capital"], 5.0
        )
        legacy = _extract_config_fields({"capital": {"target_portfolio_capital": 9.0}})
        self.assertEqual(legacy["total_capital"], 9.0)

    def test_empty_config_has_all_keys_with_none(self):
        out = _extract_config_fields({})
        self.assertIsNone(out["entry_z"])
        self.assertEqual(out["n_timescales"], 1)
        self.assertEqual(out["z_method"], "ewm")


class TestFlattenMetrics(unittest.TestCase):
    def test_known_and_extra_scalar_keys_are_kept_nested_values_dropped(self):
        out = _flatten_metrics(
            {
                "sharpe_net": 1.2,
                "n_trades": 5,
                "custom": "x",
                "flag": True,
                "nested": {"a": 1},
                "lst": [1],
            }
        )
        self.assertEqual(out, {"sharpe_net": 1.2, "n_trades": 5, "custom": "x", "flag": True})


def _write_run(root: Path, name: str, config: Any, metrics: Any = None, groups: Any = None) -> None:
    d = root / name
    d.mkdir()
    if config is not None:
        (d / "config.json").write_text(
            json.dumps(config) if not isinstance(config, str) else config
        )
    if metrics is not None:
        (d / "performance_metrics.json").write_text(json.dumps(metrics))
    if groups is not None:
        (d / "performance_group_metrics.json").write_text(json.dumps(groups))


class TestRebuildSweepResults(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_builds_one_row_per_complete_run_sorted_by_run_id(self):
        _write_run(
            self.root,
            "20240102_run",
            SINGLE,
            {"sharpe_net": 2.0, "n_trades": 7},
            {"g1": {}, "g2": {}},
        )
        _write_run(
            self.root, "20240101_run", _with(SINGLE, trader={"entry_z": 2.0}), {"sharpe_net": 1.0}
        )
        df = rebuild_sweep_results(self.root)
        self.assertEqual(list(df.index), ["20240101_run", "20240102_run"])
        self.assertEqual(df.index.name, "run_id")
        self.assertEqual(df.loc["20240102_run", "sharpe_net"], 2.0)
        self.assertEqual(df.loc["20240101_run", "entry_z"], 2.0)
        self.assertEqual(df.loc["20240102_run", "n_groups_traded"], 2)
        self.assertTrue(str(df.loc["20240102_run", "alias"]).startswith("z1.5_xz0.0"))

    def test_rows_carry_no_config_hash(self):
        # Pins a limitation worth knowing: rebuilt results have no `config_hash` column,
        # so sweep_runner's skip_existing treats every rebuilt run as "not yet run".
        _write_run(self.root, "r1", SINGLE, {"sharpe_net": 1.0})
        self.assertNotIn("config_hash", rebuild_sweep_results(self.root).columns)

    def test_incomplete_and_non_run_entries_are_skipped(self):
        _write_run(self.root, "no_metrics", SINGLE)
        _write_run(self.root, "no_config", None, {"sharpe_net": 1.0})
        _write_run(self.root, ".hidden", SINGLE, {"sharpe_net": 1.0})
        _write_run(self.root, "__cache", SINGLE, {"sharpe_net": 1.0})
        (self.root / "sweep_results.pkl").write_text("not a dir")
        _write_run(self.root, "good", SINGLE, {"sharpe_net": 3.0})
        df = rebuild_sweep_results(self.root)
        self.assertEqual(list(df.index), ["good"])

    def test_include_failed_adds_error_rows(self):
        _write_run(self.root, "no_metrics", SINGLE)
        _write_run(self.root, "broken", "{not json", {"sharpe_net": 1.0})
        _write_run(self.root, "good", SINGLE, {"sharpe_net": 3.0})
        df = rebuild_sweep_results(self.root, include_failed=True)
        self.assertEqual(sorted(df.index), ["broken", "good", "no_metrics"])
        self.assertEqual(df.loc["no_metrics", "error"], "No performance_metrics.json")
        self.assertIn("Expecting", str(df.loc["broken", "error"]))
        without = rebuild_sweep_results(self.root)
        self.assertEqual(list(without.index), ["good"])

    def test_missing_directory_and_empty_directory_give_empty_frames(self):
        self.assertTrue(rebuild_sweep_results(self.root / "nope").empty)
        self.assertTrue(rebuild_sweep_results(self.root).empty)
        self.assertIsInstance(rebuild_sweep_results(self.root), pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
