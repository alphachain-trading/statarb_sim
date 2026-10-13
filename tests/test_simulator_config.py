"""
Tests for config-layer logic that decides identity and validity of runs: group
names, z-score labels / weights, SimulatorConfig cross-field validation.
"""

import dataclasses
import unittest
from typing import Any

import pandas as pd

from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode
from src.simulator.config import (
    EntryFeatureConfig,
    FeatureSpec,
    RunConfig,
    VolSizingConfig,
    ZScoreConfig,
    group_abbrev,
    group_resolve,
)
from tests._synthetic import build_sim_config


def _z(lookback: Any = 21, rkey: str = "", method: str = "ewm", **kw: Any) -> ZScoreConfig:
    return ZScoreConfig(lookback=lookback, ddof=1, method=method, residual_key=rkey, **kw)


class TestGroupNames(unittest.TestCase):
    def test_resolve_accepts_full_and_abbreviated_names_case_insensitively(self):
        self.assertEqual(group_resolve("energy"), "energy")
        self.assertEqual(group_resolve("NRG"), "energy")
        self.assertEqual(group_resolve("Consumer_Staples"), "consumer_staples")
        self.assertEqual(group_resolve("stp"), "consumer_staples")

    def test_resolve_unknown_raises(self):
        with self.assertRaises(KeyError):
            group_resolve("crypto")

    def test_abbrev_single_name(self):
        self.assertEqual(group_abbrev("energy"), "nrg")
        self.assertEqual(group_abbrev("mat"), "mat")
        self.assertEqual(group_abbrev("energy", to_string=True), "[nrg]")

    def test_abbrev_list_forms(self):
        self.assertEqual(group_abbrev(["energy", "materials"]), ["nrg", "mat"])
        self.assertEqual(group_abbrev(["energy", "materials"], to_string=True), "nrg,mat")
        self.assertEqual(
            group_abbrev(["energy", "materials"], to_string=True, brackets=True), "[nrg,mat]"
        )


class TestZScoreConfig(unittest.TestCase):
    def test_lookbacks_scalar_and_list(self):
        self.assertEqual(_z(21).resolved_lookbacks(), [21])
        self.assertEqual(_z([21, 63]).resolved_lookbacks(), [21, 63])

    def test_timescale_label(self):
        self.assertEqual(_z(21).timescale_label, "")  # single timescale: no residual_key
        self.assertEqual(_z(21, "exp_hl63_mh126").timescale_label, "exp_hl63_mh126__zhl21")
        self.assertEqual(
            _z(63, "exp_hl63_mh126", method="rolling").timescale_label, "exp_hl63_mh126__zlb63"
        )
        # only the first lookback names the label
        self.assertEqual(_z([5, 9], "k").timescale_label, "k__zhl5")

    def test_residual_hl(self):
        self.assertEqual(_z(1, "exp_hl252_mh504_rf").residual_hl, 252)
        self.assertIsNone(_z(1, "rol_lb21").residual_hl)
        self.assertIsNone(_z(1, "").residual_hl)

    def test_weights_default_equal_and_normalised(self):
        self.assertEqual(_z([10, 20, 30, 40]).resolved_weights(), [0.25] * 4)
        self.assertEqual(_z([10, 20], weights=[1.0, 3.0]).resolved_weights(), [0.25, 0.75])
        self.assertEqual(_z(10).resolved_weights(), [1.0])

    def test_weights_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _z([10, 20], weights=[1.0]).resolved_weights()

    def test_min_periods_default_is_lookback_plus_ddof(self):
        self.assertEqual(_z(21).resolved_min_periods(lookback=21), 22)
        self.assertEqual(_z(21, min_periods=5).resolved_min_periods(lookback=21), 5)


class TestSimulatorConfigValidation(unittest.TestCase):
    def setUp(self):
        self.base = build_sim_config(["alpha"])  # z lookback 21, diagnostics lookback from bundle

    def _replace(self, **kw: Any):
        return dataclasses.replace(self.base, **kw)

    def test_z_lookback_may_not_exceed_diagnostics_lookback(self):
        diag = self.base.diagnostics
        too_long = _z(diag.lookback + 1, self.base.residual.key)
        with self.assertRaises(ValueError):
            self._replace(z_score=too_long)

    def test_rolling_residual_config_is_rejected(self):
        rolling = CausalResidualConfig(
            mode=ResidualMode.EQ_ROLLING, subtract_risk_free=True, lb=self.base.diagnostics.lookback
        )
        with self.assertRaises(ValueError) as ctx:
            self._replace(residual=rolling)
        self.assertIn("EQ_ROLLING", str(ctx.exception))

    def test_multi_timescale_requires_residual_keys_and_unique_labels(self):
        with self.assertRaises(ValueError):
            self._replace(z_score=[])
        with self.assertRaises(ValueError):
            self._replace(z_score=[_z(21, "k1"), _z(21, "")])  # missing residual_key
        with self.assertRaises(ValueError):
            self._replace(z_score=[_z(21, "k1"), _z(21, "k1")])  # duplicate label

    def test_two_sleeves_on_one_residual_key_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._replace(z_score=[_z(21, "k1"), _z(10, "k1")])
        self.assertIn("share residual_key", str(ctx.exception))

    def test_valid_multi_timescale_and_lookup_helpers(self):
        a, b = _z(21, "exp_hl63_mh126"), _z(10, "exp_hl126_mh252")
        cfg = self._replace(z_score=[a, b])
        self.assertTrue(cfg.is_multi_timescale())
        self.assertEqual(cfg.unique_residual_keys(), {"exp_hl63_mh126", "exp_hl126_mh252"})
        self.assertIs(cfg.z_score_by_residual_key()["exp_hl63_mh126"], a)
        self.assertIs(cfg.z_score_by_timescale_label()[b.timescale_label], b)
        self.assertEqual(
            cfg.z_score_configs_by_rkey(), {"exp_hl63_mh126": [a], "exp_hl126_mh252": [b]}
        )

    def test_single_timescale_helpers(self):
        self.assertFalse(self.base.is_multi_timescale())
        self.assertEqual(len(self.base.resolved_z_score_configs()), 1)


class TestSmallConfigs(unittest.TestCase):
    def test_run_config_coerces_dates_and_checks_order(self):
        rc = RunConfig(start_date="2024-01-01", end_date="2024-02-01")
        self.assertEqual(rc.start_date, pd.Timestamp("2024-01-01"))
        with self.assertRaises(ValueError):
            RunConfig(start_date="2024-02-02", end_date="2024-01-01")
        self.assertIsNone(RunConfig().start_date)

    def test_vol_sizing_bounds(self):
        VolSizingConfig(0.5, 0.5)  # equal is allowed
        with self.assertRaises(ValueError):
            VolSizingConfig(0.0, 1.0)
        with self.assertRaises(ValueError):
            VolSizingConfig(0.5, 0.4)

    def test_feature_spec_rejects_sign_param_and_empty_name(self):
        with self.assertRaises(ValueError):
            FeatureSpec("x_area_asymmetry_ewm", {"sign": 1})
        with self.assertRaises(ValueError):
            FeatureSpec("")

    def test_entry_feature_config_rejects_duplicates(self):
        spec = FeatureSpec("mean_ewm%", {"halflife": 5})
        with self.assertRaises(ValueError):
            EntryFeatureConfig(features=(spec, FeatureSpec("mean_ewm%", {"halflife": 9})))


if __name__ == "__main__":
    unittest.main()
