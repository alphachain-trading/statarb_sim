"""
Tests for the three Track A guards and the n_legs/gate consistency fix.

Each guard converts a currently-silent wrong result into a loud failure on a
configuration not exercised by any config in DEFAULT_CONFIGS or any howto
notebook today (see docs/refactor/A_spec.md section 11).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.candidates.candidate_panel import CandidatePanelResult
from src.candidates.candidate_selector import CandidateSelectionConfig, select_candidates
from src.candidates.pair_candidate_panel_creator import _fast_pair_diagnostics
from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode
from src.simulator.config import (
    ActivationConfig,
    CapitalConfig,
    DataConfig,
    ExecutionConfig,
    MRDiagnosticsConfig,
    PairSpreadTraderConfig,
    RunConfig,
    SectorDataSource,
    SimulatorConfig,
    ZScoreConfig,
)
from src.simulator.simulator_factory import _load_panels


def _minimal_sim_kwargs():
    return dict(
        data=DataConfig(),
        candidate_selection=CandidateSelectionConfig(),
        activation=ActivationConfig(),
        diagnostics=MRDiagnosticsConfig(lookback=21, compute_frequency="off"),
        trader=PairSpreadTraderConfig(),
        run=RunConfig(),
        execution=ExecutionConfig(),
        capital=CapitalConfig(total_capital=1_000_000.0),
    )


class TestGuard1AdfPvalueMaxAllNaN(unittest.TestCase):
    def test_raises_naming_setting_and_reason(self):
        panel = pd.DataFrame({
            "candidate_subtype": ["knee", "knee"],
            "is_valid": [True, True],
            "success": [True, True],
            "adf_pvalue": [np.nan, np.nan],
            "mr_score": [1.0, 2.0],
            "half_life": [10.0, 20.0],
            "candidate_type": ["pair", "pair"],
            "spread_id": ["A|B", "C|D"],
        })
        cfg = CandidateSelectionConfig(adf_pvalue_max=0.05)

        with self.assertRaises(ValueError) as cm:
            select_candidates(panel, cfg)

        msg = str(cm.exception)
        self.assertIn("adf_pvalue_max", msg)
        self.assertIn("0.05", msg)
        self.assertIn("NaN", msg)

    def test_does_not_raise_when_adf_pvalue_populated(self):
        panel = pd.DataFrame({
            "candidate_subtype": ["knee", "knee"],
            "is_valid": [True, True],
            "success": [True, True],
            "adf_pvalue": [0.01, 0.5],
            "mr_score": [1.0, 2.0],
            "half_life": [10.0, 20.0],
            "candidate_type": ["pair", "pair"],
            "spread_id": ["A|B", "C|D"],
            "asof_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        })
        cfg = CandidateSelectionConfig(adf_pvalue_max=0.05)

        result = select_candidates(panel, cfg)
        self.assertEqual(len(result.panel), 1)


class TestGuard2EqRolling(unittest.TestCase):
    def test_eq_rolling_residual_raises(self):
        residual = CausalResidualConfig(
            mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False, lb=21,
        )
        with self.assertRaises(ValueError) as cm:
            SimulatorConfig(
                z_score=ZScoreConfig(lookback=21),
                residual=residual,
                **_minimal_sim_kwargs(),
            )
        msg = str(cm.exception)
        self.assertIn("EQ_ROLLING", msg)
        self.assertIn("not yet implemented", msg)

    def test_eq_expanding_residual_does_not_raise(self):
        residual = CausalResidualConfig(
            mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False, min_lb_eq_exp=21,
        )
        cfg = SimulatorConfig(
            z_score=ZScoreConfig(lookback=21),
            residual=residual,
            **_minimal_sim_kwargs(),
        )
        self.assertIs(cfg.residual, residual)


class TestGuard3MultiSleeveOccupancy(unittest.TestCase):
    def test_multiple_z_lookbacks_under_one_residual_key_raises(self):
        with self.assertRaises(ValueError) as cm:
            SimulatorConfig(
                z_score=[
                    ZScoreConfig(residual_key="rk", lookback=21),
                    ZScoreConfig(residual_key="rk", lookback=42),
                ],
                **{**_minimal_sim_kwargs(), "diagnostics": MRDiagnosticsConfig(lookback=42, compute_frequency="off")},
            )
        msg = str(cm.exception)
        self.assertIn("residual_key='rk'", msg)
        self.assertIn("separate runs", msg)

    def test_distinct_residual_keys_do_not_raise(self):
        cfg = SimulatorConfig(
            z_score=[
                ZScoreConfig(residual_key="rk1", lookback=21),
                ZScoreConfig(residual_key="rk2", lookback=21),
            ],
            **{**_minimal_sim_kwargs(), "diagnostics": MRDiagnosticsConfig(lookback=21, compute_frequency="off")},
        )
        self.assertTrue(cfg.is_multi_timescale())

    def test_multiple_weight_models_under_one_spread_id_raises(self):
        fake_panel = pd.DataFrame({
            "spread_id": ["A|B", "A|B"],
            "weight_model": ["ols", "pca"],
            "group_id": ["g", "g"],
        })
        data_cfg = DataConfig(
            sectors=[SectorDataSource(universe_config_name="u", candidate_panel_stem="stem")],
        )
        with patch(
            "src.simulator.simulator_factory.load_candidate_panel_result",
            return_value=CandidatePanelResult(panel=fake_panel, metadata={}),
        ):
            with self.assertRaises(ValueError) as cm:
                _load_panels(data_cfg, [ZScoreConfig()])
        msg = str(cm.exception)
        self.assertIn("weight_model", msg)
        self.assertIn("separate runs", msg)

    def test_single_weight_model_does_not_raise(self):
        fake_panel = pd.DataFrame({
            "spread_id": ["A|B", "C|D"],
            "weight_model": ["ols", "ols"],
            "group_id": ["g", "g"],
        })
        data_cfg = DataConfig(
            sectors=[SectorDataSource(universe_config_name="u", candidate_panel_stem="stem")],
        )
        with patch(
            "src.simulator.simulator_factory.load_candidate_panel_result",
            return_value=CandidatePanelResult(panel=fake_panel, metadata={}),
        ):
            merged, metadata_by_key = _load_panels(data_cfg, [ZScoreConfig()])
        self.assertEqual(len(merged), 2)


class TestNLegsGateConsistency(unittest.TestCase):
    """n_legs and the too_few_active_legs gate must never disagree."""

    def _run(self, w_left: float, w_right: float) -> dict:
        rng = np.random.default_rng(0)
        spread_return = rng.normal(size=60)
        return _fast_pair_diagnostics(
            spread_return=spread_return,
            w_left=w_left,
            w_right=w_right,
            tiny_weight_threshold=1e-6,
            min_return_std=1e-8,
            min_level_std=1e-8,
            min_kappa=1e-6,
            max_half_life=126.0,
            skip_adf=True,
        )

    def test_two_active_legs_not_gated_on_leg_count(self):
        diag = self._run(w_left=1.0, w_right=-0.5)
        self.assertEqual(diag["n_legs"], 2)
        self.assertNotEqual(diag["failure_reason"], "too_few_active_legs")

    def test_one_tiny_leg_is_gated_and_reports_n_legs_1(self):
        diag = self._run(w_left=1.0, w_right=1e-9)
        self.assertEqual(diag["n_legs"], 1)
        self.assertFalse(diag["is_valid"])
        self.assertEqual(diag["failure_reason"], "too_few_active_legs")

    def test_both_tiny_legs_is_gated_and_reports_n_legs_0(self):
        diag = self._run(w_left=1e-9, w_right=1e-9)
        self.assertEqual(diag["n_legs"], 0)
        self.assertFalse(diag["is_valid"])
        self.assertEqual(diag["failure_reason"], "too_few_active_legs")


if __name__ == "__main__":
    unittest.main()
