"""
Unit tests for the ResidualMode-based PanelBatchConfig redesign.

Covers, without touching disk or market data:
  1. CausalResidualConfig.key byte-identity for the canonical decay_expanding
     reference config (must stay exp_hl504_mh1008_rf), plus the two new mode
     key shapes.
  2. min_history resolution per mode (multiplier / absolute / lookback).
  3. Construction-time validation raising loudly for the malformed cases.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.candidates.panel_batch import PanelBatchConfig
from src.simulator.config import ResidualMode, AbsOrMult


def _decay(**overrides):
    base = dict(
        residual_mode=ResidualMode.DECAY_EXPANDING,
        residual_hl=504,
        hedge_ratio_lb=252,
        mr_diag_lb=252,
        residual_min_lb_type_dec_exp=AbsOrMult.MULTIPLIER,
        residual_min_lb_dec_exp=2,
        subtract_risk_free=True,
    )
    base.update(overrides)
    return PanelBatchConfig(**base)


class TestKeyByteIdentity(unittest.TestCase):
    def test_decay_expanding_canonical_key_is_byte_identical(self):
        """The redesigned surface must reproduce exp_hl504_mh1008_rf exactly."""
        spec = _decay().resolved_specs()[0]
        self.assertEqual(spec.residual_cfg.key, "exp_hl504_mh1008_rf")

    def test_eq_expanding_key_shape(self):
        cfg = PanelBatchConfig(
            residual_mode=ResidualMode.EQ_EXPANDING,
            hedge_ratio_lb=252,
            mr_diag_lb=252,
            residual_min_lb_eq_exp=252,
            subtract_risk_free=True,
        )
        self.assertEqual(cfg.resolved_specs()[0].residual_cfg.key, "exp_mh252_rf")

    def test_eq_rolling_key_shape(self):
        cfg = PanelBatchConfig(
            residual_mode=ResidualMode.EQ_ROLLING,
            residual_lb=21,
            hedge_ratio_lb=21,
            mr_diag_lb=21,
        )
        # min_history == lookback → no _mh segment; no rf.
        self.assertEqual(cfg.resolved_specs()[0].residual_cfg.key, "rol_lb21")


class TestMinHistoryResolution(unittest.TestCase):
    def test_decay_multiplier(self):
        cfg = _decay(residual_hl=504, residual_min_lb_dec_exp=2)
        self.assertEqual(cfg.resolved_specs()[0].residual_cfg.min_history, 1008)

    def test_decay_absolute(self):
        cfg = _decay(
            residual_hl=252,
            residual_min_lb_type_dec_exp=AbsOrMult.ABSOLUTE,
            residual_min_lb_dec_exp=600,
        )
        self.assertEqual(cfg.resolved_specs()[0].residual_cfg.min_history, 600)

    def test_eq_rolling_min_history_equals_lookback(self):
        cfg = PanelBatchConfig(
            residual_mode=ResidualMode.EQ_ROLLING,
            residual_lb=63, hedge_ratio_lb=63, mr_diag_lb=63,
        )
        rc = cfg.resolved_specs()[0].residual_cfg
        self.assertEqual(rc.min_history, 63)
        self.assertEqual(rc.lookback, 63)
        self.assertIsNone(rc.half_life)

    def test_eq_expanding_min_history_absolute(self):
        cfg = PanelBatchConfig(
            residual_mode=ResidualMode.EQ_EXPANDING,
            hedge_ratio_lb=252, mr_diag_lb=252, residual_min_lb_eq_exp=800,
        )
        self.assertEqual(cfg.resolved_specs()[0].residual_cfg.min_history, 800)


class TestSweep(unittest.TestCase):
    def test_swept_hl_produces_one_spec_per_point(self):
        cfg = _decay(residual_hl=[126, 252, 504], hedge_ratio_lb=[21, 42, 63], mr_diag_lb=252)
        specs = cfg.resolved_specs()
        self.assertEqual([s.sweep_value for s in specs], [126, 252, 504])
        self.assertEqual([s.hedge_ratio_lb for s in specs], [21, 42, 63])
        self.assertEqual([s.mr_diag_lb for s in specs], [252, 252, 252])  # scalar broadcast
        self.assertEqual([s.residual_cfg.half_life for s in specs], [126, 252, 504])


class TestValidation(unittest.TestCase):
    def test_both_hl_and_lb_set_raises(self):
        with self.assertRaises(ValueError):
            _decay(residual_lb=21)  # decay mode with residual_lb also set

    def test_mode_requiring_one_but_both_unset_raises(self):
        with self.assertRaises(ValueError):
            PanelBatchConfig(
                residual_mode=ResidualMode.DECAY_EXPANDING,  # needs residual_hl
                residual_hl=None,
                hedge_ratio_lb=252, mr_diag_lb=252,
                residual_min_lb_type_dec_exp=AbsOrMult.MULTIPLIER,
                residual_min_lb_dec_exp=2,
            )

    def test_eq_expanding_with_a_timescale_raises(self):
        with self.assertRaises(ValueError):
            PanelBatchConfig(
                residual_mode=ResidualMode.EQ_EXPANDING,
                residual_hl=504,  # forbidden for eq_expanding
                hedge_ratio_lb=252, mr_diag_lb=252, residual_min_lb_eq_exp=252,
            )

    def test_mismatched_list_lengths_raise(self):
        with self.assertRaises(ValueError):
            _decay(residual_hl=[126, 252, 504], hedge_ratio_lb=[21, 42])  # 2 != 3

    def test_list_window_when_not_swept_raises(self):
        with self.assertRaises(ValueError):
            _decay(residual_hl=504, hedge_ratio_lb=[21, 42])  # scalar sweep, list window

    def test_missing_hedge_ratio_lb_raises(self):
        with self.assertRaises(ValueError):
            PanelBatchConfig(
                residual_mode=ResidualMode.DECAY_EXPANDING, residual_hl=504,
                mr_diag_lb=252,
                residual_min_lb_type_dec_exp=AbsOrMult.MULTIPLIER, residual_min_lb_dec_exp=2,
            )

    def test_decay_min_lb_fields_on_wrong_mode_raise(self):
        with self.assertRaises(ValueError):
            PanelBatchConfig(
                residual_mode=ResidualMode.EQ_EXPANDING,
                hedge_ratio_lb=252, mr_diag_lb=252, residual_min_lb_eq_exp=252,
                residual_min_lb_dec_exp=2,  # only valid for decay
            )

    def test_window_mode_derived(self):
        self.assertEqual(
            PanelBatchConfig(
                residual_mode=ResidualMode.EQ_ROLLING, residual_lb=21,
                hedge_ratio_lb=21, mr_diag_lb=21,
            ).residual_window_mode,
            "rolling",
        )
        self.assertEqual(_decay().residual_window_mode, "expanding")


if __name__ == "__main__":
    unittest.main()
