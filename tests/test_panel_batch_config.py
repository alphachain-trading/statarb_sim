"""
Unit tests for the mode-discriminated CausalResidualConfig and the
residual_configs-list-based PanelBatchConfig.

Covers, without touching disk or market data:
  1. CausalResidualConfig.key byte-identity for the canonical decay_expanding
     reference config (must stay exp_hl504_mh1008_rf), plus the two other
     mode key shapes, plus the remove_residual_pcs collision-gap fix.
  2. min_history resolution per mode (multiplier / absolute / lb-equals).
  3. CausalResidualConfig construction-time validation (mode-specific
     required/forbidden fields, no-default enforcement, from_dict contract).
  4. PanelBatchConfig construction-time validation (non-empty list,
     hedge_ratio_lb/mr_diag_lb scalar-vs-list rules) and window broadcasting.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode, AbsOrMult
from src.candidates.panel_batch import PanelBatchConfig


def _decay(**overrides):
    base = dict(
        mode=ResidualMode.DECAY_EXPANDING,
        subtract_risk_free=True,
        hl=504,
        min_lb_type_dec_exp=AbsOrMult.MULTIPLIER,
        min_lb_dec_exp=2,
    )
    base.update(overrides)
    return CausalResidualConfig(**base)


class TestKeyByteIdentity(unittest.TestCase):
    def test_decay_expanding_canonical_key_is_byte_identical(self):
        """The redesigned surface must reproduce exp_hl504_mh1008_rf exactly."""
        self.assertEqual(_decay().key, "exp_hl504_mh1008_rf")

    def test_eq_expanding_key_shape(self):
        cfg = CausalResidualConfig(
            mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=True, min_lb_eq_exp=252,
        )
        self.assertEqual(cfg.key, "exp_mh252_rf")

    def test_eq_rolling_key_shape(self):
        cfg = CausalResidualConfig(mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False, lb=21)
        # min_history == lb -> no _mh segment; no rf.
        self.assertEqual(cfg.key, "rol_lb21")

    def test_remove_residual_pcs_nonzero_produces_distinct_key(self):
        """Closes the collision gap: two configs differing only in
        remove_residual_pcs must no longer share a key."""
        base = _decay(remove_residual_pcs=0)
        pc1 = _decay(remove_residual_pcs=1)
        self.assertEqual(base.key, "exp_hl504_mh1008_rf")
        self.assertEqual(pc1.key, "exp_hl504_mh1008_pc1_rf")
        self.assertNotEqual(base.key, pc1.key)


class TestMinHistoryResolution(unittest.TestCase):
    def test_decay_multiplier(self):
        cfg = _decay(hl=504, min_lb_dec_exp=2)
        self.assertEqual(cfg.min_history, 1008)

    def test_decay_absolute(self):
        cfg = _decay(hl=252, min_lb_type_dec_exp=AbsOrMult.ABSOLUTE, min_lb_dec_exp=600)
        self.assertEqual(cfg.min_history, 600)

    def test_eq_rolling_min_history_equals_lb(self):
        cfg = CausalResidualConfig(mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False, lb=63)
        self.assertEqual(cfg.min_history, 63)
        self.assertEqual(cfg.lb, 63)
        self.assertIsNone(cfg.hl)
        # compatibility properties used unprefixed by fitting code
        self.assertEqual(cfg.lookback, 63)
        self.assertIsNone(cfg.half_life)
        self.assertEqual(cfg.window_mode, "rolling")

    def test_eq_expanding_min_history_absolute(self):
        cfg = CausalResidualConfig(
            mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False, min_lb_eq_exp=800,
        )
        self.assertEqual(cfg.min_history, 800)
        self.assertEqual(cfg.window_mode, "expanding")
        self.assertIsNone(cfg.half_life)
        self.assertIsNone(cfg.lookback)

    def test_decay_expanding_compat_properties(self):
        cfg = _decay(hl=504)
        self.assertEqual(cfg.half_life, 504)
        self.assertIsNone(cfg.lookback)
        self.assertEqual(cfg.window_mode, "expanding")


class TestCausalResidualConfigValidation(unittest.TestCase):
    def test_no_defaults_for_mode_and_subtract_risk_free(self):
        with self.assertRaises(TypeError):
            CausalResidualConfig()  # type: ignore[call-arg]

    def test_eq_rolling_requires_lb(self):
        with self.assertRaises(ValueError):
            CausalResidualConfig(mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False)

    def test_eq_rolling_forbids_hl_and_min_lb_eq_exp(self):
        with self.assertRaises(ValueError):
            CausalResidualConfig(mode=ResidualMode.EQ_ROLLING, subtract_risk_free=False, lb=21, hl=504)

    def test_decay_expanding_requires_hl_and_min_lb_fields(self):
        with self.assertRaises(ValueError):
            CausalResidualConfig(mode=ResidualMode.DECAY_EXPANDING, subtract_risk_free=True)
        with self.assertRaises(ValueError):
            CausalResidualConfig(mode=ResidualMode.DECAY_EXPANDING, subtract_risk_free=True, hl=504)

    def test_decay_expanding_forbids_lb(self):
        with self.assertRaises(ValueError):
            _decay(lb=21)

    def test_eq_expanding_requires_min_lb_eq_exp(self):
        with self.assertRaises(ValueError):
            CausalResidualConfig(mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False)

    def test_eq_expanding_forbids_lb_and_hl(self):
        with self.assertRaises(ValueError):
            CausalResidualConfig(
                mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False, min_lb_eq_exp=252, hl=504,
            )

    def test_remove_residual_pcs_negative_raises(self):
        with self.assertRaises(ValueError):
            _decay(remove_residual_pcs=-1)

    def test_from_dict_requires_mode_and_subtract_risk_free(self):
        """Clean break — no parsing of the old flat format."""
        with self.assertRaises(KeyError):
            CausalResidualConfig.from_dict({"window_mode": "expanding", "half_life": 504})

    def test_from_dict_roundtrip(self):
        cfg = _decay()
        from dataclasses import asdict
        import json
        raw = json.loads(json.dumps(asdict(cfg)))
        back = CausalResidualConfig.from_dict(raw)
        self.assertEqual(back.key, cfg.key)
        self.assertEqual(back.min_history, cfg.min_history)


class TestPanelBatchConfigValidation(unittest.TestCase):
    def test_empty_residual_configs_raises(self):
        with self.assertRaises(ValueError):
            PanelBatchConfig(residual_configs=[], hedge_ratio_lb=252, mr_diag_lb=252)

    def test_single_config_list_window_raises(self):
        with self.assertRaises(ValueError):
            PanelBatchConfig(residual_configs=[_decay()], hedge_ratio_lb=[252], mr_diag_lb=252)

    def test_mismatched_list_length_raises(self):
        rcs = [_decay(hl=h) for h in (126, 252, 504)]
        with self.assertRaises(ValueError):
            PanelBatchConfig(residual_configs=rcs, hedge_ratio_lb=[21, 42], mr_diag_lb=252)

    def test_scalar_broadcasts_across_sweep(self):
        rcs = [_decay(hl=h) for h in (126, 252, 504)]
        cfg = PanelBatchConfig(residual_configs=rcs, hedge_ratio_lb=252, mr_diag_lb=63)
        self.assertEqual(cfg.resolved_windows(), [(252, 63), (252, 63), (252, 63)])

    def test_list_windows_pair_by_index(self):
        rcs = [_decay(hl=h) for h in (126, 252, 504)]
        cfg = PanelBatchConfig(residual_configs=rcs, hedge_ratio_lb=[21, 42, 63], mr_diag_lb=252)
        self.assertEqual(cfg.resolved_windows(), [(21, 252), (42, 252), (63, 252)])

    def test_single_config_scalar_windows_ok(self):
        cfg = PanelBatchConfig(residual_configs=[_decay()], hedge_ratio_lb=252, mr_diag_lb=252)
        self.assertEqual(cfg.resolved_windows(), [(252, 252)])


if __name__ == "__main__":
    unittest.main()
