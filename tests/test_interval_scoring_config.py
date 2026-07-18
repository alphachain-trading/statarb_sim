"""
Regression test for IntervalScoringConfig.feature_weights validation.

The docstring used to promise "missing entries default to 1.0" while
SizingEngine raised KeyError instead, and nothing checked the two agreed.
A config with mismatched keys constructed fine and died at the first trade
sizing, minutes into a run.

The KeyError is correct and stays: defaulting a missing weight would
promote an intentionally-zeroed feature to full weight and silently skew
every trade's size. This just moves the failure to construction time.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.config import FeatureIntervalSpec, IntervalScoringConfig

XAA = "x_area_asymmetry_ewm"
BWG = "bw_gain_ann_norm"


def _specs():
    return (
        FeatureIntervalSpec(
            feature=XAA,
            interval_limits=(float("-inf"), -0.38, float("inf")),
            interval_weights=(1.25, 1.0),
        ),
        FeatureIntervalSpec(
            feature=BWG,
            interval_limits=(float("-inf"), 0.0, float("inf")),
            interval_weights=(1.0, 1.0),
        ),
    )


class TestIntervalScoringConfig(unittest.TestCase):
    def test_matching_keys_construct(self):
        cfg = IntervalScoringConfig(
            feature_specs=_specs(),
            feature_weights={XAA: 1.0, BWG: 0.0},
        )
        self.assertEqual(cfg.feature_weights[BWG], 0.0, "an explicit 0.0 weight must survive")

    def test_weight_without_spec_raises(self):
        """A feature_weights key naming no feature spec."""
        with self.assertRaises(ValueError) as cm:
            IntervalScoringConfig(
                feature_specs=_specs(),
                feature_weights={XAA: 1.0, BWG: 0.0, "typo_feature": 1.0},
            )
        self.assertIn("typo_feature", str(cm.exception))

    def test_spec_without_weight_raises(self):
        """A feature spec with no feature_weights entry — what SizingEngine KeyErrors on."""
        with self.assertRaises(ValueError) as cm:
            IntervalScoringConfig(
                feature_specs=_specs(),
                feature_weights={XAA: 1.0},
            )
        self.assertIn(BWG, str(cm.exception))

    def test_empty_weights_raise(self):
        """The default empty dict was never usable — SizingEngine always raised."""
        with self.assertRaises(ValueError) as cm:
            IntervalScoringConfig(feature_specs=_specs())
        self.assertIn("no feature_weights entry", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
