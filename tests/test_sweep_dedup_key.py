"""
Regression test for the sweep dedup key.

Guards the bug where skip_existing gated on SweepConfig.alias, a lossy
display string built from a subset of fields. It omits
candidate_panel_subdir and interval_scoring weights, so runs differing
only in those collide onto one alias — and skip_existing=True silently
skips the second as a false match. dedup_key hashes the fully-built
SimulatorConfig instead, which covers every field.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.config import FeatureIntervalSpec, IntervalScoringConfig, ZScoreConfig
from src.simulator.sweep_runner import SweepConfig, dedup_key


def _scoring(weight: float) -> IntervalScoringConfig:
    return IntervalScoringConfig(
        feature_specs=(
            FeatureIntervalSpec(
                feature="x_area_asymmetry_ewm",
                interval_limits=(float("-inf"), 0.0, float("inf")),
                interval_weights=(1.0, weight),
            ),
        ),
        feature_weights={"x_area_asymmetry_ewm": 1.0},
    )


class TestSweepDedupKey(unittest.TestCase):
    def test_panel_subdir_changes_dedup_key(self):
        """Same sweep on two candidate panels must not collide."""
        sweep = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm")
        self.assertNotIn("V2.pair", sweep.alias)
        self.assertNotIn("V3.pair", sweep.alias)
        self.assertEqual(
            dedup_key(sweep, "V2.pair"), dedup_key(sweep, "V2.pair"),
            "key must be stable for identical inputs",
        )
        self.assertNotEqual(
            dedup_key(sweep, "V2.pair"), dedup_key(sweep, "V3.pair"),
            "panel subdir must change the dedup key",
        )

    def test_interval_scoring_changes_dedup_key(self):
        """The collision alias hides: same alias, different interval weights."""
        a = SweepConfig(entry_z=1.8, interval_scoring=_scoring(1.25))
        b = SweepConfig(entry_z=1.8, interval_scoring=_scoring(1.75))
        self.assertEqual(a.alias, b.alias, "precondition: alias omits interval weights")
        self.assertNotEqual(
            dedup_key(a, "V4"), dedup_key(b, "V4"),
            "interval_scoring weights must change the dedup key",
        )

    def test_identical_configs_share_dedup_key(self):
        """Stability: identical configs must key the same, or skip never skips."""
        a = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm")
        b = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm")
        self.assertEqual(dedup_key(a, "V4"), dedup_key(b, "V4"))

    def test_multi_timescale_overrides_change_dedup_key(self):
        """Nested ZScoreConfig lists must reach the key too."""
        a = SweepConfig(z_score_overrides=[
            ZScoreConfig(lookback=21, method="ewm", residual_key="exp_hl126_mh252"),
        ])
        b = SweepConfig(z_score_overrides=[
            ZScoreConfig(lookback=42, method="ewm", residual_key="exp_hl126_mh252"),
        ])
        self.assertNotEqual(dedup_key(a, "V4"), dedup_key(b, "V4"))


if __name__ == "__main__":
    unittest.main()
