"""
Regression test for the sweep dedup key.

Guards the bug where skip_existing gated on SweepConfig.alias, a lossy
display string built from a subset of fields. It omits
candidate_panel_subdir and interval_scoring weights, so runs differing
only in those collide onto one alias — and skip_existing=True silently
skips the second as a false match. dedup_key hashes the fully-built
SimulatorConfig instead, which covers every field.

Also covers the single-source panel requirement: candidate_panel_subdir
lives on the SweepConfig and is required, so an unset panel raises loud
(naming the offending RUNS[i]) rather than silently running the default.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.config import FeatureIntervalSpec, IntervalScoringConfig, ZScoreConfig
from src.simulator.sweep_runner import SweepConfig, _resolve_panel_subdir, dedup_key


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
        v2 = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm", candidate_panel_subdir="V2.pair")
        v3 = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm", candidate_panel_subdir="V3.pair")
        self.assertNotIn("V2.pair", v2.alias)
        self.assertNotIn("V3.pair", v3.alias)
        self.assertNotEqual(
            dedup_key(v2), dedup_key(v3),
            "panel subdir must change the dedup key",
        )

    def test_interval_scoring_changes_dedup_key(self):
        """The collision alias hides: same alias, different interval weights."""
        a = SweepConfig(entry_z=1.8, interval_scoring=_scoring(1.25), candidate_panel_subdir="V4")
        b = SweepConfig(entry_z=1.8, interval_scoring=_scoring(1.75), candidate_panel_subdir="V4")
        self.assertEqual(a.alias, b.alias, "precondition: alias omits interval weights")
        self.assertNotEqual(
            dedup_key(a), dedup_key(b),
            "interval_scoring weights must change the dedup key",
        )

    def test_identical_configs_share_dedup_key(self):
        """Stability: identical configs must key the same, or skip never skips."""
        a = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm", candidate_panel_subdir="V4")
        b = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm", candidate_panel_subdir="V4")
        self.assertEqual(dedup_key(a), dedup_key(b))

    def test_multi_timescale_overrides_change_dedup_key(self):
        """Nested ZScoreConfig lists must reach the key too."""
        a = SweepConfig(candidate_panel_subdir="V4", z_score_overrides=[
            ZScoreConfig(lookback=21, method="ewm", residual_key="exp_hl126_mh252"),
        ])
        b = SweepConfig(candidate_panel_subdir="V4", z_score_overrides=[
            ZScoreConfig(lookback=42, method="ewm", residual_key="exp_hl126_mh252"),
        ])
        self.assertNotEqual(dedup_key(a), dedup_key(b))

    def test_unset_panel_raises(self):
        """Panel is required on the SweepConfig — unset must fail loud."""
        with self.assertRaises(ValueError) as cm:
            _resolve_panel_subdir(SweepConfig(entry_z=2.0), 0)
        self.assertIn("unset", str(cm.exception))

    def test_empty_string_panel_raises(self):
        """"" is the unset sentinel, so it raises like any other unset panel."""
        with self.assertRaises(ValueError):
            _resolve_panel_subdir(SweepConfig(entry_z=2.0, candidate_panel_subdir=""), 0)

    def test_set_panel_propagates_into_key(self):
        """A set panel reaches the hash and distinguishes panels."""
        a = SweepConfig(entry_z=2.0, candidate_panel_subdir="V4")
        b = SweepConfig(entry_z=2.0, candidate_panel_subdir="V3.pair")
        self.assertEqual(_resolve_panel_subdir(a, 0), "V4")
        self.assertNotEqual(dedup_key(a), dedup_key(b))

    def test_error_names_offending_config(self):
        """A long RUNS list needs the failure locatable — index + alias."""
        sweep = SweepConfig(entry_z=1.75)
        with self.assertRaises(ValueError) as cm:
            _resolve_panel_subdir(sweep, 7)
        msg = str(cm.exception)
        self.assertIn("RUNS[7]", msg)
        self.assertIn(sweep.alias, msg)


if __name__ == "__main__":
    unittest.main()
