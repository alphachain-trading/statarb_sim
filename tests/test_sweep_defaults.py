"""
Tests for the DEFAULT_CONFIGS registry.

Two things to prove:
  1. standard_v1 is byte-identical to the literals _build_sim_config carried
     before the registry — introducing the registry changed no behaviour.
     test_standard_v1_output_is_hash_stable is the end-to-end proof (the
     full SimulatorConfig hash is unchanged); the per-field test proves it
     at the bundle level.
  2. merge_defaults fails loud if the bundle and the sweep-derived kwargs
     ever name the same field — the guard against silent precedence bugs.
"""
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simulator.config import (
    ActivationConfig,
    CapitalConfig,
    ExecutionConfig,
    MRDiagnosticsConfig,
    PerformanceConfig,
    PersistenceConfig,
)
from src.candidates.candidate_selector import CandidateSelectionConfig
from src.simulator.sweep_defaults import (
    DEFAULT_CONFIGS,
    get_default_bundle,
    merge_defaults,
)
from src.simulator.sweep_runner import SweepConfig, dedup_key

# Hash of a canonical SweepConfig recorded before the registry existed. The
# hash covers SimulatorConfig, not SweepConfig, so extracting the defaults
# into standard_v1 must leave it byte-identical.
_PRE_REGISTRY_HASH = "73aabb33b874e0527d2b4a018d1bcc39"


class TestSweepDefaults(unittest.TestCase):
    def test_standard_v1_matches_hardcoded_literals(self):
        """Each bundle value must equal the literal _build_sim_config inlined."""
        b = DEFAULT_CONFIGS["standard_v1"]
        self.assertEqual(b["capital"], CapitalConfig(total_capital=1_000_000.0))
        self.assertEqual(b["candidate_selection"], CandidateSelectionConfig(
            allowed_candidate_subtypes=("pca",), require_is_valid=True,
        ))
        self.assertEqual(b["activation"], ActivationConfig(
            one_active_per_group=False, switch_only_when_flat=False,
        ))
        self.assertEqual(b["diagnostics"], MRDiagnosticsConfig(lookback=252, compute_frequency="off"))
        self.assertEqual(b["execution"], ExecutionConfig(
            allow_fractional_shares=False, share_rounding="nearest",
        ))
        self.assertEqual(b["performance"], PerformanceConfig(
            enabled=True, metrics_table=True, report_html=True,
            benchmark_ticker=None, annualization_factor=252, per_group_breakdown=True,
        ))
        self.assertEqual(b["persistence"], PersistenceConfig(enabled=True))

    def test_standard_v1_keys_are_exactly_the_non_swept_fields(self):
        """Guards against a field silently drifting into or out of the bundle."""
        self.assertEqual(set(DEFAULT_CONFIGS["standard_v1"]), {
            "capital", "candidate_selection", "activation",
            "diagnostics", "execution", "performance", "persistence",
        })

    def test_standard_v1_output_is_hash_stable(self):
        """End-to-end: the full SimulatorConfig hash is unchanged by the registry."""
        sweep = SweepConfig(entry_z=2.0, z_lookback=21, z_method="ewm",
                            candidate_panel_subdir="V2.pair")
        self.assertEqual(dedup_key(sweep), _PRE_REGISTRY_HASH)

    def test_overlap_guard_raises_naming_the_key(self):
        """A field set in both buckets must raise, not silently pick a winner."""
        sweep_derived = {"data": object(), "capital": object()}
        bundle = {"capital": object(), "persistence": object()}
        with self.assertRaises(ValueError) as cm:
            merge_defaults(sweep_derived, bundle, "standard_v1")
        self.assertIn("capital", str(cm.exception))

    def test_no_overlap_unions_both_buckets(self):
        out = merge_defaults({"data": 1, "run": 2}, {"capital": 3}, "standard_v1")
        self.assertEqual(out, {"data": 1, "run": 2, "capital": 3})

    def test_unknown_bundle_raises_listing_registered(self):
        with self.assertRaises(KeyError) as cm:
            get_default_bundle("does_not_exist")
        self.assertIn("standard_v1", str(cm.exception))

    def test_registry_is_read_only(self):
        """MappingProxyType — a bundle must not be mutable in place."""
        with self.assertRaises(TypeError):
            DEFAULT_CONFIGS["standard_v1"]["capital"] = None


if __name__ == "__main__":
    unittest.main()
