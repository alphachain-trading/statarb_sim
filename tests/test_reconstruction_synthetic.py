"""
Reconstruction fidelity on hermetic synthetic data (see tests/_synthetic.py).

Same contract as test_reconstruction_fidelity.py — every debug-sample row recorded by a
live run is reproduced exactly (plain float equality, no tolerance) from artifacts
persisted by that run, by a fresh signal generator — but without the gitignored
``data/`` directory, so it runs on a fresh clone.
"""

import unittest
from typing import Any, cast

import pandas as pd
import pytest

from src.simulator.candidate_signals import CandidateSignalGenerator
from src.simulator.config import DebugSampleConfig
from src.simulator.reconstruction import (
    reconstruct_candidate_view,
    reconstruct_level_and_zscore,
    reconstruct_position_view,
)
from tests._synthetic import build_sim_config, make_umd, patched_universe

GROUPS = ["alpha", "beta"]
# Spreads that trade in the C1 golden run (tests/golden/c1_main_run.json); the
# position-view coverage test below fails loudly if they ever stop trading.
SAMPLE_SPREAD_IDS = ("ALP0B|ALP2B", "BET0B|BET2B")


@pytest.mark.slow
class TestReconstructionFidelitySynthetic(unittest.TestCase):
    _scratch_dir: str
    debug_sample_df: pd.DataFrame
    open_intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]
    precomputed: dict[tuple[str, str], Any]
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]]
    selected_panel: pd.DataFrame
    signal_generator: CandidateSignalGenerator

    @classmethod
    def setUpClass(cls):
        import dataclasses
        import tempfile

        from src.candidates.candidate_panel import (
            load_candidate_panel_result,
            load_weights,
            weights_lookup_from_df,
        )
        from src.residuals.causal_residuals import load_residual_params
        from src.simulator.config import PersistenceConfig
        from src.simulator.simulation_persistence import _resolve_run_dir
        from src.simulator.simulator_factory import (
            create_simulator,
            generate_candidate_panels_by_group,
            run_from_config,
        )

        # F2 C6c (F1): this test's whole point is verifying reconstruction
        # from persisted artifacts, independently reloaded from the live
        # run's own CandidateSignalGenerator instance -- previously that
        # meant the offline run_panel_batch path, the only thing that wrote
        # candidates/weights/residual_params to disk. Now the live
        # in-simulator generation path persists them itself, into the run's
        # own run_dir (persist_live_candidate_artifacts, simulator_factory.py),
        # so this fixture runs the harness's own live-generation sim_config
        # with persistence enabled and reloads from run_dir/"candidates"
        # directly -- no run_panel_batch, no CANDIDATE_PANELS_ROOT.
        cls._scratch_dir = tempfile.mkdtemp(prefix="f2_reconstruction_fidelity_")

        umd_synth = make_umd()
        prelim_sim_config = build_sim_config(GROUPS)
        with patched_universe(umd_synth):
            panel_results = generate_candidate_panels_by_group(prelim_sim_config)
        active_groups = sorted(
            {
                group_id
                for (group_id, _residual_key), v in panel_results.items()
                if len(v["panel"]) > 0
            }
        )

        sim_config = (
            prelim_sim_config if active_groups == GROUPS else build_sim_config(active_groups)
        )
        sim_config = dataclasses.replace(
            sim_config,
            persistence=PersistenceConfig(enabled=True, output_dir=cls._scratch_dir),
            debug_sample=DebugSampleConfig(n_pairs=2, seed=0, spread_ids=SAMPLE_SPREAD_IDS),
        )

        with patched_universe(umd_synth):
            result = run_from_config(sim_config)
        cls.debug_sample_df = result.debug_sample_df()
        assert len(cls.debug_sample_df) > 0, "debug sample captured no rows"

        trades_df = result.closed_trades_df()
        cls.open_intervals = {}
        for row in cast("Any", trades_df.itertuples(index=False)):
            cls.open_intervals.setdefault(row.candidate_id, []).append(
                (
                    pd.Timestamp(cast("Any", row.entry_date)),
                    pd.Timestamp(cast("Any", row.exit_date)),
                )
            )

        # Fresh signal generator, built from persisted artifacts independently
        # of the live run's own instance -- this is the actual reconstruction
        # path, not the same object the run just used.
        assert result.run_id is not None, "persistence was enabled; run_id must be set"
        run_dir = _resolve_run_dir(sim_config.persistence, result.run_id)
        candidates_dir = run_dir / "candidates"

        residual_key = cast("Any", sim_config.residual).key
        weight_dfs = []
        cls.precomputed = {}
        panel_frames = []
        for group_id in active_groups:
            stem = f"{group_id}_{residual_key}"
            panel_result = load_candidate_panel_result(candidates_dir, stem)
            panel_frames.append(panel_result.panel)
            weight_dfs.append(load_weights(str(candidates_dir / f"{stem}_weights.parquet")))
            cls.precomputed[(group_id, residual_key)] = load_residual_params(
                str(candidates_dir / f"{stem}_residual_params.parquet")
            )

        cls.weights_lookup = weights_lookup_from_df(pd.concat(weight_dfs, ignore_index=True))
        residual_configs = {residual_key: cast("Any", sim_config.residual)}

        umd = umd_synth
        cls.selected_panel = result.selected_panel

        sim2 = create_simulator(
            config=sim_config,
            umd=umd,
            residual_configs=residual_configs,
            precomputed_residual_params=cls.precomputed,
            weights_lookup=cls.weights_lookup,
        )
        cls.signal_generator = sim2.signal_generator

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls._scratch_dir, ignore_errors=True)

    def _is_position_view(self, row) -> bool:
        for entry, exit_ in self.open_intervals.get(row.candidate_id, []):
            if entry <= row.date <= exit_:
                return True
        return False

    def test_sample_has_both_views(self):
        n_position = sum(
            self._is_position_view(r) for r in self.debug_sample_df.itertuples(index=False)
        )
        n_candidate = len(self.debug_sample_df) - n_position
        self.assertGreater(n_position, 0, "no position-view rows in the sample")
        self.assertGreater(n_candidate, 0, "no candidate-view rows in the sample")

    def test_every_row_reconstructs_exactly(self):
        n_checked = 0
        for row in cast("Any", self.debug_sample_df.itertuples(index=False)):
            state = reconstruct_level_and_zscore(
                signal_generator=self.signal_generator,
                weights_lookup=self.weights_lookup,
                group_id=row.group_id,
                residual_key=row.residual_key,
                spread_id=row.spread_id,
                asof_date=row.asof_date,
                date=row.date,
                timescale_label=row.timescale_label,
                candidate_id=row.candidate_id,
            )
            self.assertEqual(state.level, row.level, msg=f"level mismatch: {row}")
            self.assertEqual(state.z_score, row.z_score, msg=f"z_score mismatch: {row}")
            n_checked += 1
        self.assertEqual(n_checked, len(self.debug_sample_df))

    def test_position_view_reconstructs_exactly_via_dedicated_function(self):
        checked = 0
        for row in cast("Any", self.debug_sample_df.itertuples(index=False)):
            if not self._is_position_view(row):
                continue
            state = reconstruct_position_view(
                signal_generator=self.signal_generator,
                weights_lookup=self.weights_lookup,
                group_id=row.group_id,
                residual_key=row.residual_key,
                spread_id=row.spread_id,
                entry_asof_date=row.asof_date,
                date=row.date,
                timescale_label=row.timescale_label,
                candidate_id=row.candidate_id,
            )
            self.assertEqual(state.level, row.level)
            self.assertEqual(state.z_score, row.z_score)
            checked += 1
        self.assertGreater(checked, 0)

    def test_candidate_view_reconstructs_exactly_via_dedicated_function(self):
        """
        reconstruct_candidate_view always resolves the *latest* asof_date <=
        date for a spread. Most debug-sample rows are NOT that: the pair-mode
        activation rule (candidate_activation.py::_activate_pair_candidate)
        keeps every non-open ref for a spread alive, not just the open one,
        for as long as *any* position is open on that spread -- so several
        stale, superseded refs can be genuinely tracked (and captured) on the
        same date alongside the one that is actually latest. Comparing against
        the correct target: per (spread_id, date), the row with the max
        asof_date is what a fresh "candidate view" query should reproduce.
        """
        checked = 0
        one_row_dedupe = self.debug_sample_df.drop_duplicates(
            subset=["spread_id", "date", "asof_date", "candidate_id"]
        )
        for (spread_id, date), group in cast("Any", one_row_dedupe.groupby(["spread_id", "date"])):
            latest_row = cast("Any", group.loc[group["asof_date"].idxmax()])
            state = reconstruct_candidate_view(
                signal_generator=self.signal_generator,
                weights_lookup=self.weights_lookup,
                selected_panel=self.selected_panel,
                group_id=latest_row["group_id"],
                residual_key=latest_row["residual_key"],
                spread_id=spread_id,
                date=date,
                timescale_label=latest_row["timescale_label"],
            )
            self.assertIsNotNone(state)
            assert state is not None  # narrowing for mypy
            self.assertEqual(state.level, latest_row["level"])
            self.assertEqual(state.z_score, latest_row["z_score"])
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
