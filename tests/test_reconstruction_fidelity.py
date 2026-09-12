"""
F2 C5: zero-tolerance fidelity test for the reconstruction function.

Runs a real, small end-to-end simulation (reusing the Track B baseline harness's
own small-universe config, not reimplementing it) with an explicit debug_sample
of two spread_ids known to actually trade on this harness — guaranteeing real
coverage of both reconstruction views, not left to a random seed's luck.

Every debug-sample row (both an actually-open position's own ref, and any other
tracked-but-not-open ref — the pair-mode activation rule keeps every non-open ref
for a spread alive, not just the currently-latest one, for as long as any
position is open on that spread, so several genuinely-tracked, non-latest refs
can coexist on one date) is reconstructed from persisted artifacts
(weights.parquet / residual_params.parquet's in-memory form, loaded independently
of the live run's own CandidateSignalGenerator instance) via
src.simulator.reconstruction.reconstruct_level_and_zscore, and compared
check_exact (plain float equality — no tolerance) against what the live run
actually recorded. The dedicated position-view and candidate-view functions are
tested separately against the specific rows each one is actually supposed to
reproduce -- an open position's own frozen ref, and the latest ref per
(spread_id, date), respectively.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.data.returns import GroupReturnBundle
from src.residuals.causal_residuals import (
    CausalResidualConfig,
    ResidualMode,
    fit_causal_residual_model,
)
from src.simulator.candidate_signals import CandidateSignalGenerator
from src.simulator.config import DebugSampleConfig, MRDiagnosticsConfig, ZScoreConfig
from src.simulator.reconstruction import (
    reconstruct_level_and_zscore,
    reconstruct_candidate_view,
    reconstruct_position_view,
)
from src.simulator.types import CandidateRef

# Two spread_ids confirmed (via docs/refactor/B_baseline.txt) to actually trade
# on the Track B harness's committed config — guarantees non-empty position-view
# coverage, rather than hoping a random sample happens to include a trade.
SAMPLE_SPREAD_IDS = ("APA|VLO", "APD|IP")


class TestReconstructionFidelity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import dataclasses
        import b_baseline_harness as harness
        from src.simulator.simulator_factory import (
            run_from_config,
            create_simulator,
            _load_umd,
            _resolve_residual_configs,
            _load_residual_params,
            _load_weights,
            _load_panels,
        )

        _, panel_results = harness._build_panels()
        active_groups = sorted({
            group_id for (group_id, _residual_key), pr in panel_results.items()
            if len(pr.panel) > 0
        })
        sim_config = harness._build_sim_config(active_groups)
        sim_config = dataclasses.replace(
            sim_config,
            debug_sample=DebugSampleConfig(n_pairs=2, seed=0, spread_ids=SAMPLE_SPREAD_IDS),
        )

        result = run_from_config(sim_config)
        cls.debug_sample_df = result.debug_sample_df()
        assert len(cls.debug_sample_df) > 0, "debug sample captured no rows"

        trades_df = result.closed_trades_df()
        cls.open_intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for row in trades_df.itertuples(index=False):
            cls.open_intervals.setdefault(row.candidate_id, []).append(
                (pd.Timestamp(row.entry_date), pd.Timestamp(row.exit_date))
            )

        # Fresh signal generator, built from persisted artifacts independently
        # of the live run's own instance -- this is the actual reconstruction
        # path, not the same object the run just used.
        umd = _load_umd(sim_config.data)
        z_configs = sim_config.resolved_z_score_configs()
        _, metadata_by_key = _load_panels(sim_config.data, z_configs)
        residual_configs = _resolve_residual_configs(sim_config, metadata_by_key)
        cls.precomputed = _load_residual_params(sim_config.data, z_configs)
        cls.weights_lookup = _load_weights(sim_config.data, z_configs)
        cls.selected_panel = result.selected_panel

        sim2 = create_simulator(
            config=sim_config,
            umd=umd,
            residual_configs=residual_configs,
            precomputed_residual_params=cls.precomputed,
            weights_lookup=cls.weights_lookup,
        )
        cls.signal_generator = sim2.signal_generator

    def _is_position_view(self, row) -> bool:
        for entry, exit_ in self.open_intervals.get(row.candidate_id, []):
            if entry <= row.date <= exit_:
                return True
        return False

    def test_sample_has_both_views(self):
        n_position = sum(self._is_position_view(r) for r in self.debug_sample_df.itertuples(index=False))
        n_candidate = len(self.debug_sample_df) - n_position
        self.assertGreater(n_position, 0, "no position-view rows in the sample")
        self.assertGreater(n_candidate, 0, "no candidate-view rows in the sample")

    def test_every_row_reconstructs_exactly(self):
        n_checked = 0
        for row in self.debug_sample_df.itertuples(index=False):
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
        for row in self.debug_sample_df.itertuples(index=False):
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
        for (spread_id, date), group in one_row_dedupe.groupby(["spread_id", "date"]):
            latest_row = group.loc[group["asof_date"].idxmax()]
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
            self.assertEqual(state.level, latest_row["level"])
            self.assertEqual(state.z_score, latest_row["z_score"])
            checked += 1
        self.assertGreater(checked, 0)


def _bundle(seed: int = 5) -> GroupReturnBundle:
    idx = pd.bdate_range("2019-01-01", periods=60)
    rng = np.random.default_rng(seed)
    members = pd.DataFrame(
        {t: rng.normal(scale=0.01, size=60) for t in ("AAA", "BBB")},
        index=idx,
    )
    proxy = pd.Series(rng.normal(scale=0.01, size=60), index=idx, name="PROXY")
    bench = pd.Series(rng.normal(scale=0.01, size=60), index=idx, name="BENCH")
    aligned = pd.concat([members, proxy, bench], axis=1)
    rf = pd.Series(0.0, index=idx, name="RF")
    return GroupReturnBundle(
        group_id="g",
        member_returns=members,
        proxy_returns=proxy,
        benchmark_returns=bench,
        aligned_returns=aligned,
        risk_free_returns=rf,
    )


def _mutate_after(bundle: GroupReturnBundle, cutoff: pd.Timestamp) -> GroupReturnBundle:
    aligned = bundle.aligned_returns.copy()
    mask = aligned.index > cutoff
    aligned.loc[mask] = aligned.loc[mask] + 999.0
    return GroupReturnBundle(
        group_id=bundle.group_id,
        member_returns=bundle.member_returns,
        proxy_returns=bundle.proxy_returns,
        benchmark_returns=bundle.benchmark_returns,
        aligned_returns=aligned,
        risk_free_returns=bundle.risk_free_returns,
    )


def _make_generator(bundle: GroupReturnBundle, asof_date: pd.Timestamp, model) -> CandidateSignalGenerator:
    gen = CandidateSignalGenerator(
        z_score_configs={"": ZScoreConfig(lookback=5, ddof=1, method="rolling", min_periods=3)},
        diagnostics_config=MRDiagnosticsConfig(lookback=30, compute_frequency="daily"),
        residual_configs={"": CausalResidualConfig(mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False, min_lb_eq_exp=5)},
        umd=None,
    )
    gen._bundle_cache["g"] = bundle
    gen.precomputed_residual_params[("g", "")] = {asof_date: model}
    return gen


class TestReconstructionNoLookaheadAndPcGuard(unittest.TestCase):
    def setUp(self):
        self.bundle = _bundle()
        self.asof_date = self.bundle.aligned_returns.index[15]
        self.t = self.bundle.aligned_returns.index[30]
        fit_bundle = GroupReturnBundle(
            group_id=self.bundle.group_id,
            member_returns=self.bundle.member_returns,
            proxy_returns=self.bundle.proxy_returns,
            benchmark_returns=self.bundle.benchmark_returns,
            aligned_returns=self.bundle.aligned_returns.loc[:self.asof_date],
            risk_free_returns=self.bundle.risk_free_returns,
        )
        cfg = CausalResidualConfig(mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False, min_lb_eq_exp=5)
        self.model = fit_causal_residual_model(bundle=fit_bundle, date=self.asof_date, cfg=cfg)
        self.weights_lookup = {("AAA|BBB", self.asof_date): {"AAA": 1.0, "BBB": -1.0}}

    def test_perturb_the_future_on_reconstruction(self):
        gen_before = _make_generator(self.bundle, self.asof_date, self.model)
        before = reconstruct_level_and_zscore(
            signal_generator=gen_before,
            weights_lookup=self.weights_lookup,
            group_id="g",
            residual_key="",
            spread_id="AAA|BBB",
            asof_date=self.asof_date,
            date=self.t,
        )

        mutated = _mutate_after(self.bundle, self.t)
        gen_after = _make_generator(mutated, self.asof_date, self.model)
        after = reconstruct_level_and_zscore(
            signal_generator=gen_after,
            weights_lookup=self.weights_lookup,
            group_id="g",
            residual_key="",
            spread_id="AAA|BBB",
            asof_date=self.asof_date,
            date=self.t,
        )

        self.assertTrue(before.is_signal_ready)
        self.assertEqual(before.level, after.level)
        self.assertEqual(before.z_score, after.z_score)

    def test_remove_residual_pcs_raises(self):
        fit_bundle = GroupReturnBundle(
            group_id=self.bundle.group_id,
            member_returns=self.bundle.member_returns,
            proxy_returns=self.bundle.proxy_returns,
            benchmark_returns=self.bundle.benchmark_returns,
            aligned_returns=self.bundle.aligned_returns.loc[:self.asof_date],
            risk_free_returns=self.bundle.risk_free_returns,
        )
        pc_cfg = CausalResidualConfig(
            mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False, min_lb_eq_exp=5,
            remove_residual_pcs=1,
        )
        pc_model = fit_causal_residual_model(bundle=fit_bundle, date=self.asof_date, cfg=pc_cfg)
        self.assertIsNotNone(pc_model.pc_components)

        gen = _make_generator(self.bundle, self.asof_date, pc_model)
        with self.assertRaises(NotImplementedError):
            reconstruct_level_and_zscore(
                signal_generator=gen,
                weights_lookup=self.weights_lookup,
                group_id="g",
                residual_key="",
                spread_id="AAA|BBB",
                asof_date=self.asof_date,
                date=self.t,
            )


if __name__ == "__main__":
    unittest.main()
