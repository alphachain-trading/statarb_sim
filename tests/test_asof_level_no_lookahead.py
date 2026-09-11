"""
F2 C3, D3: perturb-the-future test on R2's asof-frozen level pipeline.

The in-memory residual matrix (CandidateSignalGenerator._get_asof_residuals)
holds rows after t by construction -- it is applied once over the group's
full return history and then sliced at read time (F2_spec.md R1/R2). This is
the <= t guard for the change C3 introduces: level, z_score, roll_std and the
MR diagnostics at t must not change when the returns strictly after t are
mutated. C5's reconstruction fidelity test does not replace this -- it will
check the reconstruction matches the live path, not that the live path
itself is lookahead-free.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.returns import GroupReturnBundle
from src.residuals.causal_residuals import (
    CausalResidualConfig,
    ResidualMode,
    fit_causal_residual_model,
)
from src.simulator.candidate_signals import CandidateSignalGenerator
from src.simulator.config import MRDiagnosticsConfig, ZScoreConfig
from src.simulator.types import CandidateRef

GROUP_ID = "g"
RESIDUAL_KEY = ""
N = 60
ASOF_IDX = 20   # the candidate's own refit date
T_IDX = 40      # "today" -- strictly after asof, strictly before the end of history


def _make_bundle(seed: int = 3) -> GroupReturnBundle:
    idx = pd.bdate_range("2018-01-01", periods=N)
    rng = np.random.default_rng(seed)
    members = pd.DataFrame(
        {t: rng.normal(scale=0.01, size=N) for t in ("AAA", "BBB")},
        index=idx,
    )
    proxy = pd.Series(rng.normal(scale=0.01, size=N), index=idx, name="PROXY")
    bench = pd.Series(rng.normal(scale=0.01, size=N), index=idx, name="BENCH")
    aligned = pd.concat([members, proxy, bench], axis=1)
    rf = pd.Series(0.0, index=idx, name="RF")
    return GroupReturnBundle(
        group_id=GROUP_ID,
        member_returns=members,
        proxy_returns=proxy,
        benchmark_returns=bench,
        aligned_returns=aligned,
        risk_free_returns=rf,
    )


def _residual_cfg() -> CausalResidualConfig:
    return CausalResidualConfig(mode=ResidualMode.EQ_EXPANDING, subtract_risk_free=False, min_lb_eq_exp=5)


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


def _build_generator(bundle: GroupReturnBundle, asof_date: pd.Timestamp, model) -> CandidateSignalGenerator:
    gen = CandidateSignalGenerator(
        z_score_configs={RESIDUAL_KEY: ZScoreConfig(lookback=5, ddof=1, method="rolling", min_periods=3)},
        diagnostics_config=MRDiagnosticsConfig(lookback=30, compute_frequency="daily"),
        residual_configs={RESIDUAL_KEY: _residual_cfg()},
        umd=None,  # never touched -- the bundle is pre-populated in the cache below
    )
    gen._bundle_cache[GROUP_ID] = bundle
    gen.precomputed_residual_params[(GROUP_ID, RESIDUAL_KEY)] = {asof_date: model}
    return gen


class TestAsofLevelNoLookahead(unittest.TestCase):
    def setUp(self):
        self.bundle = _make_bundle()
        self.asof_date = self.bundle.aligned_returns.index[ASOF_IDX]
        self.t = self.bundle.aligned_returns.index[T_IDX]
        # Fit the frozen model exactly as panel-build time would: on data <= asof_date only.
        fit_bundle = GroupReturnBundle(
            group_id=self.bundle.group_id,
            member_returns=self.bundle.member_returns,
            proxy_returns=self.bundle.proxy_returns,
            benchmark_returns=self.bundle.benchmark_returns,
            aligned_returns=self.bundle.aligned_returns.loc[:self.asof_date],
            risk_free_returns=self.bundle.risk_free_returns,
        )
        self.model = fit_causal_residual_model(bundle=fit_bundle, date=self.asof_date, cfg=_residual_cfg())
        self.ref = CandidateRef(
            candidate_id="c1",
            spread_id="AAA|BBB",
            group_id=GROUP_ID,
            asof_date=self.asof_date,
            members=("AAA", "BBB"),
            weights=(1.0, -1.0),
            residual_key=RESIDUAL_KEY,
            timescale_label="",
        )

    def _states_at_t(self, bundle: GroupReturnBundle):
        gen = _build_generator(bundle, self.asof_date, self.model)
        return gen.build_candidate_analytics_states(date=self.t, candidate_refs=[self.ref])

    def test_batch_analytics_unchanged_by_future_mutation(self):
        before = self._states_at_t(self.bundle)["c1"]
        after = self._states_at_t(_mutate_after(self.bundle, self.t))["c1"]

        self.assertTrue(before.is_signal_ready)
        self.assertTrue(after.is_signal_ready)
        self.assertEqual(before.level, after.level)
        self.assertEqual(before.z_score, after.z_score)
        self.assertEqual(before.roll_std, after.roll_std)
        self.assertEqual(before.roll_mean, after.roll_mean)
        self.assertEqual(before.mr_score, after.mr_score)
        self.assertEqual(before.kappa, after.kappa)
        self.assertEqual(before.half_life, after.half_life)
        self.assertEqual(before.adf_pvalue, after.adf_pvalue)

    def test_get_level_series_unchanged_by_future_mutation(self):
        before = _build_generator(self.bundle, self.asof_date, self.model).get_level_series(self.ref, self.t)
        after = _build_generator(
            _mutate_after(self.bundle, self.t), self.asof_date, self.model
        ).get_level_series(self.ref, self.t)

        pd.testing.assert_series_equal(before, after, check_exact=True)

    def test_compute_analytics_from_weights_unchanged_by_future_mutation(self):
        weights = {"AAA": 1.0, "BBB": -1.0}

        def _compute(bundle):
            gen = _build_generator(bundle, self.asof_date, self.model)
            return gen.compute_analytics_from_weights(
                date=self.t,
                asof_date=self.asof_date,
                group_id=GROUP_ID,
                candidate_id="c1",
                spread_id="AAA|BBB",
                weights_by_ticker=weights,
                residual_key=RESIDUAL_KEY,
            )

        before = _compute(self.bundle)
        after = _compute(_mutate_after(self.bundle, self.t))

        self.assertTrue(before.is_signal_ready)
        self.assertEqual(before.level, after.level)
        self.assertEqual(before.z_score, after.z_score)
        self.assertEqual(before.roll_std, after.roll_std)
        self.assertEqual(before.mr_score, after.mr_score)
        self.assertEqual(before.kappa, after.kappa)
        self.assertEqual(before.half_life, after.half_life)


if __name__ == "__main__":
    unittest.main()
