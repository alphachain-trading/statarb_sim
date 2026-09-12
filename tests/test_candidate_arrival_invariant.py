"""
The `<= t` alignment invariant (F2_spec.md §1.5): a candidate's own refit date
(`asof_date` / `entry_asof_date`) is always at or before the trading day it is
used on, and a fit at date d never sees data strictly after d. Structural today,
never asserted before this test (F2_spec.md §1.5, F2 C2).

Two independent halves:
(a) entry_asof_date <= entry_date on real closed trades, run end-to-end through
    the shipped run_panel_batch -> run_from_config path (reusing the Track B
    baseline harness's own small-universe config, not reimplementing it).
(b) a "perturb-the-future" test on _slice_fit_window: mutating returns strictly
    after the fit date must not change the sliced window, i.e. no lookahead.
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
from src.residuals.causal_residuals import CausalResidualConfig, ResidualMode, _slice_fit_window


class TestEntryAsofDateInvariant(unittest.TestCase):
    """Half (a): entry_asof_date <= entry_date on a real, small end-to-end run."""

    @classmethod
    def setUpClass(cls):
        import b_baseline_harness as harness

        _, panel_results = harness._build_panels()
        active_groups = sorted({
            group_id for (group_id, _residual_key), pr in panel_results.items()
            if len(pr.panel) > 0
        })
        if not active_groups:
            cls.trades_df = None
            return

        sim_config = harness._build_sim_config(active_groups)

        from src.simulator.simulator_factory import run_from_config
        result = run_from_config(sim_config)
        cls.trades_df = result.closed_trades_df()

    def test_has_trades(self):
        """Sanity check on the fixture: a vacuously-true invariant over zero
        trades would not be evidence of anything."""
        self.assertIsNotNone(self.trades_df)
        self.assertGreater(len(self.trades_df), 0)

    def test_entry_asof_date_le_entry_date(self):
        df = self.trades_df
        self.assertIn("entry_asof_date", df.columns)
        self.assertIn("entry_date", df.columns)
        violations = df[df["entry_asof_date"] > df["entry_date"]]
        self.assertTrue(
            violations.empty,
            f"entry_asof_date > entry_date for {len(violations)} trade(s):\n{violations}",
        )


class TestSliceFitWindowNoLookahead(unittest.TestCase):
    """Half (b): perturb-the-future on _slice_fit_window."""

    def _bundle(self) -> GroupReturnBundle:
        dates = pd.bdate_range("2020-01-01", periods=40)
        rng = np.random.default_rng(0)
        members = ["AAA", "BBB"]
        proxy = "PROXY"
        bench = "BENCH"
        cols = members + [proxy, bench]
        data = pd.DataFrame(
            rng.normal(scale=0.01, size=(len(dates), len(cols))),
            index=dates,
            columns=cols,
        )
        return GroupReturnBundle(
            group_id="g",
            member_returns=data[members],
            proxy_returns=data[proxy].rename(proxy),
            benchmark_returns=data[bench].rename(bench),
            aligned_returns=data,
            risk_free_returns=pd.Series(0.0, index=dates, name="RF"),
        )

    def _expanding_cfg(self) -> CausalResidualConfig:
        return CausalResidualConfig(
            mode=ResidualMode.EQ_EXPANDING,
            subtract_risk_free=False,
            min_lb_eq_exp=5,
        )

    def _rolling_cfg(self) -> CausalResidualConfig:
        return CausalResidualConfig(
            mode=ResidualMode.EQ_ROLLING,
            subtract_risk_free=False,
            lb=10,
        )

    def _assert_future_mutation_is_invisible(self, cfg: CausalResidualConfig):
        bundle = self._bundle()
        fit_date = bundle.aligned_returns.index[20]

        before = _slice_fit_window(bundle=bundle, date=fit_date, cfg=cfg)

        mutated = bundle.aligned_returns.copy()
        after_mask = mutated.index > fit_date
        mutated.loc[after_mask] = mutated.loc[after_mask] + 999.0
        mutated_bundle = GroupReturnBundle(
            group_id=bundle.group_id,
            member_returns=bundle.member_returns,
            proxy_returns=bundle.proxy_returns,
            benchmark_returns=bundle.benchmark_returns,
            aligned_returns=mutated,
            risk_free_returns=bundle.risk_free_returns,
        )

        after = _slice_fit_window(bundle=mutated_bundle, date=fit_date, cfg=cfg)

        pd.testing.assert_frame_equal(before, after, check_exact=True)

    def test_no_lookahead_expanding(self):
        self._assert_future_mutation_is_invisible(self._expanding_cfg())

    def test_no_lookahead_rolling(self):
        self._assert_future_mutation_is_invisible(self._rolling_cfg())


if __name__ == "__main__":
    unittest.main()
