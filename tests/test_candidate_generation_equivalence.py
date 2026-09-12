"""
F2 C6a: exact-equality test for the in-simulator candidate generator.

Compares src.simulator.candidate_generation.generate_candidates against
create_pair_candidate_panel (the function run_panel_batch itself calls) on the
Track B baseline harness's two universes (energy, materials) and its own
config — reused, not reimplemented.

Both calls are given the *same* GroupReturnBundle object, built once per group.
An earlier version rebuilt the bundle separately for each side and found a
spurious one-date mismatch (2007-04-06, a market holiday) traceable entirely to
the bundle reconstruction, not to either walk's own logic — reusing one bundle
object eliminates that whole noise category and isolates exactly the thing this
test exists to check: does the outer-date-only walk (candidate_generation.py)
produce the same candidates, weights, and residual params as the offline walk's
daily-grid one, at the dates both of them actually visit.

Covers all scored candidates (including invalid rows, which the panel schema
carries deliberately — see create_pair_candidate_panel's own docstring),
weights, and residual params, check_exact=True. Residual params are compared
only at outer (asof) dates, since the in-simulator generator deliberately does
not fit the offline path's daily grid (F2_spec.md R8/D5/P3) — D5 shows a fit at
date d is a pure function of data <= d with no state carried across the walk,
so a fit at an outer date is identical whether or not intermediate daily fits
happened; P3 shows nothing downstream reads fits at non-outer dates after C3.

create_pair_candidate_panel only exposes weight_rows/fitted_params via disk
persistence (they are local variables otherwise) — persists to a disposable,
uniquely-named subdirectory under CANDIDATE_PANELS_ROOT, cleaned up in
tearDown regardless of outcome, never touching the shared harness directory.
"""
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.candidates.candidate_panel import load_candidate_panel_result, load_weights, weights_lookup_from_df
from src.candidates.pair_candidate_panel_creator import (
    PairSpreadConfig,
    create_pair_candidate_panel,
)
from src.data.returns import build_group_return_bundle
from src.data.universe_config import UniverseConfig
from src.data.universe_loader import UniverseDataLoader
from src.residuals.causal_residuals import load_residual_params
from src.settings import CANDIDATE_PANELS_ROOT, CONFIG_UNIVERSE, DATA_UNIVERSES
from src.simulator.candidate_generation import generate_candidates

# Mirrors b_baseline_harness.py's own _build_panels() pair_cfg construction
# exactly (that harness has no standalone _pair_cfg() to import).
_PAIR_CFG = PairSpreadConfig(
    hedge_ratio_methods=["pca"],
    min_obs=252,
    min_return_std=1e-8,
    min_level_std=1e-8,
    min_kappa=1e-6,
    max_half_life=126.0,
    tiny_weight_threshold=1e-6,
)


def _bundle_for(harness, group_id: str):
    yaml_path = CONFIG_UNIVERSE / harness.UNIVERSE_NAME / f"{group_id}.yaml"
    ucfg = UniverseConfig.from_yaml(yaml_path)
    loader = UniverseDataLoader(ucfg, data_path=str(DATA_UNIVERSES), progress=False)
    # Match PanelBatchConfig's own defaults (panel_batch.py:150-152), which
    # run_panel_batch actually loads under.
    umd = loader.load(force_download=False, check_for_corruptions=False, start_after_nan=True)
    return build_group_return_bundle(
        umd=umd, group_id=group_id, field="Close", return_method="log", dropna="any",
    )


class TestCandidateGenerationEquivalence(unittest.TestCase):
    GROUPS = ("energy", "materials")

    def setUp(self):
        self._tmp_subdir = f"_f2_c6a_test_{uuid.uuid4().hex[:8]}"
        self._tmp_dir = Path(CANDIDATE_PANELS_ROOT) / self._tmp_subdir

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_rows_weights_and_residual_params_match_exactly(self):
        import b_baseline_harness as harness

        residual_cfg = harness._residual_cfg()

        for group_id in self.GROUPS:
            with self.subTest(group_id=group_id):
                bundle = _bundle_for(harness, group_id)
                stem = f"{group_id}_test"

                offline = create_pair_candidate_panel(
                    bundle=bundle,
                    residual_cfg=residual_cfg,
                    pair_cfg=_PAIR_CFG,
                    frequency="W-FRI",
                    hedge_ratio_lb=harness.HEDGE_RATIO_LB,
                    mr_diag_lb=harness.MR_DIAG_LB,
                    start_date=harness.PANEL_START_DATE,
                    max_steps=harness.MAX_STEPS,
                    progress=False,
                    persist_result=True,
                    persist_residual_params=True,
                    persist_result_dir=self._tmp_subdir,
                    persist_result_file_stem=stem,
                )
                offline_panel = offline.panel
                self.assertGreater(len(offline_panel), 0, f"{group_id}: empty offline panel")

                offline_weights = load_weights(str(self._tmp_dir / f"{stem}_weights.parquet"))
                offline_params = load_residual_params(str(self._tmp_dir / f"{stem}_residual_params.parquet"))

                rows, weight_rows, fitted_params = generate_candidates(
                    bundle=bundle,
                    residual_cfg=residual_cfg,
                    pair_cfg=_PAIR_CFG,
                    hedge_ratio_lb=harness.HEDGE_RATIO_LB,
                    mr_diag_lb=harness.MR_DIAG_LB,
                    frequency="W-FRI",
                    start_date=harness.PANEL_START_DATE,
                    max_steps=harness.MAX_STEPS,
                )

                # ── candidate rows, including invalid ones ──────────────
                new_panel = pd.DataFrame(rows)
                self.assertEqual(len(new_panel), len(offline_panel))

                sort_cols = ["asof_date", "spread_id"]
                a = offline_panel.sort_values(sort_cols, kind="stable").reset_index(drop=True)
                b = new_panel.sort_values(sort_cols, kind="stable").reset_index(drop=True)
                self.assertEqual(set(a.columns), set(b.columns))
                for col in a.columns:
                    if col == "weights":
                        continue  # JSON string column; compared via weight_rows below instead
                    left = a[col].to_numpy()
                    right = b[col].to_numpy()
                    if np.issubdtype(left.dtype, np.floating):
                        np.testing.assert_array_equal(left, right, err_msg=f"{group_id}: column {col!r} differs")
                    else:
                        mismatch = ~((left == right) | (pd.isna(left) & pd.isna(right)))
                        self.assertFalse(mismatch.any(), msg=f"{group_id}: column {col!r} differs")

                # ── weights ───────────────────────────────────────────────
                new_weights = pd.DataFrame(weight_rows)
                w_sort = ["asof_date", "spread_id", "ticker"]
                wa = offline_weights.sort_values(w_sort, kind="stable").reset_index(drop=True)
                wb = new_weights.sort_values(w_sort, kind="stable").reset_index(drop=True)
                self.assertEqual(len(wa), len(wb))
                np.testing.assert_array_equal(
                    wa["weight"].to_numpy(dtype=float), wb["weight"].to_numpy(dtype=float),
                )

                # ── residual params, at outer dates only ────────────────
                outer_dates = pd.DatetimeIndex(sorted(offline_panel["asof_date"].unique()))
                self.assertEqual(set(fitted_params.keys()), {pd.Timestamp(d) for d in outer_dates})
                for dt in outer_dates:
                    dt = pd.Timestamp(dt)
                    offline_model = offline_params[dt]
                    new_model = fitted_params[dt]
                    np.testing.assert_array_equal(offline_model.B_proxy, new_model.B_proxy)
                    np.testing.assert_array_equal(offline_model.B_stock, new_model.B_stock)
                    self.assertEqual(offline_model.members, new_model.members)
                    self.assertEqual(offline_model.proxy_name, new_model.proxy_name)
                    self.assertEqual(offline_model.bench_name, new_model.bench_name)
                    self.assertEqual(offline_model.subtract_risk_free, new_model.subtract_risk_free)


if __name__ == "__main__":
    unittest.main()
