# Track J — Retire the offline panel path

**Changes results:** possibly. `tests/test_candidate_generation_equivalence.py`
verifies the live generator (`src/simulator/candidate_generation.py`) against
`create_pair_candidate_panel`'s offline walk exactly, but only on the harness's
two committed universes (`energy_only_v1`, `materials_only_v1`). The
`make_ranking_dates` phantom-bin-label defect already on record (`found.md`,
"`make_ranking_dates` can return a resample bin label that is not an actual
data date") means the two walks can visit different outer-date sets on a
universe or frequency this test never exercises — the offline walk's daily
fit grid happens to self-correct the defect; the live walk's own explicit
filter (`candidate_generation.py:86-91`) was added specifically because it
does not get that accidental correction. Deleting the offline path removes
the thing the equivalence test checks against, so any gap the test's two
universes don't cover becomes unrecoverable after this track, not just
unverified.

**Depends on:** F2 (built the live generator, wired it in, proved neutrality
— `refactor/f2-loop-and-fidelity`, F2_spec.md R11). Should run after Track I
(flag unification touches all three of `DataConfig`, `PanelBatchConfig`, and
`CandidateGenerationConfig` while all three still exist; see README's Order
section) and before Track G (Track G's hedge-ratio change lands in the one
function both walks share, so it does not force this ordering by itself —
see the note below — but G's own six-config sweep still runs through
`sweep_runner.py`, which this track must have already migrated).

## Why this track exists

F2's C6c ("delete the offline path") was descoped after its own pre-check
found two live consumers of `run_panel_batch` with no live-generation
equivalent: `sweep_runner.py`'s simulator-level sweep and `run_me.py`'s
`residuals` stage. Deleting `run_panel_batch` out from under either is not a
cleanup commit, it is an architecture migration — this track is that
migration, split out so F2 could land C6a/C6b (the loop actually running
inside the simulator, proven neutral) without also owning this.

This is expand-contract's **contract** step. F2 did **expand** (build the
live generator, wire it in alongside the offline path) and proved the two
paths agree on the harness universes. This track is **contract**: migrate
every remaining offline-path consumer to the live path, then delete the
offline path once nothing depends on it.

## Scope

1. **Migrate `sweep_runner.py`** off `DataConfig.candidate_panel_subdir`
   (required today, `sweep_runner.py:93-95`, enforced by
   `_resolve_panel_subdir`, `:196-212`) onto `SimulatorConfig.candidate_generation`.
   `_build_sim_config` (`:217-244`) always builds a disk-based `DataConfig`
   today; it needs a live-generation branch, and `SweepConfig` needs a way to
   carry hedge_ratio_lb/mr_diag_lb/pair_cfg/residual config per run (today
   these live only in `PanelBatchConfig`, which `SweepConfig` never
   constructs).
2. **Migrate `run_me.py`'s `residuals` stage** (`run_me.py:280,289,300`,
   `_make_panel_batch_cfg`, `:218-254`) to call live generation instead of
   `run_panel_batch`. This stage currently also stamps the panel's generated
   stem into `demo_materials.yaml` (`_extract_panel_stem`, `_update_panel_stem`,
   `:160-215`) for the `simulate` stage to discover — that hand-off needs a
   live-generation equivalent or needs to disappear if `simulate` also moves
   to `candidate_generation`.
3. **Notebook 03** (`03_full_simulation_pipeline.ipynb`) — repoint its
   panel-build step (cell 5, `run_panel_batch`/`PanelBatchConfig`) at live
   generation once (1) lands, or delete the notebook if the full-pipeline
   demonstration is better served by `b_baseline_harness.py`-style code
   instead of a notebook. Decide which; do not do both.
4. **Delete notebook 01** (`01_create_candidate_panel.ipynb`). Its subject —
   building and persisting a panel via `run_panel_batch` — ceases to exist
   once (5) below lands. See "Notebook 01's content" below: nothing in it is
   lost by deleting it.
5. **Delete the offline path itself**: `run_panel_batch`, `PanelBatchConfig`
   (`src/candidates/panel_batch.py`), and `create_pair_candidate_panel`'s
   persistence and outer+daily-grid walk (`src/candidates/
   pair_candidate_panel_creator.py:733-971` — the walk and its `persist_result`/
   `persist_residual_params` machinery; `_build_pair_candidate_rows_for_date`
   and the other primitives `generate_candidates` also calls stay, since the
   live path depends on them). Delete `tests/test_panel_batch_windows.py` and
   `tests/test_panel_batch_config.py` or repoint them at the live path,
   whichever their content warrants — read them before deciding.

## Extend the equivalence test first

`tests/test_candidate_generation_equivalence.py` currently proves
`generate_candidates` matches `create_pair_candidate_panel` on
`energy_only_v1` and `materials_only_v1` only. Before step 5 (deleting the
thing it checks against), add a second universe — ideally one with an actual
data gap (a market holiday landing on a group's resample bin, per the
phantom-bin-label defect; or a ticker with a late start, per Track I's own
open question about gappy universes) — so the equivalence claim generalizes
beyond the two universes that happen to have no internal gaps over the
harness's date range. A green equivalence test on a universe that cannot
exercise the defect it exists to catch is not evidence the defect is absent
elsewhere.

## Notebook 01's content — nothing is lost

Read cell-by-cell during F2's C6c pre-check (F2_spec.md, "F2b/F4"). Five
distinct content items, all duplicated elsewhere or superseded:

1. **Panel/weights/residual_params/meta.json schema** (cell 1) — duplicates
   `F1_artifact_schema.md` §3-5 (`residual_params.parquet`, `weights.parquet`,
   `candidates.parquet` grains).
2. **"Panel creation has three independent steps"** — residual fit, hedge-ratio
   fit, MR diagnostics; steps 2 and 3 independent of each other and of the
   residual axis (cell 2) — duplicates the inline comment in
   `pair_candidate_panel_creator.py:245-248` (repeated at `:755-756`): "Two
   independent candidate-scoring windows applied to the SAME fitted model."
3. **Residual-mode reference table** — `EQ_ROLLING`/`EQ_EXPANDING`/
   `DECAY_EXPANDING`, `min_history` formulas, `decay_rolling` deliberately
   excluded (cell 3) — duplicates `causal_residuals.py:15-111` (the
   `ResidualMode` enum's own comments and `CausalResidualConfig.__post_init__`'s
   dispatch), down to the same "deliberately excluded" wording.
4. **"Stock-residual and spread-level series are reconstructed on demand,
   never persisted"** (cell 11) — duplicates `F2_spec.md` R1/R2 and
   `found.md` directly; this cell was added during F2's own doc-amendment
   step and is a restatement of F2's own record, not independent content.
5. **The panel's founding rationale** — decouple the expensive
   residual/candidate-discovery computation from cheap, many-times-repeated
   simulation sweeps (cell 1) — not duplicated, because it is not true of the
   post-migration architecture. This is the assumption F2 (C6a/C6b) and this
   track (step 5) replace, not a fact to preserve.

## G's hedge-ratio change — landing site, checked in advance

Confirmed at `pair_candidate_panel_creator.py:135-148` (`_slice_candidate_window`,
the plain `.tail(lookback)` slice `HedgeRatioConfig`'s date-distance weighting
would replace) and `:188-218` (`_compute_pair_weights`/`_pca_spread_weights`,
the OLS/PCA hedge-ratio estimation `HedgeRatioConfig`'s `DECAY_EXPANDING` mode
would weight): both are called from, and only from, `_build_pair_candidate_rows_for_date`
(`:221` onward, calling `_slice_candidate_window` at `:248` and
`_compute_pair_weights` via `pair_cfg.hedge_ratio_methods` at `:286`) — the
one function `create_pair_candidate_panel`'s offline walk and
`generate_candidates`' live walk (`candidate_generation.py:102-112`) both
call. G's change lands there, once, and both walks pick it up in lockstep —
it does not need either walk edited separately, and does not risk the two
drifting apart during the coexistence window this track exists to close.
**This does not force G to wait for this track.** The two are independent;
the ordering in the README (`F2 → I → J → G`) is about `sweep_runner.py`
being ready for G's six-run sweep, not about this shared function.

## Out of scope

- Anything F2, I, or D already decided. This track migrates callers and
  deletes now-unreferenced code; it does not re-open estimation logic.
- `_make_sqrt_w` and any other `found.md` entry not about the offline path
  itself.

## Acceptance

- `sweep_runner.py` and `run_me.py`'s `residuals` stage both run entirely
  through `candidate_generation`; no code path constructs `PanelBatchConfig`
  or calls `run_panel_batch`.
- `tests/test_candidate_generation_equivalence.py` covers at least two
  universes, one with a real data gap, before the offline path is deleted.
- Notebook 01 deleted. Notebook 03 either repointed at live generation or
  deleted, decided and stated, not both attempted.
- `run_panel_batch`, `PanelBatchConfig`, and `create_pair_candidate_panel`'s
  persistence/walk deleted. `_build_pair_candidate_rows_for_date` and the
  other primitives `generate_candidates` shares survive.
- `docs/refactor/B_baseline.txt` before and after, with any diff explained —
  this track is not assumed neutral (see "Changes results" above).
- Full suite green.
