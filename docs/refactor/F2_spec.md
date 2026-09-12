# Track F2 — spec

Read-only verification session. Per the session's precedence rule: `F2_loop_and_fidelity.md`
(as amended before this session — series/ decision, loader-wiring reassignment to Track I) is
the only authoritative document. `F_spec.md` predates the F1/F2 split and the F1 merge; where it
disagrees with the brief, the brief wins. Every `path:line` below was re-verified against current
`main` (post F1 merge, commit `5a67f77`) this session — `F_spec.md`'s and the brief's own citations
were not trusted where they could drift, and several had.

No code was changed. No branch was created.

---

## Part 0 — State after F1

### 0.1 Data flow: data load → trade entry

**Outer-date generation.** `create_pair_candidate_panel` (`src/candidates/pair_candidate_panel_creator.py:650-942`)
resolves `asof_datetimes` either from an explicit `dates` list (validated against
`make_ranking_dates`, `:718-732`) or, in the frequency-driven path used by the harness and
`run_me.py`, at `:740-744`:

```
asof_datetimes = make_ranking_dates(
    dates=aligned_index,
    frequency=frequency,
    min_history=residual_cfg.eff_min_history(),
)
```

where `aligned_index = bundle.aligned_returns.index` (`:693`). `make_ranking_dates`
(`src/utils/date_utils.py:4-22`) resamples the given index by `frequency`, keeps the last date per
period, and filters to `>= dates[min_history-1]`.

**Residual fit.** `fit_causal_residual_model` (`src/residuals/causal_residuals.py:451-547`), called
from the walk loop for every `walk_dates` entry (`pair_candidate_panel_creator.py:812-816`) and,
via the `fitted_model` parameter, reused inside `_build_pair_candidate_rows_for_date`
(`:237-241`) on outer dates only.

**Hedge fit.** `_compute_pair_weights` / `_pca_spread_weights`
(`pair_candidate_panel_creator.py:151-218`), called at `:337-342`.

**MR diagnostics.** `_fast_pair_diagnostics` (`pair_candidate_panel_creator.py:449-552`), called at
`:362-369`.

**Candidate scoring / persistence.** Candidate row dict built at `:379-406`; leg-weight rows at
`:413-435`. Persisted via `save_candidate_panel_result` / `save_weights` / `save_residual_params`
at `:913-940`, writing `{stem}.panel.parquet`, `{stem}_weights.parquet`,
`{stem}_residual_params.parquet` under `artifacts/candidate_panels/{subdir}/`.

**What the simulator reads.** `run_from_config` (`src/simulator/simulator_factory.py:143-189`):
`_load_panels` (`:328-424`) loads the panel from disk; `_load_residual_params` (`:462-515`) loads
`residual_params.parquet` into `precomputed_residual_params`; `_load_weights` (`:518-567`) loads
`weights.parquet` into `weights_lookup`; then `sim.run(panel)` (`:189`).

**How a candidate arrives.** `CandidateFilter.get_selected_on_date`
(`src/simulator/candidate_filter.py:43-50`) still uses exact equality —
`selected_panel["asof_date"] == date` (`:49`). Confirmed unchanged post-F1 (re-verified
independently by this session's own reading and by the Part 1 fork). Once matched,
`CandidateActivation` stores the same frozen `CandidateRef` (`types.py:7`, frozen dataclass) until
the position closes; it is never re-dated.

### 0.2 F1 surfaces F2 builds on

- **`weights.parquet`** — actually written as `{stem}_weights.parquet` (not literally
  `weights.parquet`) by `save_weights` (`src/candidates/candidate_panel.py:81-111`); loaded via
  `load_weights` / `weights_lookup_from_df` (`:114-131`), consumed by
  `simulator_factory.py::_load_weights` (`:518-567`).
- **`residual_params.parquet`** — `{stem}_residual_params.parquet`; `save_residual_params` /
  `load_residual_params` (`src/residuals/causal_residuals.py:275-397`), long form, one row per
  `(fit_date, ticker, factor)`. Loaded via `simulator_factory.py::_load_residual_params`
  (`:462-515`).
- **`candidates.parquet`** — actually `{stem}.panel.parquet`
  (`src/candidates/candidate_panel.py::save_candidate_panel_result`, `:203-229` /
  `load_candidate_panel_result`, `:232-249`), under `artifacts/candidate_panels/{subdir}/`.
- **`trades.parquet`'s `entry_asof_date`** — `LiveCandidatePosition.entry_asof_date`
  (`src/simulator/types.py:144`), `ClosedCandidateTrade.entry_asof_date` (`:188`); set at
  `simulator.py:643` (open) and `:813` (close); exposed via `SimulationResult.closed_trades_df()`
  (`simulator.py:136-151`). Persisted as per-year slices,
  `closed_trades/closed_trades_{year}.parquet` (`simulation_persistence.py:169-188`) — **not** a
  monolithic `trades.parquet`.
- **The daily-state reconstruction function F1 added** — `reconstruct_daily_portfolio_state`
  (`src/simulator/performance/daily_state_reconstruction.py:150-303`), called from
  `simulator.py:527-537`, feeding `generate_report` (`:539`). It reconstructs **portfolio-level
  equity/cost state** (`total_equity_gross/net`, `n_live_candidate_positions`, borrow/transaction
  costs) from raw per-ticker prices, `closed_trades_df`, and still-open positions — explicitly "no
  residual replay" (module docstring, `:17-32`). **It is not a residual- or spread-level
  reconstruction** and has no bearing on the fidelity test F2 must build; no such
  residual/spread-level reconstruction function exists anywhere yet.

### 0.3 A load-bearing finding for the rest of this document

The F1 brief's `## Layout` section describes a unified `simrun_dir/` tree — `config.json`,
`candidates.parquet`, `weights.parquet`, `residual_params.parquet`, `trades.parquet`, `series/`,
`performance/` all under one per-run directory. **This does not reflect current on-disk reality.**
Candidate-panel-side artifacts (`{stem}.panel.parquet`, `{stem}_weights.parquet`,
`{stem}_residual_params.parquet`) live under `artifacts/candidate_panels/{subdir}/`, keyed by
panel-build stem and reused across simulation runs; simulation-run-side artifacts (`config.json`,
per-year `closed_trades/`, `performance/`) live separately under
`artifacts/simulation_runs/{run_id}/`. F1 changed the **storage format** of the panel-side
artifacts (parquet, long form) but not their **location/scope** — they are still panel-build-time,
cross-run-reusable files, not per-simulation-run files. Producing the brief's unified layout is
exactly what F2 commit 1 (the loop migration) must do: only once candidate/weight/residual-param
generation moves inside `Simulator.run()` does it become natural for these to live under one
`simrun_dir`. Confirmed by `simulator_factory.py:503,554` (current panel-dir-based paths) against
the brief's Layout list.

---

## Part 1 — Loop migration (brief §1)

### 1.1 Outer dates

Confirmed as in 0.1. **Subtlety, load-bearing for commit 1:** `bundle.aligned_returns` (the index
`make_ranking_dates` is called on) is **not** the simulator's raw price index.
`build_group_return_bundle` (`src/data/returns.py:65-139`) applies `dropna="any"` by default
(`:119-120`) across that group's member + proxy + benchmark returns — dropping any date where any
one of those tickers is missing. `Simulator._market_dates()` (`simulator.py:911-920`) instead calls
`self.umd.price_matrix(field=..., group_id=None)`, returning the **raw, un-dropna'd, union-of-all-
tickers** price index (`src/data/universe_marketdata.py:18-27`) — a strict superset, NaN-preserving.

Consequence: the migrated loop cannot call `make_ranking_dates` on `self._market_dates()` directly
and expect the identical outer-date set. It must first build (or replicate) each group's own
`GroupReturnBundle.aligned_returns.index` and call `make_ranking_dates` on *that* — per group.
Outer dates are inherently per-group; `Simulator.run()` today walks one shared `dates` sequence
across all groups (`_build_simulation_dates`, `:922-943`, currently `market_dates ∩ cp_dates`,
where `cp_dates` comes from the *loaded panel's* `asof_date` column). Once there is no
externally-built panel to source `cp_dates` from, the simulator's own per-group outer-date
computation becomes the only source of that intersection. This is a real design point for commit 1,
not a triviality — flagged here, not decided.

### 1.2 Daily fits vs. outer-date candidate rows

Re-confirmed: the walk (`pair_candidate_panel_creator.py:789-838`) fits on every `walk_dates` entry
but builds candidate rows only when `dt in panel_date_set` (outer dates). **One conditional the
brief and `F_spec.md` don't surface:** `walk_dates = daily_dates if daily_dates is not None else
asof_datetimes` (`:786`) — `daily_dates` (the daily fit grid) is only computed
(`persist_residual_params=True` branch, `:762-768`) when that flag is set. In practice it always
is: `PanelBatchConfig.persist_residual_params: bool = True` by default
(`src/candidates/panel_batch.py:162`), and the baseline harness sets it explicitly `True`
(`scripts/b_baseline_harness.py:122`); no caller sets it `False`.

**What consumes the non-outer-date fits — more than a persistence nicety.** The full daily
`fitted_params` dict is loaded into `precomputed_residual_params`
(`simulator_factory.py::_load_residual_params`) and read **every simulated trading day**, keyed by
*today's* simulation date, in `CandidateSignalGenerator._get_residuals`
(`src/simulator/candidate_signals.py:944-946`): `if date in group_params: model = group_params[date]`.
So the daily residual-model grid feeds the live z-score/level computation for every tracked
candidate on every trading day — the spread *weights* stay frozen at the candidate's own
`asof_date`, but the *residual matrix* is refit daily and looked up by today's date. This refines
`F_spec.md`'s framing (its own file docstring, `pair_candidate_panel_creator.py:757-759`, calls
this "persisted for later use," which understates that it's on the critical path of every day's
signal computation, not an optional convenience). If `date not in group_params`, a live (expensive)
fit runs as fallback (`candidate_signals.py:949`).

### 1.3 Ticker/group resolution and call sites

`fit_causal_residual_model` resolves `members`/`proxy_name`/`bench_name` from
`bundle.member_returns.columns` / `bundle.proxy_returns.name` / `bundle.benchmark_returns.name`
(`causal_residuals.py:477-479`) — separate typed fields on `GroupReturnBundle`
(`src/data/returns.py:10-33`), not derivable from a flat DataFrame alone.

**All call sites (grep includes `notebooks/**/*.ipynb`, zero notebook hits):**

1. `pair_candidate_panel_creator.py:237` — `bundle` passed in as a parameter.
2. `pair_candidate_panel_creator.py:812` — same, main walk loop.
3. `candidate_signals.py:949` — `bundle = self._get_bundle(group_id)` (`:940`), lazily built/cached
   via `build_group_return_bundle`.
4. `causal_residuals.py:643`, inside `create_causal_residuals` — **this function has zero callers
   anywhere** (grepped `src/`, `tests/`, `scripts/`, `run_me.py`, notebooks: only its own
   definition and an unused import at `candidate_signals.py:19`). Dead code, not previously
   recorded in `found.md`.

**What "input returns as an argument" precisely implies.** `apply_causal_residual_model`
(`causal_residuals.py:550-620`) **already** takes `aligned_returns: pd.DataFrame` directly, not a
bundle — the brief's requirement is narrower than it first reads: only `fit_causal_residual_model`
(and `_slice_fit_window`, internally coupled to it) is bundle-shaped and needs to change. However, a
bare `aligned_returns: pd.DataFrame` is **not sufficient** on its own — the fit also needs to know
which columns are members vs. proxy vs. benchmark, currently read off the bundle's typed fields.
Any redesign needs explicit additional arguments (`members`, `proxy_name`, `bench_name`) alongside
the returns frame. This is a real design constraint the brief doesn't spell out, flagged here, not
decided.

`_slice_fit_window` (`causal_residuals.py:400-425`) has **three** callers, not one:
`candidate_signals.py:955` (external, apply-step window), plus the two internal-to-
`causal_residuals.py` callers (`fit_causal_residual_model` at `:466`, dead `create_causal_residuals`
at `:649`). A signature change here touches `candidate_signals.py:955` too.

**Callers needing to change:** sites 1, 2, 3 above; site 4 is dead and could simply be deleted
rather than updated.

### 1.4 Primitives commit 1 would call

All of `src/residuals/spreads.py` (`spread_members`, `generate_spread_ids`, `ols_beta_no_intercept`)
are pure functions on explicit arguments, no module state — **callable unchanged**.

`_pca_spread_weights`, `_compute_pair_weights`, `_fast_pair_diagnostics`
(`pair_candidate_panel_creator.py:151-218,449-552`) are pure numpy functions on explicit arrays —
**callable unchanged**, though they currently live as module-private helpers in
`pair_candidate_panel_creator.py` rather than `spreads.py`; relocating the loop needs either an
import from that module or a move into `spreads.py` (organizational, not correctness).

`apply_causal_residual_model` — already takes a flat DataFrame — **callable unchanged**.

`make_ranking_dates` — pure function on an index — **callable unchanged**, subject to 1.1's
per-group-index subtlety (which index it's fed changes the answer, not the function).

`fit_causal_residual_model` and `_slice_fit_window` — **cannot** be called unchanged; both are
bundle-shaped and the brief mandates the returns-as-argument change (1.3).

### 1.5 The `≤ t` invariant

**No residual/spread-level reconstruction path exists yet** — confirmed `daily_state_reconstruction.py`
is portfolio-level only (0.2) and is not a candidate for "the reconstruction path" here. The only
functions that resemble it today are live-path functions the simulator itself calls
(`candidate_signals.py::get_level_series`, `:869-900`; `compute_analytics_from_weights`,
`:607-696`) — not an independent, after-the-fact reconstruction verified against a frozen sample.
Building that is F2's own job (Part 3).

**Half (a) — refit date ≤ t, candidate arrives on its own outer date.** Structural today via three
facts together, never asserted: `Simulator.run()` walks `dates` chronologically ascending
(`simulator.py:241`, sourced from `_build_simulation_dates`, `:922-943`); `CandidateFilter
.get_selected_on_date` uses exact equality (`candidate_filter.py:49`); `CandidateActivation` stores
the same frozen `CandidateRef` until close, never re-dated (`candidate_activation.py:24,62-78`).
Confirms `F_spec.md`'s "structural, never asserted" finding still holds post-F1.

**Half (b) — returns passed to the fit at refit date d have max index ≤ d.** Panel-build time:
`_slice_fit_window`'s `aligned.loc[:pd.Timestamp(date)]` (`causal_residuals.py:406`) — hard
truncation. Simulation time, recompute fallback: same call, both externally
(`candidate_signals.py:955-959`) and internally inside `fit_causal_residual_model`
(`causal_residuals.py:466-470`). Simulation time, **precomputed-params path**: `model =
group_params[date]` (`candidate_signals.py:946`) is **not re-verified at read time** — it trusts
whatever was stored under key `date` in `residual_params.parquet` was itself correctly truncated at
panel-build time. The `≤t` guarantee on this path is inherited, not independently checked at
consumption — worth stating explicitly.

**Proposed tests (options, not decisions):**
- Half (a): buildable **today**, pre-migration, no new infrastructure — assert
  `(closed_trades_df["entry_asof_date"] <= closed_trades_df["entry_date"]).all()` (both columns
  exist now via F1 commit 7). Could live in a new `tests/test_candidate_arrival_invariant.py`, or
  fold into `tests/test_track_a_guards.py`. No existing test covers this (grepped `tests/` for
  `entry_asof_date` comparisons — none).
- Half (b): no existing test covers `_slice_fit_window`'s truncation (grepped `tests/` for
  "lookahead" and `_slice_fit_window` — zero hits). Proposed: a "perturb-the-future" test —
  fit at date d, mutate rows strictly after d, refit, assert identical coefficients. Stronger than
  an index-bound check since it tests the actual no-lookahead guarantee rather than the presence of
  a `.loc[:date]` call.

### 1.6 Baseline harness

Confirmed: the `## counts` section is `pr.panel["is_valid"].sum()` / `len(pr.panel)` per
`(group_id, residual_key)` (`scripts/b_baseline_harness.py:259-262`), where `pr.panel` comes from
`panel_results = run_panel_batch(cfg)` (`:126`). Trade list from `result.closed_trades_df()`
(`:238`) after `run_from_config(sim_config)` (`:235`).

`run_panel_batch` (`src/candidates/panel_batch.py`) — confirmed calling `build_group_return_bundle`
(`:297`) and `create_pair_candidate_panel` (`:321`) per group — is exactly the offline,
pre-simulation panel-build step this track proposes to relocate into the simulator.

**Yes — commit 1 relocates something the harness reads, structurally, not incidentally, and in two
separate ways:**
1. The harness calls `run_panel_batch` **directly**, not through `run_from_config`. If the offline
   panel-build path is removed (consistent with the "no persisted panel artifact" dissolution goal
   F1's brief states as context), the harness's `_build_panels()` has nothing left to call.
2. Even independent of (1): the `## counts` line needs **both valid and total row counts**
   (`n_valid`/`n_rows`) — the full scored panel including invalid rows. The simulator's own
   `selected_panel` (post `CandidateFilter.run()`) is already filtered to valid rows only per F1's
   own artifact design — it structurally cannot supply the denominator the harness's counts line
   needs. Today that full scored-panel object only exists as a byproduct of the offline walk. After
   migration, whatever replaces it inside the simulator needs to expose an equivalent full-scored
   count, or the harness's counts section needs a different source entirely.

This is a real adaptation the brief does not name as its own step. Per the task's own instructions
(and the standing rule that a harness adaptation must be its own commit, checked neutral on the
*old* path, before commit 1), this needs to land as a numbered commit ahead of the loop migration —
see Part 4.

### 1.7 Parallelization constraints

`CandidateSignalGenerator`'s mutable caches — `_bundle_cache` (`candidate_signals.py:176`, keyed by
`group_id`), `_residual_cache`/`_latest_pc_variance_ratios` (`:177-178`, keyed
`(group_id, residual_key)`), `_spread_series_cache` (`:179`, keyed by file path) — are all
group-scoped or group-derived. **No cross-group collision found.** One instance is shared across a
whole (possibly multi-group) run (`simulator_factory.py:88-97`); this is not a *correctness*
violation today (keys prevent collision), but a future per-group multiprocessing design would need
one fresh instance per worker rather than a shared one.

**No current violation**, since commit 1 does not itself implement parallelization (the brief
explicitly defers it). One structural nuance worth flagging: `RiskManager.approve`
(`risk_manager.py:29-34`) takes `sized_opens`/`live_positions` mixed **across all groups**, called
once per day (`simulator.py:401-406`) — the simulator's trading-decision loop is inherently
cross-group-coupled at the portfolio-risk level. The five parallelization constraints, as stated,
read as scoped to the candidate-scoring sub-step (residual fit → weights → diagnostics — matching
"outer at refit frequency, inner daily"), not to the whole per-day loop, which cannot be naively
parallelized per group given the shared risk manager. Worth the eventual parallelization design
being explicit about which sub-step the constraints bind.

---

## Part 2 — `series/` (brief §2)

### 2.1 Current path:line (re-verified, drift confirmed)

- **(a) Cache read path** — `candidate_signals.py`: `panel_dir` field (`:167`); gate
  `if self.panel_dir is not None:` (`:343`); `_spread_series_path` (`:539-542`, **matches** the
  brief's citation exactly — did not drift); `_try_batch_levels_from_disk`, missing-file check
  (`:576-577`, **matches** the brief's "576" citation — did not drift).
- **(b) skip-if-exists** — `series.py`: stock loop `path.exists()` at **`:180-181`**; spread loop at
  **`:221-222`**. **Drift confirmed**: the F2 brief and `found.md` cite `series.py:192-193,233-234`
  (inherited unedited from `F_spec.md`) — stale by exactly 12 lines each.
- **(c) `panel_dir` scope** — threaded from `SimulatorConfig.data.candidate_panel_subdir` through
  `simulator_factory.py:84-86` (`create_simulator`) into `CandidateSignalGenerator(panel_dir=...)`
  at `:96`.

Also stale: `F_spec.md`'s notebook citation `02_run_simulation.ipynb:1647` for the writer cell —
current line is ~1497.

### 2.2 Every writer and reader

**Writers** (`compute_and_persist_series`, `series.py:66-249`), called from:
- `scripts/b_baseline_harness.py:172` (inside `_persist_series`, `:131-178`, called from `main()`
  at `:232`).
- `run_me.py:323` (`_persist_series_multi`, `:275-329`; called at `:351,558`).
- `run_me.py:538` (`_persist_spread_series`, `:507-544`; called at `:373,561`).
- `notebooks/howto/02_run_simulation.ipynb`, cell at ~line 1497.

**Readers:**
- `candidate_signals.py:343-357,539-605` — **feeds a computation** (live z-score/level input for
  trading decisions on a cache hit).
- `tests/test_clear_stem_artifacts.py:40-75` — does **not** read series content; only asserts
  `_clear_stem_artifacts` leaves a manually-planted `series/stock/AAPL.parquet` untouched. Not a
  real consumer — a stem-scoping guard that happens to reference the directory.
- `notebooks/howto/03_full_simulation_pipeline.ipynb` — no code reads it, but its committed output
  log shows `[signals] N/N spread level series missing... recomputing from residuals` warnings —
  i.e. this notebook's committed run hit the miss branch. Display-only (a captured log).
- `notebooks/howto/01_create_candidate_panel.ipynb` — no code touches it; two markdown cells
  describe it in prose as optional/recomputable. Text-only.
- No other reader found anywhere in `src/`, `tests/`, `scripts/`, `run_me.py`, notebooks.

### 2.3 Does the baseline harness read `series/` today?

**Yes, confirmed end to end.** `_build_sim_config` sets `candidate_panel_subdir=PANEL_SUBDIR`
(`b_baseline_harness.py:184`) → `panel_dir` resolves non-`None`
(`simulator_factory.py:84-86,96`) → the disk-read branch in `_batch_analytics_for_group`
(`candidate_signals.py:343`) is taken on every call. Ordering in `main()`: `_persist_series` runs
to completion (`:232`) **before** `run_from_config` (`:235`), so every file the disk-read path looks
for already exists by simulate time.

**Consequence:** the harness's `## counts`/trade-list output today is produced via the disk-read
branch, not the recompute fallback. Deleting the cache moves the harness onto the recompute path
exclusively — the harness's "does not move a single row" check for that commit is, in effect, the
first real test that recompute reproduces exactly what disk-read had been serving.

### 2.4 Can the read-path removal land before commit 1?

**Yes — `F_spec.md`'s sequencing dependency is not real.** The recompute fallback
(`_batch_analytics_for_group:359-408`) depends only on `self._get_residuals`, which in turn depends
on `precomputed_residual_params` and `weights_lookup` — both **already loaded independently** of the
loop migration (F1 artifacts: `simulator_factory.py::_load_residual_params:169,462-515` and
`::_load_weights:174,518-567`), plus the pre-existing live-fit fallback
(`fit_causal_residual_model`, untouched by the loop migration). Nothing here depends on where the
outer/inner loop lives.

Two more pre-existing, cache-independent reconstruction pathways confirm the pattern is already
live and load-bearing: `compute_analytics_from_weights` (`:607-696`, used by `simulator.py`'s
`_apply_action`/`_build_live_fz_analytics`/diagnostics-log construction) and `get_level_series`
(`:869-900`, used by `EntryFeatureEngine`) — **neither ever checks `panel_dir`**; both always
recompute. The simulator already runs substantial live logic through exactly the "recompute
residuals @ weights → cumsum" path the brief wants as the only path, with zero dependency on
commit 1. The amended brief (no run persists series at all) doesn't change this — removing the
three writer call sites is an equally independent pure deletion.

### 2.5 What removal touches

- **`src/residuals/series.py`** — `compute_and_persist_series` (`:66-249`) is the module's only
  real product; `sanitize_spread_id`/`spread_series_filename` (`:48-56`) have **zero external
  callers** (`candidate_signals.py::_spread_series_path` reimplements the same filename logic
  inline rather than importing it). Once both writer and reader are gone, **the entire module is
  dead** and deletable outright.
- **`candidate_signals.py`** — remove `panel_dir` field (`:167`), the disk-fast-path branch in
  `_batch_analytics_for_group` (`:343-357`), `_spread_series_path`/`_load_spread_series`/
  `_try_batch_levels_from_disk` (`:539-605`), and `_spread_series_cache`/
  `_SPREAD_SERIES_CACHE_MAX` (`:31,179`).
- **`simulator_factory.py`** — remove `panel_dir` resolution and the kwarg at `:84-86,96`.
- **`run_me.py`** — remove `_persist_series_multi` (`:275-329`), `_persist_spread_series`
  (`:507-544`), and their four call sites (`:351,373,558,561`).
- **`scripts/b_baseline_harness.py`** — remove `_persist_series` (`:131-178`) and its call at
  `:232`. Per 2.3, this changes what the harness exercises — flagged to the implementation session,
  not decided here.
- **`notebooks/howto/02_run_simulation.ipynb`** — remove the writer call (~1465-1500) and a now-
  false "persisted series are reused untouched" claim (~1944); re-execute per standing rule.
- **`notebooks/howto/01_create_candidate_panel.ipynb`** — reword the two markdown cells
  (`:28-32,517-521`) from "optional persistence step" to "no run persists series; always
  reconstructed on demand."
- **`notebooks/howto/03_full_simulation_pipeline.ipynb`** — no code change, but its committed
  output (the missing-series warning log) disappears once the disk-read branch is gone; needs
  re-execution per standing rule even though no notebook *code* changes.
- **`tests/test_clear_stem_artifacts.py`** — not forced to change (tests stem-scoping, orthogonal
  to whether anything writes into `series/`), but its docstring/setup references a "separate stage
  [that] writes the shared series/ tree," which becomes counterfactual prose. A judgment call for
  the implementation session — not decided here.

---

## Part 3 — Debug sample and fidelity (brief §3 and "Spread level and reconstruction")

### 3.1 Row set

Confirmed computable at a single capture point: `tracked_refs`
(`simulator.py:288`, = new arrivals ∪ `CandidateActivation.get_active_candidates()`, which returns
the same frozen `CandidateRef` objects, `candidate_activation.py:62-78`) together with
`live_positions_by_candidate_id` (carrying `entry_asof_date`, `types.py:144`) give both the
candidate view and the position view at one point per simulated day (`simulator.py:288-317`).

Re-measured directly from the real run still on disk
(`artifacts/simulation_runs/20260713_2017_e5ffbe83/`, `meta.json`: `n_closed_trades=3791`,
`n_trading_days=5920`) — matches `F_spec.md`'s figures exactly: `selected_panel.parquet` **2,041,607**
rows (candidate-view upper bound, 1186 outer dates × 10 groups); `closed_trades/*.parquet` (3791
rows) → **3721** unique `(spread_id, entry_date)` pairs (position-view count);
`daily_portfolio_state.parquet` max concurrent positions **209**, mean **123.87**.

Harness config: candidate view = 1698+2869 = **4567** valid candidate rows; position view = **97**
trades.

**Nuance not in the brief:** `needed_refs` (`simulator.py:307-311`) means the live loop does *not*
compute real analytics for every `tracked_ref` every day — a non-live candidate whose spread is
already occupied by a live position at that timescale gets a placeholder (`:319-336`), not a real
level/z-score. "Combinations the run actually used" is therefore *narrower* than the panel-row
upper bound of 2,041,607 / 4567 — matching the brief's own framing more closely than a raw row
count would, but the true count needs measuring from `analytics_by_id`'s real (non-placeholder)
entries, not from panel row counts.

### 3.2 Capture point

Today (pre-migration), per day: candidate-view level/z-score/refit-date live in `analytics_by_id`
(`simulator.py:313-317` → `candidate_signals.py:242-320,322-408`). Per-leg residual returns exist
only transiently inside `_batch_analytics_for_group` (`:368,395` / `:401-408`) — not returned;
only `level`/`z_score` survive into `CandidateAnalyticsState`. Refit date = `ref.asof_date`
(`types.py:18`, already carried on every `CandidateRef`). Position-view uses the **same code path**
for live candidates' refs — `analytics_by_id[pos.candidate_id]` is what actually drives trading
decisions (`simulator.py:360-364`).

**Caution for the fidelity test's own design:** `simulator.py` also computes `fz_analytics_by_id`
(`:348-351` → `_build_live_fz_analytics`, `:701-720`) and `dy_a` (`:744-756`) via
`compute_analytics_from_weights`, using `pos.realized_weights_by_ticker` (actual fills) and
`_compute_effective_weights` (current MTM weights) respectively — diagnostics-log-only fields
(`LiveDiagnosticsLogEntry.fz_*`/`dy_*`), not read by the trader, and sourced from **different
weights than `weights.parquet`**. The brief's "position view, frozen weights" fidelity target is
`analytics_by_id`'s output (raw `CandidateRef.weights`, i.e. the `weights.parquet` values), **not**
these fz/dy diagnostic variants — worth being explicit about so the fidelity test doesn't target
the wrong computation.

Post-migration, the shape is unchanged (`CandidateRef` + `analytics_by_id`) — commit 1 relocates
*who* computes this, not what's captured.

### 3.3 Buildable pre-migration?

Reported as fact, not decided, per the brief's own instruction. Structurally yes: `weights.parquet`
and `residual_params.parquet` already exist today with the schema/semantics the reconstruction
needs. A fidelity test comparing `analytics_by_id`'s live level/z-score against a reconstruction
built from those two artifacts reads the same data either side of commit 1 — only *where* the
files live changes (`candidate_panels/{stem}_*.parquet` today vs. a per-run `simrun_dir` after
commit 1, per 0.3). The comparison logic (residuals-from-params @ weights → cumsum → z-score)
should survive the migration unchanged; only the test's file-path plumbing needs updating.

### 3.4 Exactness threats

No `float32` anywhere in the residual/spread/signal pipeline (grepped `src/residuals`,
`candidate_signals.py`, `pair_candidate_panel_creator.py`, `returns.py` — no hits; all float64).
Fit: `_wls_multi` (`causal_residuals.py:440-448`) uses `np.linalg.lstsq`, BLAS-backed — the brief's
guidance to compare persisted beta rather than refit is the correct mitigation; confirmed no other
lstsq/OLS variant exists for the hedge fit. Beta location: written to `weights.parquet` at
`pair_candidate_panel_creator.py:417-435`; reconstruction should read it via `weights_lookup`, the
same dict `CandidateFilter.build_candidate_refs` and `daily_state_reconstruction.py:98-105` already
consume — confirmed as the intended pattern, not re-fitting.

Index/slicing: live path slices via `.loc[:date]` (`causal_residuals.py:406`) and `.iloc[-lb:]` for
rolling mode (`:418`); `apply_causal_residual_model` is row-wise/vectorized with no incremental
state, so a full-history-at-once apply and any date-truncated apply of the same frozen model
produce identical rows for overlapping dates (confirmed by `series.py`'s own docstring, `:19-23`,
which relies on exactly this property). `np.cumsum` order: single call, `axis=0`, deterministic —
`_batch_analytics_for_group` always cumsums the whole available window at once (`:396`), same as a
from-scratch reconstruction would; no incremental-vs-batch divergence risk found.

### 3.5 Z-score state

Confirmed fresh per call. `_compute_z_score` (`candidate_signals.py:759-817`) builds `rm`/`rs` via
`.ewm()`/`.rolling()` on the full `level_series` each call; the batched path
(`_finalize_batch_states:410-535`, via `_rolling_mean_std_last`/`_ewm_mean_std_last:54-136`) is
likewise a pure function over the full level matrix with no persisted state. Repo-wide grep for any
cached/carried EWM state: **NOT FOUND**.

### 3.6 PC removal

**NOT FOUND** anywhere — no existing raise for `remove_residual_pcs > 0` in any function today.
`apply_causal_residual_model` applies PC removal unconditionally when `model.pc_components is not
None` (`causal_residuals.py:608-614`), no guard. Confirms this is entirely new code the (not-yet-
built) reconstruction function must add.

### 3.7 Leg count

Pair path is hardcoded to 2 legs (`generate_spread_ids(..., n_legs=2)`,
`pair_candidate_panel_creator.py:278-281`; `"n_legs": 2` literal at `:390`). The only >2-leg-capable
code, `build_spread_returns` (`src/residuals/spreads.py:201-284`), still has **zero callers**
repo-wide (re-confirmed) and itself hardcodes `n_legs=2` internally (`:251-252`, raising
`NotImplementedError` otherwise). No live portfolio-shaped multi-leg path exists anywhere today;
`found.md`'s existing "Dead n_legs=2 hardcode" entry still holds exactly as stated.

### 3.8 Single code path

Confirmed two implementations of the same OU-fit math (dx-on-lagged-x OLS via `np.linalg.lstsq`,
`kappa=-slope`, `half_life=ln2/kappa`, `mr_score=kappa/residual_std`):

- `pair_candidate_panel_creator.py::_fast_pair_diagnostics` (`:449-552`) — panel-build time, takes
  raw `spread_return`, cumsums internally (`:470`), truncated to the caller's `mr_diag_lb` window.
- `candidate_signals.py::_compute_mr_diagnostics` (`:819-857`) — simulator runtime, takes a
  pre-built `level_series`, truncated to `self.diagnostics_config.lookback`
  (`MRDiagnosticsConfig.lookback`, `config.py:331`).

**Sharper than the brief states:** these are not guaranteed to run over the same window —
`mr_diag_lb` (panel-build config) and `MRDiagnosticsConfig.lookback` (simulator config) are
**independently settable**. The two call sites could silently score the same spread over
different-length windows today, not merely duplicate the same computation. This is a real
"single source of truth" question beyond code duplication.

`_compute_candidate_analytics` (`candidate_signals.py:698-757`) — checked callers repo-wide
including notebooks — **has zero callers**. Dead code, not previously flagged in `found.md`.

Other duplicated `np.cumsum(spread_return)`: `_fast_pair_diagnostics:470`,
`candidate_signals.py:396,662,721(dead),896`, `series.py:240` — six independent call sites, no
shared "spread level" primitive, despite `src/analytics/spread_primitives.py` existing as exactly
this kind of shared-primitives module for other signal computations (its own docstring states a
hard rule that such logic "MUST live here"). `cumsum` itself is trivial; the substantive gap is the
OU-fit duplication above, per the brief's "Single code path" intent.

Which implementation the reconstruction should call, or whether a third shared primitive needs
extracting, is not decided here — reported as an open fact for the implementation session.

### 3.9 Fidelity test runtime estimate

Harness timings (`B_baseline.txt`): panel build 10.67s, simulate 12.98s. Row set at harness scale:
candidate view ≈4567, position view ≈97 (both order-of-magnitude smaller once the `needed_refs`
narrowing from 3.1 is applied). At ~1.7ms/call (`F_spec.md`'s measured
`apply_causal_residual_model` cost — a different group's timing, not re-measured this session, used
only as an order-of-magnitude input), raw reconstruction compute ≈ (4567+97)×1.7ms ≈ **~7.9s**.
Given per-row Python/dict-lookup overhead likely dominates at this scale (not vectorized across
candidates the way the live batch path is), a debug-sample fidelity test on the harness config is
estimated at the **same order of magnitude as the harness's existing panel-build/simulate steps**
(single digits to ~15s) — consistent with the brief's "small live run... slower than a unit test"
framing, not a qualitatively slower test. No harness-scale reconstruction was actually measured
this session (the code doesn't exist yet); this is an estimate, stated as such.

---

## Part 4 — Proposed commit order

Reported as options with dependencies and neutrality classification, per the task's instructions —
this does not decide anything the brief marks `[decided]`, and does not resolve the open items in
Parts 2 (series granularity — moot now, no series persists at all) or 3.8 (which OU-fit
implementation wins).

| # | Commit | Depends on | Could move `B_baseline.txt`? | Other check | Neutrality argument |
|---|---|---|---|---|---|
| 1 | `≤t` invariant test, half (a): assert `entry_asof_date <= entry_date` on `closed_trades_df`. | Nothing new — F1 commit 7 fields already exist. | No — pure test addition. | Full suite green. | Structural (asserts a fact already true by construction, per 1.5). |
| 2 | `≤t` invariant test, half (b): "perturb-the-future" test on `_slice_fit_window`. | Nothing new. | No. | Full suite green. | Structural (tests an existing guarantee, doesn't change behavior). |
| 3 | Delete `series/` cache read path, both writers, and the now-dead `series.py` module (Part 2.5's full removal list). | Nothing — verified independent of commit 1 (2.4). | **Yes, and must be checked explicitly** — the harness currently runs the disk-read branch (2.3); this is the first real test that recompute reproduces what disk-read was serving. | `B_baseline.txt` before/after, specifically for this commit. | **Empirical, not structural** — the two code paths (cache vs. recompute) are different implementations that are *supposed* to agree; this commit is the first thing that actually proves it, at the harness's scale. |
| 4 | Baseline-harness adaptation: change `_build_panels`'s counting so the `## counts` denominator (valid + invalid rows) doesn't depend on `run_panel_batch`'s offline output shape. | Needs enough of commit 5's target shape sketched to know what to point the harness at — see sequencing tension below. | Must be checked **neutral on the OLD (pre-migration) path** per the standing rule (commit precedes the change it's adapting for). | `B_baseline.txt` byte-identical before/after this commit alone, on unchanged simulation code. | Empirical (verified by an unchanged-output diff), not structural. |
| 5 | Outer/inner loop migration into the simulator (the brief's "commit 1"): `fit_causal_residual_model`/`_slice_fit_window` take explicit returns + members/proxy/bench args (1.3); simulator computes per-group outer dates from its own group-scoped bundle index (1.1); candidate/weight/residual-param generation moves inside `Simulator.run()`. | Commits 3 (series deletion must not still be reading the old panel-dir-scoped disk cache) and 4 (harness must already be adapted to check this). | **Yes — the one commit expected to be exercised by, and required to not move, `B_baseline.txt`.** | `B_baseline.txt` (binding), plus commits 1-2's `≤t` tests re-run against the new path. | Structural in intent ("relocates *where*, not *what*"), but per the brief's own framing this is only an expectation until verified — not provable in advance the way commits 1-3 are. |
| 6 | OU-fit single-code-path consolidation: resolve `_fast_pair_diagnostics` vs. `_compute_mr_diagnostics` duplication, and the `mr_diag_lb` vs. `MRDiagnosticsConfig.lookback` independent-window question (3.8). | Should land before or alongside commit 7, since the reconstruction function needs one canonical implementation to call. | Possibly — depends on which window/implementation is chosen as canonical; not decided here. | `B_baseline.txt` if it changes live behavior; a dedicated before/after diff if the two windows are unified. | Not classifiable without a decision this session doesn't make. |
| 7 | Debug sample capture + zero-tolerance fidelity test (candidate view and position view), per Part 3. Comparison logic per 3.3 could be prototyped as early as commit 1, but capturing what the *migrated* loop actually used naturally follows commit 5 closely, per the brief's own sequencing note. | Commits 5 (what it's fidelity-testing) and 6 (a single OU-fit implementation to reconstruct against). | No, in itself — new test infrastructure. But this is what would **catch** a regression commit 5's own baseline check missed — sequence it as close after commit 5 as possible. | The fidelity test itself, `check_exact=True`. | N/A — verification infrastructure. |
| 8 | Re-execute committed notebooks (`01`, `02`, `03`) against the new API surface; fix any notebook-only caller this uncovers. | Everything above that touches a notebook-visible surface. | N/A — verification pass. | Standing rule (README). | N/A. |

**Sequencing tension flagged, not resolved:** commit 4 (harness adaptation) is required by the
standing rules to land *before* commit 5 and be checked neutral on the *old* path, but its own
design (what to point the harness's counting at instead of `run_panel_batch`'s output) is most
naturally informed by what commit 5 will actually expose. This is a real chicken-and-egg the
implementation session needs to resolve — e.g. by generalizing the harness's counting interface
first, in a way that's agnostic to panel-build-time-vs-simulator-time sourcing, rather than fully
committing to the new shape before commit 5 exists. Reported as a fact, not decided.

---

## Out of scope — overlap notes

- Anything in F1's list.
- `start_after_nan` / `check_for_corruptions`: Track I.
- `zscore_key`, occupancy, sleeve identity: Track H.
- Cross-run reuse of anything.
- Snapshot-layer wiring of the loader call sites, `UniverseDataLoader.load`'s in-place resync, and
  the `DataConfig.universe_name` / `snapshot_id` defaults: Track I (reassigned in Step 0).
  **Overlap found:** `panel_batch.py:291` (`run_panel_batch`'s own `UniverseDataLoader.load` call)
  is one of the three call sites `found.md` reassigns to Track I. If F2 commit 5 (loop migration)
  removes or obsoletes `run_panel_batch`'s offline panel-building entirely, that call site
  disappears along with it — whoever implements Track I's wiring should coordinate with F2 rather
  than wire a call site about to be deleted.

---

## Disagreements between the brief and `F_spec.md`

1. `F_spec.md`'s framing of the daily residual fits as "persisted for later use" (echoing the
   panel-creator module's own docstring) understates that they are consumed live, every simulated
   trading day (§1.2). Not a decision conflict, but changes the stakes of getting daily-fit
   persistence right in the migrated loop.
2. `F_spec.md`'s and the (pre-Step-0) brief's `series.py` skip-if-exists citations
   (`:192-193,233-234`) are stale by 12 lines each; current is `:180-181,221-222` (§2.1).
3. `F_spec.md`'s notebook citation for the series writer cell (`02_run_simulation.ipynb:1647`) is
   stale; current is ~1497 (§2.1).
4. `F_spec.md` sequenced the series/ read-path removal *after* the loop migration, flagging the
   dependency as something to check. Verified **not real** — it can land independently, and even
   before the loop migration (§2.4).
5. `F_spec.md` Part 1 §5's "never asserted" finding for the `≤t` invariant still holds post-F1 — no
   contradiction, re-confirmed as current (§1.5).
6. Neither the brief nor `F_spec.md` names the baseline-harness adaptation the loop migration
   forces (§1.6) as its own step — worth surfacing since it changes what "commit order" for this
   track actually needs to include (§ Part 4, commit 4).
7. The brief's F1-inherited "Layout" section (unified `simrun_dir`) does not describe current
   post-F1 reality (§0.3) — not a brief/`F_spec.md` conflict per se, but a gap between what a reader
   might assume F1 already produced and what actually exists on disk today.

## Anything the brief asks for that cannot be done as described

- "The residual fit must receive its input returns as an argument, not resolve tickers internally
  by group" (brief §1) is under-specified: a flat `aligned_returns: pd.DataFrame` argument alone
  cannot supply which columns are members vs. proxy vs. benchmark — that information currently
  lives only on `GroupReturnBundle`'s typed fields. The signature change needs explicit
  `members`/`proxy_name`/`bench_name` arguments alongside the returns frame, not just "returns as
  an argument" (§1.3). Not a blocker, but the brief's phrasing alone under-specifies the actual
  change.
- The brief's acceptance criterion "`B_baseline.txt` does not move a single row, checked after
  commit 1 specifically" presumes the harness can still run commit 1 unmodified. It cannot (§1.6) —
  a harness-adaptation commit is required first, and the brief doesn't name it.

## Proposed `found.md` entries

```
## `create_causal_residuals` and `_compute_candidate_analytics` are dead code
Found during: track F2 (spec session)
Location: `src/residuals/causal_residuals.py:623-659` (`create_causal_residuals`);
`src/simulator/candidate_signals.py:698-757` (`_compute_candidate_analytics`)
What: Both have zero callers repo-wide, including notebooks (grep includes
`notebooks/**/*.ipynb` per the standing rule). Neither was previously recorded.
`create_causal_residuals` is a convenience wrapper around `fit_causal_residual_model`
+ `apply_causal_residual_model`; `_compute_candidate_analytics` is a per-candidate
scalar analytics path duplicating `_batch_analytics_for_group`'s vectorized logic.
Severity: cosmetic
Suggested track: F2 (whoever implements commit 5/1.3's signature change should
decide whether to delete or repurpose `create_causal_residuals`, since it's one of
`fit_causal_residual_model`'s four call sites and the cheapest one to just remove)

## `mr_diag_lb` and `MRDiagnosticsConfig.lookback` can silently disagree
Found during: track F2 (spec session, Part 3.8)
Location: panel-build time, `mr_diag_lb` parameter threaded through
`pair_candidate_panel_creator.py::_build_pair_candidate_rows_for_date` into
`_fast_pair_diagnostics` (`:449-552`); simulation time,
`MRDiagnosticsConfig.lookback` (`src/simulator/config.py:331`) read by
`candidate_signals.py::_compute_mr_diagnostics` (`:819-857`)
What: Both implement the same OU-fit math (dx-on-lagged-x OLS -> kappa, half_life,
mr_score) but are independently configurable, unlike a pure code-duplication
concern -- the same spread could be scored over different-length windows at
panel-build time versus at simulation time today, with no assertion tying the two
together and no caller stating they must match.
Severity: result-affecting, magnitude unmeasured
Suggested track: F2 (Part 4 commit 6 in F2_spec.md addresses the duplication; the
window-agreement question should be decided at the same time, not left implicit)

## The baseline harness exercises the `series/` disk-cache path, not the recompute path
Found during: track F2 (spec session, Part 2.3)
Location: `scripts/b_baseline_harness.py:184,232` (sets `candidate_panel_subdir`,
calls `_persist_series` before `run_from_config`); read side,
`src/simulator/candidate_signals.py:343-357,560-605`
What: Track B's own baseline harness has, since its creation, run its simulate step
through the disk-backed spread-level cache rather than the live recompute path --
`panel_dir` is set on every harness run, and `_persist_series` populates the cache
before the simulate stage reads it. `B_baseline.txt`'s neutrality checks to date
have therefore validated the cache path, not the recompute path the rest of the
simulator's live logic (`compute_analytics_from_weights`, `get_level_series`)
already exercises unconditionally.
Severity: cosmetic today (both paths are believed to agree; F2 commit 3 in
F2_spec.md is the first thing that actually proves it) -- would have been
result-affecting if they ever silently diverged
Suggested track: none -- resolved by F2's series/ deletion; recorded as a process
note about what B_baseline.txt has and hasn't been validating
```

---

## Review amendments (decided after the spec session; these override the brief and the sections above)

### R1. Spread-level definition [decided]

Today the spread level has two definitions:

- **Disk path** (`series.py`, read by `_try_batch_levels_from_disk`): residual model
  frozen at the candidate's `asof_date`, applied over the full history, times the
  frozen weights.
- **Recompute path** (`candidate_signals.py::_get_residuals`): residual model fitted
  at today's date t, times the frozen weights.

Which one runs is chosen per (group, day) by whether files exist. It is
all-or-nothing, and the only signal is a warning. `compute_analytics_from_weights` and
`get_level_series` always recompute. §2.3's "first proof the two agree" and Part 4
commit 3's neutrality claim are withdrawn: the two paths compute different things by
construction.

**Decided: frozen-at-asof is the only definition.** A candidate is the model and the
weights, both fitted at `asof_date`. Each subsequent day extends its level by one
out-of-sample residual. Reasons:

- The weights and the MR diagnostics that selected the candidate were fitted on the
  asof level.
- Under the daily model, the residual at t is in-sample, because the fit window
  includes t.
- One position has one level series, so entry z and exit z are measured on the same
  series.
- Fresh information enters through the outer refit, as new candidates.

Refitting the residual model during a hold is a research question for after the
post-G baseline, not for this track.

**Consequence:** F2 changes results in exactly one commit, C3.

### R2. One spread-level function

Every level consumer calls one function in `src/analytics/spread_primitives.py`:
- batch analytics
- `compute_analytics_from_weights`
- `get_level_series`
- the reconstruction function (C5).

The function computes residuals from the asof model, times the weights, then a
cumsum. Its apply window matches the disk path exactly, as established by pre-check
P1.

Residual matrices are held in memory, one full-history matrix per
`(group_id, residual_key, asof_date)`. A matrix is evicted when no tracked candidate
and no open position refers to it. Slicing at t is exact because the apply is row-wise
(`series.py:19-23`).

A missing asof model raises. It does not fall back to a live fit.

### R3. OU-fit consolidation leaves F2

Part 4 commit 6 is removed. Reconstructing level and z-score never calls the OU fit,
so the brief's "no third implementation" rule holds without consolidating.

The `mr_diag_lb` / `MRDiagnosticsConfig.lookback` question is result-changing. It goes
to `found.md`.

### R4. Loop migration as expand–contract

This replaces Part 4 commits 4 and 5, and resolves the sequencing tension.

- **C6a.** Build the in-simulator candidate generator, not wired in. Add an
  exact-equality test (`check_exact=True`) against `run_panel_batch` output on the
  harness universes, covering all scored candidates including invalid rows, weights,
  and residual params.
- **C6b.** Wire it in, and switch the harness `## counts` to the generator's output.
  This is neutral by C6a's test; `B_baseline.txt` must not move.
- **C6c.** Delete the offline path. Any grep supporting the deletion includes
  notebooks.

### R5. Fidelity test before the migration

Build it on the current path (§3.3), then keep it green through C6a–c.

### R6. Correction to §1.1

`_build_simulation_dates` (`simulator.py:922-943`) is a range bound, not an
intersection. It takes the market dates between the minimum and maximum panel
`asof_date`, clipped to the run window. The migration must reproduce the same start
and end.

### R7. C4 exactness caveat

The disk path derives spread returns as `np.diff(levels)`, while the recompute path
uses them directly. The two differ at rounding level, and both feed MR diagnostics. A
moved row at C4 gets explained, not assumed away.

### R8. Daily fit grid

After C3, list every remaining consumer of fits on non-outer dates. Whether the
migrated loop fits only on outer dates is decided at C6a, on that evidence.

### R9. Commit order (supersedes Part 4)

| # | Commit | Results |
|---|---|---|
| C1 | Delete dead code: `create_causal_residuals`, `_compute_candidate_analytics`, and the unused import at `candidate_signals.py:19` | neutral |
| C2 | `≤ t` tests: (a) `entry_asof_date <= entry_date`; (b) a perturb-the-future test on `_slice_fit_window` | neutral |
| C3 | R1 + R2: the single frozen-at-asof level function, all consumers repointed, a missing asof model raises | **changes results.** Explain the `B_baseline.txt` and `performance_metrics.json` diffs |
| C4 | Delete the `series/` cache, its writers, and `series.py` (§2.5 list) | expected neutral, with the R7 caveat |
| C5 | Reconstruction function; debug-sample config (pair count, seed, optional id tuple, no defaults); zero-tolerance fidelity test for candidate and position view; perturb-the-future `≤ t` test on reconstruction; `remove_residual_pcs > 0` raises | neutral |
| C6a–c | Loop migration, per R4 | neutral |
| C7 | Re-execute notebooks `01`, `02`, `03`; fix any notebook-only callers | — |

---

## Implementation pre-check

Read-only. Re-verified against current `main` (post review-amendments commits
`d1cb80b`, `22c7d25`, `53f61b6`). Measurements taken through the shipped code path
against `artifacts/simulation_runs/20260713_2017_e5ffbe83/`, not by editing a flag by
hand. Greps supporting any deletion claim include `notebooks/**/*.ipynb`.

### P1. Every spread-level consumer: residual model, apply window

**Disk path** — writer `src/residuals/series.py::compute_and_persist_series`
(`:66-249`), reader `candidate_signals.py::_try_batch_levels_from_disk`
(`:560-605`), called from `_batch_analytics_for_group` (`:343-357`).

- Residual model: frozen at the candidate's own `asof_date`. `series.py`'s Loop 2
  (`:190-243`) calls `asof_resid(group_id, rkey, asof)` (`:224`) where
  `asof = pd.Timestamp(row.series_asof)` (`:218`) is the candidate's `asof_date`
  (`:111`, `panel["series_asof"] = pd.to_datetime(panel["asof_date"])`).
  `full_residuals` (`:128-145`) looks up `params.get(pd.Timestamp(fit_date))`
  (`:137`) with `fit_date=asof`.
- Apply window: **full history, unsliced** — `full_residuals` calls
  `apply_causal_residual_model(model=model, aligned_returns=bundle.aligned_returns, ...)`
  (`:141-145`), where `bundle.aligned_returns` (`src/data/returns.py`) is the
  group's entire aligned-returns index, not passed through `_slice_fit_window`
  anywhere in `series.py`. This is exactly what R2's function must reproduce.
  The on-disk file therefore holds the full-history residual series computed
  once from the frozen model; the reader truncates to `.loc[:date]` **at read
  time** (`candidate_signals.py:588`), not at write time.

**Recompute path** — three live call sites, all routed through
`CandidateSignalGenerator._get_residuals` (`:915-975`):
1. `_batch_analytics_for_group`'s fallback, `:361`.
2. `compute_analytics_from_weights`, `:629`.
3. `get_level_series`, `:884`.
(A fourth call site, `:706` inside `_compute_candidate_analytics`, is dead code —
zero callers repo-wide including notebooks, confirmed below under P5/R9-C1 — and
is deleted at C1, not a live consumer.)

- Residual model: fitted/looked-up at **today's date** `date` (the argument
  passed in — always the simulator's current simulation day, never the
  candidate's `asof_date`, confirmed at the one place `date` originates,
  `simulator.py:313` `build_candidate_analytics_states(date=date, ...)`). Inside
  `_get_residuals`: `group_params[date]` if `date in group_params` (`:944-946`,
  the precomputed daily-fit-grid lookup), else a live fit at `date`
  (`fit_causal_residual_model(bundle=bundle, date=date, cfg=residual_config)`,
  `:949-953`). Both sub-cases fit/look-up at `date`, i.e. "today" — not a third
  definition, just two ways of arriving at the same (wrong, per R1) date.
- Apply window: `_slice_fit_window(bundle=bundle, date=date, cfg=residual_config)`
  (`:955-959`), then `apply_causal_residual_model(model=model, aligned_returns=aligned, ...)`
  (`:960-964`). `_slice_fit_window` (`causal_residuals.py:400-425`) always
  truncates to `.loc[:pd.Timestamp(date)]` (`:406`) — right-truncated at today,
  unlike the disk path's untruncated full history — and additionally, when
  `cfg.window_mode == "rolling"`, further truncates to `aligned.iloc[-cfg.lookback:]`
  (`:418`). For `window_mode == "expanding"` (the production `exp_hl504_mh1008_rf`
  key used in the measured run), no further truncation beyond `.loc[:date]`.

Since `apply_causal_residual_model` is row-wise with no window-dependent internal
state (confirmed at `causal_residuals.py:550-620` — every output row depends only
on that row's own inputs and the fixed `model` coefficients), slicing before vs.
after applying produces identical values for shared rows. The two paths differ in
(a) which model is applied — asof-frozen vs. today-fitted — and (b) which rows
exist in the output, not in how a shared row's value is computed. (a) is the
result-changing difference C3 fixes; (b) only matters for how far back the
z-score/MR-diagnostics windows can reach, unrelated to this track.

**Confirms R1/R2 exactly as written** — no drift found.

### P2. Which consumers feed a trading decision

Traced by following `analytics_by_id` (from `_batch_analytics_for_group`, disk-or-
recompute) and the three `compute_analytics_from_weights` call sites to where
their outputs are actually read in `simulator.py`.

**`_batch_analytics_for_group` output (`analytics_by_id`) — decision-feeding,
directly, in three places:**
- Entry: `self.trader.generate_actions(..., live_diagnostics_by_candidate_id=analytics_by_id)`
  (`simulator.py:360-363`). `z_score` drives entry (`_maybe_open`,
  `portfolio_mean_reversion.py:99,110`) and exit (`z_score` param of
  `_maybe_close`, same file `:78-80`, read at the trader's per-row loop,
  `simulator.py`'s `snapshot.candidate_states` built from `build_signal_frame`,
  itself sourced from `analytics_by_id`).
- Exit (deterioration stop): `portfolio_mean_reversion.py:145-153` compares
  `fz_analytics.mr_score < threshold * live_pos.entry_mr_score`, where
  `fz_analytics` is `diagnostics.get(candidate_id)` (`:76`,
  `diagnostics = live_diagnostics_by_candidate_id or {}`) — **this is
  `analytics_by_id`, not the separately-computed `fz_analytics_by_id`** (see
  below); despite the local parameter name `fz_analytics`, its `mr_score` comes
  from the frozen-weight `_batch_analytics_for_group` path, not from realized
  fill weights. A real naming collision between two different things both
  called "fz" — worth a comment or rename when C3 touches this code, not a
  correctness bug today.
- Sizing: `self.sizing_engine.size(proposed_opens=raw_opens, analytics_by_id=analytics_by_id)`
  (`simulator.py:395-398`); `sizing_engine.py:70-76` reads `a.roll_std` for vol
  normalization. **The harness exercises this**: `b_baseline_harness.py:191-193`
  sets `vol_normalize=VolSizingConfig(...)`, so `roll_std` from
  `_batch_analytics_for_group` scales every trade's notional on the harness run.

**`compute_analytics_from_weights` — three call sites, mixed:**
- `simulator.py:622-630` (`entry_analytics`, using `realized_weights_by_ticker` —
  actual fills). Its `mr_score`/`half_life`/`adf_pvalue` are persisted onto
  `LiveCandidatePosition.entry_mr_score`/`entry_half_life`/`entry_adf_pvalue`
  (`:650-652`). These **do feed a later exit decision**:
  `portfolio_mean_reversion.py:131-136` (time stop, `entry_half_life`) and
  `:145-153` (deterioration stop, `entry_mr_score`, per above). Not diagnostics-
  only — this corrects F2_spec.md §3.2's blanket "not read by the trader" framing,
  which is true for the *other two* call sites but not this one.
- `simulator.py:701-720` (`_build_live_fz_analytics` → `fz_analytics_by_id`) and
  `simulator.py:748-756` (`dy_a`) — both feed only `_build_diagnostics_log_entries`
  (`:722-787`) → `LiveDiagnosticsLogEntry` rows. Confirmed **not** passed to
  `trader.generate_actions` (which receives `analytics_by_id`, not
  `fz_analytics_by_id`, at `:363`) and not read by `sizing_engine` or
  `risk_manager`. Diagnostics-log-only, as F2_spec.md §3.2 states — for these two.

**`get_level_series` — decision-feeding when `entry_features` is configured,
not exercised by the harness today:**
- Used by `EntryFeatureEngine.compute` (`entry_feature_engine.py:171`), which
  writes into `CandidateAnalyticsState.features` (`:202`), read by
  `SizingEngine._compute_pair_notional` (`sizing_engine.py:83-89`) for
  `interval_scoring` — a `size_multiplier` of `0.0` there **drops the trade
  entirely** (`:87-88`). Structurally decision-feeding (a sizing veto), not
  diagnostics.
- **Not reached by the baseline harness**: `b_baseline_harness.py:200` sets
  `"entry_features": None`; `simulator_factory.py:110-112` only builds an
  `EntryFeatureEngine` `if config.entry_features is not None`; `simulator.py:370`
  guards the call on `self.entry_feature_engine is not None`. So `get_level_series`
  is never called on the harness run, and C3's change to it will not move
  `B_baseline.txt` through this path — but will affect any run configuring
  `entry_features`.

**Summary for predicting C3's effect on harness trades:** yes, C3 moves
`B_baseline.txt`, through two confirmed decision paths exercised by the harness —
`z_score` (entry/exit threshold) and `roll_std` (vol-normalized sizing) from
`_batch_analytics_for_group`, plus `entry_mr_score`/`entry_half_life` from the
`entry_analytics` call site feeding later exit stops. `get_level_series`'s sizing
path is real but dormant on this harness config.

### P3. Remaining consumers of fits on non-outer dates, after C3

`_get_residuals` (`candidate_signals.py:915-975`) is the only function anywhere
in the repo that looks up or fits a residual model keyed by a date other than a
candidate's own `asof_date` — confirmed by the full call-site grep under P1 (4
sites: 3 live, repointed to asof-only lookup by R2; 1 dead, deleted at C1). Grep
for `_get_residuals` across `src/`, `tests/`, `scripts/`, `run_me.py`, and
notebooks (`grep -rl` on `notebooks/`) returns zero hits outside
`candidate_signals.py` itself. **After C3, there are zero remaining consumers of
fits on non-outer (non-asof) dates** — every live level consumer moves to
asof-only lookup, and the dead one is gone.

**PC variance ratios**: `_compute_variance_ratios` (`:34-51`) runs as a side
effect of every `_get_residuals` call (`:968-973`), stored in
`self._latest_pc_variance_ratios` and exposed via `get_pc_variance_ratios`
(`:977-983`). Grepped `get_pc_variance_ratios` / `_latest_pc_variance_ratios`
across `src/`, `tests/`, `scripts/`, `run_me.py`, and notebooks: **zero callers
anywhere.** Nothing reads PC variance ratios into a decision — the whole
mechanism is already dead code, independent of C3. Not previously recorded in
`found.md`; folding into C1's dead-code deletion is the cheapest fix (C1 already
touches this file), reported here rather than decided.

**Does anything else depend on the daily fit grid surviving?** Checked
`simulator_factory.py::_load_residual_params` (`:462-515`) — a pure loader with
no outer/daily distinction, so it imposes no requirement either way. Checked
`daily_state_reconstruction.py` — confirmed (re-confirmed from F2_spec.md 0.2)
it never reads `residual_params` at all ("no residual replay"). No other
consumer found. This is exactly the evidence R8 asks C6a to use: nothing left
needs the daily grid, so fitting only on outer dates in the migrated loop is
supported by the evidence, not just plausible — reported as a fact for C6a to
act on, not decided here.

### P4. R2's in-memory cache — max concurrent keys and memory estimate

No such cache exists yet (pre-implementation), so this is measured by
reconstructing the activation-interval logic from persisted artifacts, not read
from a live counter. Method, applied to
`artifacts/simulation_runs/20260713_2017_e5ffbe83/`:

- `selected_panel.parquet` (`is_valid == True`, 2,041,607 rows) gives every
  candidate's `(group_id, spread_id, asof_date, candidate_id)`.
- `closed_trades/*.parquet` (21 year-files, 3,791 rows; 2 `candidate_id`s trade
  twice non-overlapping, handled as multiple intervals) gives each candidate's
  live-position interval(s), `[entry_date, exit_date]`.
- Per `(group_id, spread_id)`, sorted by `asof_date`, replicated
  `_activate_pair_candidate`'s exact rule (`candidate_activation.py:166-205`):
  a ref is dropped at the first later arrival for the same spread at which it is
  not currently open; if open at that moment, it survives until the next arrival
  that finds it closed.
- Aggregated per-spread intervals up to per-`(group_id, asof_date)` **key**
  intervals (a key survives as long as *any* spread sharing that asof_date is
  still tracked or open — matching R2's eviction rule, "evicted when no tracked
  candidate and no open position refers to it"), then swept the whole run for
  the maximum simultaneous count.

**Result: 164 concurrently live `(group_id, residual_key, asof_date)` keys**
(single `residual_key = exp_hl504_mh1008_rf` throughout this run), peaking on
2015-09-25 — composition: industrials 30, information_technology 26,
consumer_discretionary 21, health_care 19, consumer_staples 18, materials 14,
utilities 14, financials 12, energy 10, real_estate 0. This is a genuinely new
measurement (not previously reported anywhere in this track's documents) — 164
is well under the 209 max-concurrent-*positions* figure from §3.1, as expected,
since many candidates sharing one `(group, asof_date)` batch collapse to one key.

**Memory estimate**: using each group's member count (`config/universes/sp500_v1/
*.yaml`: 11–26 tickers, mean 21.5) and the run's full length as a conservative
per-matrix upper bound (`n_trading_days=5920` from `meta.json` — actual matrices
frozen at earlier asof-dates are shorter early in the run, so this over-
estimates), the 164-key composition above sums to 3,810 ticker-columns total.
float64, 8 bytes: `5920 × 3810 × 8 ≈ 1.80 × 10^8` bytes **≈ 172 MB** at peak.
Trivial relative to typical process memory; no eviction-policy tuning is needed
beyond what R2 already specifies.

### P5. R1–R9 vs. the code

No contradiction found. Every specific code claim in R1–R9 was re-verified and
holds exactly as written:
- R1's "daily model, residual at t is in-sample" — confirmed:
  `_slice_fit_window`'s `.loc[:pd.Timestamp(date)]` (`causal_residuals.py:406`)
  is inclusive of `date`, and `fit_causal_residual_model` fits on that window
  (`:466-470`), so when `date == t` the fit itself uses `t`'s own return row.
- R6's range-bound characterization of `_build_simulation_dates` — confirmed
  exactly: `start/end` from `market_dates`/`cp_dates` min/max
  (`simulator.py:932-933`), then `dates = market_dates[(market_dates >= start) &
  (market_dates <= end)]` (`:940`) — a bound on `market_dates`, not a set
  intersection with `cp_dates`.
- R7's `np.diff` vs. direct-returns caveat — confirmed:
  `candidate_signals.py:604` (`np.diff(levels, axis=0, prepend=0.0)`, disk path)
  vs. `:395-396` (`spread_returns = R @ W` then `cumsum`, recompute path) —
  returns-then-cumsum vs. levels-then-diff, exactly as R7 states.
- R9 C1's three deletion targets — all re-confirmed zero-callers-repo-wide
  including notebooks: `create_causal_residuals` (one definition,
  `causal_residuals.py:623`; one import, `candidate_signals.py:19`; no other
  hits), `_compute_candidate_analytics` (one definition, `candidate_signals.py:698`;
  no other hits), the `candidate_signals.py:19` import itself.
- `config.py:331`'s `MRDiagnosticsConfig.lookback` citation (used by R3/R8's
  reasoning and by `found.md`'s new entry) — confirmed, no drift.

One clarification, not a contradiction: R3 states reconstruction "never calls the
OU fit." Today's OU fit (`_compute_mr_diagnostics`) is *interleaved* with level
construction in the current code — called from inside `_finalize_batch_states`
(`:507-514`) and `compute_analytics_from_weights` (`:671-678`) — because both
functions compute level/z-score and MR diagnostics together for trading/logging
purposes. R3's claim is about the *new, dedicated* reconstruction function (C5),
whose job is exact-fidelity level/z-score reproduction only — it is correct that
this narrower function need not call the OU fit, but that's a scope decision for
C5, not a description of how the existing pipeline is structured today.

---

## Pre-check decisions

D1. C1 also deletes the PC variance-ratio mechanism: `_compute_variance_ratios`,
    `_latest_pc_variance_ratios`, `get_pc_variance_ratios`, and the side-effect call in
    `_get_residuals`. Basis: P3, zero callers anywhere, including notebooks.

D2. R2 clarified. The residual model is always the candidate's own asof model. The
    weights are the caller's: frozen candidate weights for `analytics_by_id`, realized
    fills for `entry_analytics` and `fz_analytics_by_id`, effective weights for `dy_a`.
    One function, which takes the weights as an argument.

D3. C3 includes a perturb-the-future test on R2's function, because the in-memory
    matrix holds rows after t by construction. Mutate the returns strictly after t, then
    assert that level, `z_score`, `roll_std` and the MR diagnostics at t are unchanged,
    with `check_exact=True`. This is the `≤ t` guard for the change C3 introduces.
    C5's reconstruction test does not replace it.

D4. C3's diff explanation must account for three paths:
    - `z_score` and `roll_std`, via `analytics_by_id`;
    - `entry_mr_score` and `entry_half_life`, via `entry_analytics` (time stop and
      deterioration stop);
    - for rolling-mode residual keys, the level's length changes as well (full history
      instead of the lookback window).
    State which residual keys the harness uses.

D5. C6a's outer-date-only fits are approved on P3's evidence, on one condition: show
    at `path:line` that a fit at date d is a pure function of data ≤ d, with no state
    carried across the walk (no warm start). The equivalence test then compares
    outer-date params only.

D6. Two found.md entries, added on the branch; do not fix either:
    - `portfolio_mean_reversion.py`'s `fz_analytics` parameter is populated from
      `analytics_by_id`, not from `fz_analytics_by_id` (a naming collision).
    - The deterioration stop compares a current `mr_score` computed on frozen candidate
      weights with an `entry_mr_score` computed on realized fill weights: two different
      spreads in one ratio. Severity: result-affecting, magnitude unmeasured.
      Suggested track: new.

### R10. C3's apply window (recorded, no action)

C3 changed where the level's cumsum starts. The old recompute path applied the model
to `_slice_fit_window`'s output, so the cumsum started at the slice's first row — and
in rolling mode the slice was also cut to the last `lookback` rows
(`causal_residuals.py:417`). C3 applies over the full history and truncates the level
afterwards, matching the disk path.

The resulting constant offset cancels in every consumer: MR diagnostics demean the
level and fit an intercept (`candidate_signals.py:771-781`), `adfuller`'s default
regression carries a constant, and the z-score's rolling mean absorbs it. Nothing
moves from the offset.

The substantive difference is the row count in rolling mode, where the old recompute
path computed its statistics over `lookback` rows rather than the full history. That
is the same non-determinism `SimulatorConfig.__post_init__` raises on
(`config.py:795-809`), so no reachable config is affected.

C3 removes that guard's cause: both paths now use the full history. Whether the guard
can be lifted is a separate decision, deferred — it needs the explicit span parameter
the guard's message describes. **Do not touch the guard in F2.**

### C4 result — correction: the moved field is `pair_notional`, not `realized_pnl_gross`

The C4 stop report misread `B_baseline.txt`'s column alignment. `TRADE_COLS`
(`b_baseline_harness.py:82-87`) orders columns as `trade_id, group_id, spread_id,
entry_date, exit_date, days_open, direction, pair_notional, entry_z_score,
exit_z_score, realized_pnl_gross, realized_pnl_net`. For trade `20061101:a971fcf5`,
the value that moved (`154628.238305 -> 154628.238304`) is **`pair_notional`**;
`realized_pnl_gross` (`2382.007568`) and `realized_pnl_net` (`2275.010938`) are both
unchanged.

This fits R7 more precisely than the original report claimed: `pair_notional` comes
from `SizingEngine`'s vol-normalization, which scales `base_pair_notional` by
`median_roll_std / a.roll_std` — `roll_std` is one step downstream of the level, so
the same rounding-order difference between `np.diff(levels)` and direct
`spread_return` propagates directly into it. The two PnL columns are computed from
integer share counts (rounded at execution) and are printed to six decimals on
figures in the thousands — a relative difference on the order of `1e-11` is well
below both that rounding and that print precision, so they show as unchanged even
though the same underlying floating-point noise is present there too, in principle.

### R1 impact measurement

What R1's fix (today-dated vs. asof-frozen residual model) was actually worth on
this harness, isolated from C4's own effect. Method: checked out the commit before
C3 (`9362e4a`), removed only the harness's `_persist_series` call (skip populating
the disk cache — a harness-config change, not a flag edit) so the simulator falls
through to the pre-C3 recompute path (`_get_residuals`, today-dated model), and
compared the resulting `B_baseline.txt` against the current, post-C4 baseline.

**First attempt was contaminated and discarded.** `artifacts/candidate_panels/
refactor_b_harness/series/` held ~5,235 leftover files from earlier sessions this
track. With those present, the reader (`_try_batch_levels_from_disk`, still intact
at the pre-C3 commit) found them and served the *old, already-correct-per-P1* disk
values regardless of the harness's own populate call being skipped — so the first
run measured nothing. Cleared the directory (`rm -rf .../series/`, confirmed 0 files
both before and after the corrected run) and re-ran.

**Result: large.** 97 trades (post-C4, asof-frozen) vs. **115 trades** (pre-C3
recompute, today-dated) — 37 trades exist only under the old today-dated path, 19
exist only under the new asof-frozen path, and of the 78 trades common to both by
`trade_id`, **all 78** have a different `entry_z_score` and/or `exit_z_score`. No
trade survives with identical timing and z-scores. This is not the rounding-level
effect C4 showed — the divergence is structural and compounds through the run:
`EQ_EXPANDING`'s model drifts slowly day-to-day, so any single day's today-vs-asof
difference is individually small, but once one trade's exit timing shifts by even
one day, every later occupancy/activation decision for that spread (and every
z-score computed off the resulting level series) cascades into a different
sequence — consistent with the brief's own framing of this invariant's failure mode
("plausible-looking *better* results rather than an error", `F2_loop_and_fidelity.md`
§1).

**Per E6: this is a large diff.** Reported before proceeding to C5, not folded in
silently.

### E7 — isolating model effect from cascade

Method: monkey-patched `CandidateSignalGenerator.build_candidate_analytics_states`
to capture its first call's `(date, candidate_refs, z_score per candidate_id)` and
raise immediately after, so the run stops on the first simulated day. Ran once at
the current (post-C4, asof-frozen) commit and once at the pre-C3 commit with the
same series/-cleared, populate-skipped isolation as the R1 measurement above. Both
captured the identical 127 candidates on `2006-09-08` (the harness's
`PANEL_START_DATE`, and the panel's own earliest `asof_date`).

**Result: rounding-scale.** median `|Δz|` = `2.22e-16`, 90th percentile =
`6.66e-16`, max = `1.78e-15`, 0 of 127 candidates above `1e-6` (let alone `0.01`) —
machine-epsilon noise, not a model difference.

**This is expected, and does not establish chaos, because day 1 is degenerate for
this test.** `_build_simulation_dates`'s `start = max(market_dates.min(),
cp_dates.min())` (`simulator.py:932`, R6) puts the first simulated day exactly at
the panel's earliest `asof_date` — so for every one of the 127 day-1 candidates,
`ref.asof_date == date`. The today-dated lookup (`group_params[date]`) and the
asof-frozen lookup (`group_params[ref.asof_date]`) hit the **same dict key** on day
1, for every candidate, by construction — not because the two definitions agree,
but because they have not yet had a chance to disagree. E7's own premise ("no
position divergence can have occurred yet, so any difference is the model alone")
is true, but the unstated second half — that the *model itself* has already had a
chance to diverge by day 1 — does not hold here: it can only diverge from the first
day some tracked candidate's `date` moves past its own `asof_date` (day 2 onward,
for anything still tracked from day 1; immediately, for any later-arriving
candidate evaluated on a non-arrival day).

Per E7's rule this measurement is rounding-scale, so by its letter this stops and
reports rather than recording-and-proceeding. **Superseded by E7b below**, which
measures divergence as a function of age within one run (no cascade contamination
possible) rather than relying on cross-run comparison at a single, degenerate day.
This day-1 result stands as E7b's positive control at age 0.

### E7b — model divergence as a function of age, within one run

Method: one run, at the current (post-C4, shipped) commit — no cross-run
comparison, so no cascade can contaminate the measurement. Monkey-patched
`build_candidate_analytics_states` to sample every 9th simulated day (~21 of the
188 total, spread across the run) and, for every tracked candidate on those days,
compute `z` twice: `z_asof` from the shipped `_get_asof_residuals(ref.asof_date)`
path (identical to what the run itself used), and `z_today` — measurement-only,
never fed back into the run — from the model at `date` looked up the pre-C3 way
(`group_params[date]`, then `_slice_fit_window` + `apply_causal_residual_model` +
the same `compute_spread_level`/`_compute_z_score` the shipped path uses, so only
the model/window differs). `age` = trading-day distance between `ref.asof_date` and
`date`, measured on the candidate's own group bundle index. 2,867 candidate-day
observations across 21 sampled days. Scratch script, not committed.

| age (trading days) | n | median \|Δz\| | 90th pct | max | count > 0.01 |
|---|---|---|---|---|---|
| 0 | 532 | 0 | 0 | 0 | 0 |
| 1–5 | 1456 | 0.0076 | 0.064 | 0.443 | 658 (45%) |
| 6–20 | 271 | 0.0093 | 0.071 | 0.224 | 132 (49%) |
| 21–60 | 406 | 0.0135 | 0.084 | 0.325 | 226 (56%) |
| 60+ | 202 | 0.0473 | 0.181 | 0.401 | 166 (82%) |

Largest age observed: 180 trading days.

**`|Δz|` grows with age and is model-scale well within typical holding periods.**
The harness's own mean holding period (38.9 days, from the same run's performance
table) falls in the 21–60 bucket, where the median `|Δz|` is already `0.0135` —
comfortably past the `entry_z=1.75`/`exit_z=0.0` thresholds' sensitivity, and
`count > 0.01` is a majority in every bucket past age 0. **This confirms E6's
cascade explanation and rules out chaotic floating-point amplification**: the
divergence is a real, monotonically-growing function of how stale the "today"
model is relative to the frozen asof model, not noise. Age 0 (`n=532`, `|Δz|`
exactly `0`) reproduces E7's day-1 finding within the same run and serves as its
positive control: at zero age the two lookups are arithmetically identical, which
is also the confirmation that C3's rewrite into `compute_spread_level` is
arithmetically equivalent to the old inline `R @ W` + `cumsum` it replaced — the
rewrite itself introduced no divergence; every observed `|Δz| > 0` is attributable
to the model choice (asof-frozen vs. today-dated), not to C3's refactor mechanics.

Per E7b's rule: record and proceed. R1's fix in C3 was a real, non-trivial
correction, sized here — not a cosmetic change validated only by a chaotic
harness.

### E9 — confirming the fidelity test would catch a wrong artifact, not just wrong plumbing

Before C6a: the fidelity test proves the plumbing (which dates, weights, model get
passed around) by construction, since reconstruction calls the same
`compute_analytics_from_weights` the run itself calls (R2's "single code path" —
correct, but also the common-mode blind spot: an error inside that shared function
is invisible to the test). The remaining question is whether anything *else* is
shared — specifically, whether the reconstruction-side `CandidateSignalGenerator`
reads its residual params, weights and returns from disk, or from the run's own
in-memory state.

**(a) Confirmed independent, with one caveat.** In
`tests/test_reconstruction_fidelity.py::setUpClass`:
- `umd = _load_umd(sim_config.data)` (`:96`), `_load_residual_params(sim_config.data, z_configs)`
  (`:100`), `_load_weights(sim_config.data, z_configs)` (`:101`) are called a
  *second* time, independently of the `run_from_config(sim_config)` call above them
  (`:74`) that produced the live run's own generator. Checked both loaders for
  caching that would silently hand back the same object on a second call — none:
  `_load_umd` (`simulator_factory.py:191-229`) constructs a fresh `UniverseDataLoader`
  and calls `.load()` every time, no module-level cache; `_load_residual_params`
  (`:457-511`) and `_load_weights` (`:513-562`) both go straight to
  `pd.read_parquet` per call, no caching either. `create_simulator`
  (`simulator_factory.py:45-97`) constructs a fresh `CandidateSignalGenerator(...)`
  (`:88-96`) every call, and that class's `_bundle_cache`/`_asof_residual_cache`
  fields are per-instance (`dc_field(default_factory=...)`, `candidate_signals.py`)
  — a new instance starts with empty caches regardless of what any other instance
  has cached. So `sg2` (the reconstruction-side generator) shares no bundle, no
  asof-residual cache entry, and no params/weights dict object with the run's own
  generator — everything traces back to a fresh disk read.
- **Caveat, not covered by (a)'s claim:** `cls.selected_panel = result.selected_panel`
  (`:102`) *is* taken from the live run's own in-memory `SimulationResult`, not
  reloaded from disk. This is used only by `reconstruct_candidate_view` (picking
  the latest `asof_date <= date` for a spread) — not by `reconstruct_level_and_zscore`
  or `reconstruct_position_view`, and not by weights/residual-params resolution at
  all. `test_candidate_view_reconstructs_exactly_via_dedicated_function` is
  therefore not yet a full artifacts-only check; the other tests are.

**(b) Mutation check: confirmed the test fails on a wrong artifact.** Built the
panel and ran the live simulation once (capturing `debug_sample_df` from the
original, correct weights). Used `discover_group_data_sources` (the same
resolution `_load_weights` itself uses) to find the exact `..._weights.parquet`
file backing the `energy` group — a naive "newest file by mtime in the directory"
first attempt picked up an unrelated leftover stem from an earlier session (the
same class of contamination as E8/E6's first attempt), so this matters. Perturbed
one row: `APA` in spread `APA|VLO`, `asof_date=2006-09-08`, weight `-1.0 -> 0.0`.
Re-loaded weights fresh from disk (confirmed the mutation was visible:
`weights_lookup[("APA|VLO", ...)]["APA"] == 0.0`), rebuilt a fresh
`CandidateSignalGenerator` from it, and reconstructed the same row:

- Expected (captured, correct weights): `z_score=-0.2248`, `level=-1.25e-15`.
- Reconstructed (mutated weights): `z_score=2.1077`, `level=2.59e-16`.
- **Mismatch: confirmed.** The fidelity test would have failed.

Restored the original file immediately after (confirmed via a fresh read).
Scratch script, not committed, per the task.

**Conclusion: the fidelity test is reading artifacts, not memory.** Safe to proceed
to C6a. The `selected_panel` caveat in (a) is noted for whoever next touches
`reconstruct_candidate_view`'s own test, not blocking — it affects only which
`asof_date` a candidate-view query picks, not the residual/weights computation
once one is chosen, and the dedicated position-view test plus the all-rows test
already exercise the disk-artifacts-only path for every other function.

### D5 — a fit at date d is a pure function of data ≤ d, no state across the walk

Required before proposing outer-date-only fits for C6a. Two facts together prove it:

- `fit_causal_residual_model(bundle, date, cfg)` (`causal_residuals.py:451-469`)
  takes exactly those three arguments — no `self`, no accumulator, no prior-model
  parameter. Its body computes everything fresh from `_slice_fit_window(bundle,
  date, cfg)` (`:400-425`, a pure `.loc[:date]` slice of `bundle.aligned_returns`,
  plus an `.iloc[-lookback:]` further slice in rolling mode) and `_wls_multi`
  (`:440-448`), which calls `np.linalg.lstsq(Xw, Yw, rcond=None)[0]` — a batch
  least-squares solve on the *current* window's `X`/`Y` alone, no warm-started
  initial guess, no persisted solver state carried from any other call.
- The walk loop itself threads no state between iterations:
  `pair_candidate_panel_creator.py:812-816` — `model = fit_causal_residual_model(
  bundle=bundle, date=dt, cfg=residual_cfg)` inside `for i, dt in
  enumerate(walk_dates, start=1):`. `bundle`/`residual_cfg` are the same
  loop-invariant objects on every iteration; only `dt` varies. `model` is
  reassigned fresh each iteration and stored into `fitted_params[dt]` (`:820`, a
  dict keyed by date, for later persistence) — never passed into a subsequent
  `fit_causal_residual_model` call as a warm start or otherwise.

Together: a fit at outer date `d` is bit-identical whether or not any daily fits
happened at other dates in between. Outer-date-only fitting changes nothing at
the dates that still get fit — confirmed empirically too, by C6a's own
equivalence test below (residual params compared exactly at every outer date).

### C6a — the in-simulator candidate generator, not wired in

Per R4: `src/simulator/candidate_generation.py::generate_candidates` walks outer
refit dates only (via `resolve_asof_datetimes`, extracted verbatim from
`create_pair_candidate_panel` into a shared function so both callers resolve
outer dates identically — `pair_candidate_panel_creator.py`), calling the exact
same `fit_causal_residual_model` and `_build_pair_candidate_rows_for_date` the
offline walk already calls. No daily grid, no persistence — purely in-memory,
not wired into `Simulator.run()`.

`tests/test_candidate_generation_equivalence.py` compares its output against
`create_pair_candidate_panel` (what `run_panel_batch` itself calls) on the
harness's own two universes and config, given the *same* `GroupReturnBundle`
object for both calls (eliminating any bundle-reconstruction noise — see below).
Checks, `check_exact=True`: all scored candidate rows including invalid ones (set
equality of columns, then per-column exact value equality), all weight rows, and
every outer-date residual model's `B_proxy`/`B_stock`/members/proxy/bench/
subtract_risk_free. **All pass, both groups.**

**A real, previously-latent defect surfaced building this test, not a flaw in
the generator's own logic.** First attempt (separately-rebuilt bundles per side)
showed a spurious 1-date mismatch that traced entirely to bundle reconstruction
noise; fixed by sharing one bundle object. With that fixed, a *second*,
genuine mismatch remained: `resolve_asof_datetimes` returned an extra outer date
(`2007-04-06`, Good Friday) that is not actually present in either universe's
`bundle.aligned_returns.index` — `make_ranking_dates` returns resample bin
labels, not the actual date of the row `.last()` selected (found.md: "
`make_ranking_dates` can return a resample bin label that is not an actual data
date"). The offline walk never notices, because it only builds rows on dates
also present in its own daily fit grid (also from `make_ranking_dates`, which
correctly drops the holiday's empty daily bin) — an accidental self-correction,
not a deliberate check. `generate_candidates` has no daily grid to filter
through, so it filters phantom dates itself (`asof_datetimes[asof_datetimes.
isin(bundle.aligned_returns.index)]`) — scoped to the new function only;
`make_ranking_dates`/`resolve_asof_datetimes` themselves are unchanged, per
scope (fixing the shared utility is recorded in found.md as a separate, new
track, not this one).
