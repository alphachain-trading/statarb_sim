# Track F — spec (delta against `F_artifact_layer.md`)

Read-only verification session. This is not a design document: Part 1 confirms or
corrects claims in the brief, Part 2 lays out options for the questions the brief
says must be settled here, Part 3 proposes a commit order. No code was changed.

---

## Part 1 — verification

### 1. Live weights serialization path

Live column name is **`weights`**, not `weights_json`. Written at
`src/candidates/pair_candidate_panel_creator.py:382-385`:

```
382:                "weights": serialize_weights_for_spread_id(
383:                    weights=weights,
384:                    spread_id=sid,
385:                ),
```

`serialize_weights_for_spread_id` (`src/candidates/candidate_panel.py:81-100`) does
the actual serialization, at `candidate_panel.py:100`:

```
100:    return w.to_json(double_precision=12)
```

Precision: **`double_precision=12`** (pandas `to_json`), which does not round-trip
`float64` — confirms the brief's fidelity claim directly.

Dead path confirmed: `src/residuals/spreads.py`'s `weights_json` column
(`_weights_json_from_pair_weights` at `spreads.py:97`, written by
`build_spread_returns` at `spreads.py:282,289-292`) has **zero callers** anywhere in
`src/`, `notebooks/**/*.ipynb`, `tests/`, or `run_me.py` — grep for
`build_spread_returns`, `extract_relative_weights`, `_weights_json_from_pair_weights`
turns up only their own definitions and each other. The superseded handover's
"`weights_json` belongs to a dead code path" claim is correct.

### 2. The two write-only log classes

**Correction, not a confirmation.** The two classes matching the brief's
"`(date, group_id)` grain... no aggregate total rows" description are
`DailyStateLogEntry` (`src/simulator/simulator.py:57-77`, grain `(date, group_id)`)
and `DailyPortfolioStateLogEntry` (`simulator.py:80-93`, grain `(date,)`, the
aggregate).

`DailyStateLogEntry` **is** write-only as claimed. Every field
(`active_candidate_id`, `active_z_score`, `live_candidate_id`,
`group_deployed_notional`, etc.) greps to zero hits outside `simulator.py`,
including in `notebooks/**/*.ipynb`. Populated at `simulator.py:851,874`
(`_build_daily_state_log_entries`), serialized only by its own `daily_state_df()`
(`simulator.py:175-177`), and persisted under `daily_states/` by
`flush_yearly_logs` (`simulation_persistence.py:139-155`). No other reader anywhere.

**`DailyPortfolioStateLogEntry` is NOT write-only.** Its `daily_portfolio_state_df()`
(`simulator.py:179-188`) is consumed **in the same run, in memory**, by
`generate_report` at `simulator.py:625-629`:

```
625:            performance_result = generate_report(
626:                closed_trades_df=result.closed_trades_df(),
627:                daily_portfolio_state_df=result.daily_portfolio_state_df(),
628:                cfg=perf_cfg,
629:            )
```

`src/simulator/performance/performance_basic.py` reads specific columns off that
frame by name — `total_equity_gross` (`performance_basic.py:96,254`),
`total_equity_net` (`:97,249`), `n_live_candidate_positions` (`:272`),
`daily_borrow_cost` (`:282`), `cumulative_transaction_costs` (`:276,297,325`),
`cumulative_borrow_costs` (`:277,298,325`) — to compute Sharpe/Sortino, drawdown,
average borrow cost, and average concurrent positions. This is the run's actual
`PerformanceResult`, not a later read-back of a persisted artifact. It is also read
directly by a notebook (`notebooks/howto/02_run_simulation.ipynb`, cell calling
`result.daily_portfolio_state_df()` and plotting `total_equity_net`).

The brief's "apparent decision-logic hits... resolve to same-named fields on other
classes — a naming collision, not a dependency" does **not** hold for this class:
these are genuine reads of this class's own fields by `performance_basic.py`, via
literal `df["total_equity_gross"]`-style column access on the exact frame this class
serializes to. Deleting `DailyPortfolioStateLogEntry` without first standing up an
equivalent-column `daily_state_df(run_dir)` reconstruction and repointing
`generate_report`'s caller (`simulator.py:625-629`) at it would break in-run
performance computation, not just historical analysis notebooks. This is a sequencing
constraint on the commit order (Part 3), not a reason to keep the class.

Separately: the brief's proposed replacement function is named `daily_state_df(run_dir)`
(brief, "daily_state — dropped" section). A method with that exact name,
`daily_state_df(self)`, already exists on `SimulationResult`
(`simulator.py:175-177`) and is being deleted by this same track. Same name, two
unrelated things at two different points in the track's history — worth a different
name for the reconstruction function to avoid this confusion in code review and git
blame.

### 3. Every artifact currently written by a run

Enumerated from `src/simulator/simulation_persistence.py` (the `save_simulation_run`
/ `flush_yearly_logs` path) plus one artifact writer outside that module:

| Artifact | Grain | Writer |
|---|---|---|
| `config.json` | one serialized `SimulatorConfig` | `simulation_persistence.py:86-89` |
| `selected_panel.parquet` | one row per `(spread_id, asof_date)` candidate **actually used by the run** (already filtered to `is_valid=True`/`success=True` — see Part 2 §10 measurement) | `simulation_persistence.py:93,101` |
| `daily_portfolio_state.parquet` | one row per date, whole-portfolio | `simulation_persistence.py:94,101`, via `simulator.py:179-188` |
| `performance_metrics.json` | one dict | `simulation_persistence.py:104-108` |
| `performance_group_metrics.json` | one dict per group, if present | `simulation_persistence.py:109-113` |
| `*_quantstats.html` | one HTML report | `simulation_persistence.py:370-380` (`_copy_performance_html`) |
| `meta.json` | one dict, run summary | `simulation_persistence.py:330-367` (`_build_meta`) |
| `daily_states/daily_state_{year}.parquet` | `(date, group_id)`, per-year slice | `simulation_persistence.py:139-155,172-191`, via `simulator.py:175-177` |
| `action_logs/action_log_{year}.parquet` | `(date, action)`, per-year slice | same flush path, via `action_log_df()` |
| `ticker_trade_logs/ticker_trade_log_{year}.parquet` | `(date, group_id, candidate_id, spread_id, ticker)`, per-year | same flush path, via `ticker_trade_log_df()` |
| `diagnostics/diagnostics_{year}.parquet` | `(date, trade_id, ...)`, per-year | same flush path, via `diagnostics_log_df()` |
| `closed_trades/closed_trades_{year}.parquet` | one row per closed trade, per-year | same flush path, via `closed_trades_df()` |
| `z_spectra_{group}__{rkey}__{suffix}.parquet` | `(spread_id, date)` index × `zlb` columns, opt-in (`SpectrumConfig`) | `src/simulator/z_spectrum_capture.py:330-343,346-359` (`ZSpectrumCapture._save_store`/`_save_hybrid_store`), written straight to `run_dir`, bypassing `simulation_persistence`'s artifact registry entirely |

Not addressed anywhere in the brief's Layout section or its scope list: the
`z_spectra_*.parquet` files. They're config-gated (`SpectrumConfig`), so not every
run produces them, but they are a currently-written per-run artifact with no stated
fate under the new flat layout. Flagging since the brief's Layout block doesn't
mention it and "Also in scope"/"Out of scope" don't either — worth an explicit
decision (keep as-is under `simrun_dir/`, fold into `performance/`, or defer) before
implementation, rather than rediscovering it mid-track.

### 4. `series/` artifacts

Exact key: `series/stock/{group_id}__{ticker}.parquet` (single column
`residual_return`) and `series/spread/{spread_id_sanitized}__{YYYYMMDD}.parquet`
(single column `level`), built by `sanitize_spread_id`/`spread_series_filename`
(`src/residuals/series.py:49-57`). Written by `compute_and_persist_series`
(`series.py:84-254`), called from `run_me.py:319,532` (`_persist_spread_series` /
`_persist_series_multi`) and from a notebook cell
(`notebooks/howto/02_run_simulation.ipynb:1647`). Read by
`src/simulator/candidate_signals.py:539-542,576` (`_spread_series_path`), which the
simulator's batch signal path calls when `panel_dir` is set.

Stem-reuse behaviour, confirmed at the code level: both write loops explicitly skip
if the target file already exists —

```
series.py:192-193 (stock):   if path.exists(): continue
series.py:233-234 (spread):  if path.exists(): continue
```

— documented as intentional ("incremental / resumable", `series.py:111-112`). The
filename key is `(group_id, ticker)` / `(spread_id, asof_date)` only —
**`residual_key` is not in the key** — so writing a panel under a residual config
different from whatever last populated that stem's `series/` directory silently
reuses the old files; the reader has no way to detect this. This is also asserted by
a committed test's own docstring:
`tests/test_clear_stem_artifacts.py:1-9,38` — stem-scoped clearing on a fresh panel
write deliberately **never** touches `series/` (`test_stem_scoped_clear_leaves_siblings_and_series_intact`),
so nothing in the current code path clears stale series on any kind of panel rewrite,
config change included. `found.md`'s axis-2 description is confirmed exactly as
stated, now with line-level citations.

### 5. Outer/inner loop in panel creation, and the `≤ t` alignment

Outer dates: `create_pair_candidate_panel` (`src/candidates/pair_candidate_panel_creator.py:617-723`)
resolves `asof_datetimes` either from an explicit `dates` list (validated against
`make_ranking_dates(..., frequency="B", min_history=...)`, `:671-683`) or from
`make_ranking_dates(dates=aligned_index, frequency=frequency, min_history=residual_cfg.eff_min_history())`
(`:691-695`) — `frequency` is typically `"W-FRI"` per the harness and metadata
observed in `artifacts/candidate_panels/full_v1/*.meta.json`.

Inner loop: `daily_dates` is every business day in range
(`frequency="B"`, `:713-723`); the walk (`:736-781`) fits the residual model on
**every** `walk_dates` entry but only calls `_build_pair_candidate_rows_for_date`
(builds candidate rows / weights) when `dt in panel_date_set`, i.e. only on outer
dates (`:738,768-781`).

The `≤ t` alignment is **not implemented as an explicit comparison anywhere** — grep
for `asof_date.*<=` / `<=.*asof_date` in `src/` and `tests/` turns up nothing (the
only `<=` hits are unrelated date-range filters at `pair_candidate_panel_creator.py:700,723`).
On the consuming side, `CandidateFilter.get_selected_on_date`
(`src/simulator/candidate_filter.py:45-52`) uses **exact equality**,
`selected_panel["asof_date"] == date` (`:51`) — new candidates only "arrive" on the
day that exactly matches their `asof_date`, and then persist as tracked/live
positions using their originally frozen weights
(`candidate_activation.get_active_candidates()`, `simulator.py:1196-1197`) until
closed. The invariant holds today **structurally** (chronological walk-forward
construction plus arrive-once/persist-until-closed activation), never by an
assertion or a test. This matches the brief's acceptance criterion implying no such
test exists yet — confirmed NOT FOUND.

### 6. Four config classes with zero construction sites — re-verified including notebooks

`SpreadMomentumConfig(` and `KellyConfig(` construction calls: **NOT FOUND** anywhere
in `src/`, `notebooks/**/*.ipynb`, `tests/`, `run_me.py`, or `config/*.yaml`.
`TimescaleRiskConfig(` and `CrossTimescaleEntryConfig(` construction calls: also
**NOT FOUND** anywhere. All four re-confirmed dead including notebooks.

Nuance for `TimescaleRiskConfig`/`CrossTimescaleEntryConfig`: unlike the other two,
their **fields are read** by live logic — `ts_cfg.selection` at
`src/simulator/risk_manager.py:151`, `cfg.same_sign_required` at
`src/simulator/traders/pair_spread_mean_reversion.py:275`, and both fields in a
sweep-label builder at `src/simulator/sweep_runner.py:148,184`. All three sites are
reachable only when the containing `Optional[...] = None` field
(`config.py:462,816`) is non-`None`, and nothing ever constructs it non-`None`, so
the read sites are live code, currently dead by construction rather than by absence
of a caller. Deleting the class means also deleting these three now-orphaned read
sites, not just the class body.

### 7. `start_after_nan` / `check_for_corruptions`

Two classes, confirmed:

- `DataConfig` (`src/simulator/config.py:241-264`): `check_for_corruptions: bool = True`, `start_after_nan: bool = True`
- `PanelBatchConfig` (`src/candidates/panel_batch.py:102,150-152`): `check_for_corruptions: bool = False`, `start_after_nan: bool = True`

`check_for_corruptions` disagrees (`True` vs `False`); `start_after_nan` agrees
(`True`/`True`) between these two classes specifically. Neither is overridden by any
explicit call site — `simulator_factory.py:209-210` and `panel_batch.py:293-294`
both just forward the class's own default.

**Correction/addendum:** there is a third live call site with a third, distinct
combination, not sourced from either config class:
`src/data/market_snapshot.py:277`:

```
277:        umd = loader.load(force_download=True, check_for_corruptions=True, start_after_nan=False)
```

This is a hardcoded literal in Track C's snapshot-download path, not a config-class
default — `start_after_nan=False` here, disagreeing with both `DataConfig` and
`PanelBatchConfig`. `UniverseDataLoader.load`'s own signature default
(`src/data/universe_loader.py:36`) is `check_for_corruptions=True, start_after_nan=True`,
a fourth combination coinciding with `DataConfig`'s. Whoever resolves the
two-class disagreement should be aware a third, hardcoded call site exists with yet
another combination — it's out of scope to unify unless F's config-surface reshaping
decides these three should share one source of truth, which the brief doesn't ask
for and this session isn't deciding.

---

## Part 2 — findings and options (not decided here)

### 8. `n_legs`

No live consumer today reads `n_legs` besides: notebook display cells
(`notebooks/howto/01_create_candidate_panel.ipynb`) and
`tests/test_track_a_guards.py` (tests `n_legs`'s own gate-consistency semantics,
independent of either option below).

**Decisive finding on the "reproduce the raw fit" question:** the actual
`spread_return` used for every persisted diagnostic (`kappa`, `half_life`,
`intercept`, `level_std`, `spread_return_std`, and the spread level itself) is
computed from **both legs' full weights unconditionally**, before any threshold is
applied —

```
pair_candidate_panel_creator.py:347-350
    spread_return = (
        pair_diag[left].to_numpy(dtype=float) * w_left
        + pair_diag[right].to_numpy(dtype=float) * w_right
    )
```

— and only afterward, inside `_fast_pair_diagnostics`, is `tiny_weight_threshold`
applied to produce the **diagnostic count** `n_legs`
(`pair_candidate_panel_creator.py:437`) and the `too_few_active_legs` gate
(`:454-455`). The threshold is never applied to the weights actually used to compute
`spread_return`/`level`/`kappa`/`intercept`.

- **Option A — keep `n_legs` as a real column with threshold semantics.**
  No downstream effect: `weights.parquet` stores both legs' full weights regardless,
  so reconstruction (`residuals @ weights` per the brief's "Spread level and
  reconstruction" section) reproduces the original `spread_return` bit-for-bit.
  `n_legs` stays purely a diagnostic/gate count, documented as such.

- **Option B — store only active legs in `weights.parquet`.** Breaks the
  zero-tolerance fidelity test whenever a candidate has a sub-threshold leg: the
  persisted `kappa`/`half_life`/`intercept`/`level_std` were computed from a
  spread_return that included that leg's (small but nonzero) weight, but the
  reconstruction path would compute a spread_return missing that leg entirely —
  not bit-identical, and not close either (a whole leg's return contribution is
  removed, not just rounded). This is not a hypothetical: Track A's own tests show
  `n_legs` can legitimately be 0 or 1 for a two-ticker spread
  (`tests/test_track_a_guards.py:211-219`), so sub-threshold legs occur in practice
  today. Under this option, any such candidate's persisted diagnostics would be
  provably unreconstructable from `weights.parquet` alone.

Given the fidelity test's `check_exact=True` requirement, Option B is not just a
"accept reduced reconstructability" tradeoff as the brief frames it — it actively
fails the acceptance criterion for any candidate where the threshold ever bites.
Reported per instructions rather than decided; the asymmetry above should probably
be part of that decision when it's made.

### 9. `series/` staleness axis 2

**Cost of keying by `residual_key`:** mechanical and small. Add a `residual_key`
segment to `spread_series_filename` and the stock filename builder in
`src/residuals/series.py`, and to the reader
`candidate_signals.py:539-542` (`_spread_series_path`). Existing on-disk files
without the segment become orphaned (never matched, never cleaned) rather than
silently misread — that's the actual bug fix. No other reader or test depends on the
current filename shape besides the ones just named (`test_clear_stem_artifacts.py`
only checks that `series/` is left alone by stem-clearing, not its internal naming).

**Whether the artifact is still needed at all:** `series/` is not in the brief's new
`simrun_dir` Layout list, and its stated purpose — "the simulator can *load* levels
from disk instead of recomputing... on every step" (`series.py:4-6`) — is exactly the
cross-run precomputed-cache pattern the "Decision being implemented" section rejects
("Cross-run artifact reuse is explicitly rejected... reintroduces the durable
content-addressed key tree the dissolution deletes"). `series/` is keyed by
`(group_id, ticker)` / `(spread_id, asof_date)` under a shared `panel_dir`, i.e. it
is reused across runs and across residual configs by construction — the same shape
of problem the panel dissolution exists to remove one layer up.

Whether it's still needed depends on how expensive the recompute actually is once
the outer/inner loop and residual-loadings persistence (`residual_params.parquet`)
land — see Part 2 §10's measurement: a single `apply_causal_residual_model` call
(one candidate's full-history residual reconstruction) measured **1.7 ms** on real
data (`consumer_discretionary`, 5314-row × 25-ticker aligned returns; see §10 for the
harness). At that cost, disk-caching spread levels buys little relative to the
staleness risk it carries.

- **Option A — key by `(residual_key, spread_id/ticker, ...)` and keep the cache.**
  Fixes axis 2 directly; retains the disk-cache-as-optimization pattern, but keeps a
  cross-run reuse mechanism alive in a track whose stated decision is to remove
  exactly that mechanism from the panel layer.
- **Option B — delete `series/` entirely; recompute in-memory every time**, sourced
  from `residual_params.parquet`'s persisted loadings (one matrix multiply per
  candidate, per §10's measurement). Structurally consistent with "no persisted
  panel artifact, no panel cache key, no cross-run panel reuse," and it is a live
  option: only one write path (`series.py`) and one read path
  (`candidate_signals.py:539-542,576`) touch it, per the grep above — no notebook or
  test asserts on its persisted content, only on the stem-clearing behavior around
  it.

Reported per instructions; Option B is the one structurally consistent with the rest
of the track's stated decision, but the brief says decide here, not this session.

### 10. Reconstruction cost — measured against real data

Measured on the shipped code path (`UniverseDataLoader.load`,
`build_group_return_bundle`, `apply_causal_residual_model`), not a static estimate:

- Real 10-group run `artifacts/simulation_runs/20260713_2017_e5ffbe83/`
  (`meta.json`: `n_closed_trades=3791`, `n_trading_days=5920`, ~23 years, 10 sectors):
  - `daily_portfolio_state.parquet`: max concurrent open positions **209**, mean
    **~124** across 5920 trading days.
  - `closed_trades/*.parquet` (3791 rows): **3721** unique `(spread_id, entry_date)`
    pairs — i.e. essentially one distinct spread-level series per trade, negligible
    reuse. This is the count of series a full **position-view** reconstruction over
    the entire run's history would need to build.
  - `selected_panel.parquet` (the candidate rows this run actually consumed, already
    filtered to `is_valid=True`): **2,041,607** rows, `spread_id × asof_date` all
    unique, across **1186** outer refit dates and **10** groups. This is the
    **candidate-view** upper bound if every scored candidate's level were rebuilt,
    versus the much smaller 3721 for position-view (only candidates that were
    actually opened).
- Timing, `consumer_discretionary` group (5314-row × 27-column aligned returns,
  warm data cache):
  - `UniverseDataLoader.load()` (cache hit): **0.15 s**
  - `build_group_return_bundle`: **0.007 s**
  - `apply_causal_residual_model` (one candidate's full-history residual
    reconstruction, `subtract_risk_free=True`): **1.7 ms/call**, averaged over 200
    calls against real persisted `FittedCausalResidualModel` objects from
    `artifacts/candidate_panels/full_v1/dsc_pairs_pca_W-FRI_exp_hl504_20260713_1728_residual_params.pkl`.

Extrapolating to the position-view reconstruction of the full 23-year, 10-group run:
10 groups × ~0.15 s data load ≈ 1.5 s, plus 3721 × 1.7 ms ≈ 6.3 s of residual
reconstruction, plus a cumsum per candidate (negligible, single-column numpy op) —
**on the order of 10 seconds total**, dominated by data loading, not compute. This
does not include the day-by-day equity/exposure aggregation itself (vectorizable
pandas over ≤209 concurrent positions × 5920 days ≈ 734k position-days), which is a
sub-second operation at this scale.

Candidate-view reconstruction of the **full scored set** (2,041,607 rows) at the same
per-call rate would be ~2.04M × 1.7 ms ≈ **~58 minutes** if done one candidate at a
time — this is the case to avoid; `daily_state_df(run_dir)`'s stated job
(mark-to-market of **open positions**) only needs the position-view count (3721,
~10 s), not the candidate-view count. Worth stating explicitly in the design so a
future caller doesn't reach for full candidate-view reconstruction by accident.

Caveat: this is one group's timing; `subtract_risk_free=False` groups or a group with
more tickers would differ somewhat, but the order of magnitude (single-digit
milliseconds per candidate) should hold — residual reconstruction is one dense
matrix multiply against a fit that's already been done.

### 11. Debug sample sizing

No `DebugSampleConfig` or equivalent exists yet — this is a new artifact this track
introduces, so there's no way to measure it directly; sizing depends on a config knob
(sampled pair count) the brief doesn't specify a default for. Bounding from measured
data instead:

- **Baseline harness** (`docs/refactor/B_baseline.txt`: 2 groups, `max_steps=40`
  W-FRI outer dates ≈ 40 weeks ≈ 193 trading days per group): 1698 + 2869 = 4567
  valid candidates total. If the debug sample's pair-count knob picks, say, 10-50
  pairs (order of magnitude implied by "a pair count" being a lightweight config, not
  a fraction of the panel), and each contributes up to ~193 rows (one per trading day
  it could be live, worst case = its own group's whole date range), that's
  **~2,000-10,000 rows** for the harness config — far under the brief's "a few
  million rows at most" ceiling.
- **Full run extrapolation**, using the real run above (5920 trading days, 10
  groups): the same 10-50 sampled pairs, this time each potentially live across the
  full 5920-day range in the worst case, gives **~60,000-300,000 rows** — still
  comfortably inside the stated ceiling, even before accounting for the fact that a
  candidate is only live from its own `asof_date` forward (not the full range), which
  would shrink this further.

Both estimates depend on a pair-count parameter that doesn't exist in code yet, so
this is a bound, not a measurement — reported as such rather than as a single number.

---

## Part 3 — proposed commit order

Numbered for reference; not necessarily one commit per line where a step is small
enough to fold into its neighbor, but each row is one *logical* change.

| # | Commit | Could plausibly move `B_baseline.txt`? |
|---|---|---|
| 1 | Delete `SpreadMomentumConfig`, `KellyConfig`, and the now-orphaned `TimescaleRiskConfig`/`CrossTimescaleEntryConfig` read sites (`risk_manager.py:151`, `pair_spread_mean_reversion.py:275`, `sweep_runner.py:148,184`) plus the classes themselves. Independent of everything else in this track. | **No** — zero construction sites means zero live code path touches these today (Part 1 §6). |
| 2 | Resolve `start_after_nan`/`check_for_corruptions` disagreement between `DataConfig` and `PanelBatchConfig` (and note, don't necessarily touch, the third combination at `market_snapshot.py:277`). | **Structurally could**, if the resolved value changes which tickers pass corruption/NaN checks in some universe. Needs its own explicit check beyond the baseline harness's two universes, which Track A's `found.md` entry already flagged as gap-prone (`found.md`: residual fit weighting note). |
| 3 | `residual_params.parquet`: persist loadings in long form (`fit_date`, `ticker`, `factor`, `category` dtype) replacing the `.pkl`. Fit logic itself unchanged — pure storage-format swap. | **No**, if verified byte-for-byte against the existing `.pkl` content before switching readers over. |
| 4 | `weights.parquet`: float64 column, drop `weights_json`/`to_json` serializer, decide and implement the `n_legs` option from Part 2 §8. | **No** — the live `spread_return` computation (`pair_candidate_panel_creator.py:347-350`) already uses full-precision in-memory weights; the lossy `to_json` path was never read back into any computation (Part 1 §1). This is the one commit in the whole track with a *provable* result-neutrality argument, not just an expectation — worth calling out explicitly in the commit message and covering with a test that diffs trades before/after. |
| 5 | `candidates.parquet`: persist full scored set including invalid rows, both absence classes documented, drop `adf_pvalue`. | **No** — additive persistence surface, no change to what's computed or filtered. |
| 6 | `config.json`: add `snapshot_id`, universe name; remove `meta.universe_name` redundancy from group yamls. | **No** — additive/cosmetic on the config surface. |
| 7 | `trades.parquet`: add `entry_asof_date`. Concretely: thread `ref.asof_date` (already available at `simulator.py:728`, currently dropped) through `LiveCandidatePosition` (`types.py:129+`) and `ClosedCandidateTrade` (`types.py:173+`) via the construction site at `simulator.py:745` and `_build_closed_trade` (`simulator.py:1053`). | **No** — additive field, sourced from a value that already exists in memory at the point of use; nothing about position sizing, entry timing, or PnL changes. |
| 8a | Build `daily_state_df(run_dir)`-equivalent reconstruction (rename to avoid colliding with the doomed `SimulationResult.daily_state_df()`, Part 1 §2) producing the same columns `performance_basic.py` currently reads off `daily_portfolio_state_df()`. Fidelity-test against a debug sample. | **No**, in isolation — new code, nothing wired to it yet. |
| 8b | Repoint `generate_report`'s call at `simulator.py:625-629` from `result.daily_portfolio_state_df()` to the new reconstruction. Diff `performance_metrics.json` old vs. new on the baseline harness explicitly — **`B_baseline.txt`'s `## counts` section does not include performance metrics at all** (it's candidate-validity counts + the trade list only), so it structurally cannot catch a regression here. This step needs its own before/after check that Part 3 doesn't get for free from the standing acceptance test. | **Structurally could, and `B_baseline.txt` would not show it.** Highest-risk-without-a-safety-net step in the track. |
| 8c | Delete `DailyStateLogEntry`, `DailyPortfolioStateLogEntry`, their accumulation loops (`simulator.py:291-292,506-525,600-601` etc.), and the old `daily_states`/`daily_portfolio_state` artifact writers. | **No**, once 8a/8b are verified — pure removal of now-dead code. |
| 9 | Outer/inner loop migration from panel creation into the simulator; residual fit takes input returns as an argument (per "Also in scope"). Add the explicit `≤ t` invariant test in both the simulator and the reconstruction path (Part 1 §5 confirmed no such test exists today). | **Yes — the one commit that structurally can, and is expected to be exercised by, `B_baseline.txt`.** This is the track's actual computation-path change (relocation of *where* scoring happens, not *what* is computed); an off-by-one here is exactly the "plausible-looking better result" failure mode the brief calls out. `B_baseline.txt` must not move a single row after this commit. |
| 10 | Resolve `series/` (Part 2 §9) — delete or re-key, per whatever the implementation session decides. Natural to sequence after #9 since the runtime-recompute path needs to exist first if the resolution is deletion. | **No**, either way — a cache invalidation/removal, not a computation change, provided reconstruction reproduces what the cache used to serve. |
| 11 | `debug_sample.parquet` + zero-tolerance fidelity test (candidate view and position view), depends on #3, #4, #7, #9 all being in place. | **No** in itself (new test infrastructure), but this is the commit that would **catch** a regression from #9 if #9's own baseline check somehow missed it — sequence it as close after #9 as possible, not at the very end, so it isn't just decoration on an already-trusted change. |
| 12 | Re-execute committed notebooks (`01_create_candidate_panel.ipynb`, `02_run_simulation.ipynb`, `03_full_simulation_pipeline.ipynb`) against the new API surface; fix any notebook-only caller this uncovers (per the standing `found.md` process note — Track C found one this way). | N/A — verification pass, not a code commit, though it may produce small follow-up commits if a notebook-only caller breaks. |

General note on the neutrality check per commit: for every row above, "no" means
"no live computation this session found reads the changed artifact/field for
anything other than what it's already used for" — it is not a substitute for
actually running the full test suite plus the baseline harness after each commit,
which the standing rules already require.

---

## Anything the brief asks for that cannot be done as described

Nothing found. Every numbered brief requirement checked against current code either
confirmed cleanly (Part 1) or is explicitly deferred to the implementation session as
an open option (Part 2). The one item worth flagging as "not decided by the brief at
all" rather than "cannot be done" is the `z_spectra_*.parquet` artifact (Part 1 §3) —
the brief's Layout and scope sections are silent on it, so the implementation session
will need to make a call the spec review didn't anticipate.
