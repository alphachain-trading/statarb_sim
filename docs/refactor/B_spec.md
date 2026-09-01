# Track B — spec (delta against the brief)

Read-only verification pass over `docs/refactor/B_window_admission.md`. No code
changed, no branch created. `statarb_sim_refactor_handover.md` was not consulted.

## Verify first — results

All five claims **CONFIRMED**, no `NOT FOUND`.

**1. CONFIRMED.** `src/candidates/pair_candidate_panel_creator.py:239-240`:
```python
    rr_hedge_clean = rr_hedge.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").dropna(axis=1, how="any")
    rr_diag_clean = rr_diag.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").dropna(axis=1, how="any")
```
`pair_ids` is then generated only from `rr_hedge_clean.columns` (`:249-252`), so a
ticker dropped by the shared `how="any"` dropna never enters the pair loop for that
date — no row, for either the invalid-with-reason class or a bare skip. Matches
Track A's Q12 finding independently (`A_spec.md` §12).

**2. CONFIRMED.** Both `hedge_window` and `diag_window` are built by
`_slice_candidate_window` (`:220-221`), which takes a trailing `.loc[:asof].tail(lookback)`
slice (`:114-117`).

**3. CONFIRMED.** `_slice_candidate_window` only truncates `if lookback is not None`
(`:116-117`). `hedge_ratio_lb`/`mr_diag_lb` fall back to `residual_cfg.lookback` when
omitted (`create_pair_candidate_panel:568-571`, `create_pair_candidate_panel_for_date:458-461`).
`CausalResidualConfig.lookback` (`src/residuals/causal_residuals.py:113-117`):
```python
    def lookback(self) -> int | None:
        return self.lb if self.mode == ResidualMode.EQ_ROLLING else None
```
returns `None` for `EQ_EXPANDING` and `DECAY_EXPANDING`, so a caller that omits both
windows on an expanding-mode config gets full untruncated history back to the start
of `bundle.aligned_returns`.

**4. CONFIRMED.** `_slice_candidate_window:119-120`:
```python
    if aligned.empty:
        raise ValueError(f"No aligned returns available up to {asof_datetime}.")
```
No comparison of `len(aligned)` against the requested `lookback`. A lookback of 252
requested five days into history returns a 5-row frame silently.

**5. CONFIRMED.** `ols_beta_no_intercept` (`src/residuals/spreads.py:78-94`) guards only
`denom <= min_denominator` (`:90-91`, near-zero `sum(x*x)`). `_pca_spread_weights`
(`pair_candidate_panel_creator.py:125-159`) guards only `abs(pc2[0]) < 1e-12`
(`:147-151`, degenerate PC2 eigenvector component). Neither reads `len()` of its input
anywhere. `min_obs` does not exist in the repo today (repo-wide grep, zero hits).

## 6. Shared clean matrix — call chain and per-pair replacement scope

Call chain, one `asof_datetime`, inside `_build_pair_candidate_rows_for_date`
(`pair_candidate_panel_creator.py:195-331`):

1. `_slice_candidate_window` (`:220-221`) — trailing slice of
   `bundle.aligned_returns`, all member tickers together. Not itself the defect;
   this is the source data every ticker's residual is fit from.
2. `apply_causal_residual_model` (`causal_residuals.py:399-469`) — produces
   `rr_hedge` / `rr_diag`, one residual column per member. Also not the defect — one
   fitted model applied across all tickers is correct; only the *cleaning* step after
   this is shared incorrectly.
3. **The shared clean matrix** — `:239-240` (quoted under claim 1). One
   `how="any"` dropna over every column, applied once per window, before any pair is
   considered. This is what per-pair construction replaces.
4. `hedge_cols = set(rr_hedge_clean.columns)` (`:242`) — used at `:260-261` to decide
   whether a pair's legs survived. Assumes the shared column set is the right
   membership test for *every* pair.
5. `diag_numpy = rr_diag_clean.to_numpy(...)`, `diag_col_index` (`:243-244`) — one
   shared 2-D array with a single row axis (dates), sliced by column position at
   `:278` for each pair. Assumes every pair shares the same retained date set.
6. `pair_ids = generate_spread_ids(tickers=list(rr_hedge_clean.columns), n_legs=2)`
   (`:249-252`) — **the candidate universe itself is drawn from the shared-clean
   column set**, not from the full ticker set available in `rr_hedge`. This is the
   widest-reaching assumption: a ticker dropped by any other ticker's gap is not just
   mis-fitted, it never generates a pair id.
7. **The beta fit** — `_compute_pair_weights(residual_returns=rr_hedge_clean, left, right, method)`
   (`:266-271`), reaching `ols_beta_no_intercept(y=rr_hedge_clean[left], x=rr_hedge_clean[right])`
   (`:182-186`) or `_pca_spread_weights(rr_hedge_clean, left, right)` (`:187-188`,
   itself indexing `residual_returns[[left, right]]` at `:138`). Both read two columns
   out of the one shared clean frame — by the time execution reaches here, the row set
   is already wrong for any pair whose legs didn't need a third ticker's dates dropped.
8. Diagnostics spread construction (`:278`) reads `diag_numpy[:, diag_col_index[...]]`
   — same shared-row-axis assumption as (5), on the diagnostics side.

**What per-pair construction has to replace:** steps 3, 4, 6, and 8 (5 and 7 fall out
once 3/4/6 change — there is no longer one shared frame to slice from). Concretely:
for each `(left, right)` pair, build `rr_hedge[[left, right]]` and `rr_diag[[left, right]]`
independently, `dropna(how="any")` on each 2-column frame (dropping only dates where
`left` or `right` — of *that* pair — is missing), fit weights on the pair's own
retained rows, and build the diagnostics spread from the pair's own retained diagnostics
rows. The candidate universe (step 6) must enumerate pairs from the full member set
present in `rr_hedge` (post ticker-subset filter, pre shared dropna), not from a
post-dropna column list — every two-ticker combination is now attempted, and only
fails per-pair (too few retained rows, or a raise from the weight fit).

The structural pre-check at `:260-264` (`left not in hedge_cols` / `diag_col_index`)
has no shared set to consult once (3)/(4)/(6) change; it is replaced by whatever the
per-pair `min_obs` check becomes (item 3 of the brief's scope), not by an equivalent
membership test.

**Correction carried from Track A (`A_spec.md`, "Consequences for other tracks"):**
the coverage hole is wider than this brief states. `why_invalid` covers diagnostic
gate failures only. The structural skip at `:260-264` (and the weight-fit exception at
`:266-274`, e.g. a degenerate PCA eigenvector) produce **no row and no reason** — a
distinct absence class from an invalid row. The brief's "log dropped observations per
pair per date" item must log **pre-check skips separately from ticker drops**; a
single log stream that conflates "this pair's legs weren't both present" with "this
pair was attempted and its weight fit raised" reproduces the same coverage-hole problem
the logging is meant to fix. This is a correction to the brief's "Also in scope"
section, not new scope — the logging item already exists there.

## 7. Hedge fit cost as a fraction of the panel build — measured baseline

Measured through `run_panel_batch`, materials sector, the same config Track A used for
its timing (`A_spec.md`, "Implementation verification"): `DECAY_EXPANDING hl=504`,
`min_lb_type_dec_exp=MULTIPLIER`, `min_lb_dec_exp=2` (→ `mh1008`), `subtract_risk_free=True`,
default `PairSpreadConfig` (PCA hedge ratio, `skip_adf=True`), `hedge_ratio_lb=mr_diag_lb=252`,
`frequency="W-FRI"`, `max_steps=400` → 31,200 rows, `persist_result=False`.

Method: wall-clock (`time.perf_counter`), no code edited — `_compute_pair_weights`
(the hedge-ratio fit entry point, `:162-192`) wrapped by a timing shim at the module
level from an external script, matching the "measure through the shipped code path"
rule (the shim observes, it does not alter, control flow or arguments). Two runs:

| run | total panel build | `_compute_pair_weights` wall time | fraction |
|---|---|---|---|
| 1 | 26.4s | 13.4s (31,200 calls) | 50.7% |
| 2 | 25.9s | 13.2s (31,200 calls) | 50.8% |

Total build time matches Track A's own re-timed baseline for this exact config
(`skip_adf=True`: **25.4s**, `A_spec.md` §"Implementation verification") within normal
run-to-run variance. The brief's claim — hedge fit "roughly half the post-ADF
build" — is **CONFIRMED** at ~51%, not just directionally.

This is the number to diff against once pairwise dropna lands: per-pair slicing turns
one shared `dropna` + one shared beta fit into `n_pairs` slices and fits per date.
Since the fit itself is ~51% of the (unprofiled) build and dominated by
`_pca_spread_weights`'s `np.cov`/`np.linalg.eigh` per pair (not by the dropna itself,
which cProfile shows costing far less than the fit — see per-call breakdown below),
the budget headroom is in the fit call count staying at `n_pairs` per date, same as
today; only the slicing step changes shape from one shared `dropna` to `n_pairs` small
ones.

cProfile corroboration (profiled run, `skip_adf=True`, same config, 49.1s wall under
the profiler): `_compute_pair_weights` cumtime 25.98s of 47.85s inside
`_build_pair_candidate_rows_for_date` (54%, consistent with the unprofiled 51% given
profiler overhead scales with call count and `_compute_pair_weights`/`_pca_spread_weights`
account for a large share of total calls).

## 8. Exponential weights — residual fit only, computed from row position

Exponential (half-life) weights exist in exactly one place: the residual fit.
`_make_sqrt_w` (`causal_residuals.py:277-286`), called from `fit_causal_residual_model`
(`:331`) and used in `_wls_multi` (`:289-297`, called at `:340` and `:356`):
```python
    age = np.arange(n, dtype=float)[::-1]
    w = 2.0 ** (-age / half_life)
```
`n = len(aligned)` (`:330`), and `aligned` is the fit window after `_slice_fit_window`
(`:249-274`) — a single trailing or expanding slice over **all** members at once, so
`age` is **row position within that shared window**, not date distance. Under a
trailing/expanding window with no internal gaps this is equivalent to date distance;
it stops being equivalent the moment a row is missing from that window for some
subset of tickers, which today can't happen (the fit window isn't dropna'd per
ticker) but is exactly the situation pairwise dropna in the *hedge fit* creates.

The hedge fit itself — `_compute_pair_weights` / `ols_beta_no_intercept` /
`_pca_spread_weights` — applies **no weighting at all**. Confirmed by reading every
line of `_compute_pair_weights` (`:162-192`): no `sqrt_w`, no half-life parameter in
its signature or any callee's. This matches notebook 01's own description ("Fit hedge
ratios — OLS or PCA on the fitted residuals... Unweighted").

So: exponential weights exist today, only in the residual fit, computed from row
position. The brief's item "weight by date distance, not post-dropna row position" is
about a weighting scheme that **does not exist yet** in the hedge fit — it is a
requirement for when G adds weighting there, made mandatory now (per the brief) because
pairwise dropna is what first makes per-pair gaps possible in an unweighted-today fit.
There is no existing by-position hedge-fit weighting to migrate away from; `_make_sqrt_w`
is the only precedent, and it is row-position-based, which is the wrong model to copy
forward once dropna is per-pair.

## 9. Minimal baseline harness — reusable pieces

No existing fixed small-universe test config or metrics-summary-to-text helper. What
exists and is directly reusable:

- **Small-universe runner, already the pattern used in tests.**
  `tests/test_panel_batch_windows.py:43-57` and `:84-98` call
  `run_panel_batch(PanelBatchConfig(residual_configs=[...], selected_sectors=["materials"],
  max_steps=3..5, persist_result=False, persist_residual_params=False))` and complete
  well under a second per test. `max_steps` is already the speed-cap knob (notebook 01
  uses `max_steps=8` for the same reason). No new runner code is needed for the panel
  side of the harness — call `run_panel_batch` the same way, with `max_steps` sized for
  "tens of seconds" instead of unit-test-instant, and `selected_sectors` given two
  entries instead of one (the brief asks for "two groups"; `energy` (11 equities,
  `config/universe/universe.energy_only.v1.yaml`) and `materials` (13 equities) are
  the two smallest committed universes by equity count — repo-wide
  `grep -c "kind: equity"` over `config/universe/*.yaml` — keeping the per-date pair
  count, and so the run time, small).
- **End-to-end stage pattern.** `run_me.py` (`stage_residuals`, `stage_simulate`) plus
  `config/demo_materials.yaml` is the existing single-sector, full-history, panel→simulate
  pipeline — not reusable as-is (single sector, no date cap, not "tens of seconds"), but
  its `_build_sim_config` (`:380-463`) is the reference for what a minimal
  `SimulatorConfig` needs; a harness config is a shrunk copy of this shape (two
  sectors, `run.start_date`/`end_date` or panel `max_steps` bounding the range),
  not new plumbing.
- **Trades.** `SimulationResult` from `run_from_config` (`simulator_factory.py`)
  exposes closed trades; `simulation_persistence.load_closed_trades_df` /
  `save_simulation_run` (`simulation_persistence.py:194`, `:51-125`) is the existing
  serialization path — the harness can call `closed_trades_df()` directly rather than
  round-tripping through disk, if only an in-process text diff is wanted.
- **Deterministic metrics summary.** `compute_performance`
  (`src/simulator/performance/performance_basic.py:29-`) returns
  `PerformanceResult.metrics`, a plain dict. `_METRICS_ORDER`
  (`src/simulator/performance/performance_report.py:20-65`) is already a fixed,
  ordered list of `(key, label, format_spec)` triples with a formatter,
  `_fmt_value` (`:107-119`) — together they are a ready-made deterministic
  metrics-to-text serializer; `_print_metrics_table` (`:122-149`) does the same
  iteration but to stdout via `print`. The harness's summary file is this same
  iteration writing to a file handle instead of stdout, reusing `_METRICS_ORDER`
  and `_fmt_value` directly rather than re-deriving a metric list. No
  `PerformanceConfig(report_html=True)` / quantstats dependency is needed for this —
  `metrics_table` output and `report_html` are independent flags (`generate_report`,
  `performance_report.py:86-98`).

Net: the harness is a thin script gluing three existing pieces (`run_panel_batch` with
`max_steps` + two sectors, `run_from_config` for the trades, `compute_performance` +
`_METRICS_ORDER` for the summary) — no new runner, no new metrics computation, no new
small-universe fixture beyond picking the two smallest already-committed sector YAMLs
and a `max_steps`/date-range cap.

## 10. Downstream dependence on low-`min_obs` pairs currently receiving a hedge ratio

**Nothing found that would break.** Two candidate failure points were checked
specifically because they are the places that would break first if a `(spread_id,
asof_date)` that used to have a row now has none:

- **Selection.** `_build_candidate_selection_mask` (`candidate_selector.py:243-284`)
  and `select_candidates` (`:287-`) operate row-wise on whatever rows the panel
  contains; `select_candidates` already has an explicit empty-panel path (`:294-309`).
  No assertion or code path anywhere in `candidate_selector.py`, `simulator.py`, or
  `simulator_factory.py` requires a minimum candidate count per date or per group
  (repo-wide grep for `len(selected)` / "no candidates" / "empty" near candidate
  handling: only a row-count log line, `simulator.py:257`, no raise).
- **Persistence.** `series.py` keys spread-level series by `(spread_id, asof_date)`
  (`series.py:11-16`, docstring) and iterates the panel's actual rows to write them
  (`:191-221`) — a `(spread_id, asof_date)` with no panel row simply gets no series
  file. This is already how a structurally-skipped or diagnostic-invalid pair is
  handled today (Track A §12's two absence classes); a `min_obs`-rejected pair is a
  third instance of a mechanism the persistence layer already tolerates, not a new
  shape.

**What does change, and is a result change, not a break:** `is_valid=True` pairs
fitted on very few observations (today possible down to `len(dx) >= 10`, i.e. as few
as ~11 raw observations reaching the diagnostics gates — see `_fast_pair_diagnostics`
`:385-386`) are currently selectable and tradeable. After `min_obs` lands, such a pair
produces no beta and is reported (per the brief's acceptance test), not silently
fitted — so it stops being selected and stops being traded. That is exactly item
4/5's intended effect, not a side effect to guard against.

## §8 (superseded handover) — recorded per the brief's note

The brief's "Note for the spec session" directs recording this here, not deciding it:
the superseded handover's assertion that rolling and decay-expanding modes must admit
the same `(spread_id, asof_date)` set is unachievable — rolling requires observations
inside a trailing window, decay-expanding requires them anywhere in history, so
decay-expanding always admits a strict superset. G's rolling-vs-decay-expanding
comparison must run on the **intersection** of the two admitted sets, with the
symmetric-difference size reported as a diagnostic. No code exists yet to verify this
against (G is unimplemented); this is carried forward as-is from the brief.

## Corrected claims

None of the five "verify first" claims required correction — all five confirmed
as stated. The one correction is process-level, carried in from Track A (§6 above):
the brief's logging item needs pre-check skips and ticker drops logged as two
distinct things, not one stream, or it inherits Track A's coverage-hole finding
instead of fixing it.

## What the brief asks for that cannot be done as described

Nothing. All three scope commits (harness, pairwise dropna, per-pair `min_obs`) and
both "also in scope" logging items are buildable against the code as it exists; the
only refinement is the pre-check-skip vs. ticker-drop split in §6/Corrected-claims
above, which narrows the logging item's shape rather than blocking it.

## Proposed commit order

1. **Baseline harness.** Two smallest committed sector universes (`materials`,
   `energy`), `run_panel_batch(..., max_steps=<small>)` → `run_from_config` → trades +
   `compute_performance`/`_METRICS_ORDER` written to a committed text file, current
   (tainted) numbers included. No behavior change.
2. **Pairwise dropna.** Replace the shared clean matrix (§6, steps 3/4/6/8) with
   per-pair construction; candidate universe enumerated from the full ticker set, not
   the shared-clean columns. Structural pre-check skips and ticker drops logged
   separately (§6 correction). tqdm on the per-pair loop. Measure and diff the harness
   file: report how many pairs pairwise dropna admits that the shared-matrix version
   rejected.
3. **Per-pair `min_obs`.** Mandatory, no default, absolute, counted on the pair's own
   retained observations post pairwise-dropna; applies to the hedge fit per item 4/5.
   Date-distance weighting for the hedge fit (§8: the mandatory-now item, since it has
   no existing implementation to conflict with) lands in this commit, since it only
   becomes meaningful once gaps are per-pair. Log count of pairs dropped by `min_obs`
   per date. Measure and diff the harness file: report how many rows this removes.

Each commit measured against the harness file from commit 1, per the brief's
acceptance section.
