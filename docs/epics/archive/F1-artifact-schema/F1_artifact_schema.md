# Track F1 — Artifact layer: schema

**Changes results:** no. Every commit here is a storage-format change or an additive
field.

**Depends on:** C, D, E (all merged).

First half of the dissolved Track F. F1 replaces the artifact surface; **F2** moves
the loop into the simulator and builds the fidelity test. The seam is deliberate:
everything in F1 has a result-neutrality argument that `B_baseline.txt` can check,
and the one commit that structurally changes a computation path lives in F2.

Decisions carried in from `F_spec.md` and from review are marked **[decided]**.

## Context: the dissolution

The independent candidate panel layer is dissolved. The simulator computes beta,
spread levels, MR diagnostics, and candidate selection at runtime. No persisted
panel artifact, no panel cache key, no cross-run panel reuse.

The panel's key surface was the entire upstream pipeline — residual config, hedge
config, diagnostic config, ticker set, price content, date range. Every key
component not encoded is a silent wrong result. That fragility bought avoidance of
recomputation that Track A showed to be roughly three-quarters dead weight.

**Cross-run artifact reuse is rejected**, and that principle decides several items
below.

## Layout

```
simrun_dir/
  config.json
  candidates.parquet
  weights.parquet
  residual_params.parquet
  trades.parquet
  debug_sample.parquet      # F2, only when debug sampling enabled
  series/                   # optional, see below
  performance/
```

Keys are **columns, not directories**. The hashed directory tree was residue from
the abandoned panel cache; with no cache identity to encode it solves nothing.

Conventions: flat RangeIndex, no MultiIndex; key columns on every artifact so any
concatenated frame is self-describing; `category` dtype for ids, tickers, and keys;
dates as `datetime64[ns]`; uniqueness asserted on write.

Until Track H defines `zscore_key`, the key columns are `residual_key` and
`hedge_key` only.

## Commits

Ordered per `F_spec.md` Part 3, minus step 2 (moved to Track I) and minus steps
9–12 (F2).

### 1. Delete dead config classes and their orphaned read sites

`SpreadMomentumConfig`, `KellyConfig`, `TimescaleRiskConfig`,
`CrossTimescaleEntryConfig`, plus the read sites at `risk_manager.py:151`,
`pair_spread_mean_reversion.py:275`, `sweep_runner.py:148,184`.

Zero construction sites, re-verified including notebooks. Both `KellyConfig` and
`CrossTimescaleEntryConfig` encode approaches the research record lists as failed or
neutral-to-harmful, so they are surface that looks supported and is not.

**Any grep supporting a deletion must include `notebooks/**/*.ipynb`.** Track C
found a live notebook caller for a function two prior sessions had recorded as dead.

### 2. Delete z_spectra

**[decided]** `z_spectra_*.parquet` and its capture module. Residue from an entry-
selection experiment that returned null, ported from `hierarchical-arb` unnoticed.

Before deleting, export one `z_spectra` sample plus a short result note to
`docs/research/` — the analysis may be written up later, and a post needs the data,
not the code path. Then remove the module.

### 3. `residual_params.parquet`

Long form, one row per factor loading per ticker per **`fit_date`**, with `factor`
as a category. Replaces the `.pkl`. Fit logic unchanged — pure storage-format swap.

No `hedge_key`: this is upstream of beta, and including it would duplicate the table
across the `hl` sweep.

**Parameters, not series.** A single loading fit applies across the whole window in
one matrix multiply, so every fit date produces a full revised residual history per
ticker. The series grain is `(ticker, fit_date, date)`; persisting it is four orders
of magnitude larger than persisting the loadings.

The fit grid is **daily**, not outer-date. Size accordingly.

Verify byte-for-byte against the existing `.pkl` content before switching readers
over. That verification is the result-neutrality argument for this commit.

### 4. `weights.parquet` and `n_legs`

Grain is one row per leg: `(group_id, residual_key, hedge_key, spread_id,
asof_date, ticker) -> weight float64`.

The live path writes weights via `to_json(double_precision=12)` into a column named
`weights`, which does not round-trip float64 — so the persisted beta is not the beta
the run used, and F2's zero-tolerance fidelity test cannot pass against it. Drop the
JSON serializer from the live path. `weights_json` belongs to a dead code path.

`intercept` stays on candidates: a per-spread scalar, not per-leg. Track A already
persists it.

**[decided] Store ALL legs.** `spread_return` uses both legs' full weights
regardless of `tiny_weight_threshold`, so storing only above-threshold legs would
provably break the fidelity test. In the portfolio path a pruned weight is genuinely
`0.0` and costs one row; in the pair path every weight is live.

**[decided] `n_legs` stays a real column**, documented as the count of non-zero
weights. It is not a groupby size and must not be derived as one.

**[decided] The pair-spread path does not apply `tiny_weight_threshold`.** The
threshold exists to thin large optimizer-produced portfolio spreads, where
sub-threshold weights are set to zero and `n_legs` is a genuine count. Pair spreads
have no such problem: two legs, both from the hedge fit, and a tiny beta is a real
estimate rather than noise to prune. `n_legs` is therefore always 2 in the pair path,
even at a weight of 1e-32.

This restores what the hardcoded value already said before Track A wired it to the
diagnostics function. The hardcode was accidentally right; the computed value was
the wrong one, because the same name carried two meanings.

**Measure before implementing.** The `too_few_active_legs` gate currently rejects
pair candidates using this threshold. Removing the threshold from the pair path
removes the gate's ability to fire there. Count how often it fires across the
baseline run:

- Zero → the gate is dead in the pair path and the change is result-neutral.
- Non-zero → separate result-changing commit, before/after on `B_baseline.txt`,
  and report the count. A pair with beta near zero being tradeable is then a
  question worth asking on its own.

Note this is the one commit in the track with a *provable* neutrality argument
rather than an expectation — the lossy JSON was never read back into any
computation. Say so in the commit message and cover it with a test diffing trades
before and after.

### 5. `candidates.parquet`

One row per pair per outer refit date. Key columns, `spread_id`, `asof_date`, the
OU/MR metrics, `intercept`, `is_valid`, `why_invalid`. Full scored set including
invalid rows. `adf_pvalue` dropped, since Track A stopped computing it.

The NaN pattern is structured: a pair rejected at an early gate never reaches the OU
fit, so downstream metrics are all NaN. `why_invalid` is authoritative — do not
infer the reason from which column is NaN. Only the first failure is recorded, so
conditioning on a gate means conditioning on passing all earlier ones.

**Two distinct absence classes**, and the table can only carry one. Rows marked
`is_valid=False` carry a reason. Pairs skipped by a structural pre-check — a leg
missing from the cleaned window, or a weight-computation exception — produce **no row
at all**. Track B logs pre-check skips separately from ticker drops. Any breadth,
coverage or rejection-rate denominator needs both, and this table alone cannot
supply the second. Document that on the artifact.

### 6. `config.json`

Add `snapshot_id` and the universe name — Track C landed the snapshot layer, so a
run binds to one frozen snapshot by id, and the universe name is now the directory
name under `config/universes/`.

Remove the redundant `meta.universe_name` from the group yamls. The directory name
is authoritative; the field inside each yaml is a leftover from when each yaml was
its own universe.

Also record `fit_input_tickers` per `(group_id, residual_key)`: the **union** across
fit dates of the **realized** input set, not the requested one. It is what
reconstruction must load. Per-date composition varies with dropna and belongs to
Track B's log. The load-time assert catches "different universe"; the snapshot hash
covers "same universe, different content".

### 7. `trades.parquet`: `entry_asof_date`

Thread `ref.asof_date` — already available at `simulator.py:728` and currently
dropped — through `LiveCandidatePosition` and `ClosedCandidateTrade`.

**Load-bearing.** Weights are frozen at entry for the life of a position, so
position-view reconstruction is impossible without it. It also makes candidate age
at entry computable, which is the diagnostic for the adverse-selection question: an
old active candidate is one that recent refits repeatedly rejected.

`entry_refit_date` was verified as NOT FOUND anywhere, so this field is genuinely
new.

### 8. Replace `daily_state` — three steps, not one

**[decided] The brief's original claim was wrong.**
`DailyPortfolioStateLogEntry` is **not** write-only: `generate_report` computes
Sharpe, drawdown and the rest off it at `simulator.py:625-629`. Deleting it requires
standing up the replacement first.

**8a.** Build the reconstruction function producing the columns
`performance_basic.py` currently reads. Rename to avoid colliding with the doomed
`SimulationResult.daily_state_df()`. New code, nothing wired to it — neutral in
isolation.

Measured cost on a real 23-year, 10-group run: 209 max concurrent positions, 3721
unique position-view series, ~1.7 ms per residual reconstruction, so roughly ten
seconds. That is what makes the file droppable.

**8b.** Repoint `generate_report`. **This is the highest-risk step in the track and
`B_baseline.txt` structurally cannot catch a regression here** — the baseline
carries candidate counts and the trade list, not performance metrics.

**Required:** diff `performance_metrics.json` old versus new on the baseline
harness, exactly, as a one-off check for this commit. Not a standing criterion; a
gate for this step.

**8c.** Delete `DailyStateLogEntry`, `DailyPortfolioStateLogEntry`, their
accumulation loops, and the old writers. Pure removal once 8a and 8b verify.

**No aggregate total rows anywhere.** A total is a groupby, and storing a derived
aggregate invites disagreement with its components. For the helper: summing is right
for additive quantities only — Sharpe and drawdown come from the summed equity path,
never averaged across sub-portfolios.

## `rejections.parquet` — deferred, confirmed as the real gap

The genuinely non-reconstructible thing is which proposed opens the risk manager
turned down, at which cap. A rejected trade leaves no trace in `trades.parquet`, and
Track D confirmed positions-opened/rejected counts exist nowhere.

But it only earns its place once the caps bind, and once concurrent positions
saturate, every proposed open is rejected every day — potentially tens of millions
of rows. Add it when a run shows rejections.

## Out of scope

- Outer/inner loop migration, the `≤ t` invariant test, `series/`, the debug sample
  and fidelity test. All F2.
- `start_after_nan` / `check_for_corruptions`. **Track I**, which runs before G.
- `zscore_key` and occupancy. Track H.
- Any change to a computed value, except as measured and reported under commit 4.
- The `run_me.py` config-sourcing decision and whether `demo_materials.yaml` should
  expose the ExecutionConfig cost fields. Deferred until the field set stops moving.

## Acceptance

- `docs/refactor/B_baseline.txt` does not move. The `## counts` section is the
  binding check, run after every commit.
- Commit 8b additionally requires the `performance_metrics.json` before/after diff.
- Commit 4 additionally requires the `too_few_active_legs` count, and a separate
  result-changing commit if it is non-zero.
- Full suite green.
- Uniqueness asserted on every artifact write, with a test per artifact.
- Notebooks re-executed; fix any notebook-only caller this uncovers.
