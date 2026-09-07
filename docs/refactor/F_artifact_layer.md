# Track F — Artifact layer (panel dissolution)

**Changes results:** no. It changes where things live and what is stored, not what
is computed.

**Depends on:** E. Weakly on C — the run config should record a `snapshot_id`, and
building C afterwards means touching one file twice. That is cheap, so treat it as
a preference, not a constraint. If C stalls, proceed.

This is the largest track. It stays large; the fidelity test and the weights schema
genuinely cannot be separated.

## Decision being implemented

**The independent candidate panel layer is dissolved.** The simulator computes
beta, spread levels, MR diagnostics, and candidate selection at runtime. No
persisted panel artifact, no panel cache key, no cross-run panel reuse.

The reasoning: the panel's key surface was the entire upstream pipeline — residual
config, hedge config, diagnostic config, ticker set, price content, date range.
Every key component not encoded is a silent wrong result. That fragility was being
paid to avoid recomputing work that turned out to be roughly three-quarters dead
weight, which A removes.

Cross-run artifact reuse is explicitly rejected: it reintroduces the durable
content-addressed key tree the dissolution deletes.

## Layout

```
simrun_dir/
  config.json
  candidates.parquet
  weights.parquet
  residual_params.parquet
  trades.parquet
  debug_sample.parquet      # only when debug sampling enabled
  performance/
```

Keys are **columns, not directories**. The hashed directory tree was residue from
the abandoned panel cache; with no cache identity to encode it solves nothing. Hive
partitioning within a dataset directory remains available as a storage
optimization, but the logical interface stays flat.

Conventions: flat RangeIndex, no MultiIndex; key columns on every artifact so any
concatenated frame is self-describing; `category` dtype for ids, tickers, and keys;
dates as `datetime64[ns]`; uniqueness asserted on write.

## `weights.parquet` — separate file, and why

Grain is one row per leg: `(group_id, residual_key, hedge_key, spread_id,
asof_date, ticker) -> weight float64`. Exactly `n_legs ×` the candidates row count,
so 2× today.

**This is not a stylistic choice.** The live serialization path writes weights via
pandas `to_json` with a precision cap, which does not round-trip float64 — the
persisted beta is therefore not the beta the run used. The zero-tolerance fidelity
test below cannot pass against it. A typed float64 column is the fix; raising the
JSON precision is still lossy.

Consequences:
- `weights_json` disappears from candidates. Verify which column name is actually
  live — the handover named a column that belongs to a dead code path.
- Remove the JSON serializer from the live path.
- `n_legs` becomes a groupby size and should be dropped, **unless** a candidates
  row can exist with no weight rows. Check whether rows failing the early gates
  still emit weights. If they do not, keep `n_legs` and record why.
- `intercept` stays on candidates. It is a per-spread scalar, not per-leg.

Rejected alternatives: `weight_left`/`weight_right` hard-wires to pairs trading;
indexed `weight_0..n` columns push the same restriction to a fixed leg count and
add sparse NaNs and schema churn.

## `candidates.parquet`

One row per pair per outer refit date — the dissolved panel's content, now a run
output. Carries the key columns, `spread_id`, `asof_date`, the OU/MR metrics
(kappa, half-life, mr_score, residual_std, level_std, spread_return_std),
`intercept`, `is_valid`, and `why_invalid`.

**Full scored set persisted by default**, including invalid rows. `adf_pvalue`
excluded, since A stops computing it.

Note that the NaN pattern is structured: a pair rejected at an early gate never
reaches the OU fit, so the downstream metrics are all NaN. `why_invalid` is
authoritative for the reason — do not infer it from which column is NaN. Only the
first failure is recorded, so analysis conditioning on a gate is conditioning on
passing all earlier ones.

## `residual_params.parquet`

Long form, one row per factor loading per ticker per `fit_date`, with `factor` as a
category. No `hedge_key` — this is upstream of beta, and including it would
duplicate the table across the `hl` sweep.

**Parameters, not series.** Residual series are computed by applying a single
loading fit across the whole window in one matrix multiply, so every fit date
produces a full revised residual history per ticker. The series grain is
`(ticker, fit_date, date)`; persisting it is four orders of magnitude larger than
persisting the loadings. Long form also accommodates future additions without
schema change — an intercept is just another factor value.

Note the fit grid is **daily**, not outer-date. Size accordingly.

**PC removal:** reconstruction raises when `remove_residual_pcs > 0`, with a message
saying the path is unverified and not covered by the fidelity test. The key encodes
the PC *count*, not the vectors; re-fitting recovers them deterministically from
frozen data, but sign is defined only up to a factor of minus one and
near-degenerate eigenvalues rotate the basis. Both are harmless when the component
is only projected out, but a latent unverified path returning plausible numbers is
the silent-wrong-result class. Three lines, low stakes — the default is zero and
one is confirmed harmful.

## `trades.parquet`

Existing columns plus the key columns and `entry_asof_date`.

**`entry_asof_date` is load-bearing.** Weights are frozen at entry for the life of a
position, so position-view reconstruction is impossible without it. It also makes
candidate age at entry computable, which H will want.

## `daily_state` — dropped

Do not port it. There is no capital ledger in the simulator: total capital is read
once and never decremented, exposure is recomputed each day from live positions,
and sizing reads a fixed base notional rather than current capital. Nothing in the
old log is non-derivable, so by the created-vs-derived rule it should not exist.

Its `(date, group_id)` grain with singular active-candidate slots also encodes
one-position-per-group, which is wrong.

**Replace with a `daily_state_df(run_dir)` function** producing equity and exposure
curves on demand. These are used constantly in analysis, so they must be easy —
easy is a function, not a file. The function needs mark-to-market of open positions,
so it consumes the reconstruction path and inherits its fidelity test rather than
needing its own.

**Delete both log classes.** They are write-only: populated from live objects and
read back only by their own serializers. Apparent decision-logic hits when grepping
their field names resolve to same-named fields on different classes — a naming
collision, not a dependency. Verify this before deleting.

**No aggregate total rows.** A total is a groupby. Storing a derived aggregate
invites disagreement with its components. Caveat for the helper: summing is right
for additive quantities only — Sharpe and drawdown must come from the summed equity
path, never averaged across sub-portfolios.

## `rejections.parquet` — deferred

The genuinely non-reconstructible thing is which proposed opens the risk manager
turned down, at which cap. A rejected trade leaves no trace in `trades.parquet`.

**But it only earns its place once the caps actually bind**, and once concurrent
positions saturate, every proposed open is rejected every day — potentially tens of
millions of rows. Add it when a run shows rejections. The schema is three lines
when needed.

## Spread level and reconstruction

Every refit date delivers new residuals and new hedge ratios, cumsummed into a new
spread level series over the whole history. So each `spread_id` has one level series
per refit date, and there is **no splicing** — an earlier concern about refit
discontinuities in a spliced cumulative series does not apply.

Anchoring the cumsum at zero is a **gauge freedom**, not a choice: the level enters
via a z-score against an EWM mean, so a constant shifts both and z is unchanged.

Time-decaying spread returns before cumsumming was considered and **rejected**. It
is not a re-anchoring — it imposes an OU pull toward zero before an OU model is
fitted, biasing kappa toward the imposed parameter. Fatal, independent of the extra
magic number.

Two reconstruction modes, both fidelity-tested: **candidate view** (latest refit at
or before *t*) and **position view** (frozen at `entry_asof_date`). The position
view is the special case and the more likely to break.

Z-score state is constructed fresh per call and must stay that way. Carrying EWM
state across refits would initialize a statistic on one series with state
accumulated on a different series, and would make reconstruction path-dependent.

## Debug sample and fidelity testing

**Standing rule:** any value used in a run that can be derived afterwards must have
a fidelity test against a persisted debug sample.

Config on the simulator: a pair count, a seed, and an optional explicit id tuple
that overrides the random draw. **The sampled ids are persisted with the values**,
so the sample is self-describing and the seed is irrelevant to reconstruction — no
committed fixture, no snapshot pinning, no regeneration discipline.

One artifact, one grain — keys, `spread_id`, `refit_date`, `date`, the per-leg
residual returns, the spread level, the z-score — so the tests join without merges.
A few million rows at most.

**Tolerance is zero.** `check_exact=True`, not almost-equal. Reconstruction runs the
same function on the same inputs, so results should be bit-identical. A tolerance
would hide exactly the bugs the test exists to catch: a beta applied one refit date
off, a cumsum starting one row late, a slice including an extra day. Two legitimate
threats to exactness, both avoidable — different summation order (use the same
call) and BLAS threading in the fit (compare beta from the artifact rather than
refitting). NaN equality is the one exception and is correct by default.

The test requires a small live run rather than a static fixture, so it is slower
than a unit test.

## Outer/inner loop

Migrate the loop structure from panel creation into the simulator: outer at the
configured refit frequency, inner daily.

**The `≤ t` alignment invariant** — the outer date used is at or before the trading
day, never the next — must be tested in **both** the simulator and the
reconstruction path. This is the one invariant whose failure produces
plausible-looking *better* results rather than an error, applied uniformly across
every trade.

## Also in scope

- Single code path via the spread primitives module for anything computed twice.
  The OU fit already appears in two places; do not add a third spread-level
  implementation for reconstruction.
- `fit_input_tickers` recorded in the run config per `(group_id, residual_key)`: the
  **union** across fit dates of the **realized** input set, not the requested one.
  It is what reconstruction must load. Per-date composition varies with dropna and
  belongs to B's log. The load-time assert catches "different universe"; the
  snapshot hash covers "same universe, different content".
- The residual fit must receive its input returns **as an argument**, not resolve
  tickers internally by group. The same constraint arrives from the parallelization
  direction.

## Sector parallelization — deferred, five constraints held now

Pure per-group functions; no shared RNG; disjoint artifact paths; picklable
returns; no cross-group ordering dependence, with aggregation after all groups
return. Do not hard-code the group as the sharding key inside the worker — let the
caller decide the partition.

## Out of scope

- Any change to a computed value.
- `zscore_key`. That is H. Until it exists, the key columns are `residual_key` and
  `hedge_key` only.
- Widening the occupancy grain.
- Cross-run reuse of anything.

## Acceptance

- Full suite green.
- The baseline file from B does not move.
- Fidelity test passing at `check_exact=True` for spread level in both candidate and
  position views.
- A test for the `≤ t` invariant in both the simulator and the reconstruction path.
- Uniqueness asserted on every artifact write, with a test per artifact.
- Notebooks re-executed against the branch, since the API surface changes.
