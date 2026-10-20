# Track F2 — Artifact layer: loop, series, fidelity

**Changes results:** yes, in exactly one commit (C3 in `F2_spec.md`, Review
amendments R1). Everything else must not move `B_baseline.txt`.

**Depends on:** F1.

Second half of the dissolved Track F. F1 replaced the artifact surface; F2 moves the
loop into the simulator, resolves `series/`, and builds the fidelity test that makes
the whole dissolution checkable.

Decisions carried in from `F_spec.md` and from review are marked **[decided]**.

## 1. Outer/inner loop migration

Move the loop structure from panel creation into the simulator: outer at the
configured refit frequency, inner daily. The residual fit must receive its input
returns **as an argument**, not resolve tickers internally by group — the same
constraint arrives from the parallelization direction.

**This is the track's one genuine computation-path change**, and it relocates
*where* scoring happens rather than *what* is computed. `B_baseline.txt` must not
move a single row.

**The `≤ t` alignment invariant** — the outer date used is at or before the trading
day, never the next — is structural today and never asserted (`F_spec.md` Part 1 §5).
Add an explicit test in **both** the simulator and the reconstruction path.

This is the one invariant whose failure produces plausible-looking *better* results
rather than an error, applied uniformly across every trade. An off-by-one here is
exactly the failure the fidelity test exists to catch, which is why commit 3 should
follow closely rather than sitting at the end.

## 2. `series/` — delete the cache; reconstruct on demand

**[decided]** The cache dies: the read path at `candidate_signals.py:539-542,576`,
the `skip-if-exists` at `series.py:192-193,233-234`, and the shared `panel_dir`
scope. The filename key is `(group_id, ticker)` / `(spread_id, asof_date)` with
**`residual_key` absent**, so writing a panel under a different residual config
silently reuses the old files and the reader cannot detect it. This is `found.md`'s
staleness axis 2, and it is the one axis that can produce wrong numbers today.

Deleting rather than re-keying, because `series/` as a cache is reused across runs
and across residual configs by construction — the same shape of problem the panel
dissolution removes one layer up. Re-keying would keep a cross-run reuse mechanism
alive inside a track whose stated decision is to remove exactly that.

The recompute cost is 1.7 ms per candidate's full-history residual reconstruction,
measured on real data. At that price the cache buys little against the staleness
risk it carries.

**[decided — amended before the F2 spec session; supersedes the earlier "optional
generation stays"]** No run persists series at all. Series are derived values:
the reconstruction function this track builds regenerates any residual or
spread-level series from the run's persisted artifacts, and the fidelity test
guarantees it equals what the run used. Persisting them as well would be a second
path to the same data, carrying a config flag and a granularity decision. The only
consumer ever named for them, `simrun_microscope`, does not exist in `statarb_sim`.

Plots and inspection call the reconstruction function. Every writer of persisted
series is removed, and every display-only reader is repointed to reconstruction.
`hierarchical-arb`'s `simrun_microscope` may read persisted series; that is a port
concern, not a reason to keep them here.

**Open for the spec session:** whether the read-path removal can land as its own
commit before commit 1. The recompute path already exists as the non-`panel_dir`
branch, so it may not depend on the loop migration. `F_spec.md` sequences it
after; check whether that dependency is real.

## 3. Debug sample and fidelity testing

**Standing rule:** any value used in a run that can be derived afterwards must have
a fidelity test against a persisted debug sample.

Config on the simulator: a pair count, a seed, and an optional explicit id tuple
overriding the random draw. **The sampled ids are persisted with the values**, so the
sample is self-describing and the seed is irrelevant to reconstruction — no committed
fixture, no snapshot pinning, no regeneration discipline.

One artifact, one grain — keys, `spread_id`, `refit_date`, `date`, the per-leg
residual returns, the spread level, the z-score — so the tests join without merges.

The grain is correct as stated: each candidate is permanently bound to one
`(spread_id, asof_date, residual_key)` and always uses its own frozen weights, and an
old candidate keeps producing z-scores while its position is open. Multiple refit
dates are genuinely live on the same date, so the cross product is populated rather
than counterfactual.

**Tolerance is zero.** `check_exact=True`, not almost-equal. Reconstruction runs the
same function on the same inputs, so results should be bit-identical. A tolerance
would hide exactly what the test exists to catch: a beta applied one refit date off,
a cumsum starting one row late, a slice including an extra day.

Two legitimate threats to exactness, both avoidable — different summation order (use
the same call) and BLAS threading in the fit (compare beta from the artifact rather
than refitting). NaN equality is the one exception and is correct by default.

The test needs a small live run rather than a static fixture, so it is slower than a
unit test. **Run it every time.** The bugs it catches look sporadic, which is exactly
why sampling the test itself would defeat it.

## Spread level and reconstruction

**[decided]** The residual model and the weights are both frozen at the candidate's
`asof_date`. See `F2_spec.md`, Review amendments R1.

Every refit date delivers new residuals and new hedge ratios, cumsummed into a new
spread level series over the whole history. Each `spread_id` has one level series per
refit date, and there is **no splicing** — an earlier concern about refit
discontinuities in a spliced cumulative series does not apply.

Anchoring the cumsum at zero is a **gauge freedom**, not a choice: the level enters
via a z-score against an EWM mean, so a constant shifts both and z is unchanged.

Time-decaying spread returns before cumsumming was considered and **rejected**. It is
not a re-anchoring — it imposes an OU pull toward zero before an OU model is fitted,
biasing kappa toward the imposed parameter. Fatal, independent of the extra magic
number.

Two reconstruction modes, both fidelity-tested: **candidate view** (latest refit at
or before *t*) and **position view** (frozen at `entry_asof_date`). The position view
is the special case and the more likely to break.

Z-score state is constructed fresh per call and must stay that way. Carrying EWM
state across refits would initialize a statistic on one series with state accumulated
on a different series, and would make reconstruction path-dependent.

**PC removal:** reconstruction raises when `remove_residual_pcs > 0`, with a message
saying the path is unverified and not covered by the fidelity test. The key encodes
the PC *count*, not the vectors; re-fitting recovers them deterministically from
frozen data, but sign is defined only up to a factor of minus one and near-degenerate
eigenvalues rotate the basis. Both are harmless when the component is only projected
out, but a latent unverified path returning plausible numbers is the
silent-wrong-result class. Three lines, low stakes — the default is zero and one is
confirmed harmful.

## Single code path

Anything computed twice goes through the spread primitives module. The OU fit already
appears in two places; do not add a third spread-level implementation for
reconstruction.

## Parallelization — deferred, five constraints held now

Pure per-group functions; no shared RNG; disjoint artifact paths; picklable returns;
no cross-group ordering dependence, with aggregation after all groups return. Do not
hard-code the group as the sharding key inside the worker — let the caller decide the
partition.

## Out of scope

- Anything in F1's list.
- `start_after_nan` / `check_for_corruptions`. Track I.
- `zscore_key`, occupancy, sleeve identity. Track H.
- Cross-run reuse of anything.
- Snapshot-layer wiring of the loader call sites, `UniverseDataLoader.load`'s
  in-place resync, and the `DataConfig.universe_name` / `snapshot_id` defaults.
  Track I.

## Acceptance

- `docs/refactor/B_baseline.txt` does not move a single row, checked after commit 1
  specifically.
- Fidelity test passing at `check_exact=True` for spread level in both candidate and
  position views.
- A `≤ t` invariant test in both the simulator and the reconstruction path.
- Full suite green.
- Notebooks re-executed; fix any notebook-only caller this uncovers.

## Sequencing note

Put the fidelity test as close after commit 1 as possible, not at the end. It is what
would catch a regression commit 1's own baseline check missed — at the end of the
track it is decoration on a change already trusted.
