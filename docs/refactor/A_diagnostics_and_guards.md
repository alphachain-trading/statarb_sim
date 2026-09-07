# Track A — Diagnostics defects and guards

**Changes results:** no. Every item here changes what is reported or stored, or
raises on a config that is never currently set. No number moves.

**Depends on:** nothing. Do this first.

All work is inside the candidate row builder and the config validation path.

## Verify first

Each claim below was established in an earlier investigation but is stated here as
a claim only. Re-verify with `path:line` before implementing. Report `NOT FOUND`
rather than implementing against a guess.

1. `adf_pvalue` is computed after all validity gating, feeds nothing, and is read
   by nothing in current production configuration.
2. `skip_adf` already exists as a flag.
3. `CandidateSelectionConfig.adf_pvalue_max` is never set to a non-None value
   anywhere in the repo, and the selection mask requires `adf_pvalue.notna()`
   before applying the threshold.
4. `residual_std` is computed before `kappa`, but `kappa` finiteness is gated
   first.
5. Gate 4's reason string embeds the observed length, making it dynamic.
6. The caller hardcodes `"n_legs": 2` and discards the value computed in the
   diagnostics function.
7. `intercept` is computed as part of the OU regression and dropped.
8. `failure_reason` already exists in the diagnostics return dict but is only
   written to the row when `debug=True`.
9. `ActivationConfig.candidate_max_age_days` is defined and never read anywhere.

## Scope

### Defects

- **`skip_adf=True` by default.** Re-time a panel build afterwards and record the
  number. The prior measurement was ~96s with ADF; the expectation is ~27s.
- **Swap the kappa and residual_std finiteness gates** so the reported reason is
  the upstream failure rather than a misattribution.
- **Normalize gate 4's reason** to a static `dx_too_short`. Standing rule: values
  never go in reason strings — the offending value is already in its own typed
  column and the threshold is in the run config.
- **Wire `n_legs` through** from the diagnostics function instead of hardcoding.
  Always 2 today; the gate that tests it must not be able to disagree with the
  column.
- **Persist `intercept`.** It is the OU long-run mean, already computed, free to
  store.
- **Promote `failure_reason` to a default-persisted column** named `why_invalid`.
  This is a promotion of an existing field, not new computation.
- **Delete `candidate_max_age_days`.** A config field that looks like a safety
  limit and enforces nothing is the same failure class as an unimplemented enum
  member that raises at dispatch.

### Guards

All three raise unconditionally. No opt-out flag, no warn-only mode. Each converts
a currently-silent wrong result into a loud failure, and each fires only on a
configuration that is not used today.

- **`adf_pvalue_max` with an all-NaN column.** With ADF skipped, setting the
  threshold masks out everything and returns an empty selection with no error.
  Raise, naming both the setting and the reason the column is empty.
- **`EQ_ROLLING` residual config.** The z-score's EWM history span is inherited
  from whichever upstream path supplied the series — full history on the
  disk-backed path, truncated to the residual lookback on the recompute path. So
  the same `(spread_id, date)` yields a different z-score depending on whether a
  cache existed, and the residual config silently controls a z-score property.
  Dormant only because production uses an expanding config. Raise, stating that
  the z-score span is inherited from the residual lookback and that the explicit
  span parameter is not yet implemented.
- **Multi-sleeve occupancy.** The trader's open-position dedup is keyed on
  `(spread_id, residual_key)`, which omits the z-config and the hedge config. A
  run carrying two z-lookbacks under one `residual_key`, or several hedge configs,
  will have its second sleeve silently blocked — arrival order decides which one
  trades. Raise when a run's sleeve set exceeds that grain. Message states what
  would contend and says to run them as separate runs for now.

The third guard's message must **not** reference a fix or a config field that does
not exist yet. Describe the current limitation only.

## Out of scope

- Changing any gate threshold or adding a gate.
- Reinstating ADF with a fixed `maxlag`. Noted as possible, not wanted now.
- Fixing the z-span inheritance bug. Guarded here, fixed in H.
- Widening the occupancy grain. Guarded here, designed in H.
- Any trade-duration or max-holding-days rule. Tested previously in
  `hierarchical-arb` with no improvement; the churn problem ate it.

## Acceptance

- Full suite green.
- Re-timed panel build recorded in the commit message or the spec.
- A test per guard, asserting the raise and the message content.
- A test asserting `n_legs` and the gate that reads it cannot disagree.
- The third guard is run against every config in `DEFAULT_CONFIGS` and every
  notebook config. **If it fires on a config actually in use, record that in
  `found.md`** — it means H is urgent rather than deferred.
