# Track H — Occupancy policy

**Status: deferred.** Design happens when the branch opens. This brief exists so the
branch can be reopened without rereading the design thread.

**Changes results: yes** — trade counts rise as previously-blocked entries open.

**Depends on:** `zscore_key` and the hedge `.key` from G. You cannot scope by
components that are not defined.

## Trigger

Not a position in the sequence. Open this when a run genuinely needs multiple
sleeves — combinations of residual, hedge, and z-score config — running **in one
run**. Separate single-config runs never hit the problem, and that is the more
realistic comparison anyway.

**The guard in A is the trigger.** If it fires on a config actually wanted, H stops
being deferred.

## The defect

The trader's open-position dedup is keyed on `(spread_id, residual_key)`. It omits
the z-config and the hedge config, so two sleeves differing only in those share one
slot per spread. First to arrive takes it; the other is silently blocked. Which one
wins is an artifact of arrival order, not of signal quality.

There is a second, related inconsistency: candidate activation tracks a finer grain
that includes the z-lookback, while occupancy tracks the coarser key. Two grains for
the same lifecycle.

## The design position reached, not yet settled

**The complaint is not that the tuple is too narrow.** It is that a portfolio-level
rule is buried in the trader. Widening the tuple alone would just move the hidden
assumption.

Two positions were reached in discussion and both should be revisited when the
branch opens:

**1. The rule belongs to the risk manager.** It already has the portfolio-wide view
— gross exposure, per-ticker concentration, concurrent position count — and is the
only component that sees all sleeves. The trader's occupancy check is then the wrong
*mechanism*, not a mechanism with the wrong grain: a boolean at the wrong level that
silently drops a duplicate before anything with portfolio context sees it. A risk
manager that has a say needs the proposal to reach it.

So the likely shape is: remove the trader's short-circuit, let sleeves propose
freely tagged with their sleeve identity, and make the aggregation rule a
risk-manager concern.

**2. Whatever the rule is, it must be configured explicitly**, with no default,
stated per run and recorded in the run config. "Two sleeves may hold the same
spread" is a policy, not a fact. Sometimes doubling into one spread is the intended
diversification across timescales; sometimes it is concentration nobody chose.

## Open questions for the spec session

- **Does the existing per-ticker concentration cap already cover this?** Two sleeves
  long the same spread means doubled exposure to both legs, which is exactly what a
  per-ticker cap measures. If it binds, the answer may be that no new cap is needed
  — only removal of the trader's pre-emption. That would be the cheapest possible
  version of this branch and is worth establishing before designing anything.
- **Identity rule or exposure rule?** A rule keyed on `spread_id` catches two
  sleeves on the same spread and misses two sleeves on overlapping spreads sharing a
  leg — the same risk, less visibly. A per-ticker exposure cap catches both. That
  argues for expressing this in exposure terms, which is also the shape the
  portfolio-optimization phase will want, since marginal-variance sizing subsumes
  both cases.
- **Keep the branch small.** Do not design a new correlated-position rule here. That
  is the portfolio phase, and a `spread_id`-keyed cap built now is a rule that would
  have to be unbuilt.

## Also lands here

**`zscore_key`, with an explicit history-span parameter.** This is a prerequisite for
scoping, and it carries the fix for the z-span inheritance bug that A merely guards
against: the z-score's EWM span is currently inherited from whichever upstream path
supplied the series, so the residual config silently controls a z-score property and
the same `(spread_id, date)` can differ depending on whether a cache existed. An
explicit span parameter makes the coupling impossible rather than merely corrected.

Once `zscore_key` exists, A's `EQ_ROLLING` guard can be removed.

**Sleeve identity is a tuple, not a stored string.** `(residual_key, hedge_key,
zscore_key)` used as an in-memory hashable and as a groupby. Do not persist a
concatenated fourth column — that is derived data that can disagree with its
components, the same objection as a stored portfolio total row. The three component
keys are already columns on every artifact.

Note the existing `timescale_label` is documented as residual key plus z-lookback.
If a name is needed for the tuple, avoid "timescale" — the axes include PC removal,
risk-free subtraction, and entry thresholds, none of which are timescales.

## Acceptance, when it happens

- The baseline file from B moves only in the expected direction: more trades, from
  previously-blocked entries.
- One grain everywhere — activation and occupancy agree.
- The policy is a no-default config field recorded in the run config.
- Result-neutral at today's effective policy, which is the regression test.
