# Track G — Decay-expanding hedge ratio mode

**Changes results: yes.**

**Depends on:** B (per-pair `min_obs` and date-distance weighting are prerequisites,
not merely prior work) and D (the no-default convention).

This is the original task the whole refactor grew out of.

## Motivation — not performance

The pipeline has three estimation stages: residual fit, hedge ratio, z-score.
Residual and z-score support decay-expanding weighting; the hedge ratio is
rolling-only. That is an inconsistency. Rolling also carries a window-exit
discontinuity — a departing observation drops from full weight to zero, perturbing
beta for non-market reasons.

**Calibrate expectations honestly.** The expected Sharpe effect is small, plausibly
null, possibly slightly negative if beta non-stationarity dominates. Prior on a
positive sign is roughly 55–65%, likely a few hundredths at most. The justification
is estimation-framework consistency, not a performance play. Write it up that way.

**Correction to earlier reasoning, worth stating in any writeup.** The original
argument against decay here was that it would compound with decay in the residuals
and add recency bias. That has the sign backwards: at matched effective sample size,
rolling is *strictly more* recency-biased, since it zeroes everything beyond the
lookback. The stages estimate different objects, so the weights do not multiply. The
real concern is variance propagation, which argues for a longer halflife at this
stage, not for banning decay.

**Kalman is rejected** — two parameters, one of them unidentifiable, and a
time-varying observation variance reintroduces a window. Constant-gain Kalman
reduces to EWM anyway; decay-expanding *is* the steady-state constant-gain special
case. Worth one line in a writeup.

## `HedgeRatioConfig`

New mode-discriminated config, mirroring the residual config's shape but **not**
extending it — different estimand, and extending would drag in modes and fields
that do not apply.

Fields: `mode` in `{EQ_ROLLING, DECAY_EXPANDING}`; `lb` for rolling only; `hl` for
decay-expanding only; `min_obs` mandatory with no default.

Clean break on the batch config: the scalar lookback field becomes a
`HedgeRatioConfig`. No shim.

**`EQ_EXPANDING` is excluded.** It is the infinite-halflife limit of
`DECAY_EXPANDING`, reachable as a sweep endpoint. A separate mode for a limit point
is dead surface.

**Kalman-extensible shape, no enum member.** An unimplemented enum value that raises
at dispatch is worse than absent. What to bake in now: `mode` is the discriminator
and parameter fields are per-mode rather than a flat scalar, so a future mode
carrying several parameters fits; `.key` encodes only the fields live for that
mode; validation rejects fields set for the wrong mode, loudly.

## `.key` format — readable, not hashed

Follow the residual key convention: a readable format string that `from_key` parses
back, e.g. `rol_lb252` and `dec_exp_hl87`.

The superseded handover proposed a hash digest here. **Do not.** The repo's pattern
is consistent and should stay so: config keys are readable format strings, object
identities are hashes. A hash would also break the claim that a concatenated frame
of artifacts is self-describing, since a hashed key is only interpretable via the
run config. The readable form is shorter than the digest anyway.

Version the key format so a future field addition breaks visibly rather than
silently orphaning artifacts.

## Implementation notes

**Weight by date distance to the fit date, not post-dropna row position.** Delivered
by B; verify it is actually used here. Weighting by position after dropping pulls
old observations forward and silently up-weights them.

**Keep the no-intercept fit.** Weighted demeaning is a separate change with its own
effect on beta; bundling makes the result uninterpretable.

## `hl` sweep grid

Exponential weights with halflife `hl` have an effective sample size of roughly
`2.885 · hl`. Matched against the rolling lookback gives `hl* = lb / 2.885`. **Only
at matched effective sample size is the comparison about the weight profile rather
than about sample size.**

With `lb = 252`, `hl* ≈ 87`. Grid anchored there and bracketed geometrically both
ways: `{22, 44, 87, 175, 349, 699}`.

Two readings come off it: whether `hl*` beats rolling, which tests the
window-exit-discontinuity claim; and where the optimum sits, where the long end
means beta is near-constant and the short end means it drifts. The long end must run
far enough to flatten out.

**Run the six configs as six separate runs, not one multi-config run.** A single run
carrying multiple hedge configs hits the occupancy collision that A guards against —
the second sleeve on each spread would be silently blocked. Keeping them separate is
what keeps H deferred.

`hl` stays a permanent config field, swept once, winner pinned in the default
bundle. Not per-group, not per-timescale.

## Mode comparison — intersection, not equality

The superseded handover asserts the two modes must admit the same
`(spread_id, asof_date)` set. **That is unachievable and the assert must not be
written.** Rolling needs its observations inside a trailing window; decay-expanding
needs them anywhere in history, so it admits a strict superset.

Instead: run rolling first, record its admitted set, pass it as a mask to the decay
runs, and **compare on the intersection**. Report the symmetric-difference size as a
diagnostic. That achieves what the assert was for — removing the universe confound
from the comparison — without asserting something false. One extra pass, not six.

## Paired beta diagnostic — the definite result of this work

The Sharpe comparison is **not paired**: beta changes, so the spread changes, so z
changes, so entry dates change, so the trade sets differ. Against a thin
LLN-driven edge, that test is underpowered and will likely return null regardless of
which estimator is better.

So add an estimator-level diagnostic that does not depend on trading outcomes: mean
absolute beta change per refit date, and the distribution of that change conditional
on an observation exiting the rolling window versus not.

This directly measures the discontinuity being removed, and gives a beta
non-stationarity read independent of where the `hl` optimum sits — two estimates of
the same thing. Expect this, not the Sharpe number, to be the reportable result.

## Out of scope

- Any change to the residual or z-score estimation stages.
- Per-group or per-timescale halflife.
- A beta-magnitude pair-quality filter. On the research backlog; beta is persisted,
  so the distribution is inspectable after any run with no new code.
- Cross-sectional shrinkage of beta toward a pooled estimate. Not on the backlog,
  and it would be the one change that genuinely costs parallelism — a global
  dependency at the expensive stage needing two passes with a barrier.

## Acceptance

- Full suite green.
- The baseline file from B does not move for `EQ_ROLLING` at `lb=252`. The new mode
  is additive; the incumbent path must be untouched.
- A test that fields set for the wrong mode raise.
- A `.key` round-trip test in both directions, and a test that two configs share a
  key if and only if they are semantically identical.
- A test that weighting uses date distance, constructed with a deliberate gap.
- Grid results and the paired beta diagnostic recorded, with the
  symmetric-difference sizes.
