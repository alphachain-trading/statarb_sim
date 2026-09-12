# statarb_sim refactor — track index

`statarb_sim_refactor_handover.md` is **superseded** and contains known errors (see
`_superseded_handover.md`). Do not implement from it.

Each track has a brief. A track is worked as two CC sessions: a read-only spec
session that writes `{track}_spec.md`, then a fresh implementation session that
reads brief + spec only. For small, precisely specified tracks the spec session can
be folded into the implementation session as a stop-and-report first step.

| | Track | Changes results? | Depends on | Status |
|---|---|---|---|---|
| A | Diagnostics defects + guards | no | — | merged |
| B | Window admission | **yes** | — | merged |
| C | Market snapshot | no | — | merged |
| D | Config hygiene | no | — | merged |
| E | Terminology | no | D | merged |
| F1 | Artifact schema | no | C, D, E | merged |
| F2 | Loop, series, fidelity | **yes** (one commit) | F1 | |
| I | Data preparation flags | **yes** | — | |
| J | Retire offline panel path | possibly | F2 | |
| G | Hedge mode | **yes** | B, D, I | |
| H | Occupancy policy | **yes** | G | deferred |

Track F was split after its spec session: F1 is the artifact surface, where every
commit has a neutrality argument the baseline can check; F2 is the loop migration
and the fidelity test, where the one genuine computation-path change lives.
`F_artifact_layer.md` is deleted; `F_spec.md` predates the split and covers both
halves.

Track I was split out of F — it is data preparation, not artifact layer, and it must
land before G.

Track J was split out of F2 — F2's own C6c ("delete the offline path") found two
live consumers (`sweep_runner.py`, `run_me.py`'s `residuals` stage) with no
live-generation equivalent, so migrating them and deleting `run_panel_batch` became
its own track rather than F2's last commit. See `F2_spec.md` R11 and
`J_retire_offline_panel_path.md`.

## Order

Remaining: **F2 → I → J → G**. H only when a run actually needs multiple sleeves in
one run; the guard added in Track A is the trigger.

J precedes G because J is potentially result-changing (the equivalence test that
justifies deleting the offline path covers only two universes today) — it must land
before G's post-sweep performance baseline is defined, for the same reason I does.

I must not run in parallel with J: Track I's flag unification touches all three of
`DataConfig`, `PanelBatchConfig`, and `CandidateGenerationConfig`
(`found.md`, "four combinations" entry) while all three still exist; J deletes
`PanelBatchConfig`. I must precede both J and G.

## Standing rules for every track

### Branching — read this first

**One track, one branch. Never commit to `main`.**

- Create the branch before the first commit. Commit only to the branch.
- Never fast-forward, merge, or rebase `main`. Merging is a human decision made
  after reviewing the diff.
- If you believe a change belongs on `main`, **stop and say so** rather than doing
  it.
- Finish by reporting the diff. Do not merge.

Exception: documentation is not track work. A read-only spec session commits
`{track}_spec.md` to `main` directly and creates no branch. Amendments to briefs,
`found.md` and this README made outside a track are committed to `main` directly
when the user instructs it. `found.md` entries written during a track stay on that
track's branch.

This is not a formality. Tracks I, G, and H change trading results, and a
result-changing track merged without a reviewed diff is how a tainted baseline gets
locked in permanently — silently, and with no later signal that it happened.

This rule was violated once: Track A was committed directly to `main`, twelve
commits, with the branch pointing at the same history, which defeated the review
gate entirely. It is recorded here so it does not recur.

### The rest

- **Spec before code.** No implementation without an agreed written spec.
- **Scope is exactly what the spec lists.** Anything wrong found outside scope goes
  in `docs/refactor/found.md` and is not fixed. A fix that the track has already
  decided in principle — the same defect one config layer up — is *in* scope; a
  different decision is not.
- **Where documents disagree, the spec wins.** Specs carry amendments that override
  the brief. Where a brief carries a decision made *after* its spec session, the
  brief marks it as decided and wins for that item.
- **Tests:** `python -m unittest discover -s tests`. pytest is not installed.
- **Atomic commits**, one logical change each. `main` stays clean and presentable.
- **Loud failures**, never silent fallback.
- **Report `path:line`** for any assertion about existing behaviour. Say `NOT FOUND`
  rather than inferring from a similar name.
- **Any grep supporting a deletion must include `notebooks/**/*.ipynb`.** Track C
  found a live notebook caller for a function two prior sessions had recorded as
  having zero callers, because the greps were `.py`-only. Every "dead code" finding
  made before that point is unverified for notebooks.
- **Measure through the shipped code path**, never by editing a flag by hand. A
  timing or result figure obtained any other way does not count as verification —
  Track A's first `skip_adf` timing was obtained that way and had to be redone.
- **Re-execute committed notebooks** whenever a column, an API surface, or a runtime
  changes. Committed notebooks must carry outputs from a run against current code.
- **tqdm on every expensive loop.**

## Baseline harness

`scripts/b_baseline_harness.py`, writing `docs/refactor/B_baseline.txt`. Built as
the first commit of B, reused by every track since. Before/after on a
result-changing track is `git diff` on that file, and the `## counts` section is the
binding check — candidate and trade counts are interpretable, performance metrics at
a few dozen trades are not.

**Two known blind spots.**

The baseline carries counts and the trade list, **not performance metrics**. F1's
commit 8b repoints `generate_report`, and the baseline structurally cannot catch a
regression there; that step needs its own `performance_metrics.json` before/after
diff.

The two harness universes have **no internal gaps** over the harness date range, so
anything touching NaN handling or missing data may not be exercised at all. A zero
diff there is not evidence of neutrality. This affects Track I and the `_make_sqrt_w`
entry in `found.md`. A third, deliberately gappy test universe would close it.

## What the baseline is and is not

Every performance number computed before this refactor is invalidated —
partial-window beta fits, contaminated shared-matrix dropna, an unstated `ddof`,
unreviewed cost assumptions. Mid-refactor Sharpe comparisons are uninformative and
should not be reported as findings.

What the baseline measures is whether a change moved something it should not have.
That works fine on a tainted starting point, which is why it is committed with the
pre-fix numbers in it.

A new performance baseline gets defined after G, and the headline research findings
get re-measured against clean panels. That is separate work, not a track.

## Porting to hierarchical-arb

Fixes land in `statarb_sim` first and are ported back as separate atomic commits per
fix. Port notes accumulate in `found.md`.

Known so far: `config_hash` changed twice — Track A's `candidate_max_age_days`
deletion and Track E's `sector` → `group_id` field renames, since `hash_config`
serializes field names as well as values. Harmless in `statarb_sim`, which has no
persisted run dirs; both will orphan existing runs in `hierarchical-arb`.
