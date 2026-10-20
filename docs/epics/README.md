# Epics

Epics were formerly called tracks (A–J). Status lives only on GitHub: see the
[milestone "Baseline freeze"](https://github.com/alphachain-trading/statarb_sim/milestone/1)
and the epic issues below. For the process (issue → branch → PR), see
[CONTRIBUTING.md](../../CONTRIBUTING.md).

`statarb_sim_refactor_handover.md` is **superseded** and contains known errors (see
[archive/_superseded_handover.md](archive/_superseded_handover.md)). Do not implement
from it.

## Open epics

| Formerly | Epic | Changes results? | Depends on |
|---|---|---|---|
| I | [#30 Data preparation](https://github.com/alphachain-trading/statarb_sim/issues/30) · [brief](30-data-preparation/I_data_preparation.md) | **yes** | — |
| J | [#31 Retire the offline panel path](https://github.com/alphachain-trading/statarb_sim/issues/31) · [brief](31-retire-offline-panel-path/J_retire_offline_panel_path.md) | possibly | I |
| G | [#32 Decay-expanding hedge ratio mode](https://github.com/alphachain-trading/statarb_sim/issues/32) · [brief](32-hedge-mode/G_hedge_mode.md) | **yes** | I, J |
| H | [#33 Occupancy policy (deferred)](https://github.com/alphachain-trading/statarb_sim/issues/33) · [brief](33-occupancy-policy/H_occupancy_policy.md) | **yes** | G |

Completed tracks A–F2 are in [archive/](archive/); they have no issues. Each brief
carries its former track letter.

Each epic brief is paired with a spec where one exists: a read-only spec session
writes `spec` first, then a fresh implementation session reads brief + spec only. For
small, precisely specified epics the spec session can be folded into the
implementation session as a stop-and-report first step. Where documents disagree, the
spec wins; where a brief carries a decision made *after* its spec session, the brief
marks it as decided and wins for that item.

## Order

**I → J → G.** H only when a run actually needs multiple sleeves in one run; the
guard added in former Track A is the trigger.

J precedes G because J is potentially result-changing (the equivalence test that
justifies deleting the offline path covers only two universes today) — it must land
before G's performance baseline is defined, for the same reason I does.

I must not run in parallel with J: I's flag unification touches all three of
`DataConfig`, `PanelBatchConfig`, and `CandidateGenerationConfig` while all three
still exist; J deletes `PanelBatchConfig`. I must precede both J and G.

## Working rules for this codebase

- **Loud failures**, never silent fallback.
- **Report `path:line`** for any assertion about existing behaviour. Say `NOT FOUND`
  rather than inferring from a similar name.
- **Any grep supporting a deletion must include `notebooks/**/*.ipynb`.** Track C
  found a live notebook caller for a function two prior sessions had recorded as
  having zero callers, because the greps were `.py`-only.
- **Measure through the shipped code path**, never by editing a flag by hand.
- **Re-execute committed notebooks** whenever a column, an API surface, or a runtime
  changes.
- **tqdm on every expensive loop.**

## Baseline harness

`scripts/b_baseline_harness.py`, writing [`docs/baseline/B_baseline.txt`](../baseline/B_baseline.txt).
Before/after on a result-changing change is `git diff` on that file, and the
`## counts` section is the binding check — candidate and trade counts are
interpretable, performance metrics at a few dozen trades are not.

**Two known blind spots.**

The harness carries counts and the trade list, **not performance metrics**. Any
change to `generate_report` needs its own `performance_metrics.json` before/after
diff.

The two harness universes have **no internal gaps** over the harness date range, so
anything touching NaN handling or missing data may not be exercised at all. A zero
diff there is not evidence of neutrality. A third, deliberately gappy universe would
close it: [#34](https://github.com/alphachain-trading/statarb_sim/issues/34).

## What the baseline is and is not

Every performance number computed before this refactor is invalidated —
partial-window beta fits, contaminated shared-matrix dropna, an unstated `ddof`,
unreviewed cost assumptions. Mid-refactor Sharpe comparisons are uninformative and
should not be reported as findings.

What the harness measures is whether a change moved something it should not have.
That works fine on a tainted starting point, which is why it is committed with the
pre-fix numbers in it. The new performance baseline is defined by
[#45](https://github.com/alphachain-trading/statarb_sim/issues/45), the final issue
of the milestone.

## Porting to hierarchical-arb

Fixes land in `statarb_sim` first and are ported back as separate atomic commits per
fix. Port notes are GitHub issues (label `triage`).

Known so far: `config_hash` changed twice — former Track A's `candidate_max_age_days`
deletion and former Track E's `sector` → `group_id` field renames, since `hash_config`
serializes field names as well as values. Harmless in `statarb_sim`, which has no
persisted run dirs; both will orphan existing runs in `hierarchical-arb`.
