# statarb_sim refactor — track index

Eight independent tracks. `statarb_sim_refactor_handover.md` is **superseded** and
contains known errors (see `_superseded_handover.md`). Do not implement from it.

Each track has a brief. A track is worked as two CC sessions: a read-only spec
session that writes `{track}_spec.md`, then a fresh implementation session that
reads brief + spec only.

| | Track | Changes results? | Depends on |
|---|---|---|---|
| A | Diagnostics defects + guards | no | — |
| B | Window admission | **yes** | — |
| C | Market snapshot | no | — |
| D | Config hygiene | no | — |
| E | Terminology | no | — |
| F | Artifact layer (dissolution) | no | E; C weakly |
| G | Hedge mode | **yes** | B, D |
| H | Occupancy policy | **yes** | G — deferred |

## Order

A first. Then B (with the baseline harness). Then D, E, C, F in that order. G last
of the active work. H only when a run actually needs multiple sleeves.

E generates textual conflicts with every track it overlaps. Land it alone, same
day. C and F may overlap as parallel sessions (C is a new module); nothing else
should.

## Standing rules for every track

### Branching — read this first

**One track, one branch. Never commit to `main`.**

- Create the branch before the first commit. Commit only to the branch.
- Never fast-forward, merge, or rebase `main`. Merging is a human decision made
  after reviewing the diff.
- If you believe a change belongs on `main`, **stop and say so** rather than doing
  it.
- Finish by reporting the diff. Do not merge.

This is not a formality. Tracks B, G, and H change trading results, and a
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
  the brief.
- **Tests:** `python -m unittest discover -s tests`. pytest is not installed.
- **Atomic commits**, one logical change each. `main` stays clean and presentable.
- **Loud failures**, never silent fallback.
- **Report `path:line`** for any assertion about existing behaviour. Say `NOT FOUND`
  rather than inferring from a similar name.
- **Measure through the shipped code path**, never by editing a flag by hand. A
  timing or result figure obtained any other way does not count as verification.
- **Re-execute committed notebooks** whenever a column, an API surface, or a runtime
  changes. Committed notebooks must carry outputs from a run against current code.
- **tqdm on every expensive loop.**

## Baseline harness

Built as the first commit of B, reused by G and H. A small fixed config (two
groups, short range, tens of seconds) plus a script writing trades and summary
metrics to a committed text file. Before/after on a result-changing track is then
`git diff` on that file.

It is also the acceptance test for D: config hygiene is result-neutral, so the
baseline must not move at all.

## Porting to hierarchical-arb

Fixes land in `statarb_sim` first and are ported back as separate atomic commits per
fix. Port notes accumulate in `found.md` as they are discovered.

Known so far: deleting `candidate_max_age_days` in Track A changed `config_hash`.
Harmless in `statarb_sim`, which has no persisted run dirs, but it will orphan
existing runs in `hierarchical-arb`.