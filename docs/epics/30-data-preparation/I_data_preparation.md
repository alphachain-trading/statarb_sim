# Track I — Data preparation flags

> Epic: https://github.com/alphachain-trading/statarb_sim/issues/30 · Formerly Track I.

**Changes results:** yes, but not through the flags. The flag half is inert:
`check_for_corruptions` only prints (`I_spec.md`'s H1 amendment), and the
`start_after_nan` grain is erased downstream by `build_group_return_bundle`'s
`dropna="any"` (H6 amendment, proven byte-identical and confirmed end to end).
The result-changing work is the snapshot wiring, which fixes `_merge_umds`'s
order dependence (H4 amendment).

**Depends on:** nothing structurally. **Must run before G.**

A new track, split out of F because it has nothing to do with the artifact layer. It
is data preparation, and it sits upstream of everything.

## The finding

`start_after_nan` and `check_for_corruptions` control how raw price data is cleaned
before anything else runs. There are **five combinations** in the repo, and **no
caller overrides any of them** — every path just takes whatever default it happens to
reach.

| Source | `check_for_corruptions` | `start_after_nan` |
|---|---|---|
| `DataConfig` (`config.py:241-264`) | `True` | `True` |
| `PanelBatchConfig` (`panel_batch.py:102,150-152`) | `False` | `True` |
| `market_snapshot.py:277` (hardcoded literal) | `True` | `False` |
| `UniverseDataLoader.load` signature (`universe_loader.py:36`) | `True` | `True` |
| `CandidateGenerationConfig` (F2 C6b) | inherits `PanelBatchConfig` | inherits `PanelBatchConfig` |

The fifth was added deliberately in F2's C6b. Wiring live candidate generation
into the simulator meant the generator would otherwise have inherited
`DataConfig`'s flags while the offline walk kept `PanelBatchConfig`'s. The
`pair_notional` movement F2's C6b write-up measured when this first wired up
came from reusing `run_from_config`'s single **merged** UMD for scoring, not
from the flag difference itself (`I_spec.md`'s H1 amendment: re-tested with
the flag difference alone, isolated from the merge — zero movement). Giving
`CandidateGenerationConfig` its own fields defaulting to `PanelBatchConfig`'s
values fixed the immediate neutrality problem by restoring per-group loading;
it was not, and was never going to be, a fix for the flag values themselves,
which remain genuinely unreconciled and are this track's business regardless
— see `found.md`'s four-combinations entry.

This is now the path the simulator actually uses for candidate generation, so it
is not a peripheral fifth case. It also conflicts with Track D's no-defaults rule,
which this track is the one to resolve.

So the panel build and the simulator clean the same raw data differently, and Track
C's snapshot path introduced a fourth combination as a hardcoded literal, before
F2's C6b added a fifth.

This is the failure form Track D removed, one level more abstract: not one default
too many, but five silent answers to the same semantic question, selected by which
path the data happens to take. And because it is data preparation, it acts on every
candidate, every trade, and every result.

## Why before G

G sweeps six hedge configs and compares them. If the panel build and the simulator
prepare the underlying returns differently, the comparison is contaminated before it
starts — and the effect G is looking for is thin enough that a preparation artifact
could swamp it.

## Scope

**What this track covers, in execution order.** The order is inverted from the
brief's original framing: the flag unification comes last, not first. Three of
these four items were reassigned here after the brief was written.

1. **Partial-bar guard** on the mint and download paths. A download or mint during
   a live session writes an unsettled intraday bar; `found.md` records that no
   guard exists anywhere. A snapshot minted mid-session would freeze that bar
   permanently, so this precedes the wiring rather than following it.
2. **Snapshot wiring** of the loader call sites (reassigned from F2, README commit
   `b72a82f`), detailed below. **The only result-changing work in this track.**
3. **`DataConfig.universe_name`'s default and the `snapshot_id` `None` sentinel**,
   per Track D's rule and the two `found.md` entries named below.
4. **The flag cleanup** — the five combinations above, plus
   `CandidateGenerationConfig`'s inherited defaults from F2's C6b — on whatever
   call sites remain after the wiring.

**Why the flags come last.** Items 2 and 4 touch the same loader call sites, so
unifying first means writing those call sites twice. There is also a structural
possibility worth testing rather than assuming: if the snapshot freezes raw,
untrimmed data and consumers trim at read time, the flags may collapse to one
location instead of five — dissolving the question rather than legislating an
answer. Establish whether that holds before doing item 4 by decree.

Anything else found becomes a GitHub issue (label `triage`).

This is behaviour analysis first, unification second. Do not unify before
understanding what each flag does.

**The semantics are established** — `I_spec.md` Part 1 answers both questions, and
the H1 and H6 amendments settle what each flag is worth. Do not re-derive them.

**The grain decision, settled.** `start_after_nan` keeps its per-group grain. It is
stated explicitly at every call site per Track D's rule, and the inertness is
recorded in a comment at the trim site so nobody reopens it. Per-ticker and
no-trim were both considered and are provably unobservable downstream (H6), so
they buy nothing; no-trim additionally moves `market_dates.min()` structurally.

**Then decide the model.** One source of truth for both flags, stated explicitly by
the caller per Track D's rule, with no class defaults and no hardcoded literals.

**The snapshot case deserves its own answer.** `market_snapshot.py:277` sets
`start_after_nan=False`. There is a good argument for that — a snapshot should freeze
raw data, and trimming is a run-time decision, not a storage-time one. If that is the
reasoning, it should be written down and the flag should not be settable there at
all. If it is an accident, it is a real defect: a snapshot minted with different
trimming than the runs that read it.

**Snapshot wiring of the loader call sites (reassigned from F2).** The call sites
this track unifies are still not wired to Track C's snapshot layer. Two `found.md`
entries depend on that wiring: "`UniverseDataLoader.load`'s in-place cache-hit
resync remains live…" and "`DataConfig.universe_name` reintroduces the defect Track
D removed". Wire the call sites to `ensure_market_snapshot` /
`load_market_snapshot`, decide `UniverseDataLoader.load`'s role for them, and
remove the `universe_name` default and the `snapshot_id` `None` sentinel per Track
D's rule.

Snapshot data can differ from a local cache (`found.md`, series staleness axis 1),
so the wiring is not neutral by construction. It lands in commits separate from
the flag unification, each with its own `B_baseline.txt` check.

**This wiring is the structural fix for a live, order-dependent defect, not
just a staleness-prevention measure** (`I_spec.md`'s H1/H4 amendment).
`_load_umd`'s multi-group merge (`_merge_umds`, `simulator_factory.py:452-499`)
silently discards every group-after-the-first's own independently-loaded copy
of any ticker shared across groups (e.g. `SPY`/`^IRX`, both `benchmark`/
`risk_free` for energy and materials) whenever more than one group is
simulated together — proved empirically to be order-dependent at the
trade-count level, not merely a rounding-level price effect: reversing the
harness's group list changes the trade count itself, 97 → 98. Binding every
group in a universe to one frozen snapshot download removes the mechanism
outright — there is nothing left to merge inconsistently, since every group
reads from the same one immutable capture. This makes the snapshot-wiring half
of this track carry more weight than the flag-unification half, which the same
investigation showed to be largely inert: flipping `check_for_corruptions`
alone, holding everything else fixed, moves nothing in `B_baseline.txt`.

**Reproducibility consequence for G.** No clean post-refactor baseline
(README's "A new performance baseline gets defined after G") can be defined
while a multi-group run's prices depend on which group's cache happened to be
downloaded first and on `DataConfig.groups`' list order — neither is a
property of the trading logic G exists to measure. This wiring landing before
G is not only about staleness; it is what makes G's own baseline reproducible
at all for any multi-group config.

**Measure the unification.** Whatever value wins, at least three paths change.
Use `B_baseline.txt` per commit. The flag commits are expected to be neutral and
provably so; a moved row there means something was misunderstood, not that a
preparation difference was fixed.

## Per-commit baseline checks

Every result-changing commit in this track reports its own `B_baseline.txt` diff
with an explanation, as F2's C3 did. Not one combined diff at the end.

This track and Track J are both potentially result-changing and both land before
G. If their changes arrive as one undifferentiated shift, a contaminated G
comparison cannot be attributed to a cause. Per-commit attribution is what makes
that recoverable.

## Out of scope

- Any change to the residual, hedge, or z-score estimation stages.
- Track C's snapshot mechanism itself. Only the flag it passes, and the wiring of
  its consumers (see Scope).
- The `_make_sqrt_w` position-weighting defect in `found.md`. Adjacent — also about
  gaps — but a different stage and a different fix.

## Acceptance

- One source of truth for each flag, no class defaults, no hardcoded literals.
- A test that omitting either flag raises rather than defaulting.
- `B_baseline.txt` before and after, with the difference explained. If it does not
  move, state explicitly whether that means the change is neutral or that the harness
  universes cannot exercise it.
- The `_merge_umds` order dependence gone: the harness produces an identical
  `B_baseline.txt` with the group list reversed.
- A test that a mint or download refuses to write a bar for a session still in
  progress.
- Full suite green.

## Note

The unresolved question underneath both flags is what the pipeline should do with
missing data generally. Track B decided it for the hedge fit — pairwise dropna, no
forward fill, because manufactured zero returns bias kappa upward. The same reasoning
should govern here, and if it does, this track is partly a matter of extending an
already-made decision upstream rather than making a new one.
