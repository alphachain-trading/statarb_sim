# Track I — Data preparation flags

**Changes results:** likely yes. Unifying four combinations means at least three
call paths change behaviour.

**Depends on:** nothing structurally. **Must run before G.**

A new track, split out of F because it has nothing to do with the artifact layer. It
is data preparation, and it sits upstream of everything.

## The finding

`start_after_nan` and `check_for_corruptions` control how raw price data is cleaned
before anything else runs. There are **four combinations** in the repo, and **no
caller overrides any of them** — every path just takes whatever default it happens to
reach.

| Source | `check_for_corruptions` | `start_after_nan` |
|---|---|---|
| `DataConfig` (`config.py:241-264`) | `True` | `True` |
| `PanelBatchConfig` (`panel_batch.py:102,150-152`) | `False` | `True` |
| `market_snapshot.py:277` (hardcoded literal) | `True` | `False` |
| `UniverseDataLoader.load` signature (`universe_loader.py:36`) | `True` | `True` |

So the panel build and the simulator clean the same raw data differently, and Track
C's snapshot path introduced a third combination as a hardcoded literal.

This is the failure form Track D removed, one level more abstract: not one default
too many, but four silent answers to the same semantic question, selected by which
path the data happens to take. And because it is data preparation, it acts on every
candidate, every trade, and every result.

## Why before G

G sweeps six hedge configs and compares them. If the panel build and the simulator
prepare the underlying returns differently, the comparison is contaminated before it
starts — and the effect G is looking for is thin enough that a preparation artifact
could swamp it.

## Scope

This is behaviour analysis first, unification second. Do not unify before
understanding what each flag does.

**Establish the semantics.**

- What does `start_after_nan` actually trim, and at what grain — from the first
  valid observation per ticker, or per group? A per-group start date discards early
  history for every ticker in the group because one arrived late; a per-ticker start
  produces a ragged panel. These are very different, and Track C's harness fix
  already showed a single late ticker can dominate a group.
- What does `check_for_corruptions` check, what does it do on detection, and why is
  it off in the panel build? Off in exactly one place is either a deliberate
  performance choice or an accident nobody revisited.

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

**Measure the unification.** Whatever value wins, at least three paths change. Use
`B_baseline.txt`, and note the same caveat Track B's `found.md` entry raises: the two
harness universes have no internal gaps over the harness range, so they may not
exercise NaN handling at all. A universe or date range with real gaps is needed to
see the effect. Report the count of trimmed observations per group before and after.

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
- Trimmed-observation counts per group reported for a gappy universe, not only the
  harness ones.
- Full suite green.

## Note

The unresolved question underneath both flags is what the pipeline should do with
missing data generally. Track B decided it for the hedge fit — pairwise dropna, no
forward fill, because manufactured zero returns bias kappa upward. The same reasoning
should govern here, and if it does, this track is partly a matter of extending an
already-made decision upstream rather than making a new one.
