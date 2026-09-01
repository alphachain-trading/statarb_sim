# Found out of scope

Anything wrong discovered while working a track, that is **not** in that track's
scope, goes here. It does not get fixed in that branch.

This file is what stops the snowball reforming: it makes discovery cheap and action
deliberate.

Format: one entry per item.

```
## {short title}
Found during: track {X}
Location: path:line
What: one or two sentences.
Severity: blocking / result-affecting / cosmetic
Suggested track: {X} or "new"
```

---

## Dead `n_legs=2` hardcode in `build_spread_returns`
Found during: track A
Location: `src/residuals/spreads.py:289`
What: A second hardcode of `n_legs=2`, in a function with zero callers repo-wide.
Unrelated to the live diagnostics path Track A fixed, and dead, so left alone. Delete
it if `build_spread_returns` is ever revived or removed.
Severity: cosmetic
Suggested track: new

## `config_hash` changed by `candidate_max_age_days` deletion
Found during: track A
Location: `simulation_persistence.py:32-44` (`hash_config`, hashes the full
serialized `SimulatorConfig`)
What: Deleting the unused `ActivationConfig.candidate_max_age_days` field changed
`config_hash`. Harmless in `statarb_sim`, which has no persisted run dirs, but it
will orphan existing persisted runs when Track A is ported.
Severity: result-affecting on port only
Suggested track: port note — hierarchical-arb

## Residual fit weights by row position, not date distance
Found during: track B (spec session, Q8)
Location: `_make_sqrt_w` in the residual fit
What: Exponential weights in the residual fit are computed from post-dropna row
position rather than from date distance to the fit date. Same defect class as the
requirement Track G carries for the hedge fit, one stage earlier. It matters more
here than it looks: the residual fit is expanding, so gaps accumulate over the whole
history, and weighting by position after dropping pulls old observations forward at
exactly the point where the decay is meant to be discounting them.

Whether it bites in practice depends on how gappy the input is. Track B's new
per-pair drop logging answers that for free — check it before deciding severity.

Partial answer from Track B's own baseline harness: energy_only_v1 and
materials_only_v1 (the two universes the harness runs against) have no internal
gaps over the harness's date range — commits 2 and 3 (pairwise dropna, min_obs)
both produced a zero-row diff on these two universes for exactly that reason
(`B_baseline.txt` commit messages). So position-based weighting in the residual
fit may bite rarely in practice, at least for these two sectors over this range
— not evidence it's rare everywhere; a sector or range with real data gaps could
still show it. Track B's per-pair drop logging (`pairwise_dropna` DEBUG log,
`pair_candidate_panel_creator.py`) is the instrument to check a wider sample
against before deciding severity.

Out of scope for B, which does not touch the residual fit's window handling.
Severity: result-affecting, magnitude unknown
Suggested track: new — sequence after B, since B's logging sizes the problem