# Track A — spec (delta against the brief)

Read-only verification pass over `docs/refactor/A_diagnostics_and_guards.md`. No code
changed. `statarb_sim_refactor_handover.md` was not consulted.

Amended after review. Amendments are marked **[amended]** and supersede the
verification session's original conclusions.

## Scope note: two implementations of the candidate row builder

There are two structurally near-identical diagnostics functions:

- `_fast_pair_diagnostics` in `src/candidates/pair_candidate_panel_creator.py:334-433`
  — called from `_build_pair_candidate_rows_for_date`
  (`pair_candidate_panel_creator.py:281`), which is the **live** path: reached from
  `create_pair_candidate_panel` → `panel_batch.py` → `run_panel_batch`.
- `compute_candidate_diagnostics` in `src/candidates/candidate_panel.py:109-223` —
  imported into `pair_candidate_panel_creator.py:18` but **never called** anywhere in
  the repo (confirmed by repo-wide grep). Dead code.

All nine claims below were verified against the **live** function
(`_fast_pair_diagnostics` + its caller's row construction).

**[amended] `compute_candidate_diagnostics` is deleted in this branch.** It is dead
*and* already free of two of the defects A fixes (claims 6 and 7), which makes it a
trap: anyone later grepping for the intercept fix finds it in the wrong function and
concludes the fix landed. Same failure class as `candidate_max_age_days` — code that
looks live and isn't. Remove its import at `pair_candidate_panel_creator.py:18` in the
same commit.

Before deleting, confirm it is not re-exported as public surface from
`src/candidates/__init__.py`. If it is, the deletion is still correct but the export
must go with it.

A third, unrelated hardcode of `n_legs=2` exists in `src/residuals/spreads.py:289`
(`build_spread_returns`), which also has zero callers. **Out of scope** — different
function, and deleting it is not part of this track. Record it in `found.md` instead.

## Verify first — results

**1. CONFIRMED.** `adf_pvalue` is computed last, only on the path that has already
passed every validity gate, and does not feed `is_valid`.
`pair_candidate_panel_creator.py:415-420`:
```python
    adf_pvalue = np.nan
    if not skip_adf and adfuller is not None:
        try:
```
This sits after the last gate (`half_life_gt_max`, `:410-411`) and before the `return`
at `:422`, which already has `"is_valid": True` fixed. "Read by nothing in current
production configuration" is the corollary of claim 3 — no config sets
`adf_pvalue_max`, so the selector's mask never reads the column.

**2. CONFIRMED.** `pair_candidate_panel_creator.py:81` — `skip_adf: bool = False` on
`PairSpreadConfig`, consumed at `:290`, read at `:416`.

**3. CONFIRMED.** `candidate_selector.py:69` — `adf_pvalue_max: float | None = None`.
Mask application at `:258-260`:
```python
    if cfg.adf_pvalue_max is not None:
        mask &= panel["adf_pvalue"].notna()
        mask &= panel["adf_pvalue"] <= cfg.adf_pvalue_max
```
Repo-wide grep finds only the definition, its `__post_init__` range check (`:76-78`),
and this block. No config, sweep, or notebook sets it.

**4. CONFIRMED, but the prescribed fix is wrong — see below.**
`pair_candidate_panel_creator.py:393-394`:
```python
    residual_std = float(np.std(resid, ddof=1))
    kappa = float(-slope)
```
Gate order at `:396-399`:
```python
    if not np.isfinite(kappa):
        return invalid("kappa_not_finite")
    if not np.isfinite(residual_std):
```

**[amended] Do not swap the gates.** The brief's rationale — report the upstream
failure rather than a misattribution — does not hold. `residual_std` and `kappa` are
**siblings**, both derived from the same regression: `residual_std = std(resid)` and
`kappa = -slope`. Neither is upstream of the other, so swapping the order changes which
of two equally-arbitrary labels is reported and fixes nothing. The verification session
reached the same conclusion from the other direction, noting the misattribution is not
actually reachable today.

Replace the swap with a **single combined gate that reports what was actually
non-finite**: check both, and emit a reason naming whichever failed, or both. This
removes the ordering question rather than relitigating it, and keeps `why_invalid`
truthful — which is the point of the gate work.

Keep the reason strings static per the gate-4 rule below: `kappa_not_finite`,
`residual_std_not_finite`, `kappa_and_residual_std_not_finite`. Three fixed values, no
interpolation.

**5. CONFIRMED.** `pair_candidate_panel_creator.py:384-385`:
```python
    if len(dx) < 10:
        return invalid(f"dx_length_{len(dx)}_lt_10")
```
Dynamic — embeds `len(dx)`. Normalize to a static `dx_too_short`.

Note that three further reasons are also dynamic and were not in the brief's claim:
`kappa_lt_min:...`, `half_life_le_0:...`, `half_life_gt_max:...`. **[amended] Normalize
these too.** The rule is category-wide, not specific to gate 4: values never go in
reason strings — the offending value is already in its own typed column and the
threshold is in the run config. Leaving three of four dynamic defeats the purpose,
since any groupby on `why_invalid` still shatters.

**6. CONFIRMED, with a semantic correction — see below.** Row construction hardcodes
`n_legs` at `pair_candidate_panel_creator.py:310`, while `_fast_pair_diagnostics`
computes a real value at `:356`:
```python
    n_legs = int(abs(w_left) >= tiny_weight_threshold) + int(abs(w_right) >= tiny_weight_threshold)
```
`diagnostics["n_legs"]` is never read when building the row.

**[amended] `n_legs` is not a leg count — it is a count of legs whose absolute weight
clears `tiny_weight_threshold`.** It can therefore be 0 or 1 for a two-ticker spread.
Those rows fail the `too_few_active_legs` gate and are marked invalid, but per Q12 they
**still appear in the table**, currently showing `n_legs=2` when the true value is 1 or
0.

So wiring it through changes stored values on invalid rows. Still result-neutral —
invalid rows are never traded — but it is an artifact change, not a pure cleanup. Say so
in the commit message.

**7. CONFIRMED.** `pair_candidate_panel_creator.py:389` — `intercept, slope = beta`;
`intercept` never referenced again, absent from the row dict (`:301-323`). The dead
`compute_candidate_diagnostics` already persists it (`candidate_panel.py:221-222`), so
the fix pattern exists in the sibling being deleted — copy it before deleting.

**8. CONFIRMED.** The diagnostics dict always carries `failure_reason` (the `invalid()`
closure at `:361`, and `:424` on the valid path). Row promotion is debug-gated at
`:325-326`.

**9. CONFIRMED.** `src/simulator/config.py:295` —
`candidate_max_age_days: int | None = None` on `ActivationConfig`. Sole hit repo-wide.
Never read.

No claim required a `NOT FOUND`.

## 10. Guard placement

**Guard 1 — `adf_pvalue_max` with an all-NaN column. Point of use.**
Inside `_build_candidate_selection_mask` (`candidate_selector.py:243-276`), at the
`:258-260` block. Config validation cannot know whether the panel's `adf_pvalue` column
is all-NaN — that depends on whether the panel was built with `skip_adf=True`, which
`CandidateSelectionConfig` has no visibility into. Both the config flag and the panel
column are already in scope at `:258`.

**Guard 2 — `EQ_ROLLING` residual config. `SimulatorConfig.__post_init__`.**

**[amended]** The verification session proposed `CausalResidualConfig.__post_init__`.
That placement is wrong. The defect requires **both** an `EQ_ROLLING` residual config
**and** a z-score computation: the z-score's EWM history span is inherited from
whichever path supplied the series, and a rolling residual config truncates that
series. Raising in the residual config's `__post_init__` fires on the first condition
alone, blocking residual-only work — panel construction, residual diagnostics, anything
that never reaches a z-score — which is correct today.

`EQ_ROLLING` must remain constructible. Raise in `SimulatorConfig.__post_init__`
(`config.py:849-880`), where both conditions are visible, alongside guard 3's config
half.

Sanity check: if any existing test constructs an `EQ_ROLLING` config, a
`CausalResidualConfig.__post_init__` raise breaks the suite. That is the signal, not an
argument.

**Guard 3 — multi-sleeve occupancy. Two parts, two entry points.**

- **Multiple z-lookbacks under one `residual_key` — config validation.**
  `SimulatorConfig.__post_init__` (`config.py:849-880`) already de-duplicates on
  `timescale_label` (`:875-880`) and has the grouping helper
  `z_score_configs_by_rkey()` (`:903-907`). Pure `SimulatorConfig` state.
- **Multiple hedge configs contending under one `(spread_id, residual_key)` — point of
  use.** `SimulatorConfig` carries no list of hedge configs; this is a property of the
  loaded panel. Raise in `_load_panels` (`simulator_factory.py:293-376`) right after
  the merge at `:361`, grouping by `(spread_id, residual_key, weight_model)`. A
  comparable groupby already exists for logging at `:363`.

## 11. DEFAULT_CONFIGS and howto-notebook audit — zero hits

- **`adf_pvalue_max`**: never set. `DEFAULT_CONFIGS["standard_v1"]`
  (`sweep_defaults.py:41-43`) leaves it at `None`. No notebook sets it.
- **`EQ_ROLLING`**: no active use. `DEFAULT_CONFIGS` carries no residual-shaped field
  by design (`sweep_defaults.py:4-9`). Notebook 01 commits
  `RESIDUAL_MODE = ResidualMode.DECAY_EXPANDING`; `EQ_ROLLING` appears only in a comment
  and an unselected `match` branch. Notebooks 02/03 consume a built panel by
  `residual_key` string.
- **Multiple z-lookbacks or hedge configs per run**: never. Notebooks 02/03 each
  construct one `ZScoreConfig` with a scalar `lookback`. Notebook 01 uses
  `PanelBatchConfig`'s default `hedge_ratio_methods=["pca"]` (`panel_batch.py:129`).

**Consequence: track H stays deferred.** No config in use trips guard 3. Delete the
placeholder entry from `found.md` rather than filling it in.

## 12. Two distinct absence classes in the candidates table

The brief's "early gate" was ambiguous between two mechanisms.

**Structural pre-checks skip the pair entirely — no row.**
- `:261-264` — either leg missing from the cleaned hedge or diagnostics window.
- `:266-274` — weight computation raises (`ValueError` / `LinAlgError`, e.g. a
  degenerate PCA eigenvector).
- Implicitly, `pair_ids` (`:250-253`) is built only from tickers surviving the
  hedge-window dropna (`:240`), so a dropped ticker never enters the loop for that date.

**Diagnostic validity gates still produce a row.** The row is appended unconditionally
at `:329`, outside any validity check, and every failure branch returns a same-shaped
dict via the `invalid()` closure (`:358-370`), unpacked into the row exactly like the
valid case with `is_valid=False` and NaN diagnostics.

So the table has **two absence classes**: rows marked invalid with a reason, and pairs
with no record that they were ever considered on that date.

## Consequences for other tracks

Recorded here because they were found here; the corresponding briefs are amended
separately.

**Track B — the coverage hole is wider than the brief states.** `why_invalid` covers
gate failures only. Structurally skipped pairs have no row and therefore no reason, so
B's ticker-drop logging captures only part of the second class — the exception path at
`:266-274` is not a ticker drop at all. Any breadth or rejection-rate denominator needs
both counted. B must log pre-check skips separately from ticker drops.

**Track F — `n_legs` cannot be dropped as a groupby size.** Earlier advice said it
could, on the assumption it counts legs. Per claim 6 it counts legs *above
`tiny_weight_threshold`*, so it disagrees with `weights.parquet` row count whenever a
tiny weight is stored. Either keep `n_legs` as a real column with its threshold
semantics documented, or store only active legs in `weights.parquet` and accept that
the artifact then cannot reproduce the raw fit. Decide in F; do not drop it silently.

## two corrections applied

Both incorporated above: `compute_candidate_diagnostics` deleted, guard 2 placed in
`SimulatorConfig.__post_init__`.

## Implementation verification

- Full suite green: `python -m unittest discover -s tests` — 75 tests, OK (11 new in
  `tests/test_track_a_guards.py`: one raise + one non-firing case per guard, plus
  n_legs/gate consistency at 0/1/2 active legs).
- Guard audit against `DEFAULT_CONFIGS["standard_v1"]` and all three howto notebooks:
  silent on all of them, confirming section 11's static findings hold against the live
  guard code, not just the pre-implementation analysis.
  - Guard 1 (`adf_pvalue_max` vs all-NaN column): structurally cannot fire — no config
    or notebook sets `adf_pvalue_max`.
  - Guard 2 (`EQ_ROLLING` + z-score): `residual=` is never passed into `SimulatorConfig`
    by `sweep_runner._build_sim_config` or notebook 02 — it stays `None`, so the guard
    can't reach a `window_mode == "rolling"` check regardless of `ResidualMode`.
    Notebook 01 builds panels via `PanelBatchConfig`, never a `SimulatorConfig`, and
    commits `RESIDUAL_MODE = ResidualMode.DECAY_EXPANDING`.
  - Guard 3 (multi-sleeve occupancy): verified directly by constructing a
    notebook-02-shaped `SimulatorConfig` and a notebook-03-shaped `SweepConfig` — both
    single-timescale, single-residual_key, no raise.
  - None fire on anything in use today; Track H stays deferred (no `found.md` entry
    needed).
- Re-timed panel build (materials sector, `DECAY_EXPANDING hl=504/mh1008/rf`, PCA
  hedge ratio, `hedge_ratio_lb=mr_diag_lb=252`, `W-FRI`, 400 weekly steps -> 31200
  rows), before/after `skip_adf`:
  - `skip_adf=False`: **96.0s**
  - `skip_adf=True`: **25.4s**

  Matches the brief's cited baseline (~96s) and expectation (~27s) almost exactly —
  this was very likely the original benchmark's configuration.
