# Track A — spec (delta against the brief)

Read-only verification pass over `docs/refactor/A_diagnostics_and_guards.md`. No code
changed. `statarb_sim_refactor_handover.md` was not consulted.

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
(`_fast_pair_diagnostics` + its caller's row construction). Where the dead function
disagrees, it is noted — this matters because it shows the defects are not structural
to the diagnostics logic itself, only to the live copy and its caller.

A third, unrelated hardcode of `n_legs=2` exists in `src/residuals/spreads.py:289`
(`build_spread_returns`), which also has zero callers anywhere in the repo. Out of
scope for the nine claims (different function, also dead) but noted so it isn't
mistaken for a second live call site.

## Verify first — results

**1. CONFIRMED.** `adf_pvalue` is computed last, only on the path that has already
passed every validity gate, and does not feed `is_valid`.
`pair_candidate_panel_creator.py:415-420`:
```python
    adf_pvalue = np.nan
    if not skip_adf and adfuller is not None:
        try:
```
This sits after the last gate (`half_life_gt_max`, line 410-411) and before the
`return` at line 422, which already has `"is_valid": True` fixed. "Read by nothing in
current production configuration" is the corollary of claim 3 (below) — no config sets
`adf_pvalue_max`, so `candidate_selector.py`'s mask never reads the column.

**2. CONFIRMED.** `pair_candidate_panel_creator.py:81`:
```python
    skip_adf: bool = False
```
On `PairSpreadConfig`, consumed at `pair_candidate_panel_creator.py:290` and read at
`:416`. (The dead `compute_candidate_diagnostics` also has its own `skip_adf` param,
`candidate_panel.py:119`.)

**3. CONFIRMED.** `candidate_selector.py:69`:
```python
    adf_pvalue_max: float | None = None
```
Mask application, `candidate_selector.py:258-260`:
```python
    if cfg.adf_pvalue_max is not None:
        mask &= panel["adf_pvalue"].notna()
        mask &= panel["adf_pvalue"] <= cfg.adf_pvalue_max
```
Repo-wide grep for `adf_pvalue_max` finds only its definition, its `__post_init__`
range check (`:76-78`), and this mask block — no config, sweep, or notebook sets it.
Confirmed empty also across `DEFAULT_CONFIGS` and all three howto notebooks (see Q11).

**4. CONFIRMED.** `pair_candidate_panel_creator.py:393-394`:
```python
    residual_std = float(np.std(resid, ddof=1))
    kappa = float(-slope)
```
Gate order, `:396-399`:
```python
    if not np.isfinite(kappa):
        return invalid("kappa_not_finite")
    if not np.isfinite(residual_std):
```
`residual_std` is assigned one line before `kappa`, but `kappa`'s finiteness gate runs
first — so a `kappa`-driven failure cannot be misattributed to `residual_std`, but a
`residual_std`-driven failure currently can never surface either, since `kappa`'s gate
runs first and both derive from the same regression; not actually reachable as a
distinct misattribution today, but the ordering is exactly as claimed.

**5. CONFIRMED.** `pair_candidate_panel_creator.py:384-385`:
```python
    if len(dx) < 10:
        return invalid(f"dx_length_{len(dx)}_lt_10")
```
Dynamic — embeds `len(dx)`.

**6. CONFIRMED.** Row construction hardcodes `n_legs`, `pair_candidate_panel_creator.py:310`:
```python
                "n_legs": 2,
```
while `_fast_pair_diagnostics` computes and returns a real value,
`pair_candidate_panel_creator.py:356` (assignment) and `:369`/`:432` (returned):
```python
    n_legs = int(abs(w_left) >= tiny_weight_threshold) + int(abs(w_right) >= tiny_weight_threshold)
```
`diagnostics["n_legs"]` is never read when building the row (row dict at
`:301-323` pulls `adf_pvalue`, `mr_score`, `kappa`, `half_life`, `residual_std`,
`spread_return_std`, `level_std`, `is_valid` from `diagnostics`, but not `n_legs`).
Note: the dead `compute_candidate_diagnostics` has no equivalent hardcode — it has no
caller that could hardcode `n_legs` since it has no caller at all.

**7. CONFIRMED.** `pair_candidate_panel_creator.py:389`:
```python
    intercept, slope = beta
```
`intercept` is never referenced again in the function, and the row dict (`:301-323`)
does not include it. **Disagreement**: the dead `compute_candidate_diagnostics`
already persists it, `candidate_panel.py:221-222`:
```python
        "intercept": float(intercept),
        "slope": float(slope),
```
so the fix pattern already exists in the unused sibling function.

**8. CONFIRMED.** Diagnostics dict always carries the field
(`pair_candidate_panel_creator.py:361` in the `invalid()` closure and `:424` on the
valid path). Row promotion is debug-gated, `:325-326`:
```python
            if debug:
                row["failure_reason"] = str(diagnostics["failure_reason"])
```

**9. CONFIRMED.** `src/simulator/config.py:295`:
```python
    candidate_max_age_days: int | None = None
```
Sole hit for `candidate_max_age_days` in a repo-wide grep (`.py` files) — defined on
`ActivationConfig`, never read anywhere else.

No claim required a "NOT FOUND" — all nine matched the live code as stated, modulo the
call-site note on claims 6 and 7.

## 10. Guard placement and existing validation entry points

**Guard 1 — `adf_pvalue_max` with an all-NaN column.**
Existing entry points: `CandidateSelectionConfig.__post_init__`
(`candidate_selector.py:76-78`, range check only — `0 < x <= 1`) and
`_build_candidate_selection_mask` (`candidate_selector.py:243-276`, panel-level
application at `:258-260`).
Correct place to raise: **point of use**, inside `_build_candidate_selection_mask` (or
`select_candidates` immediately before it) right where `:258-260` runs. Config
validation alone cannot know whether the panel's `adf_pvalue` column is all-NaN —
that depends on whether the panel was built with `skip_adf=True`, a fact
`CandidateSelectionConfig` has no visibility into. The check needs both the config
(`cfg.adf_pvalue_max is not None`) and the actual panel column, both already in scope
at `:258`.

**Guard 2 — `EQ_ROLLING` residual config.**
Existing entry point: `CausalResidualConfig.__post_init__`
(`causal_residuals.py:70-99`), specifically the `case ResidualMode.EQ_ROLLING:` branch
(`:74-79`).
Correct place to raise: **config validation**, in that same branch. This is a pure
property of the mode — no panel or run data is needed to know that `EQ_ROLLING` was
selected — and `__post_init__` is already the single place `PanelBatchConfig`'s
docstring (`panel_batch.py:104-108`) says all residual mode/validation logic lives.

**Guard 3 — multi-sleeve occupancy.**
This guard has two independent parts that need two different entry points, since the
brief's "z-lookback" half is config-only and the "hedge config" half is data-dependent.

- Multiple z-lookbacks under one `residual_key`: existing entry point
  `SimulatorConfig.__post_init__` (`config.py:849-880`), which already de-duplicates on
  `timescale_label` (`:875-880`) and already has the grouping helper needed,
  `z_score_configs_by_rkey()` (`config.py:903-907`). Correct place to raise:
  **config validation**, extending `__post_init__` — this is pure `SimulatorConfig`
  state, no panel needed.
- Multiple hedge configs (`weight_model`) contending under one `(spread_id,
  residual_key)`: this is a property of the *loaded candidate panel*, not of any single
  config object — `SimulatorConfig` carries no list of hedge configs. Existing entry
  point: `_load_panels` (`simulator_factory.py:293-376`), which already assembles the
  merged panel with a `residual_key` column and already computes a comparable groupby
  for logging (`group_counts = merged.groupby(["group_id", "residual_key"]).size()`,
  `:363`). Correct place to raise: **point of use**, right after the merge
  (`:361`), grouping by `(spread_id, residual_key, weight_model)` instead.

## 11. DEFAULT_CONFIGS and howto-notebook audit

**`adf_pvalue_max`**: zero hits. `DEFAULT_CONFIGS["standard_v1"]`
(`sweep_defaults.py:41-43`) sets `CandidateSelectionConfig(allowed_candidate_subtypes=("pca",),
require_is_valid=True)` — `adf_pvalue_max` left at its `None` default. No notebook sets
it (grep for `adf_pvalue_max` across all three `.ipynb` files: no matches).

**`EQ_ROLLING`**: zero *active* hits. `DEFAULT_CONFIGS` carries no residual-shaped
field at all (by design — its docstring says so, `sweep_defaults.py:4-9`). Notebook 01
(`01_create_candidate_panel.ipynb`, cell 5) sets
`RESIDUAL_MODE = ResidualMode.DECAY_EXPANDING` as the committed value, with `EQ_ROLLING`
offered only as an inline comment ("try: EQ_ROLLING, EQ_EXPANDING") and as one branch
of a `match` statement (cell 6) that is not selected. As committed, no notebook runs
`EQ_ROLLING`. Notebooks 02/03 don't touch residual mode at all — they consume an
already-built panel by `residual_key` string.

**Multiple z-lookbacks per `residual_key` / multiple hedge configs in one run**: zero
hits. `DEFAULT_CONFIGS` carries no z-score or hedge-related field. Notebook 02 and 03
each construct exactly one `ZScoreConfig` (`ZScoreConfig(lookback=Z_LOOKBACK,
method=Z_METHOD, residual_key=RESIDUAL_KEY)`, one scalar `lookback`). Notebook 01 uses
`PanelBatchConfig`'s default `pair_cfg` (`hedge_ratio_methods=["pca"]`,
`panel_batch.py:129`), not overridden — a single hedge method. No notebook or
`DEFAULT_CONFIGS` entry constructs a `z_score` list or multiple `hedge_ratio_methods`.

**Consequence for `found.md`**: the guard-3 placeholder entry
("Occupancy guard fired on a config in use") should be deleted, not filled in — the
guard does not fire on anything actually in use today. This session did not edit
`found.md`; the implementation session should remove the placeholder once guard 3 is
built and confirmed silent against these configs.

## 12. Can an early-gate failure still produce a candidates row?

Both answers are correct depending on which gate — the brief's "early gate" is
ambiguous between two different mechanisms in `_build_pair_candidate_rows_for_date`:

**Structural pre-checks skip the pair entirely — no row.** Three points in the loop
`continue` without appending anything:
- `pair_candidate_panel_creator.py:261-264` — either leg missing from the cleaned
  hedge or diagnostics window:
  ```python
              if left not in hedge_cols or right not in hedge_cols:
                  continue
  ```
- `pair_candidate_panel_creator.py:266-274` — weight computation raises
  (`ValueError`/`LinAlgError`, e.g. degenerate PCA eigenvector):
  ```python
              except (ValueError, np.linalg.LinAlgError):
                  continue
  ```
- Implicitly, `pair_ids` (`:250-253`) is built only from tickers surviving the
  hedge-window dropna (`:240`), so a ticker dropped for NaNs never enters the loop for
  that date at all.

**Diagnostic validity gates inside `_fast_pair_diagnostics` still produce a row.**
The row is appended unconditionally at `pair_candidate_panel_creator.py:329`
(`rows.append(row)`), outside any validity check, and every failure branch of
`_fast_pair_diagnostics` (`too_few_active_legs`, `spread_return_std_too_small`,
`spread_level_std_too_small`, `dx_length_..._lt_10`, `kappa_not_finite`,
`residual_std_not_finite`, `residual_std_le_0`, `kappa_lt_min:...`,
`half_life_not_finite`, `half_life_le_0:...`, `half_life_gt_max:...`) returns a
same-shaped dict via the `invalid()` closure (`:358-370`), which is then unpacked into
the row exactly like the valid case — just with `is_valid=False` and NaN diagnostic
fields. So: a pair that fails one of these gates **does** get a candidates row, marked
invalid; a pair that fails a structural precondition (missing from the cleaned window,
or a weight-computation exception) **does not** — the loop skips it and the panel has
no record it was ever considered on that date.

## Things the brief asks for that this session could not do

Nothing in the "Verify first" or Q10-12 sections was unconfirmable — no `NOT FOUND`,
no contradiction between two live call sites (only a live-vs-dead-code difference, on
claims 6 and 7, noted above since it's relevant to how the fix should read: the dead
`compute_candidate_diagnostics` already does the right thing for claim 7, and never
had the claim-6 defect in the first place since it has no caller).
