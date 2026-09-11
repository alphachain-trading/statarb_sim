# `z_spectra` sample — exported before deletion (Track F1 commit 2)

`z_spectra_*.parquet` and its capture module (`src/simulator/z_spectrum_capture.py`,
gated by `SimulatorConfig.spectrum: SpectrumConfig | None`) are being deleted as part
of Track F1's artifact-layer dissolution. Per `F1_artifact_schema.md`, it is "residue
from an entry-selection experiment that returned null, ported from `hierarchical-arb`
unnoticed" — not referenced by any notebook or test in this repo (confirmed by grep,
including `notebooks/**/*.ipynb`), and not in the new `simrun_dir` layout. This note
and the accompanying sample exist so a later write-up of that null result has the
data, not just the (now-deleted) code path.

## What the artifact is

For each opened (and, by default, closed) trade, `ZSpectrumCapture` recomputes the
z-score the trade *would* have shown at every `(residual half-life × z-score
lookback)` combination in a grid — not just the one combination actually used to
enter/exit — and persists the whole grid alongside the actual trading z-score. The
idea being tested: does the z-score computed at a *different* (rhl, zlb) pair than
the one that triggered entry carry additional predictive signal (e.g. an
early-warning or confirmation cross-check)? The module's own doc comment records an
"identity guarantee" self-check (the trading z-score must reconstruct exactly as the
weighted sum of the matching spectrum cell) — a numerical sanity check on the
capture, not a claim about the experiment's result.

## Sample

`z_spectra_sample_mat__exp_mh20_rf__entry.parquet` — one entry-spectrum file from a
run of the exact Track B/F1 baseline harness config (`scripts/b_baseline_harness.py`:
`energy`+`materials` groups, `sp500_v1` universe, `exp_mh20_rf` residual
(`EQ_EXPANDING`, `min_lb_eq_exp=20`, `subtract_risk_free=True`), `entry_z=1.75`,
`exit_z=0.0`, `max_steps=40`, panel start `2006-09-08`) with
`spectrum=SpectrumConfig()` (defaults: `zlb_values=None` → the module's own
geometric-spaced default grid of 35 lookbacks from 5 to 252 days, `record_exit=True`,
`identity_check_epsilon=1e-4`) and `persistence.enabled=True` added. Run produced the
same 97 trades as the committed `B_baseline.txt` (spectrum capture is a pure
side-channel; it does not touch the trading path), no identity-check warnings fired.

Shape: 76 rows × 35 columns. Index: `(spread_id, date)` — one row per `materials`
candidate entered under the `exp_mh20_rf` residual config, one column per zlb value
in the sweep grid (`5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 21, 22, 25, 28, 32, 35,
40, 45, 50, 56, 63, 71, 80, 89, 100, 112, 126, 142, 159, 178, 200, 225, 252`). Cell
value: the z-score that spread would have shown at that lookback, at that entry date,
under an EWM z-score with the trading config's own `(method, ddof)` at the trading
`rhl` (here the only rhl in the grid, since `residual_lookbacks=None` resolves to the
run's own trading residual half-lives).

Sibling files from the same run (`z_spectra_nrg__exp_mh20_rf__{entry,exit}.parquet`,
`z_spectra_mat__exp_mh20_rf__exit.parquet`) were not exported — same shape and
provenance, only the group/entry-vs-exit axis differs. Regenerate any of them by
rerunning the harness config above with `spectrum=SpectrumConfig()` and
`persistence.enabled=True`, against `git show <pre-deletion-commit>:src/simulator/z_spectrum_capture.py`
if the module itself is needed again.
