# Found — out of scope for the track that found it

## Track A

- **`src/residuals/spreads.py:289`** — `build_spread_returns` hardcodes
  `"n_legs": 2` in its `meta_rows` entry, the same failure class as the
  `n_legs` hardcode Track A fixed in
  `pair_candidate_panel_creator.py:310`. Different function, and this one
  has zero callers repo-wide (confirmed by grep), so deleting or fixing it
  is out of scope here. Track A only fixed the live path
  (`_fast_pair_diagnostics` / `_build_pair_candidate_rows_for_date`).

- **`src/candidates/panel_batch.py:127-131`** — `PanelBatchConfig.pair_cfg`'s
  `default_factory` explicitly pins `skip_adf=False`, overriding the new
  `PairSpreadConfig.skip_adf=True` default (see A_diagnostics_and_guards.md).
  Anyone building a panel via `PanelBatchConfig` (e.g. howto notebook 01)
  keeps paying the ADF cost by default; only direct `PairSpreadConfig()`
  construction picks up the speedup. Not fixed here — Track A's scope is
  the candidate row builder and the config validation path, not
  `PanelBatchConfig`.
