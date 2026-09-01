# Found — out of scope for the track that found it

## Track A

- **`src/residuals/spreads.py:289`** — `build_spread_returns` hardcodes
  `"n_legs": 2` in its `meta_rows` entry, the same failure class as the
  `n_legs` hardcode Track A fixed in
  `pair_candidate_panel_creator.py:310`. Different function, and this one
  has zero callers repo-wide (confirmed by grep), so deleting or fixing it
  is out of scope here. Track A only fixed the live path
  (`_fast_pair_diagnostics` / `_build_pair_candidate_rows_for_date`).

## Port notes

- **Deleting `ActivationConfig.candidate_max_age_days` changed `config_hash`.**
  `hash_config()` (`src/simulator/simulation_persistence.py:32-44`) hashes the
  full serialized `SimulatorConfig`, which includes `ActivationConfig`. Removing
  the field changes the serialized dict for every config, hence the hash.
  Harmless in `statarb_sim` — no persisted run dirs keyed by hash exist yet —
  but when this fix is ported to `hierarchical-arb`, it will orphan any
  existing persisted run directories keyed by the old `config_hash`. Port as a
  separate atomic commit and flag the hash change to whoever owns those run
  dirs before landing it.
