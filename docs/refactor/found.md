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

## ExecutionConfig cost-field values are unreviewed
Found during: track D
Location: `src/simulator/config.py` (`ExecutionConfig`), values pinned in
`src/simulator/sweep_defaults.py`'s `_STANDARD_V1["execution"]` and
`run_me.py`'s `ExecutionConfig(...)` call.
What: Track D relocated `commission_per_share` (0.005), `commission_per_order`
(0.0), `min_commission_per_order` (1.0), `max_commission_per_order` (9.79),
`max_commission_pct_of_trade` (0.01), `short_borrow_rate_annual_bps` (30.0),
and `min_abs_units` (0.5) from class defaults into explicit call sites/bundle
entries. D only relocates; it does not validate. These are the numbers that
feed directly into reported PnL (commissions and borrow), and nobody has
reviewed whether they are still the right numbers — only that they are now
visible and version-tracked rather than implicit.
Severity: result-affecting (values, not mechanism)
Suggested track: new — a research review of the cost model, after D

## demo_materials.yaml does not expose ExecutionConfig's cost fields
Found during: track D
Location: `config/demo_materials.yaml`; `run_me.py`'s `ExecutionConfig(...)`
construction (`_build_sim_config`)
What: run_me.py sources most SimulatorConfig fields from
`config/demo_materials.yaml` (risk manager, sizing, trader, run), but the
seven ExecutionConfig cost fields are now literals in run_me.py itself,
not YAML keys — same gap that already existed for these fields as invisible
class defaults, just relocated rather than closed. A reader of the public
demo cannot see the transaction cost assumptions the demo actually runs
under.
Severity: cosmetic (presentation only; run_me.py still runs under the same
values as before)
Suggested track: revisit after F, alongside the run_me.py config-sourcing
decision noted below

## run_me.py's fixture-bootstrap path is stale, and download is C's integration point
Found during: track D
Location: `run_me.py:366-388` (`_bootstrap_panel_from_fixtures`), `run_me.py:534-537`
(called from `stage_simulate`), `config/demo_materials.yaml:16-17`
What: `stage_download` (`run_me.py:102`) and `stage_residuals` (`run_me.py:316`)
both do real work today (real caching/download via `UniverseDataLoader.load`,
real panel building via `run_panel_batch`) — not fixture copies. But
`stage_simulate`'s single-sector path falls back to
`_bootstrap_panel_from_fixtures`, which copies a panel triple from
`cfg["panels"]["fixtures_dir"]` when the residuals stage hasn't produced one
yet. `demo_materials.yaml`'s own comment (`:17`) calls this "bootstrap source
until the residuals stage exists" — stale, since `stage_residuals` already
exists and is wired into `STAGES`/`main()`'s default `--stage all` order
(`run_me.py:562-588`). Worse: the committed fixture stem
(`fixtures/materials_v1/mat_pairs_pca_W-FRI_exp_hl504_20260530_1400.*`) no
longer matches `demo_materials.yaml`'s current `panels.stem`
(`..._20260713_1531`, `:15`) — so the fallback would not find its fixture file
if it were ever actually invoked against today's committed config. Currently
masked because a matching panel already exists locally under
`artifacts/candidate_panels/materials_v1/` from a prior real run.
Separately: `stage_download` is the natural integration point for Track C's
snapshot/manifest layer (C_market_snapshot.md) — it currently calls
`UniverseDataLoader.load` directly with no snapshot id or manifest, which is
exactly the gap C's scope describes.
Severity: cosmetic today (masked by a locally-built panel); would surface as
a confusing failure for a fresh clone with no `download`/`residuals` state
Suggested track: new (fixture staleness) — new for the comment; C for the
download-stage integration

## SimulatorConfig.risk_manager: RiskManagerConfig | None = None
Found during: track D
Location: `src/simulator/config.py:844` (field default); consumed at
`src/simulator/simulator_factory.py:107-110` (`if config.risk_manager is not
None: risk_manager = RiskManager(...)`, else `risk_manager = None`)
What: Result-affecting (an omitted risk_manager means portfolio gross/ticker
exposure runs uncapped) but not a numeric default, so D's "remove default,
pin value in bundle" mechanism does not apply -- there is no value to pin,
only a subsystem to require-or-not. This is a Track A style guard question
or an explicit-mode question, not a config-hygiene one, and it is the same
defect shape H will examine (a portfolio-level policy silently absent rather
than explicitly stated). Left as-is; RiskManagerConfig's own two numeric
fields (max_gross_exposure, max_ticker_exposure_pct) lost their class
defaults this track (D_spec.md Table 1 #10) -- only the None-sentinel on
SimulatorConfig itself is deferred.
Severity: result-affecting
Suggested track: new (Track A style guard) or H