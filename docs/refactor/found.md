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

Track D removed many further class defaults without changing resolved values, so D
itself leaves `config_hash` intact — but the same porting hazard applies to any
future field deletion.
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

## `ZScoreConfig.ddof` was never chosen by anyone
Found during: track D (follow-up pass, commit 4)
Location: `ZScoreConfig` in `src/simulator/config.py`; consumed by the rolling
z-score standard deviation computation
What: `ddof` feeds the live rolling z-score std directly, and **no caller anywhere
in the repo had ever stated it** — unlike every other field this track touched,
where at least one call site made the value explicit. Every z-score in every run to
date therefore rests on a Bessel correction nobody selected.

The magnitude is small at production lookbacks: at `zlb=126` the difference between
`ddof=0` and `ddof=1` is a factor of `sqrt(126/125)`, roughly 0.4% on the standard
deviation. But it applies to the threshold comparison, so it acts multiplicatively
on every entry decision and systematically in one direction, and it grows as the
z-lookback shortens.

D relocated the value and made it explicit. It did not validate it. Which value is
correct is a statistical decision, not a hygiene one.
Severity: result-affecting (value, not mechanism)
Suggested track: new — research review, alongside the ExecutionConfig cost review

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

## Config classes with zero construction sites
Found during: track D (follow-up pass, commit 6)
Location: `SpreadMomentumConfig`, `KellyConfig`, `TimescaleRiskConfig.selection`,
`CrossTimescaleEntryConfig.same_sign_required` in `src/simulator/config.py`
What: Four config surfaces with no construction site anywhere in the repo. D removed
their defaults as trivial no-risk cleanup, but the deeper question is whether they
should exist at all.

`KellyConfig` in particular: Kelly sizing was tested in `hierarchical-arb` and
abandoned. A config class for an abandoned approach is the same failure shape as
`candidate_max_age_days` — surface that looks like a supported feature and is not.
`CrossTimescaleEntryConfig` likewise encodes cross-timescale gating, which the
research record lists as neutral-or-harmful.

Deleting them is cheap and they are recoverable from git history. Decide during F,
when the config surface is being reshaped anyway.
Severity: cosmetic
Suggested track: F, or new

## `allow_long` / `allow_short` defaults deliberately left in place
Found during: track D (follow-up pass)
Location: `PairSpreadTraderConfig`, `PortfolioMeanReversionConfig` in
`src/simulator/config.py`
What: Both default to `True` and no caller anywhere overrides either. Left as-is
because, unlike `ActivationConfig`'s inverted booleans, there is no live caller
disagreement proving the default wrong.

Recorded so the omission is visible rather than looking like an oversight. If the
rule is taken as absolute — no result-affecting defaults, full stop — these two are
the remaining exceptions.
Severity: cosmetic
Suggested track: none — documented decision, revisit only if the rule tightens

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

## `SimulatorConfig.risk_manager: RiskManagerConfig | None = None`
Found during: track D
Location: `src/simulator/config.py:844` (field default); consumed at
`src/simulator/simulator_factory.py:107-110` (`if config.risk_manager is not
None: risk_manager = RiskManager(...)`, else `risk_manager = None`)
What: Result-affecting (an omitted risk_manager means portfolio gross/ticker
exposure runs uncapped) but not a numeric default, so D's "remove default,
pin value in bundle" mechanism does not apply — there is no value to pin,
only a subsystem to require-or-not. This is a Track A style guard question
or an explicit-mode question, not a config-hygiene one, and it is the same
defect shape H will examine (a portfolio-level policy silently absent rather
than explicitly stated). Left as-is; RiskManagerConfig's own two numeric
fields (max_gross_exposure, max_ticker_exposure_pct) lost their class
defaults this track (D_spec.md Table 1 #10) — only the None-sentinel on
SimulatorConfig itself is deferred.
Severity: result-affecting
Suggested track: new (Track A style guard) or H

## `start_after_nan` / `check_for_corruptions` disagree between two config classes
Found during: track D (spec session, flagged "doubtful"; not covered by the
follow-up pass, which addressed the trader-side classes only)
Location: two config classes in `src/simulator/config.py` — identify both when
picking this up
What: The same field is pre-set to different values on two different config classes.
Behaviour therefore depends on which class a run is constructed through.

This is the defect D exists to remove, one level more abstract: not one default too
many, but two defaults that contradict each other. Unlike a plain default, no bundle
value can fix it — the two classes have to agree on what the field means, or one of
them should not carry it.

Sequence note: E renames across both classes and F reshapes the config surface, so
resolving this before F risks doing it twice. But it should not survive F.
Severity: result-affecting, depends on construction path
Suggested track: new — decide before or during F

## `config_hash` changed by Track E's `sector` -> `group_id` field renames
Found during: track E
Location: `src/simulator/simulation_persistence.py:32-44` (`hash_config`, hashes
the full serialized `SimulatorConfig` via `json.dumps(..., sort_keys=True)`,
which includes field names, not just values)
What: Same porting hazard as the `candidate_max_age_days` deletion note above, a
different trigger. Track E renamed `DataConfig.selected_sectors` ->
`selected_groups`, `DataConfig.excluded_sectors` -> `excluded_groups`, and
`DataConfig.sectors` -> `groups` (plus `SectorDataSource` -> `GroupDataSource`,
which is not itself hashed but travels with the field rename). None of these
change any resolved value, but `hash_config` serializes field names, so
`config_hash` changes for every config regardless. Confirmed via
`tests/test_sweep_defaults.py::test_standard_v1_output_is_hash_stable`, whose
recorded hash needed updating (`4d36adb0...` -> `9daa71da...`) for exactly this
reason.

Harmless in `statarb_sim`, which has no persisted run dirs, but will orphan
existing persisted runs in `hierarchical-arb` when this track's field renames are
ported, the same way the Track A deletion did.
Severity: result-affecting on port only
Suggested track: port note — hierarchical-arb
