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

Addendum (track F1, commit 3): `_bootstrap_panel_from_fixtures`'s copied-suffix
list (`run_me.py:393`, `(".panel.parquet", ".meta.json", "_residual_params.pkl")`)
was deliberately left untouched when F1 commit 3 moved every real
`residual_params` writer/reader from `.pkl` to `.parquet` — this function does
a raw file copy, never reads the params through `load_residual_params`, and the
committed fixture at `fixtures/materials_v1/..._residual_params.pkl` is still
that stale pickle. Leaving it means the fallback (already masked, per above)
would now copy a `.pkl` file that `discover_group_data_sources` cannot see
(it checks for `.parquet` — `config.py`'s `residual_params_stem` resolution),
so a series-persist step downstream would silently skip residual-params-
dependent series instead of finding them, on top of the pre-existing
stem-mismatch that already stops the fixture from being found at all. Two
compounding reasons this path won't work if it's ever reached; still masked
today by the same locally-built panel. Whoever fixes the fixture staleness
above should also regenerate this fixture as `.parquet` and update the suffix
list.
Severity: cosmetic today (masked); compounds the existing entry
Suggested track: same as above — new (fixture staleness)

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

## `start_after_nan` / `check_for_corruptions` — four combinations, no caller states either
Found during: track D (spec session, flagged "doubtful"); extended by track F's spec
session, which found the fourth combination
Location: `DataConfig` and `PanelBatchConfig` in `src/simulator/config.py`;
`market_snapshot.py:277` (hardcoded literal); `UniverseDataLoader.load`'s own
signature in `src/data/universe_loader.py:36`
What: These two flags control how raw price data is cleaned before anything else
runs. There are **four** combinations in the repo, not two, and **no caller
overrides any of them** — every path takes whatever default it happens to reach.

| Source | `check_for_corruptions` | `start_after_nan` |
|---|---|---|
| `DataConfig` | `True` | `True` |
| `PanelBatchConfig` | `False` | `True` |
| `market_snapshot.py:277` | `True` | `False` |
| `UniverseDataLoader.load` signature | `True` | `True` |

So the panel build and the simulator clean the same raw data differently, and Track
C's snapshot path introduced a third combination as a hardcoded literal. This is the
defect D removes, one level more abstract: four silent answers to one semantic
question, selected by which path the data takes. Because it is data preparation, it
acts on every candidate, every trade and every result.

**Update:** split out of F into its own **Track I** (`I_data_preparation.md`),
because it is data preparation rather than artifact layer. It must land **before
G** — G sweeps six hedge configs, and a preparation difference between panel build
and simulator would contaminate a comparison whose expected effect is thin.
Severity: result-affecting, depends on construction path
Suggested track: I — before G

**Update (F2 C6b):** this stopped being purely theoretical. Wiring live candidate
generation into the simulator (`generate_candidate_panels_by_group`,
`src/simulator/simulator_factory.py`) initially reused `run_from_config`'s single
merged `umd` — loaded under `DataConfig`'s flags — for scoring too, unifying the
two combinations above by accident. That moved `pair_notional` and downstream
performance metrics by rounding-level amounts on nearly every trade against
`B_baseline.txt` (trade count/ids/dates/directions/z-scores unchanged; only the
magnitudes moved) — a measured, non-hypothetical consequence of this entry, not
just an inconsistency. Fixed for F2's purposes by giving `CandidateGenerationConfig`
its own `force_download`/`check_for_corruptions`/`start_after_nan` fields
(defaulting to `PanelBatchConfig`'s values) and having
`generate_candidate_panels_by_group` load its own per-group UMD under them,
deliberately preserving the split rather than fixing it (F2_spec.md's C6b section
has the full before/after). Track I's eventual unification now has a second call
site to cover, in addition to the original two.

## `series/` staleness has three separate axes — only one is being fixed
Found during: track E (review of the `(group_id, ticker)` re-key)
Location: `src/residuals/series.py`; the `series/` artifact directory; stem
resolution in `run_me.py` and the panel builders
What: "Series staleness" has been used as one label for three unrelated failure modes. They have different causes and different fixes, and conflating them has already led to one of them looking solved when it is not.

This entry is reconstructed from the design discussion, not verified against the code. Confirm each axis before acting on it.

Axis 1 — market data revised underneath an artifact. Corporate actions retroactively re-adjust historical closes, so the same tickers over the same range can yield different values on a later download. Causality protects against lookahead, not against input revision. Any artifact built before a re-download is silently stale, and there is currently no way to tell with justified effort.

Fixed by Track C: snapshot_id plus a per-sector content hash, with each run bound to one frozen snapshot. This is the "marrying market data to a simulator run" axis.

Axis 2 — stem reuse across different residual configs. The series/ artifacts are keyed by ticker and spread_id but NOT by residual_key. Reusing a stem while changing the residual config therefore reads back series computed under the old config, silently.

Not fixed by anything yet. This axis was deliberately left open: auto-clearing on mismatch is unsafe without a real key design, and the obvious fix — add residual_key to the key — has not been specced. Track F dissolves the panel and reshapes the artifact layer, which is the natural place to settle it.

Axis 3 — the same ticker in two groups. Residual fits are group-scoped, so a ticker in two groups has two different residual series. Keying by ticker alone cannot hold both.

Addressed by Track E, which re-keyed to (group_id, ticker). Note this was proactive, not a staleness fix: E also added an assert forbidding multi-group tickers, so the collision cannot currently occur. The re-key exists so that the storage layer needs no change if the restriction ever lifts.

The point of this entry: E's re-key touches axis 3 only. It does nothing for axis 2, and axis 2 is the one that can silently produce wrong numbers today. Do not read "series now keyed by (group_id, ticker)" as "series staleness handled."

**Update (post-F spec):** axis 2 is DECIDED. The cache is deleted rather than
re-keyed — the read path, the skip-if-exists, and the shared `panel_dir` scope all
go. Optional per-run series generation was dropped as well, before the F2 spec
session: no run persists series, and plots and inspection reconstruct them on
demand. See `F2_loop_and_fidelity.md` §2.
Severity: axis 2 is result-affecting and live; axes 1 and 3 are addressed
Suggested track: F2

## `UniverseDataLoader.load`'s in-place cache-hit resync remains live for its three existing callers
Found during: track C
Location: `src/data/universe_loader.py:36-92` (`load`, the `to_remove`/`to_add`
resync branch reachable when `_cache_exists()` is true and the yaml's ticker
set no longer matches the cached data); callers: `run_me.py`'s `stage_download`
(`run_me.py:129-130`), `src/candidates/panel_batch.py:293-302`
(`run_panel_batch`), `src/simulator/simulator_factory.py:203-211`
(`_load_umd`).
What: When a yaml's member list changes and the loader is pointed at that
sector's existing cache directory, `load()` silently drops removed tickers,
downloads added ones, rebuilds `ticker_info`/`group_info`/`membership`, and
re-persists — mutating the cache in place with no record that the event
happened. This is the same "input revised underneath an artifact" failure
Track C's snapshot layer exists to prevent, one layer down, and it is
unguarded on all three of `UniverseDataLoader.load`'s current callers.
Track C's own snapshot build path (`src/data/market_snapshot.py`) cannot
reach this branch — each group downloads into a freshly created temp
directory, so `_cache_exists()` is always false there, and
`_download_group` additionally asserts this and raises loudly
(`SnapshotResyncGuardError`) if it ever would not be. But `load()` itself
was deliberately left unmodified for its existing callers (Track C's brief
scopes wiring `run_me.py`/panel_batch/simulator_factory to the snapshot
layer to Track F), so all three keep resyncing in place today.
Severity: result-affecting, live today (not hypothetical — any local cache
whose yaml's member list is edited will resync silently on next load)
Suggested track: I (reassigned from F2 before its spec session: these are the same
three loader call sites as I's flag unification). Wire them to
`ensure_market_snapshot`/`load_market_snapshot` there, and decide
`UniverseDataLoader.load`'s role for them.

## `ensure_universe_data` was not actually dead — notebook-only caller missed by a .py-only grep
Found during: track C (universe-model correction)
Location: `src/data/universe_loader.py:454` (`ensure_universe_data`); called from
`notebooks/howto/03_full_simulation_pipeline.ipynb`
What: Both this track's first session and a follow-up grep before starting the
layout migration searched `--include=*.py` only and concluded this function had
zero callers. It has one — a notebook cell
(`ensure_universe_data(SELECTED_GROUPS)`), invisible to a .py-scoped grep.
Re-executing the howto notebooks (standing rule, required after this track's
API-surface changes) surfaced it as a `FileNotFoundError` once the yaml layout
moved. Fixed in this track (added a `universe_name` parameter, updated the yaml
path construction) rather than left as dead code. Recorded so a future
"grep --include=*.py finds zero callers" conclusion about any function is
checked against notebooks too before being treated as dead.
Severity: cosmetic (caught and fixed before landing; recorded as a process note)
Suggested track: none — process note only

## Group yaml's own `meta.universe_name` field now collides in name with the directory-level universe concept
Found during: track C (universe-model correction)
Location: every `config/universes/{universe_name}/{group_id}.yaml`'s
`meta.universe_name` key (e.g. `materials_only_v1`); read by
`UniverseConfig.universe_name` (`src/data/universe_config.py`), consumed by
`UniverseDataLoader.__init__`'s cache-dir naming and by
`market_snapshot.py`'s per-group snapshot subdirectory naming.
What: Track C's universe-model correction introduced a directory-level
`universe_name` (config/universes/{universe_name}/) that identifies a whole
group/ticker set. The pre-existing, unrelated `meta.universe_name` field
inside each group yaml (predates this track) is really a per-GROUP data-cache
key today — it was named when each yaml was still framed as a standalone
one-group "universe". Both are now called "universe_name" for historically
unrelated reasons, which reads as one concept but is two. Deliberately left
unrenamed this track: renaming would ripple into the DATA_UNIVERSES cache
directory naming convention and UniverseDataLoader, which is a separate axis
from the CONFIG-layout migration this track scoped.
Severity: cosmetic (naming clarity only; no functional collision — the two
serve disjoint purposes and never resolve against each other)
Suggested track: new, or fold into F's config-surface reshaping

**Update (track F1, commit 6):** `F1_artifact_schema.md`'s commit 6 asked to
remove `meta.universe_name` from the group yamls, describing it as
"redundant... a leftover from when each yaml was its own universe." That
characterization does not hold: re-verified directly against this entry's
own citations, `UniverseConfig.universe_name` (`universe_config.py:74-75`)
reads `self.raw["meta"]["universe_name"]` as a **required** key (no
default; `UniverseConfig.validate()` doesn't even need to touch it —
accessing the property KeyErrors if it's absent), and it is what
`UniverseDataLoader.__init__` uses to name the on-disk cache directory
(`universe_loader.py:23`, `self.cache_dir = self.data_path /
self.config.universe_name`) — the exact `data/market/universes/
materials_only_v1/`-style paths every panel build in this track has read
from. Removing the yaml key would break cache resolution for every
existing cached universe, not free up genuinely dead surface. **Declined**:
F1 did not remove `meta.universe_name`. This looks like the brief's commit
6 was written without re-checking this entry's own prior finding.
Severity unchanged from above; the "Suggested track" line still applies —
whoever picks this up needs to actually design the cache-directory-naming
migration this entry already flagged as the real blocker, not just delete
a yaml key.

## "Zero call sites" claims in this refactor were `.py`-only greps
Found during: track C (the `ensure_universe_data` notebook caller)
Location: process, not code — affects claims in `A_spec.md` and `D_spec.md`
What: `ensure_universe_data` was recorded as having zero callers repo-wide. It
had one, in a cell of `notebooks/howto/03_full_simulation_pipeline.ipynb`, which
a `.py`-only grep did not see. It surfaced as a FileNotFoundError once the yaml
layout moved.

Every other "dead code" or "zero construction sites" finding in this refactor
rests on the same kind of search and is therefore unverified for notebooks:

- Track A deleted `compute_candidate_diagnostics` on that basis. Already
  deleted, so check the notebooks for it and restore if needed.
- Track D classified `SpreadMomentumConfig`, `KellyConfig`,
  `TimescaleRiskConfig.selection` and `CrossTimescaleEntryConfig` as having no
  construction sites. **F1 commit 1 re-verified including notebooks and deleted
  all four**, along with the subsystems reachable only through them — a Kelly
  tracker in `sizing_engine.py`, a cross-timescale trading mode in the pair
  trader, and a timescale-risk policy in `risk_manager.py`.
- `build_spread_returns` (`src/residuals/spreads.py:289`) is recorded as
  callerless on the same basis and has not been re-verified.

Rule going forward: any grep supporting a deletion must include
`notebooks/**/*.ipynb`. Now a standing rule in `README.md`.
Severity: process; one instance found and fixed, one class of claim re-verified
Suggested track: check before any deletion in F2

## `hedge_key` in weights.parquet is a stand-in, not Track G's real key
Found during: track F1 (commit 4)
Location: `src/candidates/pair_candidate_panel_creator.py` (weight_rows
construction in `_build_pair_candidate_rows_for_date`); `weights.parquet`'s
grain per `F1_artifact_schema.md`: `(group_id, residual_key, hedge_key,
spread_id, asof_date, ticker) -> weight`
What: The brief specifies `hedge_key` as part of `weights.parquet`'s grain
and as one of the two live key columns in F1's Layout section ("Until
Track H defines `zscore_key`, the key columns are `residual_key` and
`hedge_key` only") — implying it should already be a real, defined concept
by F1. It is not: grepped `hedge_key` across `src/` before this commit and
found zero hits anywhere in the codebase. Track G ("hedge mode"), which
`found.md`'s residual-fit-weighting entry above describes as carrying "the
same defect class... one stage earlier" for the hedge fit, has not landed
— no `HedgeConfig` class or `.key` property exists to source a real
`hedge_key` from.

This commit set `hedge_key` to the existing hedge-ratio method string (e.g.
`"pca"`, `"ols"` — `PairSpreadConfig.hedge_ratio_methods`, the `method`
loop variable), the closest existing analogue to how `residual_key` is
`CausalResidualConfig.key`. This is sufficient for `weights.parquet`'s own
uniqueness today (multiple `hedge_ratio_methods` configured for one panel
would otherwise collide on `(spread_id, asof_date, ticker)`), but it is not
Track G's eventual hedge-fit-config key — no lookback/weighting-scheme
information is encoded, unlike `residual_key`'s `exp_hl252_mh504_rf`-style
encoding.
Severity: cosmetic today (weights.parquet is internally consistent and
correct); a real design question for whoever picks up Track G
Suggested track: G — confirm or replace this `hedge_key` value when
`HedgeConfig` is designed

## F_spec.md's "lossy weights JSON never read back into computation" claim was wrong
Found during: track F1 (commit 4)
Location: `src/simulator/candidate_filter.py:54-100` (pre-commit-4
`build_candidate_refs`, called live from `simulator.py:336`)
What: `F_spec.md` Part 3's commit-4 row and `F1_artifact_schema.md`'s commit
4 both asserted this was "the one commit in the whole track with a
*provable* neutrality argument rather than an expectation... the lossy
JSON was never read back into any computation," citing
`pair_candidate_panel_creator.py:347-350`'s in-memory full-precision
`spread_return` computation as the reason.

That citation is correct for `spread_return`/diagnostics, but the claim of
"never read back into any computation" is false for a different, more
consequential reader: `CandidateFilter.build_candidate_refs`
(`candidate_filter.py`, called live every simulated day from
`simulator.py:336`) parsed the persisted `row["weights"]` JSON string via
`json.loads` to build `CandidateRef.weights` — the actual weight frozen at
entry and used for the whole position's lifetime (sizing, PnL). This is
the live trading path, not a historical-analysis or notebook-only reader.
Neither `F_spec.md`'s Part 1 verification pass nor the brief caught this;
both apparently reasoned from the `spread_return` computation alone and
did not trace `CandidateFilter`'s own read of the panel's `weights` column.

Verified empirically instead of assuming: ran the full pipeline (fresh
`weights.parquet` write → `_load_weights` → `CandidateFilter` → live
trading) against the Track B/F1 baseline harness. Result: byte-identical
97 trades, `B_baseline.txt` unchanged from `## counts` onward. So the
JSON-precision loss (`double_precision=12`, ~12 significant digits vs.
float64's ~15-17) turned out to be immaterial at this harness's scale —
but that is an empirical fact about this one harness run, not a structural
guarantee the way the brief framed it. A run where a z-score threshold or
unit-rounding decision sits closer to a boundary could in principle differ.
Recorded because F2's zero-tolerance fidelity test's premise partly rests
on the artifact-swap commits being provably neutral; this one commit's
neutrality claim needed correcting to "verified neutral," not "provably
neutral by construction."
Severity: cosmetic (the actual F1 commit 4 change is still neutral,
verified) — the process/documentation error is the finding
Suggested track: none — process note; relevant context if F2's fidelity
test session revisits commit 4's neutrality argument

## `config_hash` has shifted nine times during this refactor
Found during: tracks A, C, E, F1, F2
Location: `src/simulator/simulation_persistence.py:32-44` (`hash_config`, which
serializes the full `SimulatorConfig` via `json.dumps(..., sort_keys=True)` — field
NAMES as well as values)
What: Because field names are hashed, any field added, removed or renamed shifts
`config_hash` for every config, even when no resolved value changes. Nine such
changes have landed, each caught by
`tests/test_sweep_defaults.py::test_standard_v1_output_is_hash_stable`:

| Track | Trigger | Recorded hash |
|---|---|---|
| A | `ActivationConfig.candidate_max_age_days` deleted | (pre-chain) |
| E | `sectors` / `selected_sectors` / `excluded_sectors` renamed to `groups` / … | `4d36adb0` -> `9daa71da` |
| C | `DataConfig.universe_name` added | `9daa71da` -> `284530b0` |
| F1.1 | `SizingConfig.kelly`, `RiskManagerConfig.timescale_risk`, `PairSpreadTraderConfig.cross_ts` deleted | `284530b0` -> `0240fcdf` |
| F1.2 | `SimulatorConfig.spectrum` / `SpectrumConfig` deleted | `0240fcdf` -> `7a5f6834` |
| F1.6 | `DataConfig.snapshot_id` added | `7a5f6834` -> `89b82e76` |
| F1.8c | `PersistenceConfig.artifacts` default tuple shortened (a value change, not a field change) | `89b82e76` -> `0bc1e2a0` |
| F2.C5 | `SimulatorConfig.debug_sample: DebugSampleConfig \| None` added | `0bc1e2a0` -> `4e493ec7` |
| F2.C6b | `SimulatorConfig.candidate_generation: CandidateGenerationConfig \| None` added | `4e493ec7` -> `312a5f07` |

Harmless in `statarb_sim`, which has no persisted run dirs. In `hierarchical-arb`
every existing persisted run becomes unreachable by hash once these are ported. The
standing port rule is one atomic commit per fix, which would orphan runs nine times
in sequence — consider porting the hash-breaking subset as a single batch and
re-keying once.

**Two port cautions that do NOT transfer automatically.** Both F1 deletions rest on
"zero construction sites in `statarb_sim`", which says nothing about the other repo:

- `KellyConfig`, `TimescaleRiskConfig`, `CrossTimescaleEntryConfig` — Kelly sizing
  was actually tested in `hierarchical-arb`, so live non-`None` construction sites
  are plausible there.
- `SpectrumConfig` — `z_spectrum_capture.py` was ported *from* `hierarchical-arb`,
  so that is exactly where a real user of the spectrum-capture experiment would be.

Re-verify in `hierarchical-arb` specifically, including notebooks, before porting
either deletion. If a construction site exists, the port is not result-neutral.
Severity: result-affecting on port only
Suggested track: port note — hierarchical-arb

## `DataConfig.universe_name` reintroduces the defect Track D removed
Found during: track F1 (review of the config_hash entries)
Location: `DataConfig` in `src/simulator/config.py`
What: Track C added `universe_name: str = "sp500_v1"` so the loader can resolve
`config/universes/{universe_name}/{group_id}.yaml` after the layout migration. That
default decides which universe a run loads — as result-affecting as anything D
removed, just not numeric. A caller who omits it silently gets `sp500_v1`, which is
the failure mode the universe-versioning work exists to prevent: the whole point of
`sp500_v8` is that the version is stated, not inherited.

`DataConfig.snapshot_id: str | None = None` (F1 commit 6) is the same shape but
benign today, since nothing constructs it non-`None`. It stops being benign the
moment the snapshot layer is wired to `stage_download`, at which point an omitted
`snapshot_id` means "no snapshot binding" rather than an error.

D's enumeration could not have caught either — both postdate it. The rule it
established should extend to strings and sentinels that select data, not only to
numbers.
Severity: result-affecting, dormant today
Suggested track: I (reassigned from F2 before its spec session), alongside wiring
the snapshot layer

## The spread level had two definitions, chosen by whether cache files existed
Found during: track F2 (review of F2_spec.md)
Location: disk path `src/residuals/series.py` + `candidate_signals.py::_try_batch_levels_from_disk`;
recompute path `candidate_signals.py::_get_residuals`
What: The disk path uses the residual model frozen at `asof_date`; the recompute
path uses the model fitted at today's date. The choice was made per (group, day) by
whether files existed, all-or-nothing, signalled only by a warning.
`compute_analytics_from_weights` and `get_level_series` always recomputed. So within one
harness run, trading z-scores used the frozen model while features and diagnostics
used the daily one.
Severity: result-affecting
Suggested track: resolved by F2 C3 (frozen-at-asof, decided).
Port note: `hierarchical-arb` has the same split, so its research results ran on
whichever path had cache hits. The May 2026 z-spectrum discrepancy between the
entry-date-model reconstruction and the simulator may have this cause — unverified.
Magnitude, measured on the F2 baseline harness (`F2_spec.md`, "R1 impact
measurement"): forcing the pre-C3 recompute (today-dated) path on this harness's 97
post-fix trades instead gave 115 trades — 37 exist only under the old path, 19 only
under the fixed path, and every one of the 78 trades common to both by `trade_id`
has a different entry and/or exit z-score. Not a rounding-level effect: the
divergence compounds through the run once any one trade's exit timing shifts.
Whatever `hierarchical-arb` research ran on the recompute path (cache miss) should
be treated as unreliable until re-run, not just re-checked.

## `mr_diag_lb` and `MRDiagnosticsConfig.lookback` can silently disagree
Found during: track F2 (spec session, Part 3.8)
Location: `pair_candidate_panel_creator.py::_fast_pair_diagnostics` (panel build, window
`mr_diag_lb`); `candidate_signals.py::_compute_mr_diagnostics` (simulation,
`MRDiagnosticsConfig.lookback`, `config.py:331`)
What: The same OU-fit math runs under two independently configurable windows, with
no assertion tying them together.
Severity: result-affecting, magnitude unmeasured
Suggested track: new, result-changing; decide which window governs before G's
baseline.

## `portfolio_mean_reversion.py`'s `fz_analytics` parameter is populated from `analytics_by_id`, not `fz_analytics_by_id`
Found during: track F2 (implementation pre-check, P2)
Location: `src/simulator/traders/portfolio_mean_reversion.py:76,81,128,148-152`
(`_maybe_close`'s `fz_analytics` parameter, read from `diagnostics.get(candidate_id)`,
`diagnostics = live_diagnostics_by_candidate_id or {}`); caller
`simulator.py:360-364` (`self.trader.generate_actions(...,
live_diagnostics_by_candidate_id=analytics_by_id)`)
What: The trader's `_maybe_close` names its parameter `fz_analytics`, matching the
naming of `simulator.py`'s own `fz_analytics_by_id` (built separately at
`simulator.py:346-351,701-720` from each open position's *realized fill* weights).
But the trader never receives `fz_analytics_by_id` — `generate_actions` is only ever
called with `live_diagnostics_by_candidate_id=analytics_by_id`, the *frozen candidate
weight* path (`_batch_analytics_for_group`). So `_maybe_close`'s `fz_analytics.mr_score`
is actually the frozen-weight `mr_score`, not a realized-fill one, despite the name.
`fz_analytics_by_id` itself is real and computed every day but is currently read only
by `_build_diagnostics_log_entries` (logging), never by any trading decision.
Severity: cosmetic (the naming is misleading; not itself a wrong computation — see the
next entry for where this naming confusion masks a real result-affecting issue)
Suggested track: new — rename one of the two, or wire `fz_analytics_by_id` into the
trader if the realized-fill mr_score was actually intended for the deterioration stop

## The deterioration stop compares mr_score from two different weight bases
Found during: track F2 (implementation pre-check, P2)
Location: `src/simulator/traders/portfolio_mean_reversion.py:145-153` (deterioration
stop: `fz_analytics.mr_score < mr_deterioration_threshold * live_pos.entry_mr_score`);
`fz_analytics.mr_score` traces to `analytics_by_id` → `_batch_analytics_for_group`,
frozen candidate weights (`CandidateRef.weights`); `live_pos.entry_mr_score` is set at
`simulator.py:650` from `entry_analytics = compute_analytics_from_weights(...,
weights_by_ticker=realized_weights_by_ticker)` — actual fill weights, which can differ
from the frozen candidate weights (rounding to whole units, partial fills).
What: The deterioration stop's ratio compares an mr_score computed on the candidate's
frozen theoretical weights (numerator, refreshed daily) against an mr_score computed
on the position's realized fill weights at entry (denominator, fixed once at entry) —
two different spreads' worth of mr_score in one ratio, not a before/after comparison
of the same spread under the same weights.
Severity: result-affecting, magnitude unmeasured
Suggested track: new

## `SimulatorConfig`'s EQ_ROLLING guard is likely liftable after F2 C3
Found during: track F2 (C3 review, R10)
Location: `src/simulator/config.py:795-809` (`SimulatorConfig.__post_init__`, the
unconditional raise for `self.residual.window_mode == "rolling"`)
What: The guard's own message describes the cause as "the z-score's EWM history span
... inherited from whichever upstream path supplied the residual series — full
history on the disk-backed path, truncated to the residual lookback on the recompute
path — so the same (spread_id, date) can yield a different z-score depending on
whether a cache existed." F2 C3 removed that cause: both the disk path and the
recompute path now apply the asof-frozen model over the full history and truncate the
level afterward (F2_spec.md R10) — there is no longer a per-path row-count difference
for the guard to be protecting against.
Severity: blocks EQ_ROLLING residual configs entirely
Suggested track: new, after F2 merges — lifting the guard still needs the explicit
span parameter the guard's own message calls for; not decided or touched in F2

## A leftover artifacts directory silently serves stale values into a before/after diff
Found during: track F2 (C4/E6 review)
Location: `artifacts/candidate_panels/{subdir}/` (e.g.
`artifacts/candidate_panels/refactor_b_harness/`, `scripts/b_baseline_harness.py`'s
own `PANEL_SUBDIR`); read by `run_from_config`'s panel/weights/residual-params
loaders (`simulator_factory.py::_load_panels`/`_load_residual_params`/
`_load_weights`, all via `discover_group_data_sources`)
What: This directory accumulates across runs — `run_panel_batch` stamps a fresh
timestamped stem on every build but never deletes an older one, so multiple stems
for the same (group_id, residual_key) can coexist indefinitely. This is exactly how
F2's own E6 measurement first went wrong: `series/` (deleted by C4, so moot for that
one sub-path) held ~5,235 files from earlier sessions this track, and the pre-C3
reader found and served them regardless of the harness's populate call being
skipped that run, making the "before" measurement silently identical to the
"after" one — a false negative, not a passing check. The mechanism is not
series/-specific: `candidate_panels/` (panel/weights files) and the
`_residual_params.parquet` files alongside them accumulate the same way and were
not cleared by C4, since nothing in F2 writes or removes them per run.
Severity: result-affecting for any before/after comparison run against this
directory without first confirming it is empty — not a live bug in shipped
trading logic, but a standing risk to the *instrument* used to verify one
Suggested track: F2 adds a guard (E8, this track) that reports and refuses to
silently proceed; Tracks I and G verify entirely via before/after diffs against
this same harness and directory, and inherit the same risk until they either reuse
the guard or adopt their own equivalent

## `make_ranking_dates` can return a resample bin label that is not an actual data date
Found during: track F2 (C6a equivalence test)
Location: `src/utils/date_utils.py:4-22` (`make_ranking_dates`) —
`s.resample(frequency).last().dropna().index` takes `.index` from the *resampled*
series, whose labels are the bin edges (e.g. every Friday for `"W-FRI"`), not the
date of whichever row inside that bin actually supplied `.last()`'s value.
What: When a bin's period label (e.g. a Friday) is itself a date with no row at
all — a market holiday, confirmed concretely: `2007-04-06` (Good Friday) is
absent from both `energy_only_v1` and `materials_only_v1`'s
`aligned_returns.index` — `.last()` still returns the most recent real value
from earlier in that bin (e.g. Thursday), but the bin's *label*, which is what
ends up in the returned `DatetimeIndex`, is the Friday. Every caller of
`make_ranking_dates` can silently receive a date that is not in the index it was
computed from.
This was completely masked in the shipped offline panel-build path: `create_pair_
candidate_panel`'s walk (`pair_candidate_panel_creator.py`, `walk_dates =
daily_dates if daily_dates is not None else asof_datetimes`) only builds
candidate rows on outer dates that are *also* present in the daily fit grid —
itself built by the same `make_ranking_dates(frequency="B", ...)`, which drops
the holiday's now-empty daily bin via its own `.dropna()`. So the phantom Friday
is silently skipped end-to-end, and the panel simply has one fewer outer date
than requested, with no signal that anything was dropped. The in-simulator
generator (`src/simulator/candidate_generation.py`, C6a) has no daily grid to
filter through, so it filters phantom dates itself
(`asof_datetimes[asof_datetimes.isin(bundle.aligned_returns.index)]`) — a local
workaround, not a fix to `make_ranking_dates` itself, which is unchanged and
still returns the same phantom labels to every other caller.
Severity: result-affecting, magnitude unmeasured beyond this one confirmed
instance — every `make_ranking_dates` caller across the panel-build and
simulator-config resolution paths (`resolved_groups`,
`DataConfig.resolved_groups`, the daily grid itself) is exposed in principle;
only the offline outer/daily interaction happens to self-correct today.
Suggested track: new — fix `make_ranking_dates` to return the actual date of the
row `.last()` selected, not the resample bin label; re-audit every caller once
fixed, since some may be relying (even accidentally, like the offline walk) on
today's masking behavior
