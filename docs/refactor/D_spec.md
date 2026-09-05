# Track D spec — Config hygiene

Delta against `docs/refactor/D_config_hygiene.md`. Read-only session: no code
changed, no branch created. This file is the only commit.

## Verify first — the four claims

**1. `hedge_ratio_lb` is an `int | None` parameter that falls back to the
residual config's lookback when None, which is itself None for expanding
configs.**

**Confirmed**, with a correction: this is a *function-parameter* default, not
a dataclass field — but it is the exact "None sentinel resolves to another
config's field" shape Rule 2 names, so it belongs in the enumeration exactly
as claim 1 describes.

- `src/candidates/pair_candidate_panel_creator.py:538-539` (inside
  `create_pair_candidates_for_date`):
  ```
  if hedge_ratio_lb is None:
      hedge_ratio_lb = residual_cfg.lookback
  ```
- Identical pair at `src/candidates/pair_candidate_panel_creator.py:649-650`
  (inside `create_pair_candidate_panel`), plus the same for `mr_diag_lb` at
  both sites (`:540-541`, `:651-652`).
- `src/residuals/causal_residuals.py:113-117`:
  ```
  def lookback(self) -> int | None:
      """Compatibility for fitting code (causal_residuals.py reads cfg.lookback
      unprefixed) — no fitting-code changes required."""
      return self.lb if self.mode == ResidualMode.EQ_ROLLING else None
  ```
  `.lb` is `EQ_ROLLING`-only (`causal_residuals.py:61`); for `EQ_EXPANDING` and
  `DECAY_EXPANDING` configs `.lookback` is unconditionally `None`. Confirmed.

**Additional finding, not in the brief:** this fallback is currently dead
code on every exercised path. `PanelBatchConfig.hedge_ratio_lb` /
`.mr_diag_lb` (`panel_batch.py:125-126`) are typed `int | list[int]` —
**not** `| None` — so a caller going through `PanelBatchConfig` /
`run_panel_batch` can never produce `hedge_ratio_lb=None` at the fallback
site. Grepping every call site of `create_pair_candidate_panel` /
`create_pair_candidates_for_date` (`src/`, `scripts/`, `run_me.py`, all three
notebooks, `tests/`) turns up exactly two callers, both internal to
`panel_batch.py:320` (through `PanelBatchConfig`, always non-`None`). No test
or notebook calls either function directly. **Killing the fallback is a
pure risk-reduction deletion on this codebase today** — not a behaviour
change on any exercised path — which simplifies the acceptance criterion:
there is nothing for it to move.

**2. The pair-spread config carries numeric defaults for the minimum return
standard deviation, the minimum level standard deviation, the minimum kappa,
and the maximum half-life.**

**Confirmed.** `src/candidates/pair_candidate_panel_creator.py:88-91`:
```
min_return_std: float = 1e-8
min_level_std: float = 1e-8
min_kappa: float = 1e-6
max_half_life: float = 126.0
```
Two more numeric defaults in the same dataclass the brief's claim text
doesn't name: `tiny_weight_threshold: float = 1e-6` (`:92`) and the
`hedge_ratio_methods` default factory, `["ols"]` (`:84-86`) — a default
*choice*, not a numeric threshold, but it is a `field(default_factory=...)`
selecting which hedge-ratio model runs, which changes every downstream
number. `skip_adf: bool = True` (`:93`) is non-numeric but is itself the
subject of the found.md-recorded Track A/D accident (see brief's "Why this
track is worth doing" section) — not re-litigated here, just noted as
already tracked.

**3. `DEFAULT_CONFIGS` exists as a registry and the bundle name enters
`config_hash`.**

**Confirmed in part, corrected in part.** The registry exists exactly as
claimed: `src/simulator/sweep_defaults.py:63-65`:
```
DEFAULT_CONFIGS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "standard_v1": _STANDARD_V1,
})
```
**Correction:** the bundle *name* does not enter `config_hash`. `hash_config`
(`src/simulator/simulation_persistence.py:32-44`) hashes
`_config_to_serializable(config)` — the fully-resolved `SimulatorConfig` —
which never carries the bundle name as a field. Only the bundle's *resolved
values* enter the hash, via `merge_defaults` splicing them into the
`SimulatorConfig(**kwargs)` call (`sweep_defaults.py:99`,
`sweep_runner.py:315-317`). Two bundles with different names but identical
resolved field values would hash identically and be indistinguishable in
`config_hash`.

The bundle name *is* recorded, but separately: `SweepConfig.defaults: str =
"standard_v1"` (`src/simulator/sweep_runner.py:110`) is a plain field, so
`run_sweep`'s per-row dict comprehension over `fields(sweep)`
(`sweep_runner.py:502-523`) writes it into the persisted results row
alongside `config_hash` (`sweep_runner.py:534`) — i.e. it's in the sweep
*results table*, not in the *hash*. This distinction matters for D's own
acceptance test: relocating a default into a bundle is invisible to
`config_hash` unless the relocated *value* changes; the bundle name itself
buys traceability in the results log, not hash identity.

**4. `PanelBatchConfig.pair_cfg` is now a required field with no default
factory, as of Track B. Confirm this and confirm no caller relies on a
removed default.**

**Confirmed on both counts.**
- `src/candidates/panel_batch.py:135`: `pair_cfg: PairSpreadConfig` — no
  default, no factory (Track B, see the file's own comment at `:128-134`).
- Every call site states it explicitly. Grep of `PanelBatchConfig(` across
  `src/`, `scripts/`, `run_me.py`, `tests/`: `src/candidates/panel_batch.py:10`
  (docstring example), `scripts/b_baseline_harness.py:103`, `run_me.py:223`,
  and ten sites across `tests/test_panel_batch_config.py` and
  `tests/test_panel_batch_windows.py` — all pass `pair_cfg=`. The one
  omission is deliberate:
  `tests/test_panel_batch_config.py:303-308` (`test_pair_cfg_has_no_default`)
  asserts the omission raises `TypeError`. No caller relies on a removed
  default.

## The enumeration

### Scope of the enumeration

"The config layer" is taken to mean every dataclass a caller constructs to
specify *what a run computes* — reachable from the four root config objects a
human actually writes into a script or notebook:

- `PanelBatchConfig` (`src/candidates/panel_batch.py`)
- `PairSpreadConfig` (`src/candidates/pair_candidate_panel_creator.py`)
- `CausalResidualConfig` (`src/residuals/causal_residuals.py`)
- `SimulatorConfig` and everything reachable through its fields
  (`src/simulator/config.py`): `DataConfig`, `CandidateSelectionConfig`
  (`src/candidates/candidate_selector.py`), `ActivationConfig`, `ZScoreConfig`,
  `MRDiagnosticsConfig`, `PairSpreadTraderConfig`,
  `PortfolioMeanReversionConfig`, `SpreadMomentumConfig`, `RunConfig`,
  `ExecutionConfig`, `CapitalConfig`, `SizingConfig`, `VolSizingConfig`,
  `KellyConfig`, `IntervalScoringConfig`, `FeatureIntervalSpec`,
  `RiskManagerConfig`, `TimescaleRiskConfig`, `CrossTimescaleEntryConfig`,
  `PerformanceConfig`, `PersistenceConfig`, `SpectrumConfig`,
  `EntryFeatureConfig`, `FeatureSpec`
- `SweepConfig` (`src/simulator/sweep_runner.py`) — **not named in the brief,
  but it is the actual production entry point for `run_sweep`,** it
  constructs `SimulatorConfig` fields from its own dataclass defaults
  (`_build_sim_config`, `sweep_runner.py:262-317`), and its own docstring
  states "Defaults = current best baseline" (`sweep_runner.py:65`) — i.e. it
  explicitly is a dataclass holding today's live numeric defaults. Excluding
  it would leave the single largest concentration of unexamined numeric
  defaults out of the audit.

**Explicitly excluded, with reason:** engine/runtime classes that *consume*
an already-built config and hold their own internal, always-zero/empty
working state — `KellyTracker`/`_KellyGroupState`, `CandidateSignalGenerator`,
`ZSpectrumCapture`, `CandidateActivation`, `SizingEngine`, `RiskManager`,
`PositionTranslator`, `CandidateFilter` (all in `src/simulator/`). Their
`field(default_factory=dict)` / `= 0` / `= 0.0` defaults initialize mutable
caches and accumulators, not caller-facing configuration — nobody states "the
Kelly tracker starts at 0 trades" the way they state `min_obs`. These were
checked (not skipped blind): grepped for `@dataclass` and `: .* = ` across
every file in `src/simulator/`, `src/candidates/`, `src/residuals/`, and each
non-Config dataclass was read. One exception surfaced and is flagged below.

**One exception found just outside the boundary, flagged because it's the
same defect shape:** `src/simulator/traders/portfolio_mean_reversion.py:31`:
```
@dataclass(slots=True)
class PortfolioMeanReversionTrader:
    config: PortfolioMeanReversionConfig
    default_abs_target_group_exposure: float = 1.0
```
This sits beside a real `*Config` object as a second, ungated numeric default
with no way to override it — `simulator_factory.py:92` constructs
`PortfolioMeanReversionTrader(config=config.trader)` and never passes this
field, at the one and only construction site in the repo (grepped). It is
**not currently exercised by B's baseline** (which runs
`PairSpreadTraderConfig` / `PairSpreadMeanReversionTrader`, never
`PortfolioMeanReversionConfig`), so it cannot move `B_baseline.txt` today —
but it is live production code, selectable via `SimulatorConfig.trader`'s
union type (`config.py:815`), and it is unambiguously the same anti-pattern.
Listed in Table 1, flagged as currently-dormant.

### Table 1 — RESULT-AFFECTING

Removed in the implementation session; every value moves into a versioned
bundle (`DEFAULT_CONFIGS`-style, or `SweepConfig`'s own defaults if that
class is brought into scope — see "Anything that cannot be done as
described" below).

| # | Location | Field / mechanism | Current value | Form |
|---|---|---|---|---|
| 1 | `pair_candidate_panel_creator.py:88` | `PairSpreadConfig.min_return_std` | `1e-8` | (a) |
| 2 | `pair_candidate_panel_creator.py:89` | `PairSpreadConfig.min_level_std` | `1e-8` | (a) |
| 3 | `pair_candidate_panel_creator.py:90` | `PairSpreadConfig.min_kappa` | `1e-6` | (a) |
| 4 | `pair_candidate_panel_creator.py:91` | `PairSpreadConfig.max_half_life` | `126.0` | (a) |
| 5 | `pair_candidate_panel_creator.py:92` | `PairSpreadConfig.tiny_weight_threshold` | `1e-6` | (a) — named in brief's scope list ("Remove numeric defaults from the pair-spread config") but not in claim 2's text |
| 6 | `pair_candidate_panel_creator.py:84-86` | `PairSpreadConfig.hedge_ratio_methods` default factory | `["ols"]` | (b) — not numeric, but selects the fit model; every real call site already overrides it to `["pca"]`, so the default is dead weight, not a live risk. Included per the brief's "not just literal scalar defaults" instruction. |
| 7 | `pair_candidate_panel_creator.py:538-541`, `:649-652` | `hedge_ratio_lb` / `mr_diag_lb` → `residual_cfg.lookback` fallback | resolves to `None` for expanding configs | (c) — claim 1. Confirmed dead on every exercised path (see above); removal is a signature change (drop the `| None = None`), not a behaviour change. |
| 8 | `config.py:840` | `SimulatorConfig.sizing` default factory | `SizingConfig(base_pair_notional=100_000.0)` | (b) — nested config construction with a hardcoded notional. Same shape as the two known `pair_cfg` instances (Rule 3: a layer above reinstating a number the layer below requires). Currently masked in practice: every production call site (`b_baseline_harness.py`, `run_me.py`, `sweep_runner.py`, both simulate-stage notebooks) passes `sizing=` explicitly, so this factory has never actually fired in a committed run — but it is live, and a new caller who omits `sizing=` gets a silent $100k notional. |
| 9 | `sweep_runner.py:274`, `run_me.py:447`, `b_baseline_harness.py:176`, notebook `02_run_simulation.ipynb` cell `d59987351d18` | bare `VolSizingConfig()` | `floor_multiplier=0.2, cap_multiplier=5.0` (`config.py:473-474`) | (b)-adjacent — every one of 4 production/notebook call sites constructs `VolSizingConfig` with **no arguments**, relying entirely on its own class defaults. No call site states these numbers. This is the widest-spread instance found: more live call sites than either known `pair_cfg` accident. |
| 10 | `config.py:743-744` | `RiskManagerConfig.max_gross_exposure`, `.max_ticker_exposure_pct` | `10.0`, `0.15` | (a) — every production call site already restates these explicitly: `b_baseline_harness.py:178-181` (literal), `sweep_runner.py:279-283` via `SweepConfig.max_gross_exposure`/`.max_ticker_exposure_pct` field defaults (`sweep_runner.py:93-94`, themselves literals), and notebook 02 (literal). `run_me.py:449-451` also states them explicitly, but *not* as a literal in the `.py` file — it wires `rsk["max_gross_exposure"]`/`rsk["max_ticker_exposure_pct"]` (`run_me.py:409,450-451`) through from `config/demo_materials.yaml:54-55` (`10.0`, `0.15`), so `run_me.py` itself is YAML-driven for this field, not hardcoded. Either way the class default is currently redundant, not live-relied-upon — but `SimulatorConfig.risk_manager: RiskManagerConfig | None = None` (`config.py:844`, see #11) means a caller who omits `risk_manager=` entirely gets **no risk manager at all**, not this default — a sharper failure mode than a wrong number. |
| 11 | `config.py:844` | `SimulatorConfig.risk_manager: RiskManagerConfig \| None = None` | `None` → risk manager step skipped entirely (`simulator_factory.py:107-110`, `simulator.py:478-479`) | (c)-adjacent. Not "resolves to another config's field," but a `None`-sentinel whose default silently disables portfolio risk constraints (unconstrained gross/ticker exposure) rather than falling back to a stated value. Doubtful call: it's a legitimate feature-off switch in form, but its *off* state is result-affecting and the "off" behaviour (no caps) is arguably the least safe default a trading simulator could ship with. Placed here rather than Table 2 because the failure mode (silently unconstrained) is exactly what this enumeration exists to catch. |
| 12 | `config.py:775` | `ExecutionConfig.commission_per_share` | `0.005` | (a) |
| 13 | `config.py:776` | `ExecutionConfig.commission_per_order` | `0.0` | (a) |
| 14 | `config.py:777` | `ExecutionConfig.min_commission_per_order` | `1.0` | (a) |
| 15 | `config.py:778` | `ExecutionConfig.max_commission_per_order` | `9.79` | (a) |
| 16 | `config.py:779` | `ExecutionConfig.max_commission_pct_of_trade` | `0.01` | (a) |
| 17 | `config.py:780` | `ExecutionConfig.short_borrow_rate_annual_bps` | `30.0` | (a) |
| 18 | `config.py:773` | `ExecutionConfig.min_abs_units` | `0.5` | (a) — "ignored if fractional=True" per its own comment, so conditionally live |
| 19 | `sweep_defaults.py:39-60` (`_STANDARD_V1`) / `sweep_runner.py:66-115` (`SweepConfig`) | The entire non-swept `_build_sim_config` literal set duplicated as `SweepConfig` field defaults: `entry_z=2.0`, `exit_z=0.0`, `z_lookback=21`, `base_pair_notional=100_000.0`, `max_ticker_exposure_pct=0.15`, `max_gross_exposure=10.0`, `start_date="2010-01-01"`, `end_date="2025-12-31"` | see values | (a) — `SweepConfig`'s own docstring: "Defaults = current best baseline" (`:65`). This is the anti-pattern's textbook description, just not phrased as a "dataclass default" in the brief because `SweepConfig` predates the brief's framing. See "Anything that cannot be done as described." |
| 20 | `candidate_selector.py:67` | `CandidateSelectionConfig.require_success` | `True` | (a) — bundle (`sweep_defaults.py:41-43`) overrides `allowed_candidate_subtypes` and `require_is_valid` but not `require_success`; it is silently inherited from the class default at the one real call site. |
| 21 | `traders/portfolio_mean_reversion.py:31` | `PortfolioMeanReversionTrader.default_abs_target_group_exposure` | `1.0` | (a) — outside the `*Config` boundary, same shape; see note above. Dormant on B's baseline today. |

**Doubtful entries within Table 1** (per instructions: ambiguous cases go
here, flagged, rather than in Table 2):

- #10, #11 (`RiskManagerConfig` defaults / `risk_manager=None`) — doubtful
  because every current call site already overrides the numbers; the risk is
  entirely in the `None`-sentinel default at the `SimulatorConfig` level, not
  in `RiskManagerConfig`'s own field defaults.
- `PanelBatchConfig.start_after_nan: bool = True` (`panel_batch.py:151`) and
  `DataConfig.start_after_nan: bool = True` (`config.py:249`) — a boolean,
  not a magic number, but it trims leading NaN rows before fitting, which
  changes what data enters the residual/hedge fit. Every call site leaves it
  at `True` (no observed disagreement), but flipping it would move
  `B_baseline.txt`. Doubtful because it reads more like a data-correctness
  flag than a tunable; included per Rule 1's "anything that affects results,"
  not per "numeric."
- `PanelBatchConfig.check_for_corruptions: bool = False`
  (`panel_batch.py:150`) vs. `DataConfig.check_for_corruptions: bool = True`
  (`config.py:248`) — **the two config classes disagree with each other on
  the default for what reads as the same flag.** Neither call site overrides
  either one. On the committed, clean datasets this makes no observed
  difference today (no corruption present to detect), but it is a
  landmine — if corrupted data ever appeared, one code path raises and the
  other silently proceeds, depending only on which config object happened to
  own the flag. Flagged, not scored as currently result-affecting, since
  nothing today can exercise the divergence.

### Table 2 — NOT RESULT-AFFECTING

Each entry justified in one line, per instructions.

| Location | Field | Default | Why not result-affecting |
|---|---|---|---|
| `panel_batch.py:138` | `PanelBatchConfig.frequency` | `"W-FRI"` | Every production call site restates it identically; genuinely categorical (which dates get sampled), not numeric — flagged here rather than silently dropped because a *different* value absolutely would move results, but no observed call site relies on the default to get today's behaviour. |
| `panel_batch.py:141-142` | `selected_sectors`, `excluded_sectors` | `None` | Selects *which* sectors run, not how any one sector's numbers are computed; every call site states the sector list explicitly already. |
| `panel_batch.py:145-146` | `universe_dir`, `data_path` | `""` (→ settings constants) | Path resolution only; same data either way. |
| `panel_batch.py:149` | `force_download` | `False` | Controls whether cached data is re-fetched from source; same on-disk data either way once cached. |
| `panel_batch.py:154-156` | `start_date`, `end_date`, `max_steps` | `None` | `None` means "no cap" — the full, unbounded computation. This is the *maximal* behaviour, not a hidden reduced one; every production call site states its own bound explicitly already because they all deliberately want a small/demo run. |
| `panel_batch.py:157` | `debug` | `False` | Gates extra logging/columns only (verified: `_build_pair_candidate_rows_for_date`'s `debug` param only adds diagnostic dict keys, doesn't change accept/reject logic). |
| `panel_batch.py:160-162` | `persist_result`, `persist_residual_params`, `persist_dir_template` | `True, True, "V2.pair"` | Persistence/output-path only; doesn't change what's computed, only whether/where it's written. |
| `config.py:243-246` | `DataConfig.candidate_panel_subdir`, `.data_path`, `.price_field`, `.return_method` | `""`, `"data"`, `"Close"`, `"log"` | Paths and the (always-used-as-is) field/return-method choice; every call site restates `price_field`/`return_method` identically where it matters. |
| `config.py:247` | `DataConfig.force_download` | `False` | Same as above. |
| `config.py:751-754` | `RunConfig.start_date`, `.end_date`, `.progress`, `.progress_step` | `None, None, False, 10` | Date bounds default to "unbounded" (maximal, not reduced); `progress`/`progress_step` are tqdm display only. |
| `config.py:771-772` | `ExecutionConfig.allow_fractional_shares`, `.share_rounding` | `False`, `"nearest"` | Already stated explicitly at every call site (`run_me.py:459-461`, `sweep_defaults.py:48-50`) — not a live default in practice, and rounding convention only matters combined with the commission numbers already in Table 1. |
| `config.py:785-791` | `PerformanceConfig` (all fields) | see file | Reporting/metrics-computation-after-the-fact only; `run_from_config` executes identically regardless of whether/how performance is reported (verified: `PerformanceConfig` is not read anywhere upstream of trade execution — only in the post-hoc `performance_report.py`). |
| `config.py:799-812` | `PersistenceConfig` (all fields) | see file | Output-writing only, post-simulation. |
| `config.py:823-826` | `SpectrumConfig` (all fields) | see file | Additional diagnostic capture (`ZSpectrumCapture`) recorded alongside the real run; verified it's a passive recorder, not consulted by the trader/sizing/risk path. |
| `config.py:293-294` | `ActivationConfig.one_active_per_group`, `.switch_only_when_flat` | `True, True` | **Not included here as a pass** — see Table 1 doubtful note below; kept out of Table 2 deliberately. |
| `causal_residuals.py:59` | `CausalResidualConfig.remove_residual_pcs` | `0` | Dataclass's own comment: "genuine 'off' state, keeps its default" — already a documented, deliberate exception, not an oversight. |
| `config.py:642` | `IntervalScoringConfig.feature_weights` default factory | `{}` (empty dict) | Guarded by `__post_init__` (`config.py:680-686`): an empty dict only validates when `feature_specs` is also empty, and `feature_specs` itself has no default (required) — the factory can't silently under-specify a live config. |
| `config.py:515` | `FeatureSpec.params` default factory | `{}` | Per-feature kwargs; empty means "use the primitive function's own signature defaults," which is a one-time, explicitly-authored choice per `FeatureSpec` instance in the (never-yet-exercised-by-B) entry-features path, not a silent fallback. |

**Note on `ActivationConfig`:** its own field defaults
(`one_active_per_group=True, switch_only_when_flat=True`,
`config.py:293-294`) are genuinely result-affecting (trading-rule booleans),
but every real call site (the `standard_v1` bundle,
`sweep_defaults.py:44-46`) already overrides both to `False, False` — the
*opposite* of the class default. This is exactly Table 1's failure mode
(a field whose class default is wrong for production) but with zero current
blast radius, since `ActivationConfig` is itself a required field on
`SimulatorConfig` (no default at that level) and nothing constructs a bare
`ActivationConfig()` outside `tests/test_track_a_guards.py`'s
`_minimal_sim_kwargs()` helper (see Q6). Listed here, not silently placed in
Table 2, because "not currently exercised" isn't the same claim as "not
result-affecting" — the class defaults themselves would move results if
relied upon, they're simply not relied upon by anything in the enumeration's
sightline. Recommend implementation session either removes the defaults
(Rule 1, straightforwardly) or at minimum flips them to match the only
values ever used.

## Answers

### 5. Effective values for reproduction, and call-site disagreement

| Field | Effective value everywhere it runs | Disagreement? |
|---|---|---|
| `PairSpreadConfig.min_return_std` | `1e-8` | None — no production/notebook call site overrides it. |
| `PairSpreadConfig.min_level_std` | `1e-8` | None |
| `PairSpreadConfig.min_kappa` | `1e-6` | None |
| `PairSpreadConfig.max_half_life` | `126.0` | None |
| `PairSpreadConfig.tiny_weight_threshold` | `1e-6` | None |
| `PairSpreadConfig.hedge_ratio_methods` | `["pca"]` | The class default (`["ols"]`) is never used — every real call site already states `["pca"]` explicitly. Nothing to pin; the *default* itself can simply be deleted with no bundle entry needed, since no caller depends on it. |
| `hedge_ratio_lb`/`mr_diag_lb` → `residual_cfg.lookback` fallback | Never triggered on any exercised path (see claim 1) | N/A — dead code, no value to pin. |
| `SimulatorConfig.sizing` factory (`base_pair_notional`) | `100_000.0` | None among production call sites, but see `SweepConfig.base_pair_notional` below — same number, different mechanism. |
| `VolSizingConfig.floor_multiplier` / `.cap_multiplier` | `0.2` / `5.0` | None — all 4 call sites use bare `VolSizingConfig()`. |
| `RiskManagerConfig.max_gross_exposure` / `.max_ticker_exposure_pct` | `10.0` / `0.15` | None — `b_baseline_harness.py` (literal), `run_me.py` (via `config/demo_materials.yaml:54-55`, not a `.py` literal), and `SweepConfig`'s own field defaults (`sweep_runner.py:93-94`, feeding `sweep_runner.py:279-283`) all agree. |
| `ExecutionConfig.commission_per_share` | `0.005` | None |
| `ExecutionConfig.commission_per_order` | `0.0` | None |
| `ExecutionConfig.min_commission_per_order` | `1.0` | None |
| `ExecutionConfig.max_commission_per_order` | `9.79` | None |
| `ExecutionConfig.max_commission_pct_of_trade` | `0.01` | None |
| `ExecutionConfig.short_borrow_rate_annual_bps` | `30.0` | None |
| `ExecutionConfig.min_abs_units` | `0.5` | None |
| `CandidateSelectionConfig.require_success` | `True` | None — only call site is the bundle, which doesn't touch it. |
| `SweepConfig.entry_z` / `.exit_z` / `.z_lookback` / `.base_pair_notional` / `.max_ticker_exposure_pct` / `.max_gross_exposure` / `.start_date` / `.end_date` | `2.0` / `0.0` / `21` / `100_000.0` / `0.15` / `10.0` / `"2010-01-01"` / `"2025-12-31"` | **Flagged, not a clean pin:** these are `SweepConfig`'s own field defaults, used only when a `RUNS` entry doesn't override them. Every historical sweep row that *did* override one is, by definition, evidence of disagreement — this table can't certify "no disagreement" for a sweep-parameter surface the way it can for a fixed-role config, because disagreement across sweep rows is the sweep's entire purpose. Pinning these into a bundle only makes sense for the *non-swept* subset (which is what `DEFAULT_CONFIGS`/`_STANDARD_V1` already does) — the swept subset should keep its per-`SweepConfig`-instance value and simply stop calling that a "default." |
| `PortfolioMeanReversionTrader.default_abs_target_group_exposure` | `1.0` | N/A — one call site, never overridden, never exercised by B. |

**Fields that cannot be pinned to one bundle value without changing
something:** none among the `*Config` classes proper — every one of them
shows single-value agreement across all production/notebook call sites.
`SweepConfig`'s own swept fields (`entry_z`, `exit_z`, `z_lookback`,
`base_pair_notional`, `max_gross_exposure`, `max_ticker_exposure_pct`,
`start_date`, `end_date`) are the one real exception, and by construction —
they are supposed to vary per run. Treat them as "no default, required per
`SweepConfig` instance" rather than "pin one bundle value."

### 6. How existing tests construct configs

Both patterns exist, split roughly along "is this a Track-A/D-style
guard/hygiene test" vs. "is this a Track B window-mechanics test":

- **Field-by-field, hand-rolled:** `tests/test_track_a_guards.py` defines a
  local `_minimal_sim_kwargs()` helper (`:36-45`) that constructs
  `DataConfig()`, `CandidateSelectionConfig()`, `ActivationConfig()`,
  `MRDiagnosticsConfig(lookback=21, compute_frequency="off")`,
  `PairSpreadTraderConfig()`, `RunConfig()`, `ExecutionConfig()`,
  `CapitalConfig(total_capital=1_000_000.0)` — every optional-field config
  except `CapitalConfig` (required) and `MRDiagnosticsConfig` is constructed
  bare, relying on class defaults. 4 `SimulatorConfig(...)` call sites in
  that file build on top of this helper. `tests/test_panel_batch_windows.py`
  (2 sites) and `tests/test_panel_batch_config.py` (10 sites) construct
  `PanelBatchConfig` field-by-field directly, each with its own module-level
  `_PAIR_CFG = PairSpreadConfig(hedge_ratio_methods=["pca"], min_obs=2)`
  constant reused across the file's test methods.
- **Bundle-based, already exists:** `tests/test_sweep_defaults.py` goes
  through `SweepConfig` → `dedup_key` → `_build_sim_config` →
  `get_default_bundle("standard_v1")` / `merge_defaults` (`:75-79`), and
  separately asserts each bundle field equals a hand-constructed comparison
  object (`:48-66`) and that the bundle's key set is exactly the seven
  non-swept fields (`:68-73`). This *is* the bundle-based test helper the
  brief's "Expected cost" section asks for — it doesn't need to be built,
  it needs to be reused more widely.

**Rough call-site count that would need updating** once `PairSpreadConfig`'s
four/five numeric fields and `ExecutionConfig`'s seven cost fields lose their
defaults: every bare `PairSpreadConfig(...)` and `ExecutionConfig(...)`
construction that doesn't already state them.

- `PairSpreadConfig(`: 7 files (`test_gap_injection.py`, `test_min_obs.py`,
  `test_pairwise_dropna.py`, `test_panel_batch_config.py`,
  `test_panel_batch_windows.py`, plus the two production sites in
  `b_baseline_harness.py` and `run_me.py`, plus both notebooks 01 and 03).
  Of these, `test_gap_injection.py`, `test_pairwise_dropna.py`,
  `test_min_obs.py`, and `tests/test_track_a_guards.py`'s pair-cfg
  construction (`:186-193`) **already state all four/five fields
  explicitly** (deliberately permissive test values) — no change needed.
  The remaining bare constructions — `test_panel_batch_config.py`'s
  `_PAIR_CFG` module constant (used ~10 times but defined once),
  `test_panel_batch_windows.py`'s `_PAIR_CFG` (defined once, used twice),
  `b_baseline_harness.py:111`, `run_me.py:235`'s `PairSpreadConfig(...)` call,
  and both notebooks — **roughly 6 distinct construction sites**, not ~20,
  because most "call sites" share one module-level constant.
- `ExecutionConfig(`: exactly 2 call sites total in the whole repo
  (`run_me.py:459`, `sweep_defaults.py:48`) plus
  `tests/test_track_a_guards.py`'s bare `ExecutionConfig()` inside
  `_minimal_sim_kwargs()` (1 site, reused by 4 tests) and
  `tests/test_sweep_defaults.py:59-61`'s comparison object (which restates
  `allow_fractional_shares`/`share_rounding` only, same gap). **4 distinct
  construction sites.**
- `VolSizingConfig()` bare: 4 sites (`sweep_runner.py:274`, `run_me.py:447`,
  `b_baseline_harness.py:176`, notebook `02_run_simulation.ipynb`).
- `RiskManagerConfig(`: already states both numeric fields everywhere it's
  constructed — 0 sites need updating for *this* field, only for whatever
  the implementation session decides about the `None`-sentinel default
  (Table 1 #11), which is a `SimulatorConfig`-level fix, not a call-site one.

Total distinct construction sites needing an update, by my count: roughly
**14-16**, not the "every test" scale the brief's "Expected cost" section
implies — because the existing `_PAIR_CFG`-module-constant pattern already
concentrates most of the blast radius into a handful of shared constants.

### 7. Do the three howto notebooks construct configs directly?

Yes, all three. Full inventory (cell IDs from each notebook's own JSON,
since cell numbers shift on re-execution):

**`01_create_candidate_panel.ipynb`**, cell `327ce92229a3`/`e129ead3962a`
(per Track B's own commits) — cell containing the panel build:
- `CausalResidualConfig(...)` — three variants (one per `ResidualMode`
  branch), all fields already stated per-mode (required fields; no
  optional-field reliance to fix).
- `PanelBatchConfig(residual_configs=[residual_cfg], hedge_ratio_lb=...,
  mr_diag_lb=..., selected_sectors=..., frequency="W-FRI",
  max_steps=MAX_STEPS, pair_cfg=PairSpreadConfig(hedge_ratio_methods=["pca"],
  min_obs=MIN_OBS), persist_result=True, persist_residual_params=True,
  persist_dir_template="howto")` — the `PairSpreadConfig(...)` inline
  leaves `min_return_std`/`min_level_std`/`min_kappa`/`max_half_life`/
  `tiny_weight_threshold` at class defaults. **1 site to update.**

**`02_run_simulation.ipynb`**, cell `d59987351d18`:
- `get_default_bundle("standard_v1")` then overrides `performance=` and
  `persistence=` — bundle-based, no change needed for the bundle-sourced
  fields.
- `DataConfig(...)`, `ZScoreConfig(...)`, `PairSpreadTraderConfig(entry_z=,
  exit_z=)`, `RunConfig(...)` — all fields relevant to results stated
  explicitly.
- `SizingConfig(base_pair_notional=BASE_PAIR_NOTIONAL,
  vol_normalize=VolSizingConfig())` — bare `VolSizingConfig()`. **1 site.**
- `RiskManagerConfig(max_gross_exposure=10.0, max_ticker_exposure_pct=0.15)`
  — already explicit, no change needed once the class default is removed
  (the call already restates the value the default currently is).

**`03_full_simulation_pipeline.ipynb`**:
- Cell `bd87066a`/`9e9f695a` region: `PanelBatchConfig(...,
  pair_cfg=PairSpreadConfig(hedge_ratio_methods=["pca"], min_obs=MIN_OBS))`
  — same gap as notebook 01. **1 site.**
- `SweepConfig(entry_z=ENTRY_Z, z_score_overrides=[...],
  candidate_panel_subdir=..., start_date=None, end_date=None)` — goes
  through `run_sweep` → `_build_sim_config` → bundle; `ExecutionConfig`,
  `VolSizingConfig` (via `sweep.vol_normalize` defaulting `True` →
  `_build_sim_config`'s bare `VolSizingConfig()`), and `RiskManagerConfig`
  numbers (via `SweepConfig`'s own `max_gross_exposure=10.0`,
  `max_ticker_exposure_pct=0.15` field defaults, never overridden by this
  call) are all inherited implicitly here. **This notebook's exposure to
  Table 1 items is entirely through `SweepConfig`'s own defaults, not
  through a config object it constructs directly** — which is exactly why
  `SweepConfig` has to be in scope (see below): fixing everything else and
  leaving `SweepConfig` alone would leave notebook 03 silently unchanged
  while notebooks 01/02 and the harness get verbose.

Total: **3 direct `PairSpreadConfig(...)` gaps (one per relevant notebook,
counting 01 and 03) + 1 `VolSizingConfig()` gap (02)**, plus notebook 03's
indirect exposure via `SweepConfig`. All three notebooks re-execute cleanly
today (per Track B's commits) and must be re-executed again once these
become required arguments, per the brief's acceptance criterion.

## Anything the brief asks for that cannot be done as described

1. **"`DEFAULT_CONFIGS`... the bundle name enters `config_hash`" (claim 3)
   is not quite true** (see above) — the *values* enter the hash, the name
   doesn't. This doesn't block anything in scope; it just means D's
   acceptance test ("the baseline file does not move") is checking the
   right thing (resolved values) via the wrong mental model (bundle
   identity) if taken literally. No spec change needed, just don't rely on
   `config_hash` changing as a signal that a bundle *name* changed — it
   won't, if the values are unchanged.

2. **`SweepConfig` is not named anywhere in the brief, but it is where the
   largest concentration of live numeric defaults actually sits** (Table 1
   #19, plus its `vol_normalize: bool` gate on the `VolSizingConfig()`
   default). The brief's scope section talks about `PairSpreadConfig`,
   the fallback, and "every `default_factory` in the config layer" — which
   textually covers `SweepConfig` (it has several `default_factory`-shaped
   `None`-then-construct patterns: `vol_normalize`, `entry_features`, etc.)
   but the brief's own "Why this track is worth doing" narrative frames the
   problem entirely around `PanelBatchConfig`/`SimulatorConfig`, never
   mentioning `SweepConfig` by name. Recommend the implementation session
   explicitly decide whether `SweepConfig`'s "Defaults = current best
   baseline" docstring is grandfathered (it's arguably *already* a
   versioned-bundle-shaped object — every `RUNS = [SweepConfig(...)]` list
   is itself an explicit override list) or whether it too loses its
   defaults. This spec does not decide it — flagging it is the enumeration's
   job, not this session's to resolve, since "changing any value" and
   "structural changes to config classes" are both explicitly out of scope
   for D per the brief.

3. **The brief's "no numeric defaults in dataclasses" (Rule 1) and "no field
   resolves to another config's field via a `None` sentinel" (Rule 2) don't
   have a rule that names `SimulatorConfig.risk_manager: RiskManagerConfig |
   None = None`'s failure mode** — a `None` default that doesn't resolve to
   *any* value, it disables a whole subsystem. Table 1 #11 is reported as
   the closest fit but the brief's three rules don't cleanly cover it;
   recommend either a fourth rule or explicit acknowledgment that this one
   is being fixed under Rule 1's spirit ("anything that affects results")
   rather than its letter ("numeric... in dataclasses").

4. **Exhaustive engine-internal dataclass sweep was done, but the enumeration
   above only reports one exception (`PortfolioMeanReversionTrader`).** I
   cannot rule out with total confidence that no other engine class has a
   similar stray default beside a `*Config` field, since "read every
   dataclass in `src/simulator/`, `src/candidates/`, `src/residuals/`" was
   done by grep-and-read rather than a mechanical AST walk restricted to
   "dataclass with a `*Config`-typed field plus at least one other defaulted
   field." A mechanical version of that specific check would be cheap to
   write in the implementation session and would be strictly more thorough
   than what this spec session did by hand.

## Proposed commit order

Structured so `B_baseline.txt` can be checked after each commit that could
plausibly move it, per the acceptance criterion, and so later commits don't
depend on earlier ones picking the wrong bundle value.

1. **Kill the dead `hedge_ratio_lb`/`mr_diag_lb` → `residual_cfg.lookback`
   fallback** (claim 1 / Table 1 #7). Pure deletion, confirmed dead on every
   exercised path — do this first since it's risk-free and shrinks the
   surface before the harder commits.
2. **Remove `PairSpreadConfig`'s five numeric defaults + the
   `hedge_ratio_methods` factory default**, add the required-field test
   (brief's acceptance criterion: "a test asserting... raises rather than
   silently defaulting"), update the ~6 real construction sites (harness,
   `run_me.py`, both notebooks, both `_PAIR_CFG` test constants) to state
   `min_return_std=1e-8, min_level_std=1e-8, min_kappa=1e-6,
   max_half_life=126.0, tiny_weight_threshold=1e-6` explicitly at each site
   or via a bundle, whichever the implementation session decides for this
   class (it isn't currently in `DEFAULT_CONFIGS`, which only covers
   `SimulatorConfig` fields). Run `B_baseline.txt` — must not move.
3. **Remove `ExecutionConfig`'s seven cost-field defaults.** Only 4
   construction sites total; update `run_me.py`, `sweep_defaults.py`'s
   `_STANDARD_V1`, and the two test sites to state
   `commission_per_share=0.005, commission_per_order=0.0,
   min_commission_per_order=1.0, max_commission_per_order=9.79,
   max_commission_pct_of_trade=0.01, short_borrow_rate_annual_bps=30.0,
   min_abs_units=0.5`. Run `B_baseline.txt` — must not move (this one
   directly touches the reported cost/PnL numbers, so it's the highest-
   value single check in this whole track).
4. **Remove `VolSizingConfig`'s two factory-invoked defaults**, update the
   4 bare-`VolSizingConfig()` call sites to state `floor_multiplier=0.2,
   cap_multiplier=5.0` explicitly. Run `B_baseline.txt` — must not move.
5. **Remove `SimulatorConfig.sizing`'s default factory** (Table 1 #8) —
   currently unexercised, so this commit should show zero baseline
   movement by construction; still worth its own commit since it's a
   distinct field with its own required-ness change.
6. **`RiskManagerConfig`'s two numeric defaults**, plus a decision on
   `SimulatorConfig.risk_manager`'s `None` default (Table 1 #10/#11) — these
   two are coupled (fixing the numeric defaults doesn't fix the "no risk
   manager at all" failure mode, and vice versa), so land together with
   whatever decision the implementation session makes about #11's
   ambiguity. Run `B_baseline.txt` — must not move (every current call
   site already states 10.0/0.15, so a move here would mean the fix was
   done wrong).
7. **`CandidateSelectionConfig.require_success`** and the
   `ActivationConfig` note — smallest, most mechanical commit, bundle
   already exists to receive it.
8. **`PortfolioMeanReversionTrader.default_abs_target_group_exposure`** —
   last, since it's dormant on B's baseline and has no way to verify via
   `B_baseline.txt` at all (portfolio mode isn't exercised). Needs its own
   direct test instead.
9. **`SweepConfig`** — deliberately last and pending the decision flagged in
   "Anything that cannot be done as described" item 2. If in scope, this is
   the largest single commit (8 fields) and the one most likely to require
   re-executing notebook 03 and touching `sweep_defaults.py`'s registry
   shape.

Each commit re-runs `python -m unittest discover -s tests` and
`python scripts/b_baseline_harness.py` before proceeding to the next, per
the brief's acceptance criteria (full suite green; baseline's `## counts`
section unchanged; the removed-default test added per commit rather than
batched at the end).
