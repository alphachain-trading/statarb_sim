# Track I — spec

Read-only spec session. Every claim about existing behaviour below carries
`path:line`; `NOT FOUND` is used rather than inferring. Every caller grep covered
`src/`, `tests/`, `scripts/`, `run_me.py`, and `notebooks/**/*.ipynb`. All
measurements were taken through the shipped code path (real `UniverseDataLoader`,
real `check_prices_for_corruption`, real committed cache data), using the project's
own `.venv` (py3.14.3).

---

## Part 1 — Semantics

### 1.1 `start_after_nan`

**What it trims, at what grain, and where.** `UniverseDataLoader.load`
(`src/data/universe_loader.py:93-106`):

```python
if start_after_nan:
    prices = umd.prices
    mask = prices.isna().any(axis=1)
    first_valid_index = mask.idxmin() if not mask.all() else None
    if first_valid_index is not None:
        umd = UniverseMarketData(prices=prices.loc[first_valid_index:], ...)
```

`umd.prices` is one DataFrame per `UniverseDataLoader` instance, with MultiIndex
columns `(ticker, field)` for **every** ticker the loader was constructed with —
members, `proxy_etf`, `benchmark`, and `risk_free` together
(`universe_config.py:93-103`, `all_symbols()`). `mask` is `True` on any row where
**any** ticker has **any** NaN in **any** field. `first_valid_index` is the first
row where every ticker has a complete row. Everything before it is dropped for
**every ticker in the group**, not per ticker.

The grain is per **group**, not per ticker, and a `UniverseDataLoader` is
constructed once per group `yaml`, one `UniverseConfig` per instance, at every one
of the five call sites (`panel_batch.py:285-286`, `simulator_factory.py:271-272`,
`simulator_factory.py:435-436`, `run_me.py:124+139`,
`universe_loader.py:479-480`). So "per group" here means: one sector's yaml
(e.g. `energy.yaml`, 14 symbols including `SPY`/`^IRX`/`XLE`) is trimmed as one
unit. A multi-group `DataConfig` loads and trims each group separately
(`_load_umd`, `simulator_factory.py:418-444`) before merging
(`_merge_umds`, `simulator_factory.py:452+`), so the per-group grain holds even in
a multi-group run — one group's late ticker cannot push another group's start
date.

**Which ticker binds, for the harness's two universes.** Measured directly against
the committed cache
(`data/market/universes/{materials,energy}_only_v1/prices_daily.parquet`,
`config/universes/sp500_v1/{materials,energy}.yaml`):

| Universe | Binding ticker | Ticker's role | Bind date | Rows discarded | Group range |
|---|---|---|---|---|---|
| `materials_only_v1` (16 tickers) | `CF` | equity member (CF Industries — `materials.yaml:67,94`, `kind: equity`) | 2005-08-11 | 3,937 / 9,198 (42.8%) | 1990-01-02 → 2026-07-13 |
| `energy_only_v1` (14 tickers) | `XLE` | `proxy_etf` (`energy.yaml:48,71`, `kind: benchmark_etf`) | 1998-12-22 | 2,269 / 9,198 (24.7%) | 1990-01-02 → 2026-07-13 |

Every other ticker in both groups (including `SPY`, `^IRX`, `WMB`, `XOM`) has data
back to 1990-01-02 or earlier. In materials, one real equity (a 2005 IPO/spinoff)
discards **43% of the group's available history for every other member**,
including tickers with 15+ years of pre-2005 history. In energy, the discipline is
worse in kind: the binding ticker is not even a traded equity — it's the sector's
own benchmark ETF (`XLE`, launched Dec 1998) — so the entire group's usable
history is capped by a benchmark's inception date, not by any tradeable member's.
This is exactly the failure mode `I_data_preparation.md`'s Scope section warns
about: "a per-group start date discards early history for every ticker in the
group because one arrived late."

Measured by replicating `universe_loader.py:93-106`'s mask logic in an ad hoc
script run against the two committed `prices_daily.parquet` files directly
(`.venv/bin/python`, no network access, real cached data) — not committed, since
it is a one-off measurement script, not a test.

### 1.2 `check_for_corruptions`

**What it checks.** `check_prices_for_corruption`
(`src/data/qc.py:14-172`), called via `UniverseMarketData.qc_warnings()`
(`universe_marketdata.py:68-69`) with all defaults
(`extreme_return_threshold=0.50`, `ignore_pre_inception=True`,
`outside_range_tolerance=0.001`). Per ticker: missing required fields, missing
OHLC rows (post-inception), missing Volume rows, `High < Low`, `Open` outside
`[Low, High]`, `Close` outside `[Low, High]` beyond a 0.1% tolerance,
non-positive OHLC, negative Volume, and `|daily return| > 50%`.

**What it does on detection.** `universe_loader.py:107-109`:

```python
if check_for_corruptions:
    for w in umd.qc_warnings():
        print(w)
```

Log only — `print()`, not `print_qc_warnings` (`qc.py:175-182`, itself also
print-only). No raise, no drop, no mutation of `umd`. Detection and correction are
fully decoupled: nothing downstream ever sees that a warning fired.

**Cost, measured through the shipped path**
(`check_prices_for_corruption(prices)` called directly on the two committed
`prices_daily.parquet` files, 9,198 rows × 80/70 columns): 0.047s (materials),
0.040s (energy). Negligible at this scale.

**Why it is off in `PanelBatchConfig`.** `git blame -L151,151
src/candidates/panel_batch.py` → `0097e44396ff9579f9cb37dc6577924c3f77895e`,
message `"initial commit: canonical materials and full-universe pipeline"`. Same
commit for `DataConfig`'s three flags (`config.py:207-209`,
`git blame -L207,209`) and effectively the same for
`UniverseDataLoader.load`'s signature default
(`universe_loader.py:36`, blamed to `3ac66ce8`, a same-era commit whose message
is an unrelated `.gitignore`/import fix, not a data-quality decision — the
content traces to the original, unlabeled introduction). **NOT FOUND**: no commit
message, PR description, or code comment anywhere in history states a reason
`PanelBatchConfig` turned this off while `DataConfig` left it on. Given the
measured cost (~40-50ms on the harness's small universes) offers no performance
justification at this scale, and nothing in the codebase says otherwise, this
reads as an unexplained inconsistency from the repository's first commit, not a
recorded deliberate choice — consistent with `found.md`'s own framing ("flagged
'doubtful'" in Track D's spec session).

### 1.3 Do the two flags interact?

**Yes, in one direction, and it is measured, not hypothetical.**
`universe_loader.py:93-109`: the trim (`start_after_nan`) runs first, then the
corruption check (`check_for_corruptions`) runs on whatever `umd` the trim step
produced. So `start_after_nan=True` hands the corruption checker a strict subset
of rows; `start_after_nan=False` hands it everything.

Measured directly (`check_prices_for_corruption` run on the full vs. the
group-trimmed price frame for both harness universes): the **same 15 (materials)
/ 8 (energy) warning issues** fire either way — `ignore_pre_inception=True`
(`qc.py:47-51`) already drops each ticker's own leading NaN run before checking,
so the group trim mostly discards rows the checker would have ignored anyway.
But two of those warnings' **row counts change**: `^IRX`'s `missing_ohlc` and
`missing_volume` counts drop from 31 (untrimmed) to 5 (materials, trimmed) / 8
(energy, trimmed). `^IRX` (13-week T-bill yield, `risk_free` role in both groups)
has internal — not just leading — missing OHLC/Volume rows scattered across its
full 1990-2026 history; most of them happen to fall before each group's own bind
date, so trimming silently removes them from what the corruption check ever
reports. This is a real, if narrow, instance of "trimming changes what
corruption detection sees": whether a given internal gap in a
non-price-representative ticker like `^IRX` gets flagged at all depends on
whether the group's *other* members happen to have started even later.

**The reverse — does corruption detection change what trimming sees?** No.
`check_for_corruptions` never mutates `umd` (1.2); it runs strictly after the
trim in the same `load()` call and its output (print statements) feeds nothing
back into `umd.prices`.

Measured by calling `check_prices_for_corruption` directly (real function, real
data) on `prices.loc[bind_date:]` vs. the full frame for both universes,
`bind_date` per 1.1's table; not committed, one-off measurement.

### 1.4 The five sources — call sites after F2

Re-verified against current `main` (post `9b53440`), not carried forward from
the pre-F2 table.

**1. `DataConfig` (`check_for_corruptions=True`, `start_after_nan=True`,
`force_download=False` — `config.py:207-209`).** Consumed by `_load_umd`
(`simulator_factory.py:411-449`), called unconditionally from `run_from_config`
(`simulator_factory.py:148`) — **even on the live-generation branch** (see
source 5 below; both loads happen in the same `run_from_config` call when
`candidate_generation` is set). Reached by: `run_me.py`'s `stage_simulate`
(`run_me.py:446,456` → `_build_sim_config`, `run_me.py:340-` builds `DataConfig`
with no flag overrides), `sweep_runner.py`'s simulate step (builds a disk-based
`DataConfig`, per F2's R11 pre-check — no `candidate_generation`), and
`scripts/b_baseline_harness.py`'s `_build_sim_config`
(`b_baseline_harness.py:165-181` — `DataConfig(...)` with no flag overrides).
No caller anywhere overrides any of the three flags — verified directly at each
`DataConfig(...)` construction site.

**2. `PanelBatchConfig` (`check_for_corruptions=False`, `start_after_nan=True`,
`force_download=False` — `panel_batch.py:150-152`).** Consumed by
`run_panel_batch`'s per-group loader call
(`panel_batch.py:286-295`). Real callers of `run_panel_batch`: `run_me.py`'s
`stage_residuals`, both branches (`run_me.py:289` multi-group,
`run_me.py:300` single-group; `_make_panel_batch_cfg`, `run_me.py:218-254`,
overrides neither flag), and notebooks `01_create_candidate_panel.ipynb` and
`03_full_simulation_pipeline.ipynb` (confirmed via
`grep -rl run_panel_batch notebooks/`).
`panel_batch.py:33` is a **docstring usage example**, not a real call site —
excluded. `sweep_runner.py` does **not** call `run_panel_batch`; it requires a
pre-built panel on disk (`SweepConfig.candidate_panel_subdir` mandatory,
`sweep_runner.py:93-95,196-212`, per F2's R11 pre-check). Tests
(`tests/test_panel_batch_windows.py:71,113`) also call it directly.

**3. `market_snapshot.py:277` (hardcoded literal: `force_download=True,
check_for_corruptions=True, start_after_nan=False`).** Inside `_download_group`
(`market_snapshot.py:250-286`), called from `_mint_snapshot`
(`:327+`), called from `ensure_market_snapshot` (`:457-490`).
**Zero production callers of `ensure_market_snapshot` or `load_market_snapshot`
exist today** — grepped across `src/`, `run_me.py`, `scripts/`; both are called
only from `tests/test_market_snapshot.py`. The snapshot layer (Track C) is built
but entirely unwired into any of the four other sources' call paths, confirming
the brief's and `found.md`'s premise.

**4. `UniverseDataLoader.load`'s own signature default
(`check_for_corruptions=True`, `start_after_nan=True` — `universe_loader.py:36`),
reached whenever a caller omits both kwargs.** Two production reachers:
`run_me.py`'s `stage_download` (`run_me.py:140`,
`loader.load(force_download=force)` — omits the other two), and
`ensure_universe_data` (`universe_loader.py:454-480`,
`.load(force_download=force_download)` at `:480` — same omission), called from
`notebooks/howto/03_full_simulation_pipeline.ipynb`
(`found.md`'s "not actually dead" entry). Numerically identical to source 1
today (both `True`/`True`), but structurally distinct: it is reached by
*omission*, not by an explicit `DataConfig` field, so unifying sources 1 and 4
to one value does not by itself remove source 4 as a *separate* place the value
is stated (or not stated) — see Part 4.

**5. `CandidateGenerationConfig` (F2 C6b; `force_download=False`,
`check_for_corruptions=False`, `start_after_nan=True`, defaulting to
`PanelBatchConfig`'s values — `config.py:817-819`).** Consumed by
`generate_candidate_panels_by_group`'s own per-group loader
(`simulator_factory.py:271-280`), wired into `run_from_config` via
`_generate_candidates_live` when `config.candidate_generation is not None`
(`simulator_factory.py:154,178`, `_generate_candidates_live` at `:319-353`
calling `generate_candidate_panels_by_group` at `:336`).

**Correction to the brief's amended text.** The brief (Commit 0.1, "The
fifth was added deliberately...") states "This is now the path the simulator
actually uses for candidate generation... not a peripheral fifth case." Verified
directly: **`config.candidate_generation` is set by exactly one production
caller today** — `scripts/b_baseline_harness.py:183`
(`_build_sim_config`). Neither `run_me.py`'s `_build_sim_config`
(`run_me.py:340-`) nor `sweep_runner.py`'s config builder sets it (grepped for
`candidate_generation` across the whole repo outside `simulator_factory.py`,
`config.py`, and `b_baseline_harness.py` — no other hits). So the live-generation
branch is **not yet the path `run_me.py` or `sweep_runner.py` runs** — it is the
path the baseline harness runs, and the baseline harness is the instrument every
result-changing track (I, J, G) verifies neutrality against. The brief's claim
is correct in the sense that matters for this track — this flag combination now
gates the numbers in `B_baseline.txt` — but "the simulator actually uses" should
not be read as "every production entry point uses"; `run_me.py`'s `simulate`
stage and `sweep_runner.py` still run the four-source (soon-to-be-five)
disk-panel path exclusively, and will continue to until Track J retires it.

**A structural fact this tracing surfaces, not in the brief's table.** When
`config.candidate_generation` is set, `run_from_config` still calls `_load_umd`
unconditionally at line 148 — under `DataConfig`'s flags — *and*
`generate_candidate_panels_by_group` separately loads its own UMD per group
under `CandidateGenerationConfig`'s flags. **A single `run_from_config` call with
live generation enabled performs two independent, differently-flagged UMD loads
of the same underlying data**, not one. Unifying the five *values* to one number
does not by itself collapse this to one *load* — that is a separate, structural
question Part 4 returns to.

---

## Part 2 — The snapshot case

### 2.1 `market_snapshot.py:277`

**What the snapshot stores.** `_download_group` (`market_snapshot.py:250-286`)
loads with `check_for_corruptions=True, start_after_nan=False`. Per
`universe_loader.py:93-111`, `start_after_nan=False` skips the trim block
entirely — the `umd` returned (and then persisted to the snapshot directory by
`_finalize_snapshot`/the manifest-building code) is the **full, untrimmed** raw
price history, leading NaNs and all, for every ticker in the group. It is not a
group-consensus-trimmed frame; it is exactly what `_download_and_build` produced.
`check_for_corruptions=True` means QC warnings print during the mint (log only,
per 1.2 — no effect on what gets stored).

**Is trimming at mint time lossy?** Yes, and destructively so. The trim in
`universe_loader.py:101-106` replaces `umd.prices` with
`prices.loc[first_valid_index:]` — a slice, not a mask; rows before the bind date
are gone from the in-memory object with no way to recover them from it. Applied
at mint time, this would mean: (a) the bind date is fixed forever by whichever
tickers happened to be in the group's yaml *at mint time* — a group with 11
members might have 10 years more history for 10 of them than the 11th ticket
allows, permanently discarded rather than recoverable; (b) two legitimate
consumers wanting different grains (e.g. a per-ticker trim vs. today's per-group
trim, once Part 4's unification picks a value) could never both be served by one
frozen artifact — trimming is a *consumer-specific* decision (which ticker set,
which grain) baked into a group-scoped, immutable, versioned file, unlike
`start_after_nan=False`, which just stores the strict superset every consumer's
own trim can be computed from afterward.

**Is the brief's argument consistent with the code?** Yes, exactly.
"Freeze raw, trim at run time" describes `start_after_nan=False` precisely: the
module docstring's own framing (`market_snapshot.py:1-79`) is about
*immutability against re-download*, not about serving pre-trimmed data — nothing
in the module ever mentions trimming as a concern, and the one place trimming
would matter (`_download_group`) explicitly disables it. This looks like a
correct, if undocumented, decision, not an accident: **recommend it be written
down explicitly** (e.g. a short comment at `market_snapshot.py:277`) rather than
left implicit, per the brief's own "if that is the reasoning, it should be
written down" — see Final section, proposed `found.md`/doc entries.

### 2.2 The three (now four) loader call sites

**Current state**, per Part 1.4: sources 1 (`DataConfig`/`_load_umd`), 2
(`PanelBatchConfig`/`run_panel_batch`), 4
(`UniverseDataLoader.load` signature default via `stage_download` and
`ensure_universe_data`), and 5 (`CandidateGenerationConfig`/
`generate_candidate_panels_by_group`) all call `UniverseDataLoader(...).load(...)`
against the **live cache** (`DATA_UNIVERSES`), never the snapshot layer. Zero of
them call `ensure_market_snapshot`/`load_market_snapshot` (Part 1.4, source 3).
The brief and `found.md` both say "three loader call sites" — that undercounts
by (at least) one after F2 C6b added `generate_candidate_panels_by_group`'s own
load (`simulator_factory.py:271-280`), which is not one of the three named in
`found.md`'s "in-place cache-hit resync" entry (`run_me.py:129-130`,
`panel_batch.py:293-302`, `simulator_factory.py:203-211` — the last of these is
`_load_umd`, a different function from `generate_candidate_panels_by_group`).
Whoever wires "the three loader call sites" needs to also decide source 4's two
reachers (`stage_download`, `ensure_universe_data`) and source 5 — that's up to
six call sites total across five source-identities, not three.

**What `load_market_snapshot` returns, and what wiring requires.**
`load_market_snapshot` (`market_snapshot.py:493-556`) reads
`prices_daily.parquet`/`ticker_info.parquet`/`group_info.parquet`/
`membership.parquet` **directly from the snapshot directory**, verifies content
and ticker-set hashes, and returns a `UniverseMarketData` — untouched by any
trim or corruption check. **Its signature has no `check_for_corruptions` or
`start_after_nan` parameter at all.** So wiring a call site to the snapshot layer
is not a drop-in replacement of `.load(...)`'s three kwargs; the trim/check logic
(`universe_loader.py:93-109`) needs to be applied somewhere, by someone, after
`load_market_snapshot` returns.

Options for where that logic lives (brief says decide the model; the *mechanism*
here is left open for whoever implements):

- **(a) Extract a free function.** Pull `universe_loader.py:93-109` out into
  e.g. `apply_data_prep_flags(umd, *, check_for_corruptions, start_after_nan) ->
  UniverseMarketData`, callable identically after either
  `UniverseDataLoader.load`'s own fetch step or after `load_market_snapshot`.
  `UniverseDataLoader.load` keeps calling it internally for the live-cache path;
  each of the (now) six call sites calls it explicitly after
  `load_market_snapshot(...)`. One implementation, two call shapes. Matches the
  brief's "one source of truth for both flags, stated explicitly by the caller."
- **(b) Add the two params to `load_market_snapshot` itself.** Duplicates
  `universe_loader.py:93-109`'s logic inside `market_snapshot.py`, or has
  `load_market_snapshot` import and call the same extracted function from (a)
  internally — functionally equivalent to (a) but hides the step from the call
  site rather than making it explicit there.
- **(c) Leave `UniverseDataLoader.load` as the only place trim/check ever runs,
  and have each wired call site do: `load_market_snapshot(...)` then construct a
  throwaway `UniverseDataLoader` purely to reuse its trim/check code path**
  (e.g. by calling a would-be `loader._apply_flags(umd, ...)` instance method).
  Avoids extracting a free function but ties every snapshot-reading call site to
  constructing a `UniverseDataLoader` it doesn't otherwise need (no cache dir,
  no download config) just to reach two lines of logic.

(a) is the option consistent with Track D's "one source of truth" framing
already decided for the *value*; it additionally makes the *mechanism* single-
sourced. (c) is the least invasive change but leaves an awkward dependency.

**`UniverseDataLoader.load`'s remaining role**, once wiring lands (any option
above): its cache-fetch machinery
(`_cache_exists`, `_download_and_build`, `_download_prices`, `_normalize_prices`,
`_build_ticker_info`/`_build_group_info`/`_build_membership`, `_persist`) is
**not** replaced by the snapshot layer — `market_snapshot.py`'s own
`_download_group` (`:264`) *constructs and uses* a `UniverseDataLoader`
internally to do the actual downloading for a mint. So `UniverseDataLoader`
itself stays; what changes is that the wired call sites stop calling its
`.load()` method for **reads** and instead call `load_market_snapshot()`. Two
sub-questions remain, options for the brief/implementer:
- Does `.load()`'s own trim/check block (lines 93-109) stay on `.load()` for the
  live-cache path (needed if any caller keeps loading live — see below), move
  entirely into the extracted function of option (a)/(b), or both (thin wrapper)?
- Does the **live-cache path** (`.load()` without a snapshot) survive as a
  legitimate mode at all after wiring, or does every one of these six call sites
  become snapshot-only? `DataConfig.snapshot_id: str | None = None`
  (`config.py:203`) is written as an **opt-in** field ("`None` = the run loaded
  market data through `UniverseDataLoader`'s live cache... true for every run
  today" — comment at `config.py:194-202`), implying dual-mode was the intended
  design — but Scope item 4 of the amended brief also flags this `None` sentinel
  as a Track D violation once wiring lands. Those two intentions are in tension:
  an opt-in `snapshot_id` needs a live-cache fallback to opt out of; Track D's
  rule wants no silently-defaulted mode selector. Whoever implements must pick
  one: keep dual-mode with `snapshot_id` required-but-nullable-with-an-explicit-
  sentinel-value (e.g. a literal string, not bare omission), or make snapshot
  binding mandatory for every wired call site (no live-cache fallback at all).
  This track's brief does not resolve it and this spec does not decide it either
  — it is squarely Scope item 4's business, flagged here because 2.2 surfaces it
  concretely.

### 2.3 Snapshot vs. local cache in practice; does the resync still fire?

**How they can differ.** The on-disk cache format is identical between a live
cache dir (`DATA_UNIVERSES/{universe_name}/`) and a snapshot's per-group dir —
both are `prices_daily.parquet` + three metadata parquets (confirmed: same four
file names, `universe_loader.py:30-34` vs. the snapshot layout in
`market_snapshot.py:44-57`). Critically, **the live cache's `prices_daily.parquet`
is always the raw, untrimmed frame too** — `_persist(umd)` is called in `load()`
(`universe_loader.py:41,92`) strictly *before* the trim/check block
(`:93-109`) in both the fresh-download and the resync branches, so trimming
never touches what's written to disk, live cache or snapshot alike. The
difference is not in what shape of data each stores; it is in **mutability**:
- **Corporate-action re-adjustment**: `yfinance`'s `auto_adjust=True`
  (`universe_loader.py:152`, `:185`) retroactively re-adjusts historical closes
  for splits/dividends. A live cache re-downloaded with `force_download=True`
  silently overwrites its one copy with whatever adjustment vintage is current
  at that moment — no record of the prior vintage survives. A snapshot instead
  mints a new, separately-named directory (`{universe_version}_{new dl_ts}/`)
  and never touches the old one — both vintages remain inspectable.
- **Ticker-set drift**: the live cache's resync branch
  (`universe_loader.py:46-92`, `to_remove`/`to_add`) mutates the cached ticker
  set **in place** the moment a yaml's membership changes, with no snapshot
  concept at all. A snapshot's universe-version model instead requires minting
  an entirely new `universe_version` for any membership change
  (`market_snapshot.py:17-25`; `_mint_snapshot` raises `UniverseChangedError` if
  the yaml drifts from the last snapshot of the same version,
  `_check_universe_unchanged`, `:210-`).

**Can the in-place cache-hit resync still fire, for callers wired to
`load_market_snapshot`?** No — structurally, not just today.
`load_market_snapshot` never touches `DATA_UNIVERSES`; it reads exclusively from
`snapshot_root` (`DATA_MARKET_SNAPSHOTS`, a disjoint directory tree,
`market_snapshot.py:97,510-511`). For call sites migrated to it, the resync
defect (`found.md`'s "in-place cache-hit resync remains live" entry) is closed
by construction, the same way it is already closed for the snapshot *mint* path
itself (`market_snapshot.py:63-72`'s guard). It remains live for whatever call
sites are **not** migrated — concretely, today, that would be `ensure_universe_data`
(notebook-only caller, source 4) if the wiring decision (2.2) leaves it
untouched, and any test that constructs a `UniverseDataLoader` against the real
`DATA_UNIVERSES` cache dir directly
(`tests/test_candidate_generation_equivalence.py:78-81` does exactly this).

**Does E10's zero-diff finding generalize?** No — and F2's own text already
scopes it narrowly, worth restating precisely. E10
(`F2_spec.md:1364-1397`) rebuilt the **same group's bundle twice with no yaml
change between builds** (in-process and cross-process) and got byte-identical
results. That is a necessary precondition for the resync branch to be a no-op:
`to_remove`/`to_add` are computed from `set(config.all_symbols()) -
set(umd.tickers())` and its reverse (`universe_loader.py:47-51`) — with an
unchanged yaml, both sets are empty and the whole resync branch
(`:53-92`) never executes; the loader takes the plain
`_load_from_disk()` path both times. E10 therefore verified determinism of
*repeated loads under a fixed config* — it says nothing about what happens when
the yaml **does** change, which is precisely the scenario the resync branch
exists to handle and the scenario `found.md`'s entry is about. Confirmed by
re-reading the resync branch itself: `to_remove` drops ticker columns from
`umd.prices` in place (`:58-60`), `to_add` downloads and concatenates new
columns (`:62-83`) — both visibly change the DataFrame that later feeds
`build_group_return_bundle`. E10's zero-diff result is real but orthogonal to
the resync question; it does not clear the resync defect, and should not be
cited as if it did.

---

## Part 3 — Gappy universe

### 3.1 What exists today for testing gaps

Grepped `tests/` for "gap": `test_gap_injection.py`, `test_pairwise_dropna.py`,
`test_panel_batch_config.py` (unrelated "collision gap" usage), `test_min_obs.py`
(comment only), `test_market_snapshot.py` (unrelated missing-ticker "gap" in a
manifest, `:260`). Read `test_gap_injection.py` and `test_pairwise_dropna.py` in
full: both inject a NaN directly into a synthetic, in-memory
`GroupReturnBundle.aligned_returns` (`test_gap_injection.py:39-70`) or a fake
residuals DataFrame (`test_pairwise_dropna.py:60-71`) and exercise
`_build_pair_candidate_rows_for_date` / `apply_causal_residual_model` /
pairwise-dropna — Track B's layer, strictly downstream of price data. **None of
this touches `UniverseDataLoader`, `check_prices_for_corruption`, or real price
data at all.** There is no existing test, fixture, or committed universe that
exercises `start_after_nan` or `check_for_corruptions` against data with an
internal (non-leading) gap. The two harness universes have zero internal gaps
over their date range (README's own "Two known blind spots" section,
`found.md`'s residual-fit-weighting entry) — confirmed again here, from the raw
data itself (Part 1.1's measurement finds only a single leading-NaN block per
group, no internal gaps, in both `prices_daily.parquet` files).

### 3.2 A gappy test universe, specified concretely

**Convention to reuse.** `test_market_snapshot.py:1-10,35-60` already
establishes the repo's pattern for gap-free-of-network fixtures: mock
`UniverseDataLoader._download_prices`, write a synthetic yaml under a temp dir,
never touch committed `config/universes/sp500_v1/`. Recommend the same pattern
here — deterministic, no network dependency, no new committed binary fixture
files, and consistent with existing test style.

**Tickers (5, one synthetic group).** Group id `test_group`, universe version
`gappy_v1` (both names free to reuse — no existing yaml under either name):

| Ticker | Role | Behaviour |
|---|---|---|
| `AAA` | member | Dense: full range, no gaps, no corruption. |
| `BBB` | member | Dense: full range, no gaps, no corruption. Paired with `AAA` as the clean control pair. |
| `LATE` | member | Joins 40% into the range (mimics `CF`/materials from 1.1) — exercises `start_after_nan`'s per-group bind. |
| `GAPPY` | member | Dense from day 1, but with a **contiguous internal gap** of 10 business days in the middle of the range (all OHLCV fields NaN for those rows) — exercises the case `start_after_nan` structurally cannot fix (it only trims a *leading* run) and that `check_for_corruptions`'s `missing_ohlc`/`missing_volume` detectors should catch. |
| `BENCH` | benchmark + proxy_etf + risk_free (reuse one dense series for all three roles, since none of this track's mechanics depend on them being distinct) | Dense, full range, no gaps. Also carries the deliberate corruption below, since QC checks run per-ticker and this keeps the corruption isolated from the gap/lateness mechanics under test. |

**Corruption, on `BENCH`, three independent defects (one row each) so each of
`check_prices_for_corruption`'s detectors is exercised at least once beyond the
`missing_ohlc`/`missing_volume` cases `GAPPY` already covers:**
- One row with `High < Low` (`high_below_low`).
- One row with `Close <= 0` (`nonpositive_price`).
- One row with `|return| > 50%` relative to the prior close (`extreme_return`).

**Date range.** `pd.bdate_range("2015-01-05", periods=600)` — about 2.4 years of
business days. Sized so a 252-day trailing window (the harness's own
`hedge_ratio_lb`/`mr_diag_lb`) clears well before the range ends, in case 3.3's
"fold into the harness" option is taken — not required for the unit-level flag
tests in 3.1's gap but cheap to provide for free.

**Concrete parameters** (all fixed, seeded, reproducible):
- `LATE` starts at index 240 (business day ~2015-12-11), NaN before that —
  binds the group's `start_after_nan` trim to that date, discarding 240/600 = 40%
  of the other four tickers' history.
- `GAPPY`'s internal gap: indices 400-409 inclusive (10 business days, roughly
  2016-08 given the above start), all five OHLCV fields set to `NaN`. Chosen to
  fall **after** `LATE`'s bind date, so the gap survives the group-level
  `start_after_nan` trim and is still present in the trimmed frame — the whole
  point of the fixture is a gap `start_after_nan` cannot remove.
- `BENCH`'s three defects: row 100 gets `High` and `Low` swapped
  (`high_below_low`); row 200 gets `Close = -1.0` (`nonpositive_price`); row 300
  gets `Close` multiplied by 2.0 relative to the prior day
  (`extreme_return`, a +100% one-day move, comfortably over the 50% default
  threshold).
- Base price level: all series start at 100.0 with small deterministic daily
  log-returns from a fixed `numpy.random.default_rng` seed (e.g. `42`), so the
  fixture is exactly reproducible without being committed as data.

**How it gets built and committed.** A new test module,
e.g. `tests/test_i_gappy_universe.py`, following `test_market_snapshot.py`'s
shape: a module-level `_fake_download_prices` (or reuse/import
`test_market_snapshot.py`'s if it's factored into a shared test helper — minor
implementation choice) building the above five tickers' data from the fixed
seed/parameters, a `tempfile.TemporaryDirectory()` holding a synthetic
`gappy_v1/test_group.yaml` (schema per `universe_config.py:43-71`'s
`validate()` — `meta.universe_name`, `hierarchy.root`, one `groups` entry with
`members: [AAA, BBB, LATE, GAPPY]`, `proxy_etf`/`benchmark`/`risk_free: BENCH`,
and a `tickers` dict entry for all five), and a
`UniverseDataLoader._download_prices` patch (mirroring
`test_market_snapshot.py:43-60`) returning the fixed synthetic frame instead of
calling `yfinance`. The test itself is the "commit" — no new binary fixture
under `data/market/universes/` or `config/universes/`, nothing for
`git diff` to show as a data file, matching how every other synthetic-universe
test in this repo already works.

Tests to write against it (satisfying the acceptance criterion "trimmed-
observation counts per group reported for a gappy universe"):
- `start_after_nan=True` trims exactly 240 rows off every ticker in the group;
  `=False` trims none. Assert the exact row count, not just "some rows gone."
- `GAPPY`'s internal 10-day gap survives `start_after_nan=True` (it's after the
  bind date) and is present in `umd.prices` — i.e., `start_after_nan` does not
  and cannot fix an internal gap, only a leading one.
- `check_for_corruptions=True` reports exactly the expected warnings: `GAPPY`'s
  `missing_ohlc`/`missing_volume` (10 rows each), and `BENCH`'s three defects
  (`high_below_low`, `nonpositive_price`, `extreme_return`), each with the
  expected row count; `check_for_corruptions=False` reports nothing (no
  `print()` calls — assertable via `unittest.mock.patch("builtins.print")` or by
  calling `check_prices_for_corruption` directly and asserting on the returned
  list, bypassing the print layer).
- The two flags' interaction (1.3): with `start_after_nan=True`, does the
  trimmed frame still show `GAPPY`'s full 10-row gap in the corruption warnings
  (it should, since the gap is entirely after the bind date by construction)?
  This is the fixture's own self-check that the gap and the bind date don't
  interact in a way that would mask the gap.

### 3.3 Should the gappy universe join the baseline harness, or stay separate?

Trade-off, not decided here:

**Join `scripts/b_baseline_harness.py` as a third universe** (alongside energy
and materials):
- *For*: it becomes part of the standing, committed, run-every-track instrument;
  any future track's `B_baseline.txt` diff would automatically cover gap
  handling, not just the two gap-free sectors. Directly satisfies "the two
  harness universes ... may not exercise NaN handling at all" (README's known
  blind spot) for every subsequent track, not just this one.
- *Against*: the harness's whole design is "fixed and small on purpose" using
  **real, committed cache data** for two **real** sectors
  (`b_baseline_harness.py:17-25`). A synthetic universe with fabricated tickers
  and hand-picked corruption is a different kind of fixture — it would need its
  own committed cache directory (breaking the "no new binary fixture" property
  3.2 recommends) or the harness's `UniverseDataLoader` construction would need
  a mock-injection seam it doesn't have today (it calls the real loader against
  real `DATA_UNIVERSES` paths, `b_baseline_harness.py:74`,
  `_build_sim_config`). Folding it in also means every future track's
  `B_baseline.txt` diff must now explain *three* groups' worth of change
  instead of two, for tracks that have nothing to do with data preparation.
- *Consequence either way*: the harness's `## counts`/`## trades` sections
  (README's "binding check") are keyed by `(group_id, residual_key)`
  (`b_baseline_harness.py:278-282`) — adding a group is additive to those
  sections, not a reshaping, so the mechanical cost of joining is low if the
  "against" concerns are otherwise resolved.

**Stay a separate fixture** (3.2's proposal, a standalone test module):
- *For*: keeps the harness's real-data character and small size intact; the
  gappy universe's job (prove the flags behave correctly in the presence of a
  gap) is fully served by direct unit tests against `UniverseDataLoader`/
  `check_prices_for_corruption`, without needing the whole
  candidate-generation-through-simulation pipeline the harness exercises.
- *Against*: every *future* result-changing track still inherits the harness's
  gap-blindness for its own `B_baseline.txt` diffs — this track closes the gap
  only for itself, not for G, H, or whatever comes after.

No recommendation given here per the brief's instruction to give the trade-off,
not decide.

---

## Part 4 — Commit order

For each commit: what it does, whether it can move `B_baseline.txt`, what its
neutrality argument (if any) rests on, and whether that argument is structural
or empirical.

| # | Commit | Moves `B_baseline.txt`? | Argument | Structural or empirical? |
|---|---|---|---|---|
| 1 | Behaviour analysis only (this spec) — no code change | No | N/A | — |
| 2 | Build the gappy test universe + tests (Part 3) | No — new test file, touches no production code | Neutral by construction: a new test file cannot change any call site's behaviour | Structural |
| 3 | Report trimmed-observation counts per group for the harness's two universes, before any unification (Part 1.1's numbers), as a committed measurement (e.g. appended to this spec's evidence, or a small script output) | No | N/A — pure measurement | — |
| 4 | Pick and state the one value for `check_for_corruptions`/`start_after_nan`/`force_download`, remove all three class defaults, require every one of the (now six) call sites to pass all three explicitly; add the "omitting either flag raises" test (acceptance criterion) | **Likely yes** | Depends entirely on which value is picked — see below | Empirical (must be checked against `B_baseline.txt`, cannot be argued structurally, since at least four of five sources disagree today) |
| 5 | Snapshot wiring: extract the trim/check mechanism (2.2 option a/b/c) and repoint the (up to six) call sites at `ensure_market_snapshot`/`load_market_snapshot`, resolving `UniverseDataLoader.load`'s remaining role and the `snapshot_id`/live-cache-fallback question (2.2's open sub-question) | Likely yes, separately from commit 4 | The live cache and a freshly-minted snapshot can differ (2.3: corporate-action re-adjustment, or if any resync had silently fired before this commit) — not guaranteed byte-identical even though both store raw, untrimmed data | Empirical — must diff `B_baseline.txt` against a real snapshot mint of the harness universes, not assumed neutral |
| 6 | `CandidateGenerationConfig`'s own three fields: either delete them and have `generate_candidate_panels_by_group` use the same single source of truth as everything else (per Track D's rule, once commit 4 exists), or state explicitly why this config still needs its own copy | Yes, if deleted and the unified value differs from `PanelBatchConfig`'s current defaults (which is what `CandidateGenerationConfig` mirrors today) | Same as commit 4 — depends on the chosen value | Empirical |
| 7 | `DataConfig.universe_name`'s default and `snapshot_id`'s `None` sentinel (Scope item 4, `found.md`'s two entries) — remove the string default per Track D's rule; resolve the `snapshot_id` opt-in-vs-mandatory question 2.2 raised | Possibly, if removing the default changes what any caller resolves to | `universe_name="sp500_v1"` is today the *only* universe that exists (`ls config/universes/`) — removing the default is neutral **only** as long as every caller already states `sp500_v1` explicitly; verify this before claiming neutrality | Needs verification — likely structural given only one universe exists today, but must be checked per-caller, not assumed |

Per the amendment already committed to the brief (`82d04a9`), **each of commits
4-7 gets its own `B_baseline.txt` diff and explanation**, not one combined diff
at the end.

**Does any single flag value make every path agree without changing current
behaviour?** No — checked directly. The five sources' current values:

| Source | `check_for_corruptions` | `start_after_nan` | `force_download` |
|---|---|---|---|
| `DataConfig` | `True` | `True` | `False` |
| `PanelBatchConfig` | `False` | `True` | `False` |
| `market_snapshot.py:277` | `True` | `False` | `True` |
| `UniverseDataLoader.load` signature | `True` | `True` | `False` |
| `CandidateGenerationConfig` | `False` | `True` | `False` |

`start_after_nan` already agrees at `True` across four of five (everything
except the snapshot mint, which has its own, separately-argued reason to differ
— 2.1 — and would likely stay a special case even after unification, since it is
explicitly about *not* trimming at storage time). `force_download` agrees at
`False` across four of five, with the mint's `True` again separately justified
(a mint is definitionally a fresh download). **`check_for_corruptions` has a
real, live split with no value that avoids changing behaviour**: `DataConfig`
and `UniverseDataLoader.load`'s signature (sources 1 and 4) are `True`;
`PanelBatchConfig` and `CandidateGenerationConfig` (sources 2 and 5) are
`False`. Since `check_for_corruptions` only ever prints (1.2) and never mutates
data, picking either value cannot change `B_baseline.txt`'s `## counts`/
`## trades`/`## performance metrics` sections **by itself** — but flipping it
for sources 2/5 (currently `False`) to `True` would newly print QC warnings on
every panel build and candidate-generation run, which is an observable output
change (stdout) even if not a numeric one, and flipping sources 1/4 (currently
`True`) to `False` would silently stop printing warnings that exist today. So:
**no value is neutral in every respect**, but `check_for_corruptions` specifically
is neutral for the *baseline harness's binding check* regardless of which value
wins, since that check only covers counts/trades/metrics, not stdout. This is
the track's central fact: the real, unavoidable behaviour change is confined to
`start_after_nan`'s value at the one path that currently differs
(`market_snapshot.py`, deliberately) and to whichever of `check_for_corruptions`'s
two live values does not win — everything else already agrees.

---

## Out of scope — noted, not investigated

- `_make_sqrt_w`, residual/hedge/z-score estimation stages, and Track C's
  snapshot *mechanism* itself (its hashing, immutability, and mint logic) were
  not re-examined beyond what was needed to answer the flag/wiring questions
  above.
- Retiring the offline panel path (`run_panel_batch`, `PanelBatchConfig`) is
  Track J's job; this spec traces `PanelBatchConfig`'s current callers (Part
  1.4) because Track I must unify its flags before J deletes it, but does not
  evaluate J's own retirement plan.

---

## Anything the brief asks for that cannot be done as described

Nothing in the brief's instructions was undoable. Two places where the brief's
own prior text needed correction rather than confirmation are called out inline
above: the "three loader call sites" count (now at least four production reachers
of `UniverseDataLoader.load`, plus the notebook-only `ensure_universe_data`,
Part 2.2), and the "the simulator actually uses" framing for
`CandidateGenerationConfig` (true only for `b_baseline_harness.py` today, not for
`run_me.py` or `sweep_runner.py`, Part 1.4).

---

## Proposed `found.md` entries

### `market_snapshot.py:277`'s `start_after_nan=False` is undocumented at the call site
Found during: track I (spec session)
Location: `src/data/market_snapshot.py:277`
What: The choice to freeze raw (untrimmed) data at mint time and defer trimming
to run time is the right call (Part 2.1 — confirmed consistent with the module's
own immutability framing), but nothing at the call site itself says so; a future
reader could as easily conclude it was a leftover default. A one-line comment
citing the reasoning (raw storage, consumer-specific trimming) would close the
gap the brief flagged as a live risk ("if it is an accident, it is a real
defect").
Severity: cosmetic
Suggested track: I, as part of whichever commit does the snapshot wiring

### `found.md`'s "three loader call sites" undercounts after F2 C6b
Found during: track I (spec session, Part 2.2)
Location: `found.md`'s "`UniverseDataLoader.load`'s in-place cache-hit resync
remains live" entry, and the brief's Scope section referencing "the three
loader call sites"
What: Both predate F2 C6b's addition of
`generate_candidate_panels_by_group`'s own separate UMD load
(`simulator_factory.py:271-280`) and were never updated to include it, or to
include `ensure_universe_data`'s notebook-reachable load
(`universe_loader.py:454-480`, called from
`notebooks/howto/03_full_simulation_pipeline.ipynb`). Anyone wiring "the three"
to the snapshot layer without re-deriving the current call-site list from
scratch will miss at least two.
Severity: cosmetic (a documentation undercount, not a code defect) — but could
cause a result-affecting miss if acted on literally during implementation
Suggested track: I — resolved by this spec's Part 1.4/2.2 re-tracing; update
`found.md`'s entry text once Track I's implementation session lands, to point at
whichever call sites remain unwired

### `DataConfig.snapshot_id`'s opt-in design and Track D's no-defaults rule are in tension
Found during: track I (spec session, Part 2.2)
Location: `src/simulator/config.py:194-203` (`snapshot_id: str | None = None`,
comment framing it as an opt-in: "`None` = the run loaded market data through
`UniverseDataLoader`'s live cache... true for every run today")
What: The field's own comment frames live-cache-vs-snapshot as a legitimate
dual mode, selected by whether `snapshot_id` is `None`. But `found.md`'s
existing "`DataConfig.universe_name` reintroduces the defect Track D removed"
entry (and this track's Scope item 4) treats the `None` sentinel itself as the
Track D violation to remove once wiring lands. Both cannot be fully true at
once: a `None`-means-opt-out design needs *some* silent default to opt out of;
a strict no-silent-default rule needs snapshot binding to be mandatory (or the
opt-out spelled with an explicit, non-`None` sentinel value) everywhere it's
reachable. Not resolved here — Part 2.2 lays out the fork; whoever implements
the wiring (Part 4 commit 5/7) must pick a side.
Severity: result-affecting, dormant until wiring lands (same severity note as
the existing `found.md` entry this extends)
Suggested track: I — same commit that does the snapshot wiring and the
`universe_name`/`snapshot_id` cleanup (Scope item 4)

---

## Summary of open decisions (for the implementation session)

1. Which value wins for `check_for_corruptions` (the only source with a live,
   unresolved split — Part 4). `start_after_nan` and `force_download` already
   agree everywhere except the deliberately-different snapshot mint.
2. Mechanism for applying trim/check after `load_market_snapshot` — extract a
   free function (recommended, 2.2 option a), duplicate into
   `market_snapshot.py` (b), or reuse `UniverseDataLoader` as a thin wrapper (c).
3. Whether the live-cache path survives wiring as a real dual mode, or every
   wired call site becomes snapshot-only (2.2) — resolves how
   `DataConfig.snapshot_id`'s `None` default squares with Track D's rule.
4. Whether `CandidateGenerationConfig` keeps its own three fields post-
   unification, or is deleted in favor of one shared source (Part 4 commit 6).
5. Whether `DataConfig.universe_name`'s `"sp500_v1"` default can be removed
   as a pure formality (only one universe exists) or needs per-caller
   verification first (Part 4 commit 7) — leaning structural but unverified.
6. Gappy universe: join the baseline harness or stay a standalone fixture
   (Part 3.3) — trade-off given, not decided.

Contradicting-the-brief items are both in Part 1.4/2.2: the "three loader call
sites" undercount, and "the simulator actually uses" overstating
`CandidateGenerationConfig`'s current reach.
