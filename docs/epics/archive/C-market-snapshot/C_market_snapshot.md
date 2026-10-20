# Track C — Market data snapshot layer

**Changes results:** no. New module plus a loader change. Nothing reads it until F.

**Depends on:** nothing.

## Problem

Reloading market data repeatedly has made it impossible to determine with justified
effort which artifacts are stale. Every simulator run must be bound to a frozen
block of market data.

**The key insight, which corrects an earlier position:** causality does not imply
history stability. Causal fits guarantee that row *t* does not read data after *t*.
They say nothing about whether the data at *t* is the same data tomorrow. Corporate
actions retroactively re-adjust historical closes — same tickers, same range,
different values. Causality protects against lookahead, not against input revision.
These are separate failure modes and only the first is addressed by construction.

Consequence: prefix and append-only checks on market data are unimplementable
without keeping the old price content for comparison, i.e. a second copy. Rejected.

## The universe model (correction; supersedes the original scope section below)

The original scope section (kept below, struck through in spirit if not in
markdown) left "universe" underspecified: it talked about a `universe_version`
identity and a `mint_universe_version` that copies "the universe config
directory," without saying what that directory contained or how a
`universe_version` maps to a set of groups. Implementation surfaced the gap
(a bare name has no group set without something to derive it from) and
settled it as follows. This section is authoritative; where the rest of this
document still describes the old, dropped design (a `groups` parameter on
`ensure_market_snapshot`, `mint_universe_version` copying a directory), that
text is historical and superseded.

**A universe is a directory of group yamls, one per group:**

    config/universes/{universe_name}/{group_id}.yaml

The directory name IS the universe name. A universe name identifies exactly
one set of groups and tickers — any change to either (a group added or
removed, a ticker list edited) requires a new universe version (e.g.
`sp500_v8` -> `sp500_v9`), never an edit in place. There is no
`mint_universe_version` helper: the directory itself is the versioned
record (copy it by hand, or with a plain filesystem copy, to start a new
version with edits) — nothing in the snapshot layer copies it, because
copying would just give the yamls a second place to drift.

**A snapshot freezes one download of one universe.** Two snapshots of the
same universe_version may differ in realized end date and in price values
(corporate actions re-adjust history) — `dl_ts` and the content hash handle
that, unchanged from the original design. What two snapshots of the same
universe_version may NOT differ in is group/ticker composition: mint time
enforces this by diffing the current directory's yaml content against the
verbatim yamls stored in the most recent snapshot of the same
universe_version, and raises (rather than silently drifting what a
universe_version means) if they differ. With no prior snapshot there is
nothing to compare against — the first mint is definitional.

**A simulator run is bound to one frozen snapshot and may use any SUBSET of
that universe's groups.** Group selection is a run concern
(`DataConfig.selected_groups`/`excluded_groups`), not a snapshot concern — a
snapshot always covers every group in its universe directory, in full.

## Scope

**Snapshot identity:** `{universe_version}_{dl_ts}`, e.g. `sp500_v3_20260825T143012`.
`dl_ts` is a download timestamp, sortable and filesystem-safe. It records a
download *event*, not a data property.

**Layout** puts `dl_ts` outermost, not per-group, so a run config names one
snapshot id plus a group list and overlap between runs is visible by string
comparison.

**One `dl_ts` per download session**, applied to every group in the universe. This
resolves ragged end dates for free — same session, same resolution point, no
alignment check needed within a snapshot.

**Manifest** carries: resolved group configs verbatim as they were at download
time; per-group content hash; per-group ticker-set hash; per-group realized date
coverage; the requested config as written alongside the resolved range; the
timestamp.

Two hashes because they answer different questions — content hash catches
retroactive adjustment, ticker-set hash distinguishes "same universe, different
vintage" from "different universe". Ticker edits are the most frequent reason the
YAMLs change.

**Immutability:** write to a temp path, atomic rename into position, then remove
write permission. The write path must *refuse* to target an existing snapshot
directory, loudly. Convention alone will not hold.

**Adding a group to an existing snapshot is not an operation that exists.** There
is no append path. A universe directory's group set only ever grows or shrinks by
becoming a new universe_version (a new directory), which requires a fresh mint —
one download event, one resolution point, uniform coverage, no per-group alignment
checks ever. This is enforced at mint time (see "the universe model" above), not
merely documented as a convention.

**Mitigation:** download parallel across groups and resumable, so minting is
unremarkable. Reluctance to mint is what leads to someone hacking a group into an
existing snapshot by hand.

**Incomplete download fails loudly and completely.** A ticker listed in a group
yaml (member, proxy_etf, benchmark, or risk_free) that returns no data aborts the
mint — this is the yfinance survivorship problem surfacing as a loud failure
rather than a silent gap. Every group is attempted before raising, so the error
lists every missing ticker across every group, grouped by group_id, not just the
first one found. Nothing is written on an aborted mint.

**YAML consistency:** a YAML edit must never invalidate a frozen snapshot or a
completed run. It also must never silently be picked up by a later mint under the
same universe_version — see "the universe model" above. The manifest carries the
resolved config verbatim, so nothing reads the YAML again after download.

## API

- `ensure_market_snapshot(universe_version, force=False) -> list[str]` — returns
  chronologically sorted snapshot ids, newest first. Always a list, so there is no
  branch at the call site. Never silently picks "latest" for the caller. No
  `groups` parameter — the group set comes from the universe directory
  (`config/universes/{universe_version}/`) by construction.
- `load_market_snapshot(snapshot_id) -> MarketData` — recomputes both manifest
  hashes and raises on mismatch. Keep it raising rather than warning; a warning
  scrolls past in a sweep.

**Never resolve on a prefix.** Full snapshot id or nothing. "Latest v3" would
destroy reproducibility.

## Out of scope

- Recording `snapshot_id` in run configs. That is F.
- Ticker-set diff acknowledgement at download. Deferred; note it in the spec.
- Parallel/resumable download. Deferred, but design the download path so it can be
  added without restructuring.

## Acceptance

- Full suite green.
- A test that writing to an existing snapshot directory raises.
- A test that a mutated snapshot file causes `load_market_snapshot` to raise.
- A test that `ensure_market_snapshot` returns a list of length one after a fresh
  download and never resolves a partial id.
- A test that an edited group yaml raises at mint time against an existing
  snapshot of that universe name.
- A test that a missing ticker aborts the mint and that the error lists every
  missing ticker, not just the first.
- A test that the first mint of a new universe name succeeds with no prior
  snapshot to compare against.

## Rejected alternative — do not rediscover

Copying price data into each run directory. It works and makes reconstruction
airtight, but a 200-config sweep copies identical data 200 times, and every run
becomes a data island with no shared identity — "were these two runs built on the
same data?" becomes a hash comparison rather than a string comparison. A versioned
store plus the resolved id and hash in the run config gives the same guarantee more
cheaply. An `export_run` helper can bundle a run with its price slice if
portability is ever needed.

Also rejected, discovered during implementation: `ensure_market_snapshot` taking
an explicit `groups` list decoupled from `universe_version` (the layer's first
version). It worked, but left `universe_version` a bare label with nothing
actually tying it to a group set — two calls with the same `universe_version` and
different `groups` were both "valid" and produced different snapshots, which is
exactly the ambiguity a versioned identity is supposed to remove. The directory
model above removes the parameter entirely instead of tightening its validation.
