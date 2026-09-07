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

## Scope

**Snapshot identity:** `{universe_version}_{dl_ts}`, e.g. `sp500_v3_20260825T143012`.
`dl_ts` is a download timestamp, sortable and filesystem-safe. It records a
download *event*, not a data property.

**Layout** puts `dl_ts` outermost, not per-sector, so a run config names one
snapshot id plus a sector list and overlap between runs is visible by string
comparison.

**One `dl_ts` per download session**, applied to every sector in it. This resolves
ragged end dates for free — same session, same resolution point, no alignment check
needed within a snapshot.

**Manifest** carries: resolved sector configs verbatim as they were at download
time; per-sector content hash; per-sector ticker-set hash; per-sector realized date
coverage; the requested config as written alongside the resolved range; the
timestamp.

Two hashes because they answer different questions — content hash catches
retroactive adjustment, ticker-set hash distinguishes "same universe, different
vintage" from "different universe". Ticker edits are the most frequent reason the
YAMLs change.

**Immutability:** write to a temp path, atomic rename into position, then remove
write permission. The write path must *refuse* to target an existing snapshot
directory, loudly. Convention alone will not hold.

**Adding a sector raises.** No silent download. The solution is minting a new
snapshot with the desired sector set, which re-downloads everything. Cost accepted
deliberately: one download event, one resolution point, uniform coverage, no
per-sector alignment checks ever. Appending a sector to an existing snapshot would
put two adjustment vintages in one snapshot and destroy the meaning of the hash.

**Mitigation:** download parallel across sectors and resumable, so minting is
unremarkable. Reluctance to mint is what leads to someone hacking a sector into an
existing snapshot by hand.

**YAML consistency:** a YAML edit must never invalidate a frozen snapshot or a
completed run. Consistency checks and warnings only. The manifest carries the
resolved config verbatim, so nothing reads the YAML again after download.

## API

- `ensure_market_snapshot(universe_version, force=False) -> list[str]` — returns
  chronologically sorted snapshot ids, newest first. Always a list, so there is no
  branch at the call site. Never silently picks "latest" for the caller.
- `load_market_snapshot(snapshot_id) -> MarketData` — recomputes the content hash
  and raises on mismatch. Keep it raising rather than warning; a warning scrolls
  past in a sweep.
- `mint_universe_version(from_version) -> str` — copies the universe config
  directory and bumps the version. Manual copying invites drift.

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

## Rejected alternative — do not rediscover

Copying price data into each run directory. It works and makes reconstruction
airtight, but a 200-config sweep copies identical data 200 times, and every run
becomes a data island with no shared identity — "were these two runs built on the
same data?" becomes a hash comparison rather than a string comparison. A versioned
store plus the resolved id and hash in the run config gives the same guarantee more
cheaply. An `export_run` helper can bundle a run with its price slice if
portability is ever needed.
