# `statarb_sim_refactor_handover.md` — SUPERSEDED

**Do not implement from that document.** It was drafted from memory of a long design
conversation, and a subsequent code investigation found several of its claims wrong.
Implementing from it reintroduces every error below.

Keep it for the reasoning in its sections 1–3 and its dead-options table, which
remain sound and are partly carried into these briefs.

## Known errors

**`weights_json`.** The column name belongs to a dead code path with no callers. The
live path serializes under a different column name, via pandas `to_json` with a
precision cap that does not round-trip float64 — so the persisted beta is not the
beta the run used, and the zero-tolerance fidelity test cannot pass against it.
Superseded by `weights.parquet` in track F.

**The `daily_state` column list.** Invented. The real log has a `(date, group_id)`
grain with no residual or hedge key, and none of the listed capital fields. There is
no capital ledger at all: total capital is read once and never decremented, exposure
is recomputed daily from live positions, and sizing reads a fixed base notional.
Nothing in the log is non-derivable. Superseded — the artifact is dropped in F and
replaced by a function.

**The mode set-equality assert.** Section 8 requires the rolling and decay-expanding
hedge modes to admit the same `(spread_id, asof_date)` set. Unachievable: rolling
needs its observations inside a trailing window, decay-expanding needs them anywhere
in history, so it admits a strict superset. Superseded by intersection comparison in
track G.

**`infinite_leverage`.** Does not exist. The bypass is a null risk-manager config,
which skips all five risk checks at once, not a capital-specific one — and it is
never exercised anywhere in the repo.

**The hashed `.key` for the hedge config.** Section 8 proposes a digest. It breaks
the repo's consistent convention — config keys readable, object identities hashed —
and breaks the self-describing-artifact claim made in section 4 of the same
document. Superseded by a readable key in track G.

**`asof_date` used for two grains.** The residual fit grid is daily; candidates sit
at outer refit dates. Both carried the same column name. Superseded by the `fit_date`
rename in track E.

**Section 8's dropna framing.** The claim that the existing `how="any"` dropna is
already scoped to full history is wrong — both windows are trailing slices today.
The trap is real but latent, reached only when a lookback of None falls back to an
expanding residual config's null lookback. Corrected in track B.

**Section 9's independence claim.** Unconstrained mode does not make sub-portfolios
independent. Occupancy dedup omits the z-config, so sleeves collide at the trader
regardless of any capital or risk setting. Guarded in A, fixed in H.

**Section 15's open question** — one portfolio or one per group — is answered: one
per run, and the per-sleeve decomposition is derivable rather than stateful.

## Corrections it records that remain valid

The five items in its final section each overturned a stated position and are still
worth reading, particularly that the spread level series does depend on beta, and
that causality protects against lookahead but not against input revision.
