    # Track D — Config hygiene

**Changes results:** no. If the baseline moves, the track is wrong.

**Depends on:** nothing. Do it before E, so renames land on a config surface that
has stopped moving.

Amended after Tracks A and B, which each found one instance of the defect this track
exists to eliminate systematically. Amendments marked **[amended]**.

## Principle

A numeric that affects results must be stated explicitly by the caller, not
inherited from a dataclass default, a docstring, or another config's field. A
default that lives in code is a magic number that disappears and is forgotten.

Two rules:

1. **No numeric defaults in dataclasses** for anything that affects results.
2. **No field resolves to another config's field via a `None` sentinel.**

**[amended] A third rule, from the two cases found so far:**

3. **No layer above may reinstate a default the layer below deliberately omits.**
   Both known instances took this form — the field itself was correct, and a
   containing config's `default_factory` put the magic number back one level up.

Defaults do not vanish — they move into named, versioned bundles in
`DEFAULT_CONFIGS`, jointly changeable under one bundle name. **[corrected in
review, confirmed by D_spec.md]** The bundle *name* does not itself enter
`config_hash` — only the bundle's resolved values do, via
`merge_defaults` splicing them into the `SimulatorConfig(**kwargs)` call.
Two bundles with different names but identical resolved values would hash
identically. So every number is explicit at the call site and its *value*
is version-tracked in the run identity — "version-tracked" via the bundle
name overstates it; the name buys traceability in a sweep's persisted
results row (`SweepConfig.defaults`), not hash identity.

This is an extension of an existing convention, not a new one: `subtract_risk_free`,
`hedge_ratio_lb`, and `mr_diag_lb` already have no defaults.

## Why this track is worth doing systematically

**[amended]** Two instances have been found so far, both by accident:

- **`skip_adf=False`** in `panel_batch.py`'s `pair_cfg` default factory, overriding
  the `PairSpreadConfig` default that Track A had just flipped to `True`. Found only
  because A re-timed a panel build and the speedup was missing. Every production
  panel build was still computing ADF.
- **`min_obs=10`** in the same default factory, put there during Track B. `min_obs`
  is documented as mandatory with no default precisely because a rolling 252-day fit
  running on 40 observations is not a 252-day fit — and a default of 10 keeps that
  true for everything between 10 and 252.

Neither was found by looking. Both were found because a track happened to touch the
same file. That is the argument for enumerating the whole surface once rather than
waiting for the next accident.

Note the pattern: **`PanelBatchConfig`'s `pair_cfg` default factory was the site of
both.** Start there.

## Verify first

Claims only. Re-verify with `path:line`.

1. `hedge_ratio_lb` is an `int | None` parameter that falls back to the residual
   config's lookback when None, which is itself None for expanding configs.
2. The pair-spread config carries numeric defaults for the minimum return standard
   deviation, the minimum level standard deviation, the minimum kappa, and the
   maximum half-life.
3. `DEFAULT_CONFIGS` exists as a registry and the bundle name enters `config_hash`.
4. **[amended]** `PanelBatchConfig.pair_cfg` is now a required field with no
   default factory, as of Track B. Confirm this and confirm no caller relies on a
   removed default.

**The enumeration is the deliverable of the spec session.** List every remaining
result-affecting numeric default in the config layer, with `path:line`, before
changing any. Include default factories and `None`-sentinel fallbacks, not just
literal scalar defaults — both known instances were factories, not scalars.

Report separately any default that is **not** result-affecting (log levels, tqdm
flags, output paths). Those stay. The distinction is whether the value can change a
number in the baseline.

## Scope

- **Kill the `hedge_ratio_lb → residual lookback` fallback.** The anti-pattern in its
  purest form, and also the dropna trap B addressed from the other side.
- **Remove numeric defaults from the pair-spread config**, including the maximum
  half-life. That one in particular is a research choice sitting in a dataclass
  default.
- **Move every removed default into the versioned bundles**, reproducing today's
  values exactly.
- **`hedge_ratio_lb = 252`** across all non-test call sites. Verify this still holds
  and pin it in the bundle rather than leaving it at each call site.
- **[amended] Audit every `default_factory` in the config layer**, not only scalar
  defaults. Both known instances hid there.

## Out of scope

- Changing any value. This track relocates numbers; it does not tune them.
- Adding new config fields. `zscore_key` is H.
- Structural changes to the config classes beyond removing defaults.
- **[amended]** Deciding whether `min_obs` should equal `hedge_ratio_lb`. B left it
  caller-stated, which is correct; picking a canonical value is a research question,
  and the bundle is where it will be pinned when answered.

## Expected cost

Construction becomes verbose and tests become noisy, because every test config must
now state everything. That cost is the point — but mitigate it by having tests
construct from a bundle rather than field by field.

## Acceptance

- **The baseline file from B does not move at all.** This is the acceptance test.
  Any movement means a default was relocated with the wrong value.

  **[amended]** Note the baseline now leads with candidate and trade counts, with
  performance metrics in a separate section at the end. For D, the counts are the
  binding check — a relocated default that changed a gate threshold shows up there
  first.
- Full suite green.
- A test asserting that constructing the relevant configs without the required
  fields raises rather than silently defaulting.
- No result-affecting numeric default remains in the config dataclasses **or in any
  default factory**. The spec's enumeration is the checklist, and every item on it
  must be either removed or explicitly justified as non-result-affecting.
- **[amended]** Re-execute the three howto notebooks. Every removed default becomes
  a required argument, so notebook configs change even though results must not.
