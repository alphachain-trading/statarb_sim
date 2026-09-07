# Track B — Window admission

**Changes results: yes.** This track cleans a tainted baseline. Everything
downstream inherits whatever it produces, which is why it comes before the
artifact layer rather than after.

**Depends on:** nothing.

Amended after the spec session. Amendments are marked **[amended]** and supersede
the original brief. See `B_spec.md` for the verification detail.

## Verify first

Claims only. Re-verify with `path:line`. All five were confirmed in the spec
session with no corrections.

1. The hedge and diagnostic windows are built with
   `.dropna(axis=0, how="any").dropna(axis=1, how="any")` — a ticker with a single
   NaN anywhere in the window is dropped from all columns for that window, and the
   pair loop then skips every pair involving it. **No row is produced**, so the
   candidates table does not contain "all pairs at date *t*".
2. Both windows are trailing slices today, truncated by an explicit lookback.
3. The window slicer only truncates when the lookback is not None, and a None
   lookback falls back to the residual config's lookback — which is itself None for
   expanding-mode configs, giving full untruncated history.
4. The slicer raises only when the slice is empty. A window shorter than the
   requested lookback passes through silently at whatever length is available.
5. Neither the OLS beta helper nor the PCA weights helper checks sample size —
   both guard only a near-zero denominator or eigenvector component.

## The defect

Items 4 and 5 together mean that near the start of history, a hedge ratio can be
fitted on a handful of observations and returned as if it were a 252-observation
estimate. Nothing reports it. Every result computed to date includes these rows.

Item 1 is a separate defect: a coverage hole that makes any denominator computed
from the candidates table wrong — rejection rates, breadth counts, coverage. This
directly threatens the effective-breadth work planned for the portfolio phase,
which is precisely a denominator question.

Item 3 is a latent trap rather than a live defect. It matters because the natural
implementation of a decay-expanding hedge mode passes no lookback, which reaches
full-history `how="any"` dropna and drops every ticker with a single gap in its
entire history.

## Scope

**[amended] Three commits, not four.** The date-distance weighting item has moved
out of scope — see below. Measure after each.

**1. Baseline harness.** A fixed config — small universe, short range, tens of
seconds — plus a script writing trades and summary metrics to a committed text
file. This is the instrument for the rest of the track and is reused by G and H.
Commit it before changing any behaviour, with the current (tainted) numbers in it.

**[amended]** Per the spec session, reuse what exists rather than writing fresh:
`run_panel_batch` with `max_steps` is already the pattern in
`tests/test_panel_batch_windows.py`, and `compute_performance` with `_METRICS_ORDER`
covers the metrics summary. No new runner and no new metrics code.

Have the harness report **build time** as well as results. The hedge fit measured
at ~51% of a panel build, so per-pair slicing roughly doubles the dominant cost in
the worst case. That is acceptable at current runtimes but is the number to watch.

**2. Pairwise dropna.** The hedge fit for a spread needs only its own legs.
Dropping a third ticker because it has a gap is an artifact of building one shared
clean matrix for all pairs at once. Build each pair's fit input from its own legs,
dropping only dates where a leg *of that pair* is missing.

Under a trailing window this is a modest change; over full history it is the
difference between working and not. `B_spec.md` Q6 identifies the shared clean
matrix and the pair-universe generation off its columns as what per-pair
construction replaces, and lists the four spots that assume one shared column set.
Put tqdm on the loop.

If the cost regresses badly, report it rather than optimizing speculatively. The
fix would be slicing once per ticker pair rather than per pair-date, not abandoning
pairwise.

**3. Per-pair `min_obs`.** Mandatory, no default, absolute, counted on the pair's
own retained observations. This is the fix for items 4 and 5.

Forward-filling gaps was considered and **rejected**: it manufactures zero returns,
and flat runs bias kappa upward, creating apparent mean reversion. Do not
reintroduce it.

Nothing downstream assumes a low-observation pair keeps producing a hedge ratio —
selection and series persistence already tolerate absent rows.

## Also in scope

- **Log dropped observations per pair per date.** Logging only — the point is to
  make the coverage hole measurable before anyone computes a breadth denominator
  from the table.
- **Log the count of pairs dropped by `min_obs` per date**, so the size of the
  baseline taint is visible rather than inferred.
- **[amended] Log structural pre-check skips separately from ticker drops.**
  Carried forward from Track A's verification. The candidates table has two
  distinct absence classes: rows marked `is_valid=False` with a reason, and pairs
  skipped entirely that leave no record. The exception path at
  `pair_candidate_panel_creator.py:266-274` is a weight-computation failure, not a
  ticker drop. Counting them together reproduces the same coverage hole this
  logging exists to expose.

## Out of scope

- **[amended] Date-distance weighting.** The original brief carried this as an item
  to fix. The spec session established that **the hedge fit is currently unweighted
  entirely** — exponential weights exist only in the residual fit. So there is
  nothing to correct here.

  It becomes a **requirement on Track G**, which introduces weighting at this stage
  for the first time. G's brief already states it; the point is that B does not
  discharge it. When G adds weights they must be computed from date distance to the
  fit date, not from post-dropna row position — and pairwise dropna is what makes
  that mandatory rather than merely correct, since gaps become per-pair.

- The decay-expanding hedge mode itself. That is G.
- Any change to the residual fit's own window handling. Note that the residual fit's
  `_make_sqrt_w` computes weights from row position — the same defect class, a
  different stage. Recorded in `found.md`, not fixed here.
- A NaN-fraction threshold as an alternative to pairwise. Pairwise is the decision;
  a threshold is a different computation and would need its own case.

## Acceptance

- Full suite green.
- Baseline file committed before and after, with the diff explained: how many
  rows the `min_obs` fix removes, and how many pairs pairwise dropna admits that
  the shared-matrix version rejected.
- A test that a pair with a clean history is unaffected by a third ticker's gap.
- A test that a pair with fewer than `min_obs` retained observations produces no
  hedge ratio and is reported, not silently fitted.
- `min_obs` has no default anywhere.
- Build time recorded before and after commit 2.

## Note for the spec session

§8 of the superseded handover asserts that the rolling and decay-expanding modes
must admit the same `(spread_id, asof_date)` set. **That assert is unachievable.**
Rolling requires its observations inside a trailing window; decay-expanding
requires them anywhere in history, so it admits a strict superset. The comparison
in G runs on the **intersection** of the two admitted sets, with the
symmetric-difference size reported as a diagnostic.