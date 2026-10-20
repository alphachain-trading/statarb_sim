# Track E — Terminology

**Changes results:** no. Mechanical, wide, and conflict-generating.

**Depends on:** D, so renames land on a config surface that has stopped moving.

**Land this alone and land it fast.** It touches nearly every file the other tracks
touch, so it produces merge conflicts rather than logical ones. Do not let it age
on a branch while other work proceeds, and do not run another session concurrently.

## The date collision

Three distinct concepts currently share or blur one name.

- **`fit_date`** — the date a residual model was fitted. **Daily.** This is the
  rename that removes the collision.
- **`asof_date`** — the date a candidate's weights were fitted. Coincides with
  outer-loop dates, but is a property of the candidate, not of the loop.
- **`date`** — a simulation day.

Residual models are looked up by exact calendar date on a daily grid, while
candidates sit at outer refit dates. Today both carry `asof_date`, which makes two
different grains joinable by accident.

**Outer-loop iteration dates get no column anywhere.** They are a loop schedule,
not a property of a row. The distinction is load-bearing: an outer date that
produced no candidates has no `asof_date` in the table, which is precisely the
coverage hole B logs. Keeping the terms separate is what makes that hole visible
instead of silent.

## Scope

- `asof_date` on residual params → `fit_date`.
- `asof_date` retained on candidates and weights, with the definition above written
  into the docstring so it does not drift back.
- `sector` → `group_id` throughout the artifact and simulator layers. A **group** is
  a set of tickers within which pairs may form. The market data layer stays
  "sector", because those are genuinely S&P sector downloads.
- Any `entry_refit_date` → `entry_asof_date`, for consistency with the above. It is
  the asof_date of the candidate that was entered.

## Multi-group ticker assert

Add while renaming, since it is the same conceptual change. Currently a ticker
belongs to exactly one group. Assert this at universe composition and raise if
violated.

The message should explain why: multi-group tickers would require residual params
keyed by `(group_id, ticker)` rather than ticker, and duplicate residual fits per
ticker — a ticker in two groups gets two different residual series because the fit
is group-scoped, so cross-group analysis must either pick one or double-count.

Key artifacts by `(group_id, ticker)` anyway. It costs nothing now and means the
storage layer needs no change when the restriction lifts.

## Out of scope

- Any schema change beyond renaming. New columns and dropped tables are F.
- `sleeve_key`, `zscore_key`, or any sub-portfolio identity naming. That is H, and
  introducing the vocabulary here would commit to a design that has not been made.
- Changing what any date value *is*. Only what it is called.

## Acceptance

- Full suite green.
- The baseline file from B does not move.
- No remaining use of `asof_date` for a residual fit date; no remaining use of
  `sector` in the artifact or simulator layers.
- A test that the multi-group ticker assert raises, with the message content
  checked.

## Note

Do the rename with a tool that understands the code, not a text substitution.
`sector` in particular appears legitimately in the market data layer and must
survive there.
