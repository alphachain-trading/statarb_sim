# Contributing

Status lives only on GitHub. Docs reference issues and epics by full link, never by
copying their state.

## Terms

- **Milestone**: a goal with a set of issues, e.g.
  [Baseline freeze](https://github.com/alphachain-trading/statarb_sim/milestone/1).
- **Epic**: an issue with the label `epic` and a task list (`- [ ] #n`) of its issues.
  Its brief lives in `docs/epics/<issue#>-<slug>/`. Epics were formerly called tracks
  (A–J).
- **Issue**: one unit of work or one finding.
- **Branch / PR**: the vehicle for one logical change and its review.
- **Regression harness (reference run)**: `scripts/b_baseline_harness.py` and its
  committed output `docs/baseline/B_baseline.txt` (`B_baseline`). It shows whether a
  change moved results; it does not measure performance.
- **Performance baseline**: the milestone goal, defined by the final issue of the
  milestone. Not to be confused with the regression harness.

## Flow

issue → branch → commits → PR → merge.

1. Read the issue including comments before starting.
2. Branch from `main`, commit in small self-contained steps, open a PR.
3. **Merging is a human decision.** Claude creates the PR and merges only after
   explicit approval.

## Conventions

- One branch per logical change, usually one issue; closely related small issues may
  share one. Branch name: `<type>/<issue#>-<slug>` (e.g. `fix/3-residual-date-weights`).
  Trivial docs/chore changes may go without an issue (branch `<type>/<slug>`).
- Commits reference issues with `refs #n`, or `Fixes #n` when they resolve one.
  Formatting changes are never mixed with logic changes.
- Labels: `triage` (found outside the current task, awaiting prioritisation),
  `p1`–`p3` (priority), `epic`. Discarded issues are closed as "not planned".
- Findings outside the task are not fixed: open an issue with label `triage` and a
  priority suggestion.
- Substantive decisions about an issue (approach, intended behavior, scope, discard,
  priority rationale) are recorded as a comment on that issue: decision plus
  reasoning in 1–3 sentences. Pure bookkeeping (labels, milestones, task lists) gets
  no comment; the GitHub timeline already records it.
- Everything in the repo (code, comments, docs, commits, issues) is in English.

## Quality

Before opening a PR run, from the project root:

```
ruff check . && ruff format --check . && mypy . && python -m pytest -q
```

`pre-commit` (ruff) runs on commit. Tests are never weakened or skipped to get green.

Fixes that change backtest results (those in the milestone *Baseline freeze*) always
go in a separate PR, each with its own regression-harness diff
(`git diff docs/baseline/B_baseline.txt`; the `## counts` section is binding).

## PR description

- What and why.
- `Fixes #n`.
- Regression-harness diff summary, or "no diff".
