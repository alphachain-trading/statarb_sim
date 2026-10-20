# Findings moved to GitHub Issues

Out-of-scope findings are tracked as GitHub issues, not in this file:
<https://github.com/alphachain-trading/statarb_sim/issues?q=label%3Atriage>

- Milestone **Baseline freeze**: findings whose fix changes backtest trades, counts or
  metrics and must land before the post-G performance baseline is defined.
  <https://github.com/alphachain-trading/statarb_sim/milestone/1>
- Priorities: labels `p1`, `p2`, `p3`.
- Workflow: found outside the task's scope -> `gh issue create` with label `triage`,
  priority/milestone proposal in the text; fix commits say `Fixes #<n>`.

The former contents of this file (entries cited by name in the track specs and briefs,
e.g. "four combinations", `_merge_umds`, `make_ranking_dates`) are in git history before
the commit that introduced this pointer: `git log --follow -- docs/refactor/found.md`.
Entries that were already resolved or deliberately rejected have no issue.
