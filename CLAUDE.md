# statarb_sim

Statistical arbitrage simulator (pairs candidates → residuals → simulator).

## Setup
- Activate the venv: `source .venv/bin/activate` (Python 3.14, project minimum 3.12)
- Dev tools: `pip install -r requirements-dev.txt`
- Once: `pre-commit install` — **pre-commit with ruff is active** (ruff-check `--fix`, ruff-format)

## Commands (from the project root)
- Lint: `ruff check .`
- Format: `ruff format .`
- Typecheck: `mypy .`
- Tests: `python -m pytest -q` (~100 s; single file: `python -m pytest -q tests/<file>.py`)

## Structure
- `src/`: `data/` (universe, snapshots, QC), `residuals/`, `candidates/`, `analytics/`, `simulator/` (incl. `traders/`, `performance/`), `types/`, `utils/`
- `tests/`: unittest style under pytest; `pythonpath = [".", "scripts"]` is set in `pyproject.toml` (no `sys.path` hacks)
- `scripts/`, `run_me.py`: entry points; `config/`, `fixtures/`, `docs/`: configuration, test data, specs

## Conventions
- Exceptions only via `# noqa: <code>  # reason` or `# type: ignore[<code>]  # reason`; no blanket ignores in the configuration.
- Matrix notation (`X`, `Y`, `B`, `T`, …) in `src/residuals`, `src/analytics`, `src/simulator/candidate_signals.py` is deliberately allowed with `noqa: N803/N806`.
- ruff does not format `*.md` (`extend-exclude`), since docs contain code quotes.

## Findings
- Do not fix findings outside the task; create them with `gh issue create` and the label `triage`, with a proposal for priority (`p1`–`p3`) and, if applicable, milestone (e.g. `Baseline freeze`) and the reasoning in the text.
- Commits that resolve an issue contain `Fixes #<n>`.

## Decisions
- When you make a decision with me about an issue (priority, approach, discard,
  intended behavior), record it immediately in the issue via `gh issue comment`:
  decision + reasoning in 1–3 sentences.
