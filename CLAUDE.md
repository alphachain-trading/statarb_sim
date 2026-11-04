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
- `scripts/`, `run_me.py`: entry points; `config/`, `fixtures/`, `docs/` (`docs/epics/`: epic briefs and specs, `docs/baseline/`: regression-harness output): configuration, test data, specs
- `docs/glossary.md`: domain terms as implemented; `docs/adr/`: architecture decision records (template and rules in `docs/adr/README.md`)

## Glossary
@docs/glossary.md

## Conventions
- Exceptions only via `# noqa: <code>  # reason` or `# type: ignore[<code>]  # reason`; no blanket ignores in the configuration.
- Matrix notation (`X`, `Y`, `B`, `T`, …) in `src/residuals`, `src/analytics`, `src/simulator/candidate_signals.py` is deliberately allowed with `noqa: N803/N806`.
- ruff does not format `*.md` (`extend-exclude`), since docs contain code quotes.

## Workflow
@CONTRIBUTING.md

## Claude-specific rules
- Before working on an issue: read `gh issue view <n> --comments`. GitHub state wins over docs.
- Findings outside the task: `gh issue create` with label `triage` and a priority suggestion (`p1`–`p3`, milestone if applicable); do not fix them.
- Record such decisions made with me immediately via `gh issue comment` (see CONTRIBUTING.md).
- Create PRs with `gh pr create`; run `gh pr merge` only after my explicit approval.
- Use terms as defined in `docs/glossary.md`. If a term is missing, ambiguous or conflicts with the glossary, stop and ask; propose a definition, do not add it yourself.
- Never change an existing glossary definition or ADR without my explicit approval.
- If a decision with me affects more than one module or the overall architecture, propose an ADR draft in `docs/adr/` using the template. Issue-specific decisions stay issue comments.
- Glossary and ADR changes go in their own commit and are listed in the PR description.
