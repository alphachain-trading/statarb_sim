# statarb_sim

Statistical-Arbitrage-Simulator (Pairs-Kandidaten → Residuen → Simulator).

## Setup
- venv aktivieren: `source .venv/bin/activate` (Python 3.14, Projekt-Minimum 3.12)
- Dev-Tools: `pip install -r requirements-dev.txt`
- Einmalig: `pre-commit install` — **pre-commit mit ruff aktiv** (ruff-check `--fix`, ruff-format)

## Befehle (aus dem Projekt-Root)
- Lint: `ruff check .`
- Format: `ruff format .`
- Typecheck: `mypy .`
- Tests: `python -m pytest -q` (~100 s; einzelne Datei: `python -m pytest -q tests/<datei>.py`)

## Struktur
- `src/`: `data/` (Universe, Snapshots, QC), `residuals/`, `candidates/`, `analytics/`, `simulator/` (inkl. `traders/`, `performance/`), `types/`, `utils/`
- `tests/`: unittest-Stil unter pytest; `pythonpath = [".", "scripts"]` steht in `pyproject.toml` (keine `sys.path`-Hacks)
- `scripts/`, `run_me.py`: Einstiegspunkte; `config/`, `fixtures/`, `docs/`: Konfig, Testdaten, Specs

## Konventionen
- Ausnahmen nur per `# noqa: <Code>  # Begründung` bzw. `# type: ignore[<code>]  # Begründung`; keine pauschalen Ignores in der Konfiguration.
- Matrixnotation (`X`, `Y`, `B`, `T`, …) in `src/residuals`, `src/analytics`, `src/simulator/candidate_signals.py` ist mit `noqa: N803/N806` bewusst erlaubt.
- ruff formatiert `*.md` nicht (`extend-exclude`), da Docs Code-Zitate enthalten.

## Bekannter Stand
- `mypy .` meldet 7 Fehler (Portfolio-Trader/`Trader`-Protocol, siehe `src/simulator/traders/portfolio_mean_reversion.py`, `src/simulator/simulator_factory.py:104`) — echte Inkonsistenz, noch nicht entschieden.
