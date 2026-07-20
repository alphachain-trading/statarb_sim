from pathlib import Path
# settings.py lives in src/, so the repo root is two levels up (src/ -> root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "config"
CONFIG_UNIVERSE = CONFIG_ROOT / 'universe'
CONFIG_SP500 = CONFIG_ROOT / 'sp500_constituents'
DATA_ROOT = PROJECT_ROOT / "data"
MARKET_ROOT = DATA_ROOT / "market"
DATA_UNIVERSES = MARKET_ROOT / "universes"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
WALKFORWARD_ROOT = ARTIFACTS_ROOT / "walkforward"
SPREAD_STATE_PANELS_ROOT = ARTIFACTS_ROOT / "spread_state_panels"
CAUSAL_RESIDUALS_ROOT = ARTIFACTS_ROOT / "causal_residuals"
CANDIDATE_PANELS_ROOT = ARTIFACTS_ROOT/ "candidate_panels"
SIMULATION_RUNS_ROOT = ARTIFACTS_ROOT / "simulation_runs"
PERFORMANCE_REPORTS = ARTIFACTS_ROOT / "performance_reports"
TRADE_ANATOMY_ROOT = ARTIFACTS_ROOT / "trade_anatomy"
