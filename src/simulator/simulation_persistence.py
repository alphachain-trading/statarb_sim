from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulator.config import PersistenceConfig, SimulatorConfig

from settings import SIMULATION_RUNS_ROOT


# ---------------------------------------------------------------------------
# Run ID
# ---------------------------------------------------------------------------

def make_run_id(config: SimulatorConfig) -> str:
    """
    Generate a chronologically sortable run ID.

    Format: YYYYMMDD_HHMM_{config_hash[:8]}
    """
    ts = pd.Timestamp.now()
    h = hash_config(config)[:8]
    return f"{ts:%Y%m%d_%H%M}_{h}"


def hash_config(config: SimulatorConfig) -> str:
    """
    Deterministic blake2b hash of the full config.

    Produces a 32-char hex digest. Configs with identical parameters
    produce identical hashes regardless of creation time.
    """
    config_dict = _config_to_serializable(config)
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.blake2b(
        canonical.encode("utf-8"),
        digest_size=16,
    ).hexdigest()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_simulation_run(
    result: Any,
    config: SimulatorConfig,
    *,
    performance_result: Any | None = None,
    run_id: str | None = None,
) -> Path:
    """
    Persist a simulation run to disk.

    Parameters
    ----------
    result
        SimulationResult from Simulator.run().
    config
        The SimulatorConfig used for the run.
    performance_result
        Optional PerformanceResult from generate_report().
    run_id
        Optional explicit run ID. If None, generated from config.

    Returns
    -------
    Path to the run directory.
    """
    if run_id is None:
        run_id = make_run_id(config)

    run_dir = _resolve_run_dir(config.persistence, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = set(config.persistence.artifacts)

    # Config (always saved as reference)
    if "config" in artifacts:
        _save_json(
            run_dir / "config.json",
            _config_to_serializable(config),
        )

    # DataFrame artifacts — monolithic only (flushable artifacts written per-year by flush_yearly_logs)
    _df_artifacts = {
        "selected_panel": ("selected_panel", lambda: result.selected_panel),
        "daily_portfolio_state": ("daily_portfolio_state", lambda: result.daily_portfolio_state_df()),
    }

    for artifact_key, (filename, df_fn) in _df_artifacts.items():
        if artifact_key in artifacts:
            df = df_fn()
            if not df.empty:
                df.to_parquet(run_dir / f"{filename}.parquet")

    # Performance metrics
    if "performance_metrics" in artifacts and performance_result is not None:
        _save_json(
            run_dir / "performance_metrics.json",
            performance_result.metrics,
        )
        if performance_result.group_metrics:
            _save_json(
                run_dir / "performance_group_metrics.json",
                performance_result.group_metrics,
            )

    # Performance HTML report — copy if it was generated
    if "performance_report" in artifacts and performance_result is not None:
        _copy_performance_html(config, run_dir)

    # Meta — summary for quick scanning
    meta = _build_meta(
        run_id=run_id,
        config=config,
        result=result,
        performance_result=performance_result,
        artifacts=artifacts,
    )
    _save_json(run_dir / "meta.json", meta)

    return run_dir


# ---------------------------------------------------------------------------
# Yearly flush
# ---------------------------------------------------------------------------

# Artifacts written as per-year parquet slices and cleared from memory
_FLUSHABLE_ARTIFACTS = {"daily_state", "action_log", "ticker_trade_log", "diagnostics", "closed_trades"}

_FLUSHABLE_DF_FNS: dict[str, Any] = {
    "daily_state": lambda r: r.daily_state_df(),
    "action_log": lambda r: r.action_log_df(),
    "ticker_trade_log": lambda r: r.ticker_trade_log_df(),
    "diagnostics": lambda r: r.diagnostics_log_df(),
    "closed_trades": lambda r: r.closed_trades_df(),
}

# Per-year flushable artifacts each live in their own subdir inside the run dir,
# keeping the run root uncluttered (dozens of yearly slices otherwise pile up flat).
_ARTIFACT_SUBDIRS: dict[str, str] = {
    "daily_state": "daily_states",
    "action_log": "action_logs",
    "ticker_trade_log": "ticker_trade_logs",
    "diagnostics": "diagnostics",
    "closed_trades": "closed_trades",
}


def _yearly_slices(run_dir: Path, artifact_key: str) -> list[Path]:
    """
    Per-year parquet slices for a flushable artifact, newest layout first.

    Prefers the dedicated subdir (current layout); falls back to slices stored
    flat in the run root (older runs) so previously-persisted runs stay loadable.
    """
    subdir = run_dir / _ARTIFACT_SUBDIRS.get(artifact_key, artifact_key)
    files = sorted(subdir.glob(f"{artifact_key}_*.parquet"))
    if files:
        return files
    return sorted(run_dir.glob(f"{artifact_key}_*.parquet"))


def flush_yearly_logs(
    result: Any,
    year: int,
    run_dir: Path,
    artifacts: set[str],
) -> None:
    """
    Write per-year parquet slices for all flushable artifacts.
    Called at each year-end checkpoint before clearing in-memory lists.

    Each artifact is written into its own subdir (e.g. ``daily_states/``) to keep
    the run root uncluttered.
    """
    for artifact_key in _FLUSHABLE_ARTIFACTS:
        if artifact_key in artifacts:
            df = _FLUSHABLE_DF_FNS[artifact_key](result)
            if not df.empty:
                subdir = run_dir / _ARTIFACT_SUBDIRS[artifact_key]
                subdir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(subdir / f"{artifact_key}_{year}.parquet")


def load_closed_trades_df(run_dir: Path) -> pd.DataFrame:
    """Reconstruct full closed_trades DataFrame from per-year parquet slices."""
    yearly_files = _yearly_slices(run_dir, "closed_trades")
    if not yearly_files:
        p = run_dir / "closed_trades.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in yearly_files], ignore_index=True)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_simulation_run(
    run_dir: str | Path,
) -> dict[str, Any]:
    """
    Load a persisted simulation run.

    Returns a dict with:
    - "meta": dict from meta.json
    - "config": dict from config.json (raw, not deserialized to dataclass)
    - "performance_metrics": dict or None
    - DataFrame keys: "selected_panel", "closed_trades", etc. (pd.DataFrame or None)
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out: dict[str, Any] = {}

    # Meta
    meta_path = run_dir / "meta.json"
    out["meta"] = _load_json(meta_path) if meta_path.exists() else {}

    # Config
    config_path = run_dir / "config.json"
    out["config"] = _load_json(config_path) if config_path.exists() else {}

    # Performance metrics
    perf_path = run_dir / "performance_metrics.json"
    out["performance_metrics"] = _load_json(perf_path) if perf_path.exists() else None

    group_perf_path = run_dir / "performance_group_metrics.json"
    out["performance_group_metrics"] = _load_json(group_perf_path) if group_perf_path.exists() else None

    # Monolithic artifacts
    for name in ["selected_panel", "daily_portfolio_state"]:
        parquet_path = run_dir / f"{name}.parquet"
        out[name] = pd.read_parquet(parquet_path) if parquet_path.exists() else None

    # Flushable artifacts — per-year slices concatenated, falling back to monolithic
    for name in ["closed_trades", "daily_state", "action_log", "ticker_trade_log", "diagnostics"]:
        yearly_files = _yearly_slices(run_dir, name)
        if yearly_files:
            out[name] = pd.concat([pd.read_parquet(f) for f in yearly_files], ignore_index=True)
        else:
            parquet_path = run_dir / f"{name}.parquet"
            out[name] = pd.read_parquet(parquet_path) if parquet_path.exists() else None

    return out


def list_simulation_runs(
    runs_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Scan simulation runs directory and return summary DataFrame.

    Reads meta.json from each run directory to build a summary table
    for quick comparison of runs.
    """
    if runs_dir is None:
        runs_dir = Path(SIMULATION_RUNS_ROOT)
    else:
        runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        return pd.DataFrame()

    rows = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        meta = _load_json(meta_path)
        meta["run_dir"] = str(d)
        rows.append(meta)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_run_dir(cfg: PersistenceConfig, run_id: str) -> Path:
    if cfg.output_dir:
        return Path(cfg.output_dir) / run_id
    return Path(SIMULATION_RUNS_ROOT) / run_id


def _config_to_serializable(config: SimulatorConfig) -> dict[str, Any]:
    """Convert config to a JSON-serializable dict."""
    return _to_serializable(asdict(config))


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj


def _save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_meta(
    *,
    run_id: str,
    config: SimulatorConfig,
    result: Any,
    performance_result: Any | None,
    artifacts: set[str],
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": pd.Timestamp.now().isoformat(),
        "config_hash": hash_config(config),
        "artifacts": sorted(artifacts),
        "universe": config.data.universe_config_name,
        "candidate_panel_stem": config.data.candidate_panel_stem,
    }

    # Summary stats from result
    if hasattr(result, "closed_trades"):
        meta["n_closed_trades"] = len(result.closed_trades)
    if hasattr(result, "daily_portfolio_state_log"):
        meta["n_trading_days"] = len(result.daily_portfolio_state_log)

    # Run dates
    meta["run_start_date"] = str(config.run.start_date) if config.run.start_date else None
    meta["run_end_date"] = str(config.run.end_date) if config.run.end_date else None

    # Key performance stats
    if performance_result is not None and hasattr(performance_result, "metrics"):
        m = performance_result.metrics
        for key in [
            "sortino_net", "sharpe_net", "max_drawdown_net",
            "total_return_net", "win_rate_net", "n_trades",
        ]:
            if key in m:
                meta[key] = m[key]

    return meta


def _copy_performance_html(config: SimulatorConfig, run_dir: Path) -> None:
    """Copy performance HTML report to run directory if it exists."""
    perf_cfg = config.performance
    if not perf_cfg.report_html:
        return

    source_dir = Path(perf_cfg.report_output_dir) if perf_cfg.report_output_dir else Path(".")
    for html_file in source_dir.glob("*_quantstats.html"):
        dest = run_dir / html_file.name
        if not dest.exists():
            shutil.copy2(html_file, dest)