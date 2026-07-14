"""
Rebuild sweep results from persisted run directories.

Replaces the fragile sweep_results.pkl append mechanic with a
scan-and-rebuild approach. Each run directory contains config.json
and performance_metrics.json — these are the source of truth.

Usage:
    from sweep_rebuild import rebuild_sweep_results

    df = rebuild_sweep_results()                    # uses SIMULATION_RUNS_ROOT
    df = rebuild_sweep_results("/path/to/runs")     # explicit path

    # Re-save as pkl for backward compat with view_* helpers
    df.to_pickle("/path/to/runs/sweep_results.pkl")
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from settings import SIMULATION_RUNS_ROOT


def rebuild_sweep_results(
    runs_dir: str | Path | None = None,
    *,
    include_failed: bool = False,
) -> pd.DataFrame:
    """
    Scan all run directories and build a consolidated sweep results DataFrame.

    Parameters
    ----------
    runs_dir
        Root directory containing run subdirectories. Defaults to SIMULATION_RUNS_ROOT.
    include_failed
        If True, include run dirs that lack performance_metrics.json (as error rows).

    Returns
    -------
    DataFrame indexed by run_id with config + performance columns.
    """
    if runs_dir is None:
        runs_dir = Path(SIMULATION_RUNS_ROOT)
    else:
        runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        print(f"[rebuild] Directory not found: {runs_dir}")
        return pd.DataFrame()

    rows = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Skip non-run directories (like __pycache__, .git, etc.)
        run_id = run_dir.name
        if run_id.startswith(".") or run_id.startswith("__"):
            continue

        config_path = run_dir / "config.json"
        metrics_path = run_dir / "performance_metrics.json"
        group_metrics_path = run_dir / "performance_group_metrics.json"

        if not config_path.exists():
            n_skip += 1
            continue

        if not metrics_path.exists():
            if include_failed:
                n_fail += 1
                rows.append({
                    "run_id": run_id,
                    "error": "No performance_metrics.json",
                })
            else:
                n_skip += 1
            continue

        try:
            config = _load_json(config_path)
            metrics = _load_json(metrics_path)

            row = {
                "run_id": run_id,
                "alias": _build_alias_from_config(config),
                **_extract_config_fields(config),
                **_flatten_metrics(metrics),
            }

            # Add group-level metrics if available
            if group_metrics_path.exists():
                group_metrics = _load_json(group_metrics_path)
                row["n_groups_traded"] = len(group_metrics)

            rows.append(row)
            n_ok += 1

        except Exception as e:
            n_fail += 1
            if include_failed:
                rows.append({
                    "run_id": run_id,
                    "error": str(e),
                })

    print(f"[rebuild] Scanned {n_ok + n_skip + n_fail} dirs: "
          f"{n_ok} OK, {n_skip} skipped, {n_fail} failed")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("run_id")

    # Sort by run_id (chronological since IDs are timestamp-based)
    df = df.sort_index()

    return df


# ── Internal helpers ─────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_config_fields(config: dict) -> dict:
    """Extract sweep-relevant fields from config.json."""
    out = {}

    # Trader config
    trader = config.get("trader", {})
    out["entry_z"] = trader.get("entry_z") or trader.get("entry_z_threshold")
    out["exit_z"] = trader.get("exit_z") or trader.get("exit_z_threshold")
    out["max_holding_days"] = trader.get("max_holding_days")
    out["exit_rule"] = _serialize_if_complex(trader.get("exit_rule"))

    # Z-score config
    z_cfg = config.get("z_score", {})
    if isinstance(z_cfg, list):
        # Multi-timescale
        out["z_score_overrides"] = str([
            f"{zc.get('residual_key', '')}:zhl{zc.get('lookback', '?')}"
            for zc in z_cfg
        ])
        out["z_method"] = z_cfg[0].get("method", "ewm") if z_cfg else "ewm"
        out["z_lookback"] = [zc.get("lookback") for zc in z_cfg]
        out["n_timescales"] = len(z_cfg)
    else:
        out["z_method"] = z_cfg.get("method", "ewm")
        out["z_lookback"] = z_cfg.get("lookback")
        out["z_score_overrides"] = None
        out["n_timescales"] = 1

    # Risk manager
    rm = config.get("risk_manager") or {}
    out["max_gross_exposure"] = rm.get("max_gross_exposure")
    out["max_ticker_exposure_pct"] = rm.get("max_ticker_exposure_pct")

    # Timescale risk
    ts_risk = rm.get("timescale_risk")
    out["timescale_risk"] = _serialize_if_complex(ts_risk)

    # Sizing (new schema) with legacy fallback (base_pair_notional/kelly in risk_manager)
    sizing = config.get("sizing") or {}
    out["base_pair_notional"] = sizing.get("base_pair_notional") or rm.get("base_pair_notional")
    vol_cfg = sizing.get("vol_normalize")
    out["vol_normalize"] = (vol_cfg is not None) if sizing else rm.get("vol_normalize")
    kelly = sizing.get("kelly") or rm.get("kelly")
    out["kelly"] = _serialize_if_complex(kelly)

    # Cross-ts entry config
    cross_ts = trader.get("cross_timescale_entry")
    out["cross_ts"] = _serialize_if_complex(cross_ts)

    # Data config
    data = config.get("data", {})
    out["excluded_sectors"] = data.get("excluded_sectors")
    out["selected_sectors"] = data.get("selected_sectors")

    # Run config
    run = config.get("run", {})
    out["start_date"] = run.get("start_date")
    out["end_date"] = run.get("end_date")

    # Capital (new: total_capital, legacy: target_portfolio_capital)
    capital = config.get("capital", {})
    out["total_capital"] = capital.get("total_capital") or capital.get("target_portfolio_capital")

    return out


def _flatten_metrics(metrics: dict) -> dict:
    """Flatten performance_metrics.json into sweep-compatible column names."""
    # Map from metrics JSON keys to sweep DataFrame column names
    key_map = {
        "total_return_net": "total_return_net",
        "total_return_gross": "total_return_gross",
        "annual_return_net": "annual_return_net",
        "annual_return_gross": "annual_return_gross",
        "sharpe_net": "sharpe_net",
        "sortino_net": "sortino_net",
        "sortino_gross": "sortino_gross",
        "calmar_net": "calmar_net",
        "max_drawdown_net": "max_drawdown_net",
        "max_drawdown_gross": "max_drawdown_gross",
        "n_trades": "n_trades",
        "n_groups_traded": "n_groups_traded",
        "n_trading_days": "n_trading_days",
        "avg_concurrent_positions": "avg_concurrent_positions",
        "avg_holding_days": "avg_holding_days",
        "avg_holding_days_wins": "avg_holding_days_wins",
        "avg_holding_days_losses": "avg_holding_days_losses",
        "win_rate_net": "win_rate_net",
        "loss_rate_net": "loss_rate_net",
        "win_rate_gross": "win_rate_gross",
        "profit_factor_net": "profit_factor_net",
        "mean_gain_per_win_net": "mean_gain_per_win_net",
        "mean_loss_per_loss_net": "mean_loss_per_loss_net",
        "mean_gain_per_win_gross": "mean_gain_per_win_gross",
        "mean_loss_per_loss_gross": "mean_loss_per_loss_gross",
        "avg_abs_entry_z_score": "avg_abs_entry_z_score",
        "avg_entry_z_score": "avg_entry_z_score",
        "avg_exit_z_score": "avg_exit_z_score",
        "total_gross_pnl": "total_gross_pnl",
        "total_net_pnl": "total_net_pnl",
        "total_transaction_costs": "total_transaction_costs",
        "total_borrow_costs": "total_borrow_costs",
        "total_costs": "total_costs",
        "cost_drag_bps": "cost_drag_bps",
        "avg_cost_per_trade": "avg_cost_per_trade",
        "avg_daily_borrow_cost": "avg_daily_borrow_cost",
    }

    out = {}
    for json_key, col_name in key_map.items():
        if json_key in metrics:
            out[col_name] = metrics[json_key]

    # Also capture any keys not in the map (future-proofing)
    for k, v in metrics.items():
        if k not in key_map and isinstance(v, (int, float, str, bool)):
            out[k] = v

    return out


def _build_alias_from_config(config: dict) -> str:
    """Best-effort alias reconstruction from config.json."""
    parts = []

    trader = config.get("trader", {})
    entry_z = trader.get("entry_z") or trader.get("entry_z_threshold")
    exit_z = trader.get("exit_z") or trader.get("exit_z_threshold") or 0.0
    if entry_z is not None:
        parts.append(f"z{entry_z}")
    parts.append(f"xz{exit_z}")

    z_cfg = config.get("z_score", {})
    if isinstance(z_cfg, list):
        # Multi-timescale
        parts.append(f"mt{len(z_cfg)}")
        method = z_cfg[0].get("method", "ewm") if z_cfg else "ewm"
        parts.append(method)

        # Residual half-lives and z-score lookbacks
        rhls = []
        zhls = []
        for zc in z_cfg:
            rkey = zc.get("residual_key", "")
            # Extract hl from rkey like "exp_hl63_mh126"
            for part in rkey.split("_"):
                if part.startswith("hl") and part[2:].isdigit():
                    rhls.append(part[2:])
                    break
            lb = zc.get("lookback")
            if lb is not None:
                zhls.append(str(lb))
        if rhls:
            parts.append("rhl" + "+".join(rhls))
        if zhls:
            parts.append("zhl" + "+".join(zhls))
    else:
        parts.append(f"mt1")
        method = z_cfg.get("method", "ewm")
        parts.append(method)
        lb = z_cfg.get("lookback")
        if lb is not None:
            # Extract rhl from residual_key
            rkey = z_cfg.get("residual_key", "")
            for part in rkey.split("_"):
                if part.startswith("hl") and part[2:].isdigit():
                    parts.append(f"rhl{part[2:]}")
                    break
            parts.append(f"zhl{lb}")

    # Risk manager
    rm = config.get("risk_manager") or {}
    tick = rm.get("max_ticker_exposure_pct")
    if tick is not None:
        parts.append(f"tick{tick:.2f}")
    bpn = sizing.get("base_pair_notional") or rm.get("base_pair_notional")
    if bpn is not None:
        parts.append(f"bpn{int(bpn / 1000)}k")
    gx = rm.get("max_gross_exposure")
    if gx is not None:
        parts.append(f"gx{gx:.1f}")

    # Kelly (new: sizing.kelly, legacy: risk_manager.kelly)
    sizing = config.get("sizing") or {}
    kelly = sizing.get("kelly") or rm.get("kelly")
    if kelly is not None:
        k_parts = [f"f{kelly.get('fraction', 0.5):.1f}"]
        if kelly.get("half_life") is not None:
            k_parts.append(f"hl{kelly['half_life']}")
        k_parts.append("sec" if kelly.get("per_sector", True) else "glob")
        parts.append("kelly-" + "+".join(k_parts))

    # Timescale risk
    ts_risk = rm.get("timescale_risk")
    if ts_risk is not None:
        tr_parts = []
        if ts_risk.get("max_timescales_per_spread") is not None:
            tr_parts.append(f"mts{ts_risk['max_timescales_per_spread']}")
        tr_parts.append(ts_risk.get("selection", "max_abs_z"))
        parts.append("tsr-" + "+".join(tr_parts))

    # Sectors
    data = config.get("data", {})
    excluded = data.get("excluded_sectors")
    if excluded:
        parts.append("ex-" + ",".join(sorted(excluded)))
    else:
        parts.append("all")

    # Date range
    run = config.get("run", {})
    start = run.get("start_date")
    end = run.get("end_date")
    if start:
        parts.append(f"S{pd.Timestamp(start).strftime('%Y%m%d')}")
    if end:
        parts.append(f"E{pd.Timestamp(end).strftime('%Y%m%d')}")

    return "_".join(parts)


def _serialize_if_complex(val) -> str | None:
    """Serialize dicts/lists to string, pass through None."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return str(val)
    return val