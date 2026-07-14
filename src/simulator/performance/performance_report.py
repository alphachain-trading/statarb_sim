from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulator.config import PerformanceConfig
from src.simulator.performance.performance_basic import PerformanceResult, compute_performance

try:
    import quantstats as qs
    _HAS_QUANTSTATS = True
except ImportError:
    _HAS_QUANTSTATS = False


_METRICS_ORDER = [
    # Return
    ("total_return_net",            "Total Return (Net)",           ".2%"),
    ("total_return_gross",          "Total Return (Gross)",         ".2%"),
    ("annual_return_net",           "Annual Return (Net)",          ".2%"),
    ("annual_return_gross",         "Annual Return (Gross)",        ".2%"),
    # Risk-adjusted
    ("sharpe_net",                  "Sharpe Ratio (Net)",           ".3f"),
    ("sortino_net",                 "Sortino Ratio (Net)",          ".3f"),
    ("sortino_gross",               "Sortino Ratio (Gross)",        ".3f"),
    ("calmar_net",                  "Calmar Ratio (Net)",           ".3f"),
    # Drawdown
    ("max_drawdown_net",            "Max Drawdown (Net)",           ".2%"),
    ("max_drawdown_gross",          "Max Drawdown (Gross)",         ".2%"),
    # Trades
    ("n_trades",                    "# Trades",                     "d"),
    ("n_groups_traded",             "# Groups Traded",              "d"),
    ("n_trading_days",              "# Trading Days",               "d"),
    ("avg_concurrent_positions",    "Avg Concurrent Positions",     ".2f"),
    ("avg_holding_days",            "Avg Holding Period (days)",    ".1f"),
    ("avg_holding_days_wins",       "Avg Holding — Wins (days)",    ".1f"),
    ("avg_holding_days_losses",     "Avg Holding — Losses (days)",  ".1f"),
    # Win/loss
    ("win_rate_net",                "Win Rate (Net)",               ".1%"),
    ("loss_rate_net",               "Loss Rate (Net)",              ".1%"),
    ("win_rate_gross",              "Win Rate (Gross)",             ".1%"),
    ("profit_factor_net",           "Profit Factor (Net)",          ".3f"),
    ("mean_gain_per_win_net",       "Mean Gain / Win (Net $)",      ".2f"),
    ("mean_loss_per_loss_net",      "Mean Loss / Loss (Net $)",     ".2f"),
    ("mean_gain_per_win_gross",     "Mean Gain / Win (Gross $)",    ".2f"),
    ("mean_loss_per_loss_gross",    "Mean Loss / Loss (Gross $)",   ".2f"),
    # Entry/exit quality
    ("avg_abs_entry_z_score",       "Avg |Entry Z-Score|",          ".3f"),
    ("avg_entry_z_score",           "Avg Entry Z-Score",            ".3f"),
    ("avg_exit_z_score",            "Avg Exit Z-Score",             ".3f"),
    # PnL
    ("total_gross_pnl",             "Total Gross PnL ($)",          ".2f"),
    ("total_net_pnl",               "Total Net PnL ($)",            ".2f"),
    # Costs
    ("total_transaction_costs",     "Total Transaction Costs ($)",  ".2f"),
    ("total_borrow_costs",          "Total Borrow Costs ($)",       ".2f"),
    ("total_costs",                 "Total Costs ($)",              ".2f"),
    ("cost_drag_bps",               "Cost Drag (bps)",              ".1f"),
    ("avg_cost_per_trade",          "Avg Cost / Trade ($)",         ".2f"),
    ("avg_daily_borrow_cost",       "Avg Daily Borrow Cost ($)",    ".4f"),
]


def generate_report(
    closed_trades_df: pd.DataFrame,
    daily_portfolio_state_df: pd.DataFrame,
    cfg: PerformanceConfig,
    *,
    title: str = "Strategy Performance",
) -> PerformanceResult:
    """
    Compute metrics, optionally print table, optionally write quantstats HTML report.

    Returns the PerformanceResult for programmatic access regardless of cfg flags.
    """
    result = compute_performance(
        closed_trades_df=closed_trades_df,
        daily_portfolio_state_df=daily_portfolio_state_df,
        cfg=cfg,
    )

    if cfg.metrics_table:
        _print_metrics_table(result.metrics, title=title)

        if cfg.per_group_breakdown and result.group_metrics:
            for group_id, gm in sorted(result.group_metrics.items()):
                _print_metrics_table(gm, title=f"  Group: {group_id}", trade_only=True)

    if cfg.report_html:
        _write_quantstats_report(
            returns_net=result.returns_net,
            cfg=cfg,
            title=title,
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_value(value: Any, fmt: str) -> str:
    if value is None or (isinstance(value, float) and __import__("math").isnan(value)):
        return "n/a"
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    try:
        if fmt == "d":
            return f"{int(value):,}"
        return format(value, fmt)
    except (ValueError, TypeError):
        return str(value)


def _print_metrics_table(
    metrics: dict[str, Any],
    *,
    title: str,
    trade_only: bool = False,
) -> None:
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print(f"{'─' * 52}")

    rows = _METRICS_ORDER if not trade_only else [
        r for r in _METRICS_ORDER
        if r[0] in {
            "n_trades", "win_rate_net", "loss_rate_net", "win_rate_gross",
            "profit_factor_net", "avg_holding_days",
            "mean_gain_per_win_net", "mean_loss_per_loss_net",
            "total_gross_pnl", "total_net_pnl", "total_costs",
            "avg_entry_z_score", "avg_abs_entry_z_score", "avg_exit_z_score",
        }
    ]

    for key, label, fmt in rows:
        if key not in metrics:
            continue
        value_str = _fmt_value(metrics[key], fmt)
        print(f"  {label:<38} {value_str:>10}")

    print(f"{'─' * 52}")


def _write_quantstats_report(
        returns_net: pd.Series,
        cfg: PerformanceConfig,
        title: str,
) -> None:
    import warnings
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
    warnings.filterwarnings('ignore', category=UserWarning, message='.*findfont.*')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    if not _HAS_QUANTSTATS:
        print(
            "\n[performance] quantstats not installed — skipping HTML report. "
            "Install with: pip install quantstats"
        )
        return

    if returns_net.empty:
        print("\n[performance] No returns data — skipping HTML report.")
        return

    # Sanitize returns: remove NaN, inf
    returns_clean = returns_net.dropna().copy()
    returns_clean = returns_clean[np.isfinite(returns_clean)]

    if returns_clean.empty or len(returns_clean) < 10:
        print("\n[performance] Insufficient valid return data — skipping HTML report.")
        return

    # IMPORTANT: Preserve the datetime index for quantstats compatibility
    # Do NOT reset_index() — quantstats needs datetime index to match with benchmark
    returns_clean = returns_clean.astype(np.float64)

    out_dir = Path(cfg.report_output_dir) if cfg.report_output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "_").replace(":", "")
    out_path = out_dir / f"{safe_title}_quantstats.html"

    benchmark = cfg.benchmark_ticker or None

    qs.extend_pandas()
    qs.reports.html(
        returns_clean,
        benchmark=benchmark,
        output=str(out_path),
        title=title,
        download_filename=str(out_path),
    )

    print(f"\n[performance] HTML report written to: {out_path}")