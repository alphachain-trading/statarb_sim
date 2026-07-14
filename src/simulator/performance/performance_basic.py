from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.simulator.config import PerformanceConfig
from src.simulator.performance.performance_objective import sortino_ratio


@dataclass(slots=True)
class PerformanceResult:
    """
    Output of compute_performance().

    metrics         — portfolio-level scalar metrics dict
    group_metrics   — per-group metrics dict keyed by group_id (empty if cfg.per_group_breakdown=False)
    returns_gross   — daily gross return series (fraction of total_target_capital)
    returns_net     — daily net return series (fraction of total_target_capital)
    """
    metrics: dict[str, Any]
    group_metrics: dict[str, dict[str, Any]]
    returns_gross: pd.Series
    returns_net: pd.Series


def compute_performance(
    closed_trades_df: pd.DataFrame,
    daily_portfolio_state_df: pd.DataFrame,
    cfg: PerformanceConfig,
) -> PerformanceResult:
    """
    Compute all performance metrics from simulation output DataFrames.

    Parameters
    ----------
    closed_trades_df
        From SimulationResult.closed_trades_df().
    daily_portfolio_state_df
        From SimulationResult.daily_portfolio_state_df().
    cfg
        PerformanceConfig controlling annualization and breakdown options.
    """
    returns_gross, returns_net = _extract_returns(daily_portfolio_state_df)

    metrics = _compute_metrics(
        closed_trades_df=closed_trades_df,
        daily_df=daily_portfolio_state_df,
        returns_gross=returns_gross,
        returns_net=returns_net,
        cfg=cfg,
        group_id=None,
    )

    group_metrics: dict[str, dict[str, Any]] = {}
    if cfg.per_group_breakdown and not closed_trades_df.empty and "group_id" in closed_trades_df.columns:
        for group_id, group_trades in closed_trades_df.groupby("group_id"):
            group_metrics[str(group_id)] = _compute_trade_metrics(
                trades=group_trades,
                label=str(group_id),
            )

    return PerformanceResult(
        metrics=metrics,
        group_metrics=group_metrics,
        returns_gross=returns_gross,
        returns_net=returns_net,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_returns(daily_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Convert equity curve to daily percentage return series.

    Returns are computed as pct_change of the equity curve so that
    compounding them reproduces the equity curve exactly.  This is
    required for QuantStats (which compounds) and is also the correct
    input for Sharpe / Sortino / drawdown calculations.

    The series is trimmed to start at the first non-zero return so that
    pre-trading periods (e.g. residual model warmup) don't dilute
    annualized metrics.
    """
    if daily_df.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    df = daily_df.set_index("date").sort_index()

    gross = df["total_equity_gross"].pct_change().fillna(0.0)
    net = df["total_equity_net"].pct_change().fillna(0.0)

    # Trim to first active day (first non-zero return in either series)
    active_mask = (gross != 0.0) | (net != 0.0)
    if active_mask.any():
        first_active = active_mask.idxmax()
        gross = gross.loc[first_active:]
        net = net.loc[first_active:]

    gross.name = "returns_gross"
    net.name = "returns_net"
    return gross, net


def _compute_metrics(
    *,
    closed_trades_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    returns_gross: pd.Series,
    returns_net: pd.Series,
    cfg: PerformanceConfig,
    group_id: str | None,
) -> dict[str, Any]:
    trade_metrics = _compute_trade_metrics(
        trades=closed_trades_df,
        label="portfolio",
    )
    equity_metrics = _compute_equity_metrics(
        daily_df=daily_df,
        returns_gross=returns_gross,
        returns_net=returns_net,
        cfg=cfg,
    )
    return {**trade_metrics, **equity_metrics}


def _compute_trade_metrics(
    *,
    trades: pd.DataFrame,
    label: str,
) -> dict[str, Any]:
    if trades.empty:
        return _empty_trade_metrics()

    t = trades.copy()

    n_trades = int(len(t))

    # Win/loss on net PnL
    wins_net = t[t["realized_pnl_net"] > 0.0]
    losses_net = t[t["realized_pnl_net"] <= 0.0]
    win_rate_net = len(wins_net) / n_trades if n_trades > 0 else float("nan")
    loss_rate_net = len(losses_net) / n_trades if n_trades > 0 else float("nan")
    mean_gain_net = float(wins_net["realized_pnl_net"].mean()) if not wins_net.empty else float("nan")
    mean_loss_net = float(losses_net["realized_pnl_net"].mean()) if not losses_net.empty else float("nan")

    # Win/loss on gross PnL
    wins_gross = t[t["realized_pnl_gross"] > 0.0]
    losses_gross = t[t["realized_pnl_gross"] <= 0.0]
    win_rate_gross = len(wins_gross) / n_trades if n_trades > 0 else float("nan")
    mean_gain_gross = float(wins_gross["realized_pnl_gross"].mean()) if not wins_gross.empty else float("nan")
    mean_loss_gross = float(losses_gross["realized_pnl_gross"].mean()) if not losses_gross.empty else float("nan")

    # Profit factor (net)
    total_gains = float(wins_net["realized_pnl_net"].sum()) if not wins_net.empty else 0.0
    total_losses = float(losses_net["realized_pnl_net"].sum()) if not losses_net.empty else 0.0
    profit_factor = (
        total_gains / abs(total_losses)
        if total_losses != 0.0
        else float("inf") if total_gains > 0.0 else float("nan")
    )

    # Holding periods
    avg_holding_days = float(t["days_open"].mean()) if "days_open" in t.columns else float("nan")
    avg_holding_wins = float(wins_net["days_open"].mean()) if not wins_net.empty and "days_open" in t.columns else float("nan")
    avg_holding_losses = float(losses_net["days_open"].mean()) if not losses_net.empty and "days_open" in t.columns else float("nan")

    # Entry/exit z-scores
    avg_entry_z = float(t["entry_z_score"].dropna().mean()) if "entry_z_score" in t.columns else float("nan")
    avg_exit_z = float(t["exit_z_score"].dropna().mean()) if "exit_z_score" in t.columns else float("nan")
    avg_abs_entry_z = float(t["entry_z_score"].dropna().abs().mean()) if "entry_z_score" in t.columns else float("nan")

    # Cost metrics
    total_txn_costs = float(t["transaction_costs"].sum()) if "transaction_costs" in t.columns else float("nan")
    total_borrow_costs = float(t["borrow_costs"].sum()) if "borrow_costs" in t.columns else float("nan")
    total_costs = total_txn_costs + total_borrow_costs
    total_gross_pnl = float(t["realized_pnl_gross"].sum())
    total_net_pnl = float(t["realized_pnl_net"].sum())
    avg_cost_per_trade = total_costs / n_trades if n_trades > 0 else float("nan")

    # Groups traded (if column present)
    n_groups = int(t["group_id"].nunique()) if "group_id" in t.columns else float("nan")

    return {
        "n_trades": n_trades,
        "n_groups_traded": n_groups,
        "total_gross_pnl": total_gross_pnl,
        "total_net_pnl": total_net_pnl,
        "total_transaction_costs": total_txn_costs,
        "total_borrow_costs": total_borrow_costs,
        "total_costs": total_costs,
        "avg_cost_per_trade": avg_cost_per_trade,
        "profit_factor_net": profit_factor,
        "win_rate_net": win_rate_net,
        "loss_rate_net": loss_rate_net,
        "win_rate_gross": win_rate_gross,
        "mean_gain_per_win_net": mean_gain_net,
        "mean_loss_per_loss_net": mean_loss_net,
        "mean_gain_per_win_gross": mean_gain_gross,
        "mean_loss_per_loss_gross": mean_loss_gross,
        "avg_holding_days": avg_holding_days,
        "avg_holding_days_wins": avg_holding_wins,
        "avg_holding_days_losses": avg_holding_losses,
        "avg_entry_z_score": avg_entry_z,
        "avg_abs_entry_z_score": avg_abs_entry_z,
        "avg_exit_z_score": avg_exit_z,
    }


def _compute_equity_metrics(
    *,
    daily_df: pd.DataFrame,
    returns_gross: pd.Series,
    returns_net: pd.Series,
    cfg: PerformanceConfig,
) -> dict[str, Any]:
    if daily_df.empty or returns_net.empty:
        return _empty_equity_metrics()

    ann = cfg.annualization_factor
    df = daily_df.set_index("date").sort_index()

    # Align daily_df to the trimmed returns (first active trading day onward)
    first_active = returns_net.index[0]
    df = df.loc[first_active:]

    n_days = int(len(returns_net))
    n_months = max(n_days / 21, 1.0)

    # Sortino (net and gross)
    sortino_net = sortino_ratio(returns_net, annualization_factor=ann)
    sortino_gross = sortino_ratio(returns_gross, annualization_factor=ann)

    # Sharpe (net)
    r = returns_net.dropna().to_numpy(dtype=float)
    sharpe_net = (
        float(np.mean(r) / np.std(r, ddof=1)) * float(np.sqrt(ann))
        if len(r) > 1 and np.std(r, ddof=1) > 0.0
        else float("nan")
    )

    # Drawdown
    equity_net = df["total_equity_net"]
    rolling_max = equity_net.cummax()
    drawdown = (equity_net - rolling_max) / rolling_max.replace(0.0, float("nan"))
    max_drawdown = float(drawdown.min())

    equity_gross = df["total_equity_gross"]
    rolling_max_gross = equity_gross.cummax()
    drawdown_gross = (equity_gross - rolling_max_gross) / rolling_max_gross.replace(0.0, float("nan"))
    max_drawdown_gross = float(drawdown_gross.min())

    # Calmar (net annualized return / abs max drawdown)
    total_return_net = float((equity_net.iloc[-1] / equity_net.iloc[0]) - 1.0) if equity_net.iloc[0] != 0.0 else float("nan")
    annual_return_net = float((1.0 + total_return_net) ** (ann / n_days) - 1.0) if np.isfinite(total_return_net) else float("nan")
    calmar_net = (
        annual_return_net / abs(max_drawdown)
        if np.isfinite(max_drawdown) and max_drawdown != 0.0 and np.isfinite(annual_return_net)
        else float("nan")
    )

    total_return_gross = float((equity_gross.iloc[-1] / equity_gross.iloc[0]) - 1.0) if equity_gross.iloc[0] != 0.0 else float("nan")
    annual_return_gross = float((1.0 + total_return_gross) ** (ann / n_days) - 1.0) if np.isfinite(total_return_gross) else float("nan")

    # Avg concurrent positions
    avg_concurrent = float(df["n_live_candidate_positions"].mean()) if "n_live_candidate_positions" in df.columns else float("nan")

    # Cost drag
    capital = float(df["total_capital"].iloc[0])
    cum_txn = float(df["cumulative_transaction_costs"].iloc[-1]) if "cumulative_transaction_costs" in df.columns else float("nan")
    cum_borrow = float(df["cumulative_borrow_costs"].iloc[-1]) if "cumulative_borrow_costs" in df.columns else float("nan")
    total_costs_all = cum_txn + cum_borrow
    cost_drag_bps = (total_costs_all / capital) * 10_000 if capital > 0.0 else float("nan")

    # Avg trades per month (from daily_df n_live proxy — actual from closed_trades handled in trade metrics)
    avg_daily_borrow = float(df["daily_borrow_cost"].mean()) if "daily_borrow_cost" in df.columns else float("nan")

    return {
        "n_trading_days": n_days,
        "sharpe_net": sharpe_net,
        "sortino_net": sortino_net,
        "sortino_gross": sortino_gross,
        "calmar_net": calmar_net,
        "total_return_net": total_return_net,
        "total_return_gross": total_return_gross,
        "annual_return_net": annual_return_net,
        "annual_return_gross": annual_return_gross,
        "max_drawdown_net": max_drawdown,
        "max_drawdown_gross": max_drawdown_gross,
        "avg_concurrent_positions": avg_concurrent,
        "cumulative_transaction_costs": cum_txn,
        "cumulative_borrow_costs": cum_borrow,
        "cost_drag_bps": cost_drag_bps,
        "avg_daily_borrow_cost": avg_daily_borrow,
    }


def _empty_trade_metrics() -> dict[str, Any]:
    keys = [
        "n_trades", "n_groups_traded", "total_gross_pnl", "total_net_pnl",
        "total_transaction_costs", "total_borrow_costs", "total_costs",
        "avg_cost_per_trade", "profit_factor_net",
        "win_rate_net", "loss_rate_net", "win_rate_gross",
        "mean_gain_per_win_net", "mean_loss_per_loss_net",
        "mean_gain_per_win_gross", "mean_loss_per_loss_gross",
        "avg_holding_days", "avg_holding_days_wins", "avg_holding_days_losses",
        "avg_entry_z_score", "avg_abs_entry_z_score", "avg_exit_z_score",
    ]
    return {k: float("nan") for k in keys} | {"n_trades": 0, "n_groups_traded": 0}


def _empty_equity_metrics() -> dict[str, Any]:
    keys = [
        "n_trading_days", "sharpe_net", "sortino_net", "sortino_gross",
        "calmar_net", "total_return_net", "total_return_gross",
        "annual_return_net", "annual_return_gross",
        "max_drawdown_net", "max_drawdown_gross",
        "avg_concurrent_positions",
        "cumulative_transaction_costs", "cumulative_borrow_costs",
        "cost_drag_bps", "avg_daily_borrow_cost",
    ]
    return {k: float("nan") for k in keys} | {"n_trading_days": 0}