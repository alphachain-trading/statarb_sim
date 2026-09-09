"""
Reconstruct the daily portfolio equity/cost curve from persisted trade
records, replacing DailyPortfolioStateLogEntry's day-by-day accumulation
during simulation (Track F1 commit 8).

DailyPortfolioStateLogEntry is not write-only: generate_report computes
Sharpe, drawdown, and the rest off the columns it produces
(performance_basic.py's _extract_returns / _compute_equity_metrics). This
module stands up a replacement that produces the same columns from
closed_trades_df + the still-open positions at run end, so that log can be
dropped (commit 8c) once this is wired in (8b) and verified.

Deliberately named to avoid colliding with the doomed
SimulationResult.daily_state_df() (a different, already-write-only method
being deleted in the same commit).

Design: raw per-ticker prices, not residual/spread levels. Every column
performance_basic.py reads (total_equity_gross, total_equity_net,
n_live_candidate_positions, cumulative_transaction_costs,
cumulative_borrow_costs, daily_borrow_cost, total_capital) is, in the live
code, computed from LiveCandidatePosition.units_by_ticker /
entry_prices_by_ticker against the raw price matrix
(_mark_to_market_live_positions, _accrue_borrow_costs in simulator.py) --
never from a residual model. So reconstruction needs no residual replay:
just re-derive each closed trade's units_by_ticker / entry_prices_by_ticker
(via the SAME PositionTranslator.translate_open + ExecutionEngine.
execute_deltas the live run used, fed weights.parquet's full-precision
weights) and replay the exact day-by-day bookkeeping order the live loop
used. Still-open positions at run end need no reconstruction at all --
LiveCandidatePosition already carries units_by_ticker, entry_prices_by_ticker,
and accumulated_transaction_cost (entry commission only, since it hasn't
closed) directly.

Sequencing this replay must match exactly (verified against
Simulator.run(), simulator.py's per-day loop):
  1. Mark-to-market positions already open BEFORE today (today's own new
     opens are not in the set yet).
  2. Accrue today's borrow cost from that post-MTM state (so a position
     opened today pays no borrow cost today; a position closing today still
     pays borrow cost today, since it is removed only in step 3 below).
  3. Process today's closes: credit realized PnL and the exit-side
     commission to the cumulative totals; remove from the open set.
  4. Process today's opens: construct units/entry prices, add to the open
     set with zero unrealized PnL; credit the entry-side commission.
  5. Snapshot: total_unrealized_pnl / total_gross_value /
     n_live_candidate_positions from the now-updated open set.

A closed trade with entry_date=E, exit_date=X therefore appears in the
equity snapshot (step 5) on every day in [E, X-1] -- not X, since it is
removed in step 3 before step 5 runs -- but accrues borrow cost (step 2) on
every day in [E+1, X], since step 2 runs before step 3 removes it on X but
after it was added on E in step 4. This one-day offset between the two
windows is easy to get backwards; it is exactly what the day-by-day replay
below reproduces by construction, rather than by a derived date-range
formula.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulator.actions import OpenCandidateAction
from src.simulator.execution import ExecutionEngine
from src.simulator.position_translator import PositionTranslator
from src.simulator.types import CandidateRef, LiveCandidatePosition


@dataclass(slots=True)
class _ReplayPosition:
    units_by_ticker: dict[str, int]
    entry_prices_by_ticker: dict[str, float]
    direction: float
    unrealized_pnl: float = 0.0
    gross_value: float = 0.0


def _reconstruct_entry_fill(
    *,
    trade_row: pd.Series,
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]],
    price_matrix: pd.DataFrame,
    position_translator: PositionTranslator,
    execution_engine: ExecutionEngine,
) -> tuple[dict[str, int], dict[str, float], float]:
    """
    Re-derive (units_by_ticker, entry_prices_by_ticker, entry_commission)
    for one closed trade, via the exact same PositionTranslator /
    ExecutionEngine the live run used -- not a reimplementation of their
    math, the same code, fed weights.parquet's full-precision weights.
    """
    spread_id = str(trade_row["spread_id"])
    entry_date = pd.Timestamp(trade_row["entry_date"])
    entry_asof_date = pd.Timestamp(trade_row["entry_asof_date"])
    members = tuple(spread_id.split("|"))

    weight_dict = weights_lookup.get((spread_id, entry_asof_date))
    if weight_dict is None:
        raise KeyError(
            f"No weights found for spread_id={spread_id!r} entry_asof_date={entry_asof_date} "
            f"-- weights.parquet is missing this (spread_id, asof_date) key, needed to "
            f"reconstruct trade_id={trade_row['trade_id']!r}'s entry fill."
        )
    weights = tuple(float(weight_dict[m]) for m in members)

    if entry_date not in price_matrix.index:
        raise KeyError(f"entry_date={entry_date} not found in the price matrix.")
    entry_prices = price_matrix.loc[entry_date].dropna()

    action = OpenCandidateAction(
        candidate_id=str(trade_row["candidate_id"]),
        group_id=str(trade_row["group_id"]),
        spread_id=spread_id,
        direction=float(trade_row["direction"]),
        pair_notional=float(trade_row["pair_notional"]),
    )
    ref = CandidateRef(
        candidate_id=str(trade_row["candidate_id"]),
        spread_id=spread_id,
        group_id=str(trade_row["group_id"]),
        asof_date=entry_asof_date,
        members=members,
        weights=weights,
    )

    deltas = position_translator.translate_open(
        action=action, candidate_ref=ref, current_prices=entry_prices,
    )
    exec_res = execution_engine.execute_deltas(deltas=deltas, current_prices=entry_prices)

    units_by_ticker = {f.ticker: int(f.filled_delta_units) for f in exec_res.fills}
    entry_prices_by_ticker = {f.ticker: f.fill_price for f in exec_res.fills}
    entry_commission = sum(f.commission for f in exec_res.fills)
    return units_by_ticker, entry_prices_by_ticker, entry_commission


def _mark_to_market(pos: _ReplayPosition, current_prices: pd.Series) -> None:
    gross_value = 0.0
    unrealized_pnl = 0.0
    for ticker, units in pos.units_by_ticker.items():
        px_now = float(current_prices.loc[ticker])
        px_entry = float(pos.entry_prices_by_ticker[ticker])
        gross_value += abs(units * px_now)
        unrealized_pnl += units * (px_now - px_entry)
    pos.gross_value = gross_value
    pos.unrealized_pnl = float(unrealized_pnl)


def reconstruct_daily_portfolio_state(
    *,
    dates: pd.DatetimeIndex,
    price_matrix: pd.DataFrame,
    closed_trades_df: pd.DataFrame,
    final_live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
    weights_lookup: dict[tuple[str, pd.Timestamp], dict[str, float]],
    position_translator: PositionTranslator,
    execution_engine: ExecutionEngine,
    total_capital: float,
    short_borrow_rate_annual_bps: float,
) -> pd.DataFrame:
    """
    Reconstruct the columns performance_basic.py reads off
    SimulationResult.daily_portfolio_state_df(): date, total_capital,
    total_equity_gross, total_equity_net, n_live_candidate_positions,
    daily_borrow_cost, cumulative_transaction_costs,
    cumulative_borrow_costs.

    Day-by-day replay (see module docstring for the exact step order this
    reproduces), driven by trade-open/trade-close events instead of live
    trader decisions. dates must be the same full simulation date range
    Simulator.run() walked; price_matrix must be the same price field
    matrix (e.g. Close) it used.
    """
    if dates.empty:
        return pd.DataFrame(columns=[
            "date", "total_capital", "total_equity_gross", "total_equity_net",
            "n_live_candidate_positions", "daily_borrow_cost",
            "cumulative_transaction_costs", "cumulative_borrow_costs",
        ])

    rate = short_borrow_rate_annual_bps / 10_000.0 / 252.0

    # Pre-reconstruct every closed trade's entry fill once, up front.
    opens_by_date: dict[pd.Timestamp, list[tuple[str, _ReplayPosition, float]]] = {}
    closes_by_date: dict[pd.Timestamp, list[tuple[str, float, float]]] = {}
    # candidate_id -> (open_date, exit_commission) not needed beyond the
    # per-date lists above; realized_pnl_gross and total transaction_costs
    # come straight from closed_trades_df, no re-derivation needed for
    # those two (only the entry/exit commission SPLIT is re-derived).

    for _, row in closed_trades_df.iterrows():
        units, entry_prices, entry_commission = _reconstruct_entry_fill(
            trade_row=row,
            weights_lookup=weights_lookup,
            price_matrix=price_matrix,
            position_translator=position_translator,
            execution_engine=execution_engine,
        )
        pos = _ReplayPosition(
            units_by_ticker=units,
            entry_prices_by_ticker=entry_prices,
            direction=float(row["direction"]),
        )
        entry_date = pd.Timestamp(row["entry_date"])
        exit_date = pd.Timestamp(row["exit_date"])
        candidate_id = str(row["candidate_id"])
        realized_pnl_gross = float(row["realized_pnl_gross"])
        total_transaction_costs = float(row["transaction_costs"])
        exit_commission = total_transaction_costs - entry_commission

        opens_by_date.setdefault(entry_date, []).append((candidate_id, pos, entry_commission))
        closes_by_date.setdefault(exit_date, []).append(
            (candidate_id, realized_pnl_gross, exit_commission)
        )

    # Still-open positions at run end: no reconstruction, read directly.
    for candidate_id, live_pos in final_live_positions_by_candidate_id.items():
        pos = _ReplayPosition(
            units_by_ticker=dict(live_pos.units_by_ticker),
            entry_prices_by_ticker=dict(live_pos.entry_prices_by_ticker),
            direction=live_pos.direction,
        )
        opens_by_date.setdefault(
            pd.Timestamp(live_pos.entry_date), []
        ).append((candidate_id, pos, live_pos.accumulated_transaction_cost))

    open_positions: dict[str, _ReplayPosition] = {}
    cumulative_transaction_costs = 0.0
    cumulative_borrow_costs = 0.0

    rows: list[dict] = []
    for date in dates:
        date = pd.Timestamp(date)
        current_prices = price_matrix.loc[date].dropna()

        # Step 1-2: MTM positions already open before today, then accrue
        # today's borrow cost from that post-MTM state.
        for pos in open_positions.values():
            _mark_to_market(pos, current_prices)

        daily_borrow_cost = sum(
            pos.gross_value * rate for pos in open_positions.values() if pos.direction < 0.0
        )
        cumulative_borrow_costs += daily_borrow_cost

        # Step 3: today's closes.
        for candidate_id, realized_pnl_gross, exit_commission in closes_by_date.get(date, []):
            open_positions.pop(candidate_id, None)
            cumulative_transaction_costs += exit_commission
        # cumulative_realized_pnl is folded straight into total_equity_gross
        # below via a running total, not tracked as its own persisted column
        # (performance_basic.py never reads it separately).

        # Step 4: today's opens.
        for candidate_id, pos, entry_commission in opens_by_date.get(date, []):
            open_positions[candidate_id] = pos
            cumulative_transaction_costs += entry_commission

        # Step 5: snapshot.
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions.values())
        n_live = len(open_positions)

        rows.append({
            "date": date,
            "daily_borrow_cost": float(daily_borrow_cost),
            "cumulative_transaction_costs": float(cumulative_transaction_costs),
            "cumulative_borrow_costs": float(cumulative_borrow_costs),
            "n_live_candidate_positions": n_live,
            "total_unrealized_pnl": float(total_unrealized_pnl),
        })

    df = pd.DataFrame(rows)

    # cumulative_realized_pnl: a running sum of closed trades' realized_pnl_gross,
    # credited on the exit date -- computed once, vectorized, rather than inside
    # the loop above (same result: additive, order-independent within a date).
    if not closed_trades_df.empty:
        realized_by_date = (
            closed_trades_df.assign(exit_date=pd.to_datetime(closed_trades_df["exit_date"]))
            .groupby("exit_date")["realized_pnl_gross"].sum()
        )
        df["cumulative_realized_pnl"] = (
            realized_by_date.reindex(df["date"], fill_value=0.0).cumsum().to_numpy()
        )
    else:
        df["cumulative_realized_pnl"] = 0.0

    df["total_capital"] = float(total_capital)
    df["total_equity_gross"] = (
        df["total_capital"] + df["cumulative_realized_pnl"] + df["total_unrealized_pnl"]
    )
    df["total_equity_net"] = (
        df["total_equity_gross"]
        - df["cumulative_transaction_costs"]
        - df["cumulative_borrow_costs"]
    )

    return df[[
        "date", "total_capital", "total_equity_gross", "total_equity_net",
        "n_live_candidate_positions", "daily_borrow_cost",
        "cumulative_transaction_costs", "cumulative_borrow_costs",
    ]]
