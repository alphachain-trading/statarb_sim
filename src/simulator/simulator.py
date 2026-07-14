from __future__ import annotations

import dataclasses
import os
import psutil
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

from src.data.universe_marketdata import UniverseMarketData
from src.simulator.actions import CandidateAction, CloseCandidateAction, OpenCandidateAction
from src.simulator.candidate_activation import CandidateActivation
from src.simulator.candidate_filter import CandidateFilter
from src.simulator.candidate_signals import CandidateSignalGenerator
from src.simulator.config import PerformanceConfig, SimulatorConfig
from src.simulator.execution import ExecutionEngine
from src.simulator.position_translator import PositionTranslator
from src.simulator.entry_feature_engine import EntryFeatureEngine
from src.simulator.risk_manager import RiskManager
from src.simulator.sizing_engine import SizingEngine
from src.simulator.traders.protocol import Trader
from src.simulator.types import (
    CandidateAnalyticsState,
    CandidateMarketSnapshot,
    CandidateRef,
    ClosedCandidateTrade,
    ExecutionFill,
    LiveCandidatePosition,
)
from src.simulator.z_spectrum_capture import ZSpectrumCapture
from src.utils.sim_logger import logger


@dataclass(slots=True, frozen=True)
class ActionLogEntry:
    date: pd.Timestamp
    action: CandidateAction


@dataclass(slots=True, frozen=True)
class TickerTradeLogEntry:
    date: pd.Timestamp
    action_kind: str
    group_id: str
    candidate_id: str
    spread_id: str
    ticker: str
    requested_delta_units: float
    filled_delta_units: float
    fill_price: float
    traded_notional: float


@dataclass(slots=True, frozen=True)
class DailyStateLogEntry:
    date: pd.Timestamp
    group_id: str

    active_candidate_id: str | None
    active_spread_id: str | None
    active_z_score: float | None
    active_mr_score: float | None
    active_half_life: float | None
    active_adf_pvalue: float | None

    live_candidate_id: str | None
    live_spread_id: str | None
    live_z_score: float | None

    is_live: bool
    pair_notional: float | None          # per-trade notional at entry (from SizingEngine)
    group_deployed_notional: float       # sum of pair_notional for all live positions in group
    unrealized_pnl: float
    days_open: int | None


@dataclass(slots=True, frozen=True)
class DailyPortfolioStateLogEntry:
    date: pd.Timestamp
    total_capital: float                 # CapitalConfig.total_capital (reference, not deployed)
    cumulative_realized_pnl: float
    total_unrealized_pnl: float
    total_equity_gross: float
    total_equity_net: float
    total_gross_value: float             # sum of |units × price| across all live positions
    deployed_notional_by_group: dict[str, float]  # bookkeeping: sum of pair_notional per group
    n_live_candidate_positions: int
    daily_borrow_cost: float
    cumulative_transaction_costs: float
    cumulative_borrow_costs: float
    residual_pc1_variance: float | None = None


@dataclass(slots=True, frozen=True)
class LiveDiagnosticsLogEntry:
    date: pd.Timestamp
    trade_id: str
    group_id: str
    candidate_id: str
    days_open: int

    entry_mr_score: float | None
    entry_half_life: float | None
    active_mr_score: float | None
    active_half_life: float | None

    fz_z_score: float | None
    fz_mr_score: float | None
    fz_half_life: float | None
    fz_adf_pvalue: float | None

    dy_z_score: float | None
    dy_mr_score: float | None
    dy_half_life: float | None
    dy_adf_pvalue: float | None

    weight_drift_l2: float | None
    weight_drift_cos: float | None


@dataclass(slots=True)
class SimulationResult:
    selected_panel: pd.DataFrame
    action_log: list[ActionLogEntry]
    ticker_trade_log: list[TickerTradeLogEntry]
    daily_state_log: list[DailyStateLogEntry]
    daily_portfolio_state_log: list[DailyPortfolioStateLogEntry]
    diagnostics_log: list[LiveDiagnosticsLogEntry]
    closed_trades: list[ClosedCandidateTrade]
    final_live_positions_by_candidate_id: dict[str, LiveCandidatePosition]
    snapshots_by_date: dict[pd.Timestamp, CandidateMarketSnapshot] = field(default_factory=dict)

    performance: Any | None = None
    run_id: str | None = None

    def action_log_df(self) -> pd.DataFrame:
        rows = []
        for entry in self.action_log:
            action = entry.action
            rows.append(
                {
                    "date": pd.Timestamp(entry.date),
                    "action_kind": action.action_kind,
                    "group_id": getattr(action, "group_id", pd.NA),
                    "candidate_id": getattr(action, "candidate_id", pd.NA),
                    "spread_id": getattr(action, "spread_id", pd.NA),
                    "reason": getattr(action, "reason", pd.NA),
                    "z_score": getattr(action, "z_score", pd.NA),
                    "direction": getattr(action, "direction", pd.NA),
                    "pair_notional": getattr(action, "pair_notional", pd.NA),
                }
            )
        return pd.DataFrame(rows).convert_dtypes()

    def ticker_trade_log_df(self) -> pd.DataFrame:
        rows = [
            {
                "date": pd.Timestamp(e.date),
                "action_kind": e.action_kind,
                "group_id": e.group_id,
                "candidate_id": e.candidate_id,
                "spread_id": e.spread_id,
                "ticker": e.ticker,
                "requested_delta_units": e.requested_delta_units,
                "filled_delta_units": e.filled_delta_units,
                "fill_price": e.fill_price,
                "traded_notional": e.traded_notional,
            }
            for e in self.ticker_trade_log
        ]
        return pd.DataFrame(rows).convert_dtypes()

    def daily_state_df(self) -> pd.DataFrame:
        rows = [asdict(entry) for entry in self.daily_state_log]
        return pd.DataFrame(rows).convert_dtypes()

    def daily_portfolio_state_df(self) -> pd.DataFrame:
        rows = []
        for entry in self.daily_portfolio_state_log:
            d = asdict(entry)
            # Flatten deployed_notional_by_group into columns for parquet compatibility
            by_group = d.pop("deployed_notional_by_group", {})
            for group_id, val in by_group.items():
                d[f"deployed_{group_id}"] = val
            rows.append(d)
        return pd.DataFrame(rows).convert_dtypes()

    def diagnostics_log_df(self) -> pd.DataFrame:
        rows = [asdict(entry) for entry in self.diagnostics_log]
        return pd.DataFrame(rows).convert_dtypes()

    def closed_trades_df(self) -> pd.DataFrame:
        rows = []
        for trade in self.closed_trades:
            d = asdict(trade)
            # Flatten entry_features → ef.<name> columns
            ef = d.pop("entry_features", {}) or {}
            for k, v in ef.items():
                d[f"ef.{k}"] = v
            # Flatten entry_feature_scores → efsc.<name> columns
            efsc = d.pop("entry_feature_scores", {}) or {}
            for k, v in efsc.items():
                d[f"efsc.{k}"] = v
            # entry_size_multiplier → ef_multiplier
            d["ef_multiplier"] = d.pop("entry_size_multiplier", 1.0)
            rows.append(d)
        return pd.DataFrame(rows).convert_dtypes()


@dataclass(slots=True)
class Simulator:
    config: SimulatorConfig
    umd: UniverseMarketData

    candidate_filter: CandidateFilter
    candidate_activation: CandidateActivation
    signal_generator: CandidateSignalGenerator
    trader: Trader
    position_translator: PositionTranslator
    execution_engine: ExecutionEngine
    sizing_engine: SizingEngine
    entry_feature_engine: EntryFeatureEngine | None
    risk_manager: RiskManager | None

    _spectrum_capture: ZSpectrumCapture | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.spectrum is not None:
            self._spectrum_capture = ZSpectrumCapture(
                config=self.config.spectrum,
                signal_generator=self.signal_generator,
                z_score_configs={
                    zc.timescale_label: zc
                    for zc in self.config.resolved_z_score_configs()
                },
            )

    def run(
        self,
        candidate_panel: pd.DataFrame,
    ) -> SimulationResult:
        run_id = None
        run_dir = None

        if self.config.persistence.enabled:
            from src.simulator.simulation_persistence import make_run_id, _resolve_run_dir
            run_id = make_run_id(self.config)
            run_dir = _resolve_run_dir(self.config.persistence, run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.configure(run_dir / "sim.log")

        logger.log(f"[sim] Received panel with {len(candidate_panel)} rows")
        import time as _time; _t = _time.time()
        selected = self.candidate_filter.run(candidate_panel)
        logger.log(f"[sim] Filtered to {len(selected)} rows in {_time.time() - _t:.1f}s")

        if selected.empty:
            return SimulationResult(
                selected_panel=selected,
                action_log=[],
                ticker_trade_log=[],
                daily_state_log=[],
                daily_portfolio_state_log=[],
                diagnostics_log=[],
                closed_trades=[],
                final_live_positions_by_candidate_id={},
                snapshots_by_date={},
            )

        market_dates = self._market_dates()
        dates = self._build_simulation_dates(
            market_dates=market_dates,
            selected_panel=selected,
        )

        logger.log("[run] Extracting price matrix...")
        import time as _time
        _t_pm = _time.time()
        price_matrix = self.umd.price_matrix(field=self.signal_generator.price_field)
        logger.log(f"[run] Price matrix extracted in {_time.time() - _t_pm:.1f}s — shape {price_matrix.shape}")

        logger.log("[run] Starting preflight validation...")
        self._preflight_validate(selected, price_matrix, dates)
        logger.log("[run] Preflight done.")

        live_positions_by_candidate_id: dict[str, LiveCandidatePosition] = {}
        action_log: list[ActionLogEntry] = []
        ticker_trade_log: list[TickerTradeLogEntry] = []
        daily_state_log: list[DailyStateLogEntry] = []
        daily_portfolio_state_log: list[DailyPortfolioStateLogEntry] = []
        diagnostics_log: list[LiveDiagnosticsLogEntry] = []
        closed_trades: list[ClosedCandidateTrade] = []
        all_closed_trades: list[ClosedCandidateTrade] = []  # survives year-end flush

        cumulative_realized_pnl = 0.0
        cumulative_transaction_costs = 0.0
        cumulative_borrow_costs = 0.0

        last_date_by_year: dict[int, pd.Timestamp] = {}
        for _d in dates:
            last_date_by_year[pd.Timestamp(_d).year] = pd.Timestamp(_d)

        if self.config.run.progress:
            from tqdm import tqdm
            date_iter = tqdm(dates, desc="Simulating", smoothing=0.1, unit="step")
        else:
            date_iter = dates

        label_to_meta = {
            zc.timescale_label: (zc.residual_hl, zc.resolved_lookbacks()[0])
            for zc in self.config.resolved_z_score_configs()
            if zc.timescale_label
        }

        for _i, date in enumerate(date_iter, 1):
            date = pd.Timestamp(date)
            if self.config.run.progress and hasattr(date_iter, 'set_postfix'):
                date_iter.set_postfix({"date": str(date.date())}, refresh=False)

            if _i % 100 == 0:
                mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
                logger.log(
                    f"[mem] step={_i} {mb:.0f} MB | "
                    f"closed_trades={len(closed_trades)} | "
                    f"action_log={len(action_log)} | "
                    f"daily_state={len(daily_state_log)} | "
                    f"portfolio_state={len(daily_portfolio_state_log)} | "
                    f"snapshots={len(live_positions_by_candidate_id)}"
                )

            current_prices = price_matrix.loc[date].dropna()

            self._increment_days_open(live_positions_by_candidate_id=live_positions_by_candidate_id)
            self._mark_to_market_live_positions(
                live_positions_by_candidate_id=live_positions_by_candidate_id,
                current_prices=current_prices,
            )

            daily_borrow_cost = self._accrue_borrow_costs(
                live_positions_by_candidate_id=live_positions_by_candidate_id,
            )
            cumulative_borrow_costs += daily_borrow_cost

            selected_today = self.candidate_filter.get_selected_on_date(
                selected_panel=selected,
                date=date,
            )
            new_arrivals = self.candidate_filter.build_candidate_refs(selected_today)

            self.candidate_activation.process_new_arrivals(
                selected_refs=new_arrivals,
                is_flat_by_candidate_id=self._build_is_flat_map(
                    live_positions_by_candidate_id=live_positions_by_candidate_id,
                ),
                open_spread_ids={
                    (pos.spread_id, pos.timescale_label)
                    for pos in live_positions_by_candidate_id.values()
                },
            )

            tracked_refs = self._build_tracked_refs(new_arrivals=new_arrivals)
            tracked_ref_by_id = {ref.candidate_id: ref for ref in tracked_refs}

            diag_freq = self.config.diagnostics.compute_frequency
            has_new_arrivals = len(new_arrivals) > 0
            skip_diagnostics = (
                diag_freq == "off"
                or (diag_freq == "weekly" and not has_new_arrivals)
            )

            activation_frame = self.candidate_activation.build_activation_frame(
                new_arrivals=new_arrivals,
            )

            live_candidate_ids = set(live_positions_by_candidate_id.keys())
            live_spread_keys = {
                (pos.spread_id, pos.timescale_label)
                for pos in live_positions_by_candidate_id.values()
            }
            needed_refs = [
                ref for ref in tracked_refs
                if ref.candidate_id in live_candidate_ids
                or (ref.spread_id, ref.timescale_label) not in live_spread_keys
            ]

            analytics_by_id = self.signal_generator.build_candidate_analytics_states(
                date=date,
                candidate_refs=needed_refs,
                skip_diagnostics=skip_diagnostics,
            )

            for ref in tracked_refs:
                if ref.candidate_id not in analytics_by_id:
                    analytics_by_id[ref.candidate_id] = CandidateAnalyticsState(
                        candidate_id=ref.candidate_id,
                        group_id=ref.group_id,
                        spread_id=ref.spread_id,
                        date=date,
                        z_score=None,
                        level=None,
                        roll_mean=None,
                        roll_std=None,
                        adf_pvalue=None,
                        mr_score=None,
                        kappa=None,
                        half_life=None,
                        is_signal_ready=False,
                        residual_key=ref.residual_key,
                    )

            signal_frame = self.signal_generator.build_signal_frame(
                date=date,
                activation_frame=activation_frame,
                candidate_refs=tracked_refs,
                precomputed_analytics=analytics_by_id,
            )

            if skip_diagnostics:
                fz_analytics_by_id: dict[str, CandidateAnalyticsState] = {}
            else:
                fz_analytics_by_id = self._build_live_fz_analytics(
                    date=date,
                    live_positions_by_candidate_id=live_positions_by_candidate_id,
                )

            snapshot = CandidateMarketSnapshot(
                date=date,
                current_prices=current_prices,
                candidate_states=signal_frame,
            )

            # ── Pipeline: propose → size → approve → execute ──────────────
            proposed_opens = self.trader.generate_actions(
                snapshot=snapshot,
                live_positions_by_candidate_id=live_positions_by_candidate_id,
                live_diagnostics_by_candidate_id=analytics_by_id,
            )

            closes = [a for a in proposed_opens if isinstance(a, CloseCandidateAction)]
            raw_opens = [a for a in proposed_opens if isinstance(a, OpenCandidateAction)]

            # Compute entry features for proposed opens (before sizing)
            if self.entry_feature_engine is not None and raw_opens:
                self.entry_feature_engine.compute(
                    proposed_opens=raw_opens,
                    analytics_by_id=analytics_by_id,
                    candidate_refs_by_id=tracked_ref_by_id,
                    date=date,
                )

            # Execute closes first (risk-reducing, no sizing needed)
            for action in closes:
                _n_before = len(closed_trades)
                realized_pnl_delta, action_txn_cost = self._apply_action(
                    date=date,
                    action=action,
                    tracked_ref_by_id=tracked_ref_by_id,
                    analytics_by_id=analytics_by_id,
                    current_prices=current_prices,
                    live_positions_by_candidate_id=live_positions_by_candidate_id,
                    closed_trades=closed_trades,
                    ticker_trade_log=ticker_trade_log,
                )
                all_closed_trades.extend(closed_trades[_n_before:])
                cumulative_realized_pnl += realized_pnl_delta
                cumulative_transaction_costs += action_txn_cost
                action_log.append(ActionLogEntry(date=date, action=action))

            # Size proposed opens (drops zero-notional trades)
            sized_opens = self.sizing_engine.size(
                proposed_opens=raw_opens,
                analytics_by_id=analytics_by_id,
            )

            # Apply portfolio constraints
            if self.risk_manager is not None:
                approved_sized = self.risk_manager.approve(
                    sized_opens=sized_opens,
                    live_positions=live_positions_by_candidate_id,
                    current_prices=current_prices,
                )
            else:
                approved_sized = sized_opens

            # Execute approved opens
            for action, pair_notional, feature_scores, size_multiplier in approved_sized:
                sized_action = action.with_notional(pair_notional)
                realized_pnl_delta, action_txn_cost = self._apply_action(
                    date=date,
                    action=sized_action,
                    tracked_ref_by_id=tracked_ref_by_id,
                    analytics_by_id=analytics_by_id,
                    current_prices=current_prices,
                    live_positions_by_candidate_id=live_positions_by_candidate_id,
                    closed_trades=closed_trades,
                    ticker_trade_log=ticker_trade_log,
                    entry_feature_scores=feature_scores,
                    entry_size_multiplier=size_multiplier,
                )
                cumulative_realized_pnl += realized_pnl_delta
                cumulative_transaction_costs += action_txn_cost
                action_log.append(ActionLogEntry(date=date, action=sized_action))

            daily_state_log.extend(
                self._build_daily_state_log_entries(
                    date=date,
                    analytics_by_id=analytics_by_id,
                    live_positions_by_candidate_id=live_positions_by_candidate_id,
                )
            )

            diagnostics_log.extend(
                self._build_diagnostics_log_entries(
                    date=date,
                    analytics_by_id=analytics_by_id,
                    fz_analytics_by_id=fz_analytics_by_id,
                    live_positions_by_candidate_id=live_positions_by_candidate_id,
                    current_prices=current_prices,
                )
            )

            daily_portfolio_state_log.append(
                self._build_daily_portfolio_state_log_entry(
                    date=date,
                    live_positions_by_candidate_id=live_positions_by_candidate_id,
                    cumulative_realized_pnl=cumulative_realized_pnl,
                    daily_borrow_cost=daily_borrow_cost,
                    cumulative_transaction_costs=cumulative_transaction_costs,
                    cumulative_borrow_costs=cumulative_borrow_costs,
                )
            )

            if (
                self.config.persistence.enabled
                and run_dir is not None
                and date == last_date_by_year[date.year]
            ):
                from src.simulator.simulation_persistence import save_simulation_run, flush_yearly_logs

                if label_to_meta:
                    enriched = []
                    for trade in closed_trades:
                        meta = label_to_meta.get(trade.timescale_label)
                        if meta:
                            trade = dataclasses.replace(trade, trigger_rhl=meta[0], trigger_zlb=meta[1])
                        enriched.append(trade)
                    closed_trades = enriched
                    n = len(enriched)
                    all_closed_trades[-n:] = enriched

                _ckpt_result = SimulationResult(
                    selected_panel=selected,
                    action_log=action_log,
                    ticker_trade_log=ticker_trade_log,
                    daily_state_log=daily_state_log,
                    daily_portfolio_state_log=daily_portfolio_state_log,
                    diagnostics_log=diagnostics_log,
                    closed_trades=closed_trades,
                    final_live_positions_by_candidate_id=live_positions_by_candidate_id,
                )
                flush_yearly_logs(
                    result=_ckpt_result,
                    year=date.year,
                    run_dir=run_dir,
                    artifacts=set(self.config.persistence.artifacts),
                )
                save_simulation_run(
                    result=_ckpt_result,
                    config=self.config,
                    performance_result=None,
                    run_id=run_id,
                )
                daily_state_log.clear()
                action_log.clear()
                ticker_trade_log.clear()
                diagnostics_log.clear()
                closed_trades.clear()
                logger.log(f"[sim] Year {date.year} checkpoint: flushed logs, memory freed")
                if self._spectrum_capture is not None:
                    self._spectrum_capture.save(run_dir)

        if label_to_meta:
            enriched = []
            for trade in closed_trades:
                meta = label_to_meta.get(trade.timescale_label)
                if meta:
                    trade = dataclasses.replace(trade, trigger_rhl=meta[0], trigger_zlb=meta[1])
                enriched.append(trade)
            closed_trades = enriched
            n = len(enriched)
            if n:
                all_closed_trades[-n:] = enriched

        result = SimulationResult(
            selected_panel=selected,
            action_log=action_log,
            ticker_trade_log=ticker_trade_log,
            daily_state_log=daily_state_log,
            daily_portfolio_state_log=daily_portfolio_state_log,
            diagnostics_log=diagnostics_log,
            closed_trades=all_closed_trades,
            final_live_positions_by_candidate_id=live_positions_by_candidate_id,
        )

        performance_result = None

        if self.config.performance.enabled:
            from src.simulator.performance.performance_report import generate_report

            perf_cfg = self.config.performance
            if run_dir is not None and perf_cfg.report_html:
                run_dir.mkdir(parents=True, exist_ok=True)
                perf_cfg = PerformanceConfig(
                    enabled=perf_cfg.enabled,
                    metrics_table=perf_cfg.metrics_table,
                    report_html=perf_cfg.report_html,
                    report_output_dir=str(run_dir),
                    benchmark_ticker=perf_cfg.benchmark_ticker,
                    annualization_factor=perf_cfg.annualization_factor,
                    per_group_breakdown=perf_cfg.per_group_breakdown,
                )

            performance_result = generate_report(
                closed_trades_df=result.closed_trades_df(),
                daily_portfolio_state_df=result.daily_portfolio_state_df(),
                cfg=perf_cfg,
            )

        if self.config.persistence.enabled:
            from src.simulator.simulation_persistence import save_simulation_run, flush_yearly_logs
            if run_dir is not None and daily_state_log:
                last_date = pd.Timestamp(dates[-1])
                flush_yearly_logs(
                    result=result,
                    year=last_date.year,
                    run_dir=run_dir,
                    artifacts=set(self.config.persistence.artifacts),
                )
            save_simulation_run(
                result=result,
                config=self.config,
                performance_result=performance_result,
                run_id=run_id,
            )
            if self._spectrum_capture is not None:
                if run_dir is not None:
                    self._spectrum_capture.save(run_dir)
                else:
                    import warnings
                    warnings.warn(
                        "[Simulator] spectrum capture configured but persistence disabled "
                        "— spectra not saved.",
                        stacklevel=2,
                    )

        result.performance = performance_result
        result.run_id = run_id if self.config.persistence.enabled else None
        logger.close()
        return result

    def _apply_action(
        self,
        *,
        date: pd.Timestamp,
        action: CandidateAction,
        tracked_ref_by_id: dict[str, CandidateRef],
        analytics_by_id: dict[str, CandidateAnalyticsState],
        current_prices: pd.Series,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        closed_trades: list[ClosedCandidateTrade],
        ticker_trade_log: list[TickerTradeLogEntry],
        entry_feature_scores: dict[str, float] | None = None,
        entry_size_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        """Returns (realized_pnl_delta, action_transaction_cost)."""

        if isinstance(action, OpenCandidateAction):
            ref = tracked_ref_by_id.get(action.candidate_id)
            if ref is None:
                raise ValueError(f"Missing CandidateRef for candidate_id={action.candidate_id!r}")

            ideal_deltas = self.position_translator.translate_open(
                action=action,
                candidate_ref=ref,
                current_prices=current_prices,
            )
            exec_res = self.execution_engine.execute_deltas(
                deltas=ideal_deltas,
                current_prices=current_prices,
            )
            if not exec_res.has_fills:
                return 0.0, 0.0

            ticker_trade_log.extend(
                self._build_ticker_trade_log_entries(
                    date=date,
                    action=action,
                    fills=exec_res.fills,
                )
            )

            units_by_ticker = {f.ticker: int(f.filled_delta_units) for f in exec_res.fills}
            entry_prices_by_ticker = {f.ticker: f.fill_price for f in exec_res.fills}

            realized_weights_by_ticker = self._compute_realized_weights(
                units_by_ticker=units_by_ticker,
                entry_prices_by_ticker=entry_prices_by_ticker,
            )

            # Gross value at entry
            gross_value = sum(
                abs(int(units) * entry_prices_by_ticker[ticker])
                for ticker, units in units_by_ticker.items()
            )

            entry_analytics = self.signal_generator.compute_analytics_from_weights(
                date=date,
                group_id=action.group_id,
                candidate_id=action.candidate_id,
                spread_id=action.spread_id,
                weights_by_ticker=realized_weights_by_ticker,
                residual_key=ref.residual_key,
                timescale_label=ref.timescale_label,
            )

            raw_w = dict(zip(ref.members, ref.weights))
            gross_norm = sum(abs(v) for v in raw_w.values())
            model_weights_by_ticker = {t: v / gross_norm for t, v in raw_w.items()} if gross_norm > 0 else raw_w

            if self._spectrum_capture is not None:
                self._spectrum_capture.capture_entry(
                    date=date,
                    spread_id=action.spread_id,
                    group_id=action.group_id,
                    weights_by_ticker=model_weights_by_ticker,
                    residual_key=ref.residual_key,
                    trading_z_score=action.z_score,
                    trading_z_cfg=self.signal_generator._get_z_score_config(ref.timescale_label),
                )

            open_txn_cost = sum(f.commission for f in exec_res.fills)

            live_positions_by_candidate_id[action.candidate_id] = LiveCandidatePosition(
                trade_id=self._make_trade_id(action.candidate_id, date),
                candidate_id=action.candidate_id,
                group_id=action.group_id,
                spread_id=action.spread_id,
                direction=action.direction,
                pair_notional=action.pair_notional,
                gross_value=gross_value,
                entry_date=pd.Timestamp(date),
                days_open=0,
                entry_z_score=action.z_score,
                unrealized_pnl=0.0,
                units_by_ticker=units_by_ticker,
                entry_prices_by_ticker=entry_prices_by_ticker,
                realized_weights_by_ticker=realized_weights_by_ticker,
                entry_mr_score=entry_analytics.mr_score,
                entry_half_life=entry_analytics.half_life,
                entry_adf_pvalue=entry_analytics.adf_pvalue,
                entry_features=dict(analytics_by_id[action.candidate_id].features) if action.candidate_id in analytics_by_id else {},
                entry_feature_scores=dict(entry_feature_scores) if entry_feature_scores else {},
                entry_size_multiplier=entry_size_multiplier,
                entry_z_components=action.z_components,
                accumulated_transaction_cost=open_txn_cost,
                accumulated_borrow_cost=0.0,
                residual_key=ref.residual_key,
                timescale_label=ref.timescale_label,
            )
            return 0.0, open_txn_cost

        if isinstance(action, CloseCandidateAction):
            live_pos = live_positions_by_candidate_id.get(action.candidate_id)
            if live_pos is None:
                return 0.0, 0.0

            ideal_deltas = self.position_translator.translate_close(
                action=action,
                live_position=live_pos,
            )
            exec_res = self.execution_engine.execute_deltas(
                deltas=ideal_deltas,
                current_prices=current_prices,
            )
            ticker_trade_log.extend(
                self._build_ticker_trade_log_entries(
                    date=date,
                    action=action,
                    fills=exec_res.fills,
                )
            )

            close_txn_cost = sum(f.commission for f in exec_res.fills)
            live_pos.accumulated_transaction_cost += close_txn_cost

            closed_trade = self._build_closed_trade(
                date=date,
                action=action,
                live_position=live_pos,
                current_prices=current_prices,
            )
            closed_trades.append(closed_trade)
            live_positions_by_candidate_id.pop(action.candidate_id, None)

            if self._spectrum_capture is not None:
                self._spectrum_capture.capture_exit(
                    date=date,
                    spread_id=live_pos.spread_id,
                    group_id=live_pos.group_id,
                    weights_by_ticker=live_pos.realized_weights_by_ticker,
                    residual_key=live_pos.residual_key,
                )

            # Feed outcome to SizingEngine Kelly tracker
            self.sizing_engine.record_closed_trade(
                group_id=closed_trade.group_id,
                pnl_net=closed_trade.realized_pnl_net,
            )

            return float(closed_trade.realized_pnl_gross), close_txn_cost

        raise NotImplementedError(f"Unsupported action type: {type(action).__name__}")

    def _build_daily_state_log_entries(
        self,
        *,
        date: pd.Timestamp,
        analytics_by_id: dict[str, CandidateAnalyticsState],
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
    ) -> list[DailyStateLogEntry]:
        rows: list[DailyStateLogEntry] = []
        live_by_group: dict[str, list[LiveCandidatePosition]] = {}
        for pos in live_positions_by_candidate_id.values():
            live_by_group.setdefault(pos.group_id, []).append(pos)

        group_ids = sorted(
            set(self.candidate_activation.active_by_group.keys())
            | set(live_by_group.keys())
        )

        for group_id in group_ids:
            active_ref = self.candidate_activation.get_active_candidate(group_id)
            live_positions = live_by_group.get(group_id, [])
            active_a = None if active_ref is None else analytics_by_id.get(active_ref.candidate_id)
            group_deployed = sum(pos.pair_notional for pos in live_positions)

            if not live_positions:
                rows.append(
                    DailyStateLogEntry(
                        date=pd.Timestamp(date),
                        group_id=group_id,
                        active_candidate_id=None if active_ref is None else active_ref.candidate_id,
                        active_spread_id=None if active_ref is None else active_ref.spread_id,
                        active_z_score=None if active_a is None else active_a.z_score,
                        active_mr_score=None if active_a is None else active_a.mr_score,
                        active_half_life=None if active_a is None else active_a.half_life,
                        active_adf_pvalue=None if active_a is None else active_a.adf_pvalue,
                        live_candidate_id=None,
                        live_spread_id=None,
                        live_z_score=None,
                        is_live=False,
                        pair_notional=None,
                        group_deployed_notional=0.0,
                        unrealized_pnl=0.0,
                        days_open=None,
                    )
                )
            else:
                for live_pos in live_positions:
                    live_a = analytics_by_id.get(live_pos.candidate_id)
                    rows.append(
                        DailyStateLogEntry(
                            date=pd.Timestamp(date),
                            group_id=group_id,
                            active_candidate_id=None if active_ref is None else active_ref.candidate_id,
                            active_spread_id=None if active_ref is None else active_ref.spread_id,
                            active_z_score=None if active_a is None else active_a.z_score,
                            active_mr_score=None if active_a is None else active_a.mr_score,
                            active_half_life=None if active_a is None else active_a.half_life,
                            active_adf_pvalue=None if active_a is None else active_a.adf_pvalue,
                            live_candidate_id=live_pos.candidate_id,
                            live_spread_id=live_pos.spread_id,
                            live_z_score=None if live_a is None else live_a.z_score,
                            is_live=True,
                            pair_notional=live_pos.pair_notional,
                            group_deployed_notional=group_deployed,
                            unrealized_pnl=live_pos.unrealized_pnl,
                            days_open=live_pos.days_open,
                        )
                    )

        return rows

    def _build_live_fz_analytics(
        self,
        *,
        date: pd.Timestamp,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
    ) -> dict[str, CandidateAnalyticsState]:
        out: dict[str, CandidateAnalyticsState] = {}
        for pos in live_positions_by_candidate_id.values():
            if not pos.realized_weights_by_ticker:
                continue
            out[pos.candidate_id] = self.signal_generator.compute_analytics_from_weights(
                date=date,
                group_id=pos.group_id,
                candidate_id=pos.candidate_id,
                spread_id=pos.spread_id,
                weights_by_ticker=pos.realized_weights_by_ticker,
                residual_key=pos.residual_key,
                timescale_label=pos.timescale_label,
            )
        return out

    def _build_diagnostics_log_entries(
        self,
        *,
        date: pd.Timestamp,
        analytics_by_id: dict[str, CandidateAnalyticsState],
        fz_analytics_by_id: dict[str, CandidateAnalyticsState],
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        current_prices: pd.Series,
    ) -> list[LiveDiagnosticsLogEntry]:
        rows: list[LiveDiagnosticsLogEntry] = []

        for pos in live_positions_by_candidate_id.values():
            if not pos.realized_weights_by_ticker:
                continue

            active_ref = self.candidate_activation.get_active_candidate(pos.group_id)
            active_a = None if active_ref is None else analytics_by_id.get(active_ref.candidate_id)

            fz_a = fz_analytics_by_id.get(pos.candidate_id)
            if fz_a is None:
                continue

            effective_weights = self._compute_effective_weights(
                units_by_ticker=pos.units_by_ticker,
                current_prices=current_prices,
            )
            dy_a = self.signal_generator.compute_analytics_from_weights(
                date=date,
                group_id=pos.group_id,
                candidate_id=pos.candidate_id,
                spread_id=pos.spread_id,
                weights_by_ticker=effective_weights,
                residual_key=pos.residual_key,
                timescale_label=pos.timescale_label,
            ) if effective_weights else None

            drift_l2, drift_cos = self._compute_weight_drift(
                entry_weights=pos.realized_weights_by_ticker,
                effective_weights=effective_weights,
            )

            rows.append(
                LiveDiagnosticsLogEntry(
                    date=pd.Timestamp(date),
                    trade_id=pos.trade_id,
                    group_id=pos.group_id,
                    candidate_id=pos.candidate_id,
                    days_open=pos.days_open,
                    entry_mr_score=pos.entry_mr_score,
                    entry_half_life=pos.entry_half_life,
                    active_mr_score=None if active_a is None else active_a.mr_score,
                    active_half_life=None if active_a is None else active_a.half_life,
                    fz_z_score=fz_a.z_score,
                    fz_mr_score=fz_a.mr_score,
                    fz_half_life=fz_a.half_life,
                    fz_adf_pvalue=fz_a.adf_pvalue,
                    dy_z_score=None if dy_a is None else dy_a.z_score,
                    dy_mr_score=None if dy_a is None else dy_a.mr_score,
                    dy_half_life=None if dy_a is None else dy_a.half_life,
                    dy_adf_pvalue=None if dy_a is None else dy_a.adf_pvalue,
                    weight_drift_l2=drift_l2,
                    weight_drift_cos=drift_cos,
                )
            )

        return rows

    def _build_daily_portfolio_state_log_entry(
        self,
        *,
        date: pd.Timestamp,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        cumulative_realized_pnl: float,
        daily_borrow_cost: float,
        cumulative_transaction_costs: float,
        cumulative_borrow_costs: float,
    ) -> DailyPortfolioStateLogEntry:
        total_capital = self.config.capital.total_capital
        total_unrealized_pnl = 0.0
        total_gross_value = 0.0
        deployed_notional_by_group: dict[str, float] = {}

        for pos in live_positions_by_candidate_id.values():
            total_unrealized_pnl += float(pos.unrealized_pnl)
            total_gross_value += float(pos.gross_value)
            deployed_notional_by_group[pos.group_id] = (
                deployed_notional_by_group.get(pos.group_id, 0.0) + pos.pair_notional
            )

        total_equity_gross = total_capital + cumulative_realized_pnl + total_unrealized_pnl
        total_costs = cumulative_transaction_costs + cumulative_borrow_costs
        total_equity_net = total_equity_gross - total_costs

        pc1_var = None
        for gid in set(pos.group_id for pos in live_positions_by_candidate_id.values()):
            for rkey in self.config.unique_residual_keys():
                ratios = self.signal_generator.get_pc_variance_ratios(gid, rkey)
                if ratios is not None and len(ratios) > 0:
                    v = ratios[0]
                    if pc1_var is None or v > pc1_var:
                        pc1_var = v

        return DailyPortfolioStateLogEntry(
            date=pd.Timestamp(date),
            total_capital=float(total_capital),
            cumulative_realized_pnl=float(cumulative_realized_pnl),
            total_unrealized_pnl=float(total_unrealized_pnl),
            total_equity_gross=float(total_equity_gross),
            total_equity_net=float(total_equity_net),
            total_gross_value=float(total_gross_value),
            deployed_notional_by_group=deployed_notional_by_group,
            n_live_candidate_positions=len(live_positions_by_candidate_id),
            daily_borrow_cost=float(daily_borrow_cost),
            cumulative_transaction_costs=float(cumulative_transaction_costs),
            cumulative_borrow_costs=float(cumulative_borrow_costs),
            residual_pc1_variance=pc1_var,
        )

    def _build_closed_trade(
        self,
        *,
        date: pd.Timestamp,
        action: CloseCandidateAction,
        live_position: LiveCandidatePosition,
        current_prices: pd.Series,
    ) -> ClosedCandidateTrade:
        realized_pnl_gross = 0.0
        for ticker, units in live_position.units_by_ticker.items():
            px_exit = float(current_prices.loc[ticker])
            px_entry = float(live_position.entry_prices_by_ticker[ticker])
            realized_pnl_gross += float(units) * (px_exit - px_entry)

        transaction_costs = live_position.accumulated_transaction_cost
        borrow_costs = live_position.accumulated_borrow_cost
        realized_pnl_net = realized_pnl_gross - transaction_costs - borrow_costs

        return ClosedCandidateTrade(
            trade_id=live_position.trade_id,
            candidate_id=live_position.candidate_id,
            group_id=live_position.group_id,
            spread_id=live_position.spread_id,
            entry_date=pd.Timestamp(live_position.entry_date),
            exit_date=pd.Timestamp(date),
            days_open=int(live_position.days_open),
            direction=live_position.direction,
            pair_notional=live_position.pair_notional,
            entry_z_score=live_position.entry_z_score,
            exit_z_score=action.z_score,
            realized_pnl_gross=float(realized_pnl_gross),
            transaction_costs=float(transaction_costs),
            borrow_costs=float(borrow_costs),
            realized_pnl_net=float(realized_pnl_net),
            entry_z_components=live_position.entry_z_components,
            exit_z_components=action.z_components,
            residual_key=live_position.residual_key,
            timescale_label=live_position.timescale_label,
            entry_features=dict(live_position.entry_features),
            entry_feature_scores=dict(live_position.entry_feature_scores),
            entry_size_multiplier=live_position.entry_size_multiplier,
        )

    def _build_ticker_trade_log_entries(
        self,
        *,
        date: pd.Timestamp,
        action: CandidateAction,
        fills: tuple[ExecutionFill, ...],
    ) -> list[TickerTradeLogEntry]:
        return [
            TickerTradeLogEntry(
                date=pd.Timestamp(date),
                action_kind=action.action_kind,
                group_id=action.group_id,
                candidate_id=action.candidate_id,
                spread_id=action.spread_id,
                ticker=f.ticker,
                requested_delta_units=f.requested_delta_units,
                filled_delta_units=f.filled_delta_units,
                fill_price=f.fill_price,
                traded_notional=f.traded_notional,
            )
            for f in fills
        ]

    def _preflight_validate(
        self,
        selected_panel: pd.DataFrame,
        price_matrix: pd.DataFrame,
        dates: list[pd.Timestamp],
    ) -> None:
        import time as _time
        t0 = _time.time()
        logger.log("[preflight] Validating ticker coverage...")

        panel_tickers: set[str] = set()
        if "left_ticker" in selected_panel.columns and "right_ticker" in selected_panel.columns:
            panel_tickers = set(selected_panel["left_ticker"].unique()) | set(selected_panel["right_ticker"].unique())
        elif "members" in selected_panel.columns:
            for members_str in selected_panel["members"].dropna().unique():
                if isinstance(members_str, str):
                    panel_tickers.update(members_str.split("|"))
        elif "spread_id" in selected_panel.columns:
            for sid in selected_panel["spread_id"].unique():
                panel_tickers.update(str(sid).split("|"))

        if not panel_tickers:
            logger.log("[preflight] No tickers found in panel — skipping validation")
            return

        price_tickers = set(price_matrix.columns)
        missing_entirely = panel_tickers - price_tickers
        if missing_entirely:
            raise ValueError(
                f"[preflight] {len(missing_entirely)} tickers in candidate panel "
                f"have NO price data: {sorted(missing_entirely)}"
            )

        tickers_sorted = sorted(panel_tickers)
        price_slice = price_matrix.loc[dates[0]:dates[-1], tickers_sorted]
        coverage = price_slice.notna().mean()

        last_5 = price_matrix.loc[:dates[-1], tickers_sorted].tail(5)
        missing_at_end = [t for t in tickers_sorted if last_5[t].isna().all()]
        if missing_at_end:
            raise ValueError(
                f"[preflight] {len(missing_at_end)} tickers have no price data in "
                f"the last 5 trading days (ending {dates[-1].strftime('%Y-%m-%d')}). "
                f"Re-download universe data. Tickers: {missing_at_end}"
            )

        sparse = coverage[coverage < 0.95]
        if not sparse.empty:
            logger.log(f"[preflight] Warning: {len(sparse)} tickers have >5% missing prices:")
            for ticker, frac in sparse.items():
                logger.log(f"  {ticker}: {frac:.1%} coverage")

        elapsed = _time.time() - t0
        logger.log(f"[preflight] Validated {len(panel_tickers)} tickers across {len(dates)} dates in {elapsed:.1f}s — OK")

    def _market_dates(self) -> pd.DatetimeIndex:
        px = self.umd.price_matrix(field=self.signal_generator.price_field)
        idx = px.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("UMD price index must be a DatetimeIndex.")
        if not idx.is_monotonic_increasing:
            raise ValueError("UMD price index must be sorted ascending.")
        if idx.has_duplicates:
            raise ValueError("UMD price index must not contain duplicates.")
        return pd.DatetimeIndex(idx.unique())

    def _build_simulation_dates(
        self,
        *,
        market_dates: pd.DatetimeIndex,
        selected_panel: pd.DataFrame,
    ) -> pd.DatetimeIndex:
        cp_dates = pd.DatetimeIndex(pd.to_datetime(selected_panel["asof_date"]).unique()).sort_values()
        if len(cp_dates) == 0:
            return pd.DatetimeIndex([])

        start = max(market_dates.min(), cp_dates.min())
        end = min(market_dates.max(), cp_dates.max())

        if self.config.run.start_date is not None:
            start = max(start, self.config.run.start_date)
        if self.config.run.end_date is not None:
            end = min(end, self.config.run.end_date)

        dates = market_dates[(market_dates >= start) & (market_dates <= end)]
        if len(dates) == 0:
            raise ValueError("No simulation dates left after intersecting UMD, candidate panel, and run config.")
        return dates

    def _build_tracked_refs(
        self,
        *,
        new_arrivals: list[CandidateRef],
    ) -> list[CandidateRef]:
        by_id = {
            ref.candidate_id: ref
            for ref in self.candidate_activation.get_active_candidates()
        }
        for ref in new_arrivals:
            by_id[ref.candidate_id] = ref
        return list(by_id.values())

    @staticmethod
    def _build_is_flat_map(
        *,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
    ) -> dict[str, bool]:
        return {candidate_id: False for candidate_id in live_positions_by_candidate_id}

    @staticmethod
    def _increment_days_open(
        *,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
    ) -> None:
        for pos in live_positions_by_candidate_id.values():
            pos.days_open += 1

    def _mark_to_market_live_positions(
        self,
        *,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        current_prices: pd.Series,
    ) -> None:
        for pos in live_positions_by_candidate_id.values():
            gross_value = 0.0
            unrealized_pnl = 0.0
            for ticker, units in pos.units_by_ticker.items():
                px_now = float(current_prices.loc[ticker])
                px_entry = float(pos.entry_prices_by_ticker[ticker])
                gross_value += abs(int(units) * px_now)
                unrealized_pnl += int(units) * (px_now - px_entry)

            pos.gross_value = gross_value
            pos.unrealized_pnl = float(unrealized_pnl)
            pos.min_unrealized_pnl = min(pos.min_unrealized_pnl, pos.unrealized_pnl)

    def _accrue_borrow_costs(
        self,
        *,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
    ) -> float:
        rate_annual_bps = self.config.execution.short_borrow_rate_annual_bps
        if rate_annual_bps <= 0.0:
            return 0.0

        daily_borrow_total = 0.0
        for pos in live_positions_by_candidate_id.values():
            if pos.direction >= 0.0:
                continue
            daily_borrow = pos.gross_value * (rate_annual_bps / 10_000) / 252
            pos.accumulated_borrow_cost += daily_borrow
            daily_borrow_total += daily_borrow

        return daily_borrow_total

    @staticmethod
    def _compute_realized_weights(
        *,
        units_by_ticker: dict[str, int],
        entry_prices_by_ticker: dict[str, float],
    ) -> dict[str, float]:
        notionals = {
            ticker: float(units) * entry_prices_by_ticker[ticker]
            for ticker, units in units_by_ticker.items()
        }
        gross_notional = sum(abs(v) for v in notionals.values())
        if gross_notional <= 0.0:
            return {}
        return {ticker: v / gross_notional for ticker, v in notionals.items()}

    @staticmethod
    def _compute_effective_weights(
        *,
        units_by_ticker: dict[str, int],
        current_prices: pd.Series,
    ) -> dict[str, float]:
        notionals = {}
        for ticker, units in units_by_ticker.items():
            if ticker not in current_prices.index:
                continue
            notionals[ticker] = float(units) * float(current_prices.loc[ticker])

        gross_notional = sum(abs(v) for v in notionals.values())
        if gross_notional <= 0.0:
            return {}
        return {ticker: v / gross_notional for ticker, v in notionals.items()}

    @staticmethod
    def _compute_weight_drift(
        *,
        entry_weights: dict[str, float],
        effective_weights: dict[str, float],
    ) -> tuple[float | None, float | None]:
        if not entry_weights or not effective_weights:
            return None, None

        tickers = sorted(set(entry_weights.keys()) | set(effective_weights.keys()))
        w_entry = np.array([entry_weights.get(t, 0.0) for t in tickers], dtype=float)
        w_eff = np.array([effective_weights.get(t, 0.0) for t in tickers], dtype=float)

        l2 = float(np.linalg.norm(w_entry - w_eff))

        norm_entry = float(np.linalg.norm(w_entry))
        norm_eff = float(np.linalg.norm(w_eff))
        if norm_entry > 0.0 and norm_eff > 0.0:
            cos_sim = float(np.dot(w_entry, w_eff) / (norm_entry * norm_eff))
            cos_distance = 1.0 - cos_sim
        else:
            cos_distance = None

        return l2, cos_distance

    @staticmethod
    def _make_trade_id(candidate_id: str, entry_date: pd.Timestamp) -> str:
        ts = pd.Timestamp(entry_date)
        date_part = f"{ts:%Y%m%d}"
        time_part = f"T{ts:%H%M}" if (ts.hour or ts.minute) else ""
        return f"{date_part}{time_part}:{candidate_id[:8]}"

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        s = max(0, int(round(seconds)))
        d, s = divmod(s, 86400)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        parts: list[str] = []
        if d:
            parts.append(f"{d}d")
        if h:
            parts.append(f"{h}h")
        if m:
            parts.append(f"{m}m")
        if s or not parts:
            parts.append(f"{s}s")
        return " ".join(parts[:3])
