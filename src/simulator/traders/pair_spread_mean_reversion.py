from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.simulator.actions import CandidateAction, CloseCandidateAction, OpenCandidateAction
from src.simulator.config import PairSpreadTraderConfig
from src.simulator.types import CandidateAnalyticsState, CandidateMarketSnapshot, LiveCandidatePosition


@dataclass(slots=True)
class PairSpreadMeanReversionTrader:
    """
    Pair spread trader for multi-position mean reversion.

    Responsibilities:
    - Decide which spreads to enter (signal quality gates, z-threshold)
    - Decide direction (+1.0 long, -1.0 short)
    - Decide which positions to exit (z-cross, time/exit rules)

    Does NOT decide size — that is SizingEngine's responsibility.
    OpenCandidateAction.pair_notional is left as None; SizingEngine populates it.

    Supports two activation modes:

    Mode A — Independent (config.cross_ts is None, default):
        Each (spread_id, residual_key) decides independently.

    Mode B — Cross-timescale gated (config.cross_ts set):
        Entry only fires when cross-ts conditions are met across timescales.

    Exit is always per-position (own rkey's z-score).
    """

    config: PairSpreadTraderConfig

    def generate_actions(
        self,
        snapshot: CandidateMarketSnapshot,
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        live_diagnostics_by_candidate_id: dict[str, CandidateAnalyticsState] | None = None,
    ) -> list[CandidateAction]:
        actions: list[CandidateAction] = []

        df = snapshot.candidate_states
        if df.empty:
            return actions

        actions.extend(self._generate_closes(df, live_positions_by_candidate_id, live_diagnostics_by_candidate_id))

        if self.config.cross_ts is not None:
            actions.extend(self._generate_opens_cross_ts(df, live_positions_by_candidate_id, live_diagnostics_by_candidate_id))
        else:
            actions.extend(self._generate_opens_independent(df, live_positions_by_candidate_id, live_diagnostics_by_candidate_id))

        return actions

    # ── Closes ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_rhl(residual_key: str) -> int | None:
        for part in residual_key.split("_"):
            if part.startswith("hl") and part[2:].isdigit():
                return int(part[2:])
        return None

    def _resolve_max_hold(self, residual_key: str) -> int | None:
        mhd = self.config.max_holding_days
        if mhd is None:
            return None
        if isinstance(mhd, int):
            return mhd
        if isinstance(mhd, dict):
            return mhd.get(residual_key)
        if isinstance(mhd, str):
            rhl = self._extract_rhl(residual_key)
            if rhl is None:
                return None
            try:
                return round(eval(mhd, {"__builtins__": {}}, {"rhl": rhl}))
            except Exception:
                return None
        return None

    def _resolve_exit_rule(self, residual_key: str) -> str | None:
        rule = self.config.exit_rule
        if rule is None:
            return None
        if isinstance(rule, str):
            return rule
        if isinstance(rule, dict):
            return rule.get(residual_key)
        return None

    def _eval_exit_rule(self, rule: str, pos: LiveCandidatePosition) -> bool:
        rhl = self._extract_rhl(pos.residual_key)
        notional = pos.pair_notional
        if notional <= 0:
            return False

        pnl_pct = pos.unrealized_pnl / notional * 100
        min_pnl_pct = pos.min_unrealized_pnl / notional * 100

        env = {
            "min_pnl_pct": min_pnl_pct,
            "days_open": pos.days_open,
            "rhl": rhl if rhl is not None else 0,
            "pnl_pct": pnl_pct,
        }

        try:
            expr = rule.replace(" AND ", " and ").replace(" OR ", " or ")
            return bool(eval(expr, {"__builtins__": {}}, env))
        except Exception:
            return False

    def _generate_closes(
        self,
        df: "pd.DataFrame",
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        live_diagnostics_by_candidate_id: dict[str, "CandidateAnalyticsState"] | None = None,
    ) -> list[CloseCandidateAction]:
        if not live_positions_by_candidate_id:
            return []

        closes: list[CloseCandidateAction] = []
        already_closed: set[str] = set()

        def _spread_z_components(cid: str) -> tuple:
            if not live_diagnostics_by_candidate_id:
                return ()
            diag = live_diagnostics_by_candidate_id.get(cid)
            if diag is None:
                return ()
            return diag.z_components

        # Exit rule / time stops
        for cid, pos in live_positions_by_candidate_id.items():
            rule = self._resolve_exit_rule(pos.residual_key)
            if rule is not None and self._eval_exit_rule(rule, pos):
                z = float(df.at[cid, "z_score"]) if cid in df.index else None
                closes.append(CloseCandidateAction(
                    candidate_id=pos.candidate_id,
                    group_id=pos.group_id,
                    spread_id=pos.spread_id,
                    reason="exit_rule",
                    z_score=z,
                    z_components=_spread_z_components(cid),
                ))
                already_closed.add(cid)
                continue

            max_hold = self._resolve_max_hold(pos.residual_key)
            if max_hold is not None and pos.days_open >= max_hold:
                z = float(df.at[cid, "z_score"]) if cid in df.index else None
                closes.append(CloseCandidateAction(
                    candidate_id=pos.candidate_id,
                    group_id=pos.group_id,
                    spread_id=pos.spread_id,
                    reason=f"time_stop_{max_hold}d",
                    z_score=z,
                    z_components=_spread_z_components(cid),
                ))
                already_closed.add(cid)

        # Z-cross exits
        for cid, pos in live_positions_by_candidate_id.items():
            if cid in already_closed:
                continue
            if cid not in df.index:
                continue

            row = df.loc[cid]
            if not row.get("is_signal_ready", False):
                continue

            z = row.get("z_score")
            if z is None:
                continue
            z = float(z)

            exit_triggered = (
                (pos.direction > 0 and z >= -self.config.exit_z) or
                (pos.direction < 0 and z <= self.config.exit_z)
            )

            if exit_triggered:
                closes.append(CloseCandidateAction(
                    candidate_id=pos.candidate_id,
                    group_id=pos.group_id,
                    spread_id=pos.spread_id,
                    reason="z_cross",
                    z_score=z,
                    z_components=_spread_z_components(cid),
                ))

        return closes

    # ── Opens: Mode A ────────────────────────────────────────────────────────

    def _generate_opens_independent(
        self,
        df: "pd.DataFrame",
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        live_diagnostics_by_candidate_id: dict[str, "CandidateAnalyticsState"] | None = None,
    ) -> list[OpenCandidateAction]:
        open_spread_keys: set[tuple[str, str]] = {
            (pos.spread_id, pos.residual_key)
            for pos in live_positions_by_candidate_id.values()
        }

        if "residual_key" not in df.columns:
            rkeys = []
            for cid in df.index:
                parts = str(cid).rsplit("__", 1)
                rkeys.append(parts[1] if len(parts) == 2 else "")
            df = df.copy()
            df["residual_key"] = rkeys

        mask = df["is_active"] & df["is_signal_ready"]
        if live_positions_by_candidate_id:
            mask = mask & ~df.index.isin(live_positions_by_candidate_id)

        if open_spread_keys:
            spread_rkey_pairs = list(zip(df["spread_id"], df["residual_key"]))
            occupied = [pair in open_spread_keys for pair in spread_rkey_pairs]
            mask = mask & ~np.array(occupied)

        if not mask.any():
            return []

        return self._threshold_entries(df.loc[mask], live_diagnostics_by_candidate_id)

    # ── Opens: Mode B ────────────────────────────────────────────────────────

    def _generate_opens_cross_ts(
        self,
        df: "pd.DataFrame",
        live_positions_by_candidate_id: dict[str, LiveCandidatePosition],
        live_diagnostics_by_candidate_id: dict[str, "CandidateAnalyticsState"] | None = None,
    ) -> list[OpenCandidateAction]:
        open_spread_keys: set[tuple[str, str]] = {
            (pos.spread_id, pos.residual_key)
            for pos in live_positions_by_candidate_id.values()
        }

        if "residual_key" not in df.columns:
            rkeys = []
            for cid in df.index:
                parts = str(cid).rsplit("__", 1)
                rkeys.append(parts[1] if len(parts) == 2 else "")
            df = df.copy()
            df["residual_key"] = rkeys

        mask = df["is_active"] & df["is_signal_ready"]
        if live_positions_by_candidate_id:
            mask = mask & ~df.index.isin(live_positions_by_candidate_id)
        if open_spread_keys:
            spread_rkey_pairs = list(zip(df["spread_id"], df["residual_key"]))
            occupied = [pair in open_spread_keys for pair in spread_rkey_pairs]
            mask = mask & ~np.array(occupied)

        eligible = df.loc[mask]
        if eligible.empty:
            return []

        qualifying_ids: list[str] = []
        cfg = self.config.cross_ts

        for spread_id, group in eligible.groupby("spread_id"):
            z_scores = group["z_score"].to_numpy(dtype=float)
            abs_z = np.abs(z_scores)

            if cfg.same_sign_required:
                if not (np.all(z_scores > 0) or np.all(z_scores < 0)):
                    continue

            if cfg.min_abs_z_all is not None:
                if np.min(abs_z) < cfg.min_abs_z_all:
                    continue

            if cfg.mean_abs_z_min is not None:
                if np.mean(abs_z) < cfg.mean_abs_z_min:
                    continue

            qualifying_ids.extend(group.index.tolist())

        if not qualifying_ids:
            return []

        return self._threshold_entries(eligible.loc[qualifying_ids], live_diagnostics_by_candidate_id)

    # ── Shared: apply entry z-threshold and emit actions ─────────────────────

    def _threshold_entries(
        self,
        candidates: "pd.DataFrame",
        live_diagnostics_by_candidate_id: dict[str, "CandidateAnalyticsState"] | None = None,
    ) -> list[OpenCandidateAction]:
        """
        Apply entry_z threshold, signal quality gates, and emit OpenCandidateActions.

        pair_notional is left as None — SizingEngine populates it before execution.
        """
        z_scores = candidates["z_score"].to_numpy(dtype=float)
        entry_z = self.config.entry_z

        long_mask = z_scores <= -entry_z if self.config.allow_long else None
        short_mask = z_scores >= entry_z if self.config.allow_short else None

        if long_mask is not None and short_mask is not None:
            entry_mask = long_mask | short_mask
        elif long_mask is not None:
            entry_mask = long_mask
        elif short_mask is not None:
            entry_mask = short_mask
        else:
            return []

        if not entry_mask.any():
            return []

        entry_candidates = candidates[entry_mask]
        entry_z_vals = z_scores[entry_mask]
        spread_ids = entry_candidates["spread_id"].to_numpy()
        group_ids = entry_candidates["group_id"].to_numpy()
        cand_ids = entry_candidates.index.to_numpy()

        has_rkey = "residual_key" in entry_candidates.columns
        rkeys = entry_candidates["residual_key"].to_numpy() if has_rkey else None

        opens: list[OpenCandidateAction] = []

        for i in range(len(entry_candidates)):
            z = float(entry_z_vals[i])
            rkey = str(rkeys[i]) if rkeys is not None else ""
            cid_str = str(cand_ids[i])

            diag = live_diagnostics_by_candidate_id.get(cid_str) if live_diagnostics_by_candidate_id else None
            z_comps = diag.z_components if diag is not None else ()

            if self.config.allow_long and z <= -entry_z:
                opens.append(OpenCandidateAction(
                    candidate_id=cid_str,
                    group_id=str(group_ids[i]),
                    spread_id=str(spread_ids[i]),
                    direction=+1.0,
                    pair_notional=None,    # SizingEngine will populate
                    reason="entry_long",
                    z_score=z,
                    residual_key=rkey,
                    z_components=z_comps,
                ))
            elif self.config.allow_short and z >= entry_z:
                opens.append(OpenCandidateAction(
                    candidate_id=cid_str,
                    group_id=str(group_ids[i]),
                    spread_id=str(spread_ids[i]),
                    direction=-1.0,
                    pair_notional=None,    # SizingEngine will populate
                    reason="entry_short",
                    z_score=z,
                    residual_key=rkey,
                    z_components=z_comps,
                ))

        return opens
