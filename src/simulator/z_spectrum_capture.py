from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.residuals.causal_residuals import CausalResidualConfig
from src.simulator.candidate_signals import (
    CandidateSignalGenerator,
    _ewm_mean_std_last,
    _rolling_mean_std_last,
)
from src.simulator.config import SpectrumConfig, ZScoreConfig

_SECTOR_ABBREV = {
    "consumer_discretionary": "dsc",
    "consumer_staples": "stp",
    "energy": "nrg",
    "financials": "fin",
    "health_care": "hlt",
    "industrials": "ind",
    "information_technology": "tec",
    "materials": "mat",
    "real_estate": "rle",
    "utilities": "utl",
}


def _default_zlb_values() -> np.ndarray:
    return np.unique(np.round(np.geomspace(5, 252, 35)).astype(int))


@dataclass
class ZSpectrumCapture:
    """
    Captures z-score spectra at trade entry (and optionally exit) during simulation.

    For each opened trade, computes z-scores across all (rhl × zlb) combinations
    and persists one parquet per (sector, rhl) at simulation end.

    Files: z_spectra_{sector}__{rkey}__{entry|exit}.parquet
    Index: (spread_id, date)
    Columns: zlb values (int)

    Identity guarantee: the trading z_score must equal the weighted sum of spectrum
    cells at (trigger_rhl, trading_zlbs). Warns if divergence exceeds epsilon.
    """

    config: SpectrumConfig
    signal_generator: CandidateSignalGenerator
    # trading residual_key -> ZScoreConfig, used for identity check and method/ddof
    z_score_configs: dict[str, ZScoreConfig]

    # Resolved at __post_init__
    _rhl_to_rkey: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _zlb_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), init=False, repr=False)
    _rows_entry: dict[tuple[str, str], list[dict]] = field(default_factory=dict, init=False, repr=False)
    _rows_exit: dict[tuple[str, str], list[dict]] = field(default_factory=dict, init=False, repr=False)
    _rows_hybrid_entry: dict[str, list[dict]] = field(default_factory=dict, init=False, repr=False)
    _rows_hybrid_exit: dict[str, list[dict]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Build rhl -> residual_key map from existing trading residual configs
        trading_rhl_to_rkey: dict[int, str] = {
            rc.half_life: rkey
            for rkey, rc in self.signal_generator.residual_configs.items()
        }

        # Resolve target rhls
        if self.config.residual_lookbacks is None:
            target_rhls = sorted(trading_rhl_to_rkey.keys())
        else:
            target_rhls = sorted(self.config.residual_lookbacks)

        # Map rhls to residual_keys; register spectrum-only rhls into signal_generator
        for rhl in target_rhls:
            if rhl in trading_rhl_to_rkey:
                self._rhl_to_rkey[rhl] = trading_rhl_to_rkey[rhl]
            else:
                # Spectrum-only rhl: create config and inject into signal_generator
                rc = CausalResidualConfig(
                    window_mode="expanding",
                    half_life=rhl,
                    min_history=rhl * 2,
                    remove_residual_pcs=0,
                )
                rkey = rc.key
                # inject — _get_residuals will use live-fit fallback for these
                self.signal_generator.residual_configs[rkey] = rc
                self._rhl_to_rkey[rhl] = rkey

        # Collect all trading zlb values to guarantee inclusion
        all_trading_zlbs: list[int] = []
        for zc in self.z_score_configs.values():
            all_trading_zlbs.extend(zc.resolved_lookbacks())

        # Resolve zlb sweep values
        if self.config.zlb_values is None:
            base = _default_zlb_values()
        else:
            base = np.array(sorted(set(self.config.zlb_values)), dtype=int)

        self._zlb_values = np.unique(
            np.concatenate([base, np.array(all_trading_zlbs, dtype=int)])
        ).astype(int)

        # Warn about non-persisted rhls (will use slower live fitting)
        precomputed_rkeys: set[str] = {
            rkey for (_, rkey) in self.signal_generator.precomputed_residual_params
        }
        for rhl, rkey in sorted(self._rhl_to_rkey.items()):
            if rkey not in precomputed_rkeys:
                warnings.warn(
                    f"[ZSpectrumCapture] rhl={rhl} (key={rkey!r}) has no precomputed "
                    f"residual params — live fitting will run (slower).",
                    stacklevel=3,
                )

    # ── Public capture interface ─────────────────────────────────────────────

    def capture_entry(
            self,
            *,
            date: pd.Timestamp,
            spread_id: str,
            group_id: str,
            weights_by_ticker: dict[str, float],
            residual_key: str,
            trading_z_score: float | None,
            trading_z_cfg: ZScoreConfig,
    ) -> None:
        self._capture(
            date=date,
            spread_id=spread_id,
            group_id=group_id,
            weights_by_ticker=weights_by_ticker,
            trading_residual_key=residual_key,
            trading_z_score=trading_z_score,
            trading_z_cfg=trading_z_cfg,
            rows_store=self._rows_entry,
            hybrid_store=self._rows_hybrid_entry,
            direction_sign=1.0,
        )

    def capture_exit(
            self,
            *,
            date: pd.Timestamp,
            spread_id: str,
            group_id: str,
            weights_by_ticker: dict[str, float],
            residual_key: str,
    ) -> None:
        if not self.config.record_exit:
            return
        self._capture(
            date=date,
            spread_id=spread_id,
            group_id=group_id,
            weights_by_ticker=weights_by_ticker,
            trading_residual_key=residual_key,
            trading_z_score=None,
            trading_z_cfg=self.z_score_configs.get(residual_key),
            rows_store=self._rows_exit,
            hybrid_store=self._rows_hybrid_exit,
            direction_sign=1.0,
        )

    def save(self, run_dir: str | Path) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self._save_store(self._rows_entry, run_dir, suffix="entry")
        if self.config.record_exit:
            self._save_store(self._rows_exit, run_dir, suffix="exit")
        self._save_hybrid_store(self._rows_hybrid_entry, run_dir, suffix="entry")
        if self.config.record_exit:
            self._save_hybrid_store(self._rows_hybrid_exit, run_dir, suffix="exit")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _capture(
            self,
            *,
            date: pd.Timestamp,
            spread_id: str,
            group_id: str,
            weights_by_ticker: dict[str, float],
            trading_residual_key: str,
            trading_z_score: float | None,
            trading_z_cfg: ZScoreConfig | None,
            rows_store: dict[tuple[str, str], list[dict]],
            hybrid_store: dict[str, list[dict]],
            direction_sign: float = 1.0,
    ) -> None:
        tickers = list(weights_by_ticker.keys())
        w = np.array([weights_by_ticker[t] for t in tickers], dtype=float)
        w = w * direction_sign

        # Collect spread return series per rhl for hybrid computation
        spread_returns_by_rhl: dict[int, np.ndarray] = {}

        for rhl, rkey in self._rhl_to_rkey.items():
            try:
                residuals = self.signal_generator._get_residuals(group_id, rkey, date)
            except Exception:
                continue

            if any(t not in residuals.columns for t in tickers):
                continue

            spread_returns = residuals.loc[:, tickers].to_numpy(dtype=float) @ w   # (T,)
            spread_returns_by_rhl[rhl] = spread_returns
            levels = np.cumsum(spread_returns).reshape(-1, 1)                       # (T, 1)

            if rkey == trading_residual_key and trading_z_cfg is not None:
                method = trading_z_cfg.method
                ddof = trading_z_cfg.ddof
            else:
                method = "ewm"
                ddof = 1

            row: dict = {"spread_id": spread_id, "date": date}
            for zlb in self._zlb_values:
                row[int(zlb)] = self._compute_z(levels, int(zlb), method, ddof)

            rows_store.setdefault((group_id, rkey), []).append(row)

            if (
                rkey == trading_residual_key
                and trading_z_score is not None
                and trading_z_cfg is not None
            ):
                self._identity_check(row, trading_z_score, trading_z_cfg, spread_id, date)

        # ── Hybrid spread computation ─────────────────────────────────────────
        if len(spread_returns_by_rhl) >= 2:
            rhls = sorted(spread_returns_by_rhl.keys())
            # Align on common tail length
            T_common = min(len(spread_returns_by_rhl[rhl]) for rhl in rhls)
            if T_common >= 2:
                # Stack aligned tails: (T_common, n_rhls)
                aligned = np.stack(
                    [spread_returns_by_rhl[rhl][-T_common:] for rhl in rhls],
                    axis=1,
                )  # (T_common, n_rhls)

                # Time-decay weights w_i(tau): tau=0 is most recent (index T-1)
                # tau runs from T_common-1 (oldest) to 0 (newest)
                rhl_arr = np.array(rhls, dtype=float)  # (n_rhls,)
                tau = np.arange(T_common - 1, -1, -1, dtype=float)  # (T_common,) oldest→newest
                # raw weights: (T_common, n_rhls)
                raw_w = np.exp(-tau[:, None] / rhl_arr[None, :])
                norm_w = raw_w / raw_w.sum(axis=1, keepdims=True)

                # Hybrid returns: (T_common,)
                r_hybrid = (aligned * norm_w).sum(axis=1)
                S_hybrid = np.cumsum(r_hybrid).reshape(-1, 1)  # (T_common, 1)

                hybrid_row: dict = {"spread_id": spread_id, "date": date}
                for zlb in self._zlb_values:
                    hybrid_row[int(zlb)] = self._compute_z(S_hybrid, int(zlb), "ewm", 1)

                hybrid_store.setdefault(group_id, []).append(hybrid_row)

    @staticmethod
    def _compute_z(
        levels: np.ndarray,   # (T, 1)
        zlb: int,
        method: str,
        ddof: int,
    ) -> float | None:
        min_p = max(zlb // 2, 2)
        if method == "ewm":
            m, s = _ewm_mean_std_last(levels, halflife=zlb, min_periods=min_p)
        else:
            m, s = _rolling_mean_std_last(levels, window=zlb, min_periods=min_p, ddof=ddof)
        m_val, s_val = float(m[0]), float(s[0])
        level = float(levels[-1, 0])
        if np.isnan(m_val) or np.isnan(s_val) or s_val <= 0.0:
            return None
        return float((level - m_val) / s_val)

    def _identity_check(
        self,
        row: dict,
        trading_z_score: float,
        z_cfg: ZScoreConfig,
        spread_id: str,
        date: pd.Timestamp,
    ) -> None:
        """
        Reconstruct trading z_score from spectrum cells and warn on divergence.

        For blended z_score: sum(w_i * spectrum[trigger_rhl, zlb_i]).
        All trading zlb_i are guaranteed present in zlb_values by __post_init__.
        """
        lookbacks = z_cfg.resolved_lookbacks()
        weights = z_cfg.resolved_weights()

        reconstructed = 0.0
        for lb, w in zip(lookbacks, weights):
            cell = row.get(int(lb))
            if cell is None:
                return  # insufficient history — skip check silently
            reconstructed += w * cell

        diff = abs(reconstructed - trading_z_score)
        if diff > self.config.identity_check_epsilon:
            warnings.warn(
                f"[ZSpectrumCapture] Identity check FAILED: "
                f"spread={spread_id} date={date} "
                f"reconstructed={reconstructed:.6f} trading={trading_z_score:.6f} "
                f"diff={diff:.2e} > epsilon={self.config.identity_check_epsilon:.2e}",
                stacklevel=3,
            )

    @staticmethod
    def _save_store(
        rows_store: dict[tuple[str, str], list[dict]],
        run_dir: Path,
        suffix: str,
    ) -> None:
        for (group_id, rkey), rows in rows_store.items():
            if not rows:
                continue
            df = pd.DataFrame(rows).set_index(["spread_id", "date"])
            df = df[sorted(df.columns, key=int)]
            abbrev = _SECTOR_ABBREV.get(group_id, group_id)
            fname = f"z_spectra_{abbrev}__{rkey}__{suffix}.parquet"
            df.to_parquet(run_dir / fname)
            print(f"[ZSpectrumCapture] Saved {fname} ({len(df)} rows × {len(df.columns)} zlb)")

    @staticmethod
    def _save_hybrid_store(
        hybrid_store: dict[str, list[dict]],
        run_dir: Path,
        suffix: str,
    ) -> None:
        for group_id, rows in hybrid_store.items():
            if not rows:
                continue
            df = pd.DataFrame(rows).set_index(["spread_id", "date"])
            df = df[sorted(df.columns, key=int)]
            abbrev = _SECTOR_ABBREV.get(group_id, group_id)
            fname = f"z_spectra_{abbrev}__hybrid__{suffix}.parquet"
            df.to_parquet(run_dir / fname)
            print(f"[ZSpectrumCapture] Saved {fname} ({len(df)} rows × {len(df.columns)} zlb)")