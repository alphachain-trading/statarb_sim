from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.returns import GroupReturnBundle


def _to_daily_rf(rf_annual_pct: pd.Series, index: pd.Index) -> pd.Series:
    """
    Convert annualized yield in % (e.g. 5.2) to daily log return,
    aligned and forward-filled to the target index.
    """
    rf = rf_annual_pct.reindex(index).ffill()
    return np.log(1.0 + rf / 100.0) / 252.0


def _subtract_rf(df: pd.DataFrame, rf_daily: pd.Series) -> pd.DataFrame:
    """Subtract daily RF from every column of df (aligned on index)."""
    return df.subtract(rf_daily.reindex(df.index), axis=0)


@dataclass(frozen=True)
class CausalResidualConfig:
    window_mode: str = "rolling"          # "rolling" | "expanding"
    lookback: int = 252                   # rolling window size (ignored when expanding)
    half_life: int | None = None
    min_history: int | None = None
    remove_residual_pcs: int = 0          # number of leading PCs to remove from residuals (0 = disabled)
    subtract_risk_free: bool = False      # subtract daily RF (^IRX) from all returns before fitting

    def __post_init__(self) -> None:
        if self.window_mode not in ("rolling", "expanding"):
            raise ValueError(
                f"window_mode must be 'rolling' or 'expanding', got {self.window_mode!r}"
            )
        if self.window_mode == "rolling" and self.lookback <= 0:
            raise ValueError("lookback must be positive when window_mode='rolling'.")
        if self.half_life is not None and self.half_life <= 0:
            raise ValueError("half_life must be positive.")
        if self.min_history is not None and self.min_history <= 0:
            raise ValueError("min_history must be positive.")
        if (
            self.window_mode == "rolling"
            and self.min_history is not None
            and self.min_history < self.lookback
        ):
            raise ValueError("min_history must be >= lookback.")
        if self.remove_residual_pcs < 0:
            raise ValueError("remove_residual_pcs must be >= 0.")

    def eff_min_history(self) -> int | None:
        if self.min_history is not None:
            return self.min_history
        if self.window_mode == "rolling":
            return self.lookback
        return None

    @classmethod
    def from_dict(cls, raw: dict) -> CausalResidualConfig:
        """
        Build from a metadata dict (e.g. from meta.json).

        Handles legacy formats where window_mode may be absent.
        """
        window_mode = raw.get("window_mode")
        lookback = raw.get("lookback", 252)
        half_life = raw.get("half_life")
        min_history = raw.get("min_history", 252)
        remove_residual_pcs = raw.get("remove_residual_pcs", 0)
        subtract_risk_free = raw.get("subtract_risk_free", False)

        if window_mode is None:
            window_mode = "rolling" if lookback is not None else "expanding"

        return cls(
            window_mode=window_mode,
            lookback=lookback if lookback is not None else 252,
            half_life=half_life,
            min_history=min_history,
            remove_residual_pcs=remove_residual_pcs,
            subtract_risk_free=subtract_risk_free,
        )

    @property
    def key(self) -> str:
        """
        Canonical short key identifying this residual configuration.

        Used as the matching key between panels and ZScoreConfig,
        and as part of position identity in multi-timescale mode.

        The three residual modes map to distinct key shapes:
            eq_rolling      -> rol_lb21            (rol_lb21_mh40 if min_history != lookback)
            eq_expanding    -> exp_mh252
            decay_expanding -> exp_hl504_mh1008    (+ _rf when subtract_risk_free)

        The decay_expanding shape is byte-identical to the historical format
        (e.g. exp_hl504_mh1008_rf); panels persisted before the ResidualMode
        redesign resolve to the same key.
        """
        rf = "_rf" if self.subtract_risk_free else ""
        if self.window_mode == "rolling":
            # eq_rolling: equal-weight rolling fit window
            mh = f"_mh{self.min_history}" if self.min_history != self.lookback else ""
            return f"rol_lb{self.lookback}{mh}{rf}"
        # expanding
        if self.half_life:
            # decay_expanding
            return f"exp_hl{self.half_life}_mh{self.min_history}{rf}"
        # eq_expanding
        return f"exp_mh{self.min_history}{rf}"


@dataclass(frozen=True)
class FittedCausalResidualModel:
    members: list[str]
    proxy_name: str
    bench_name: str
    B_proxy: np.ndarray   # shape (2, 1)
    B_stock: np.ndarray   # shape (3, n_stocks)
    # Optional PC removal: eigenvectors to project out of residuals
    pc_components: np.ndarray | None = None  # shape (n_pcs, n_stocks) — rows are unit eigenvectors
    subtract_risk_free: bool = False


def _slice_fit_window(
    bundle: GroupReturnBundle,
    date: pd.Timestamp,
    cfg: CausalResidualConfig,
) -> pd.DataFrame:
    aligned = bundle.aligned_returns.copy()
    aligned = aligned.loc[:pd.Timestamp(date)]

    if aligned.empty:
        raise ValueError("No data after slicing by date.")

    min_history = cfg.eff_min_history()
    if min_history is not None and len(aligned) < min_history:
        raise ValueError(
            f"Need at least {min_history} observations up to {date}, got {len(aligned)}."
        )

    if cfg.window_mode == "rolling":
        aligned = aligned.iloc[-cfg.lookback:]

    # expanding: use all data up to date (no truncation)

    if aligned.empty:
        raise ValueError("No data left after applying window.")

    return aligned


def _make_sqrt_w(n: int, half_life: int | None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")

    if half_life is None:
        return np.ones(n, dtype=float)

    age = np.arange(n, dtype=float)[::-1]
    w = 2.0 ** (-age / half_life)
    return np.sqrt(w)


def _wls_multi(
    Y: np.ndarray,
    X: np.ndarray,
    sqrt_w: np.ndarray,
) -> np.ndarray:
    Xw = X * sqrt_w[:, None]
    Yw = Y * sqrt_w[:, None]
    B = np.linalg.lstsq(Xw, Yw, rcond=None)[0]
    return B


def fit_causal_residual_model(
    bundle: GroupReturnBundle,
    date: pd.Timestamp,
    cfg: CausalResidualConfig | None = None,
) -> FittedCausalResidualModel:
    """
    Fit the causal residualization model using only data up to `date`.

    The fit window is determined by:
    - min_history: minimum total history required before fitting is allowed
    - lookback: estimation window length; None means expanding
    - half_life: optional exponential decay within the fit window

    All returns are converted to excess returns (minus daily RF) before fitting.
    """
    if cfg is None:
        cfg = CausalResidualConfig()

    aligned = _slice_fit_window(
        bundle=bundle,
        date=pd.Timestamp(date),
        cfg=cfg,
    )

    # Convert to excess returns before fitting (only if enabled)
    if cfg.subtract_risk_free:
        rf_daily = _to_daily_rf(bundle.risk_free_returns, aligned.index)
        aligned = _subtract_rf(aligned, rf_daily)

    members = list(bundle.member_returns.columns)
    proxy_name = bundle.proxy_returns.name
    bench_name = bundle.benchmark_returns.name

    n = len(aligned)
    sqrt_w = _make_sqrt_w(n=n, half_life=cfg.half_life)

    y_proxy = aligned[proxy_name].to_numpy(dtype=float)
    x_bench = aligned[bench_name].to_numpy(dtype=float)

    X_proxy = np.column_stack([
        np.ones(n, dtype=float),
        x_bench,
    ])
    B_proxy = _wls_multi(
        Y=y_proxy[:, None],
        X=X_proxy,
        sqrt_w=sqrt_w,
    )

    proxy_fit = X_proxy @ B_proxy
    proxy_res = y_proxy - proxy_fit[:, 0]

    X_common = np.column_stack([
        np.ones(n, dtype=float),
        x_bench,
        proxy_res,
    ])
    Y = aligned[members].to_numpy(dtype=float)

    B_stock = _wls_multi(
        Y=Y,
        X=X_common,
        sqrt_w=sqrt_w,
    )

    if B_proxy.shape != (2, 1):
        raise ValueError(f"Unexpected B_proxy shape: {B_proxy.shape}")
    if B_stock.shape != (3, len(members)):
        raise ValueError(
            f"Unexpected B_stock shape: {B_stock.shape}, expected (3, {len(members)})"
        )

    # Estimate leading PCs of residuals for optional removal
    pc_components = None
    if cfg.remove_residual_pcs > 0:
        # Compute residuals on the fit window to estimate PCs
        Y_fit = X_common @ B_stock
        resid_fit = Y - Y_fit

        # Apply same exponential weighting for PC estimation
        resid_weighted = resid_fit * sqrt_w[:, None]

        # Covariance of weighted residuals
        cov = np.cov(resid_weighted, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order; take top k from the end
        n_pcs = min(cfg.remove_residual_pcs, len(members))
        # Each row is a unit eigenvector (principal component direction)
        pc_components = eigenvectors[:, -n_pcs:][:, ::-1].T  # shape (n_pcs, n_stocks)

    return FittedCausalResidualModel(
        members=members,
        proxy_name=proxy_name,
        bench_name=bench_name,
        B_proxy=B_proxy,
        B_stock=B_stock,
        pc_components=pc_components,
        subtract_risk_free=cfg.subtract_risk_free,
    )


def apply_causal_residual_model(
    model: FittedCausalResidualModel,
    aligned_returns: pd.DataFrame,
    rf_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Apply a previously fitted causal residual model to any aligned return slice.

    This is the crucial primitive for proper out-of-sample testing:
    fit on train history once, then apply unchanged to test data.

    Parameters
    ----------
    rf_series
        Raw annualized yield in % (e.g. bundle.risk_free_returns).
        Required when model.subtract_risk_free is True; ignored otherwise.
    """
    aligned = aligned_returns.copy()

    if aligned.empty:
        raise ValueError("aligned_returns is empty.")

    required_cols = [model.bench_name, model.proxy_name, *model.members]
    missing = [c for c in required_cols if c not in aligned.columns]
    if missing:
        raise ValueError(f"aligned_returns is missing required columns: {missing}")

    aligned = aligned[required_cols].copy()

    # Subtract RF only when the model was fitted with subtract_risk_free=True
    if model.subtract_risk_free:
        if rf_series is None:
            raise ValueError(
                "model was fitted with subtract_risk_free=True but rf_series was not provided."
            )
        rf_daily = _to_daily_rf(rf_series, aligned.index)
        aligned = _subtract_rf(aligned, rf_daily)

    n = len(aligned)
    y_proxy = aligned[model.proxy_name].to_numpy(dtype=float)
    x_bench = aligned[model.bench_name].to_numpy(dtype=float)

    X_proxy = np.column_stack([
        np.ones(n, dtype=float),
        x_bench,
    ])
    proxy_fit = X_proxy @ model.B_proxy
    proxy_res = y_proxy - proxy_fit[:, 0]

    X_common = np.column_stack([
        np.ones(n, dtype=float),
        x_bench,
        proxy_res,
    ])
    Y = aligned[model.members].to_numpy(dtype=float)
    Y_fit = X_common @ model.B_stock
    residuals = Y - Y_fit

    # Remove leading PCs estimated during fit (out-of-sample projection)
    if model.pc_components is not None:
        # pc_components shape: (n_pcs, n_stocks)
        # For each PC direction v, remove the projection: resid -= (resid @ v) * v
        # Equivalent to: resid = resid - resid @ V^T @ V  where V = pc_components
        projections = residuals @ model.pc_components.T  # (T, n_pcs)
        residuals = residuals - projections @ model.pc_components  # (T, n_stocks)

    return pd.DataFrame(
        residuals,
        index=aligned.index,
        columns=model.members,
    )


def create_causal_residuals(
    bundle: GroupReturnBundle,
    date: pd.Timestamp | None = None,
    cfg: CausalResidualConfig | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper:
    fit the causal residual model up to `date`,
    then apply it to the same fit window.

    This yields a window-consistent residual-return matrix suitable for:
    - spread diagnostics
    - pair analysis
    - portfolio optimization on one trailing window
    """
    if cfg is None:
        cfg = CausalResidualConfig()
    if date is None:
        raise ValueError("date must be provided for causal residual creation.")

    date = pd.Timestamp(date)

    model = fit_causal_residual_model(
        bundle=bundle,
        date=date,
        cfg=cfg,
    )

    aligned = _slice_fit_window(
        bundle=bundle,
        date=date,
        cfg=cfg,
    )

    return apply_causal_residual_model(
        model=model,
        aligned_returns=aligned,
        rf_series=bundle.risk_free_returns if cfg.subtract_risk_free else None,
    )