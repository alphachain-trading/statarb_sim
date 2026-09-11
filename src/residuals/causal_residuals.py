from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from src.data.returns import GroupReturnBundle


class ResidualMode(str, Enum):
    """Residual-fitting mode for CausalResidualConfig.

    decay_rolling is deliberately excluded — time-decay + rolling window is
    redundant/ambiguous with decay_expanding.
    """
    EQ_ROLLING = "eq_rolling"          # equal-weight, rolling fit window (lb)
    EQ_EXPANDING = "eq_expanding"      # equal-weight, expanding fit window
    DECAY_EXPANDING = "decay_expanding"  # half-life decay, expanding fit window


class AbsOrMult(str, Enum):
    """How min_lb_dec_exp is interpreted for DECAY_EXPANDING's min_history."""
    ABSOLUTE = "absolute"      # min_history = min_lb_dec_exp directly
    MULTIPLIER = "multiplier"  # min_history = hl * min_lb_dec_exp


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


@dataclass
class CausalResidualConfig:
    """
    Canonical, mode-structured residual-fitting config.

    Scope is step 1 only (residual fitting) — hedge-ratio / MR-diagnostics
    windows are PanelBatchConfig's concern. Persisted, keyed, and
    reconstructed via from_dict. Mode-specific fields and subtract_risk_free
    carry no defaults: each must be stated explicitly, since both are
    material to the computation and to the resulting key.
    """

    mode: ResidualMode
    subtract_risk_free: bool          # no default — material to computation and to key
    remove_residual_pcs: int = 0      # genuine "off" state, keeps its default

    lb: int | None = None                          # EQ_ROLLING only
    hl: int | None = None                          # DECAY_EXPANDING only
    min_lb_type_dec_exp: AbsOrMult | None = None   # DECAY_EXPANDING only
    min_lb_dec_exp: int | None = None              # DECAY_EXPANDING only
    min_lb_eq_exp: int | None = None               # EQ_EXPANDING only

    window_mode: Literal["rolling", "expanding"] = field(init=False)
    min_history: int = field(init=False)

    def __post_init__(self) -> None:
        self.window_mode = "rolling" if self.mode == ResidualMode.EQ_ROLLING else "expanding"

        match self.mode:
            case ResidualMode.EQ_ROLLING:
                if self.lb is None:
                    raise ValueError("EQ_ROLLING requires lb")
                if self.hl is not None or self.min_lb_eq_exp is not None:
                    raise ValueError("EQ_ROLLING: hl / min_lb_eq_exp must be None")
                self.min_history = self.lb
            case ResidualMode.DECAY_EXPANDING:
                if self.hl is None:
                    raise ValueError("DECAY_EXPANDING requires hl")
                if self.min_lb_type_dec_exp is None or self.min_lb_dec_exp is None:
                    raise ValueError("DECAY_EXPANDING requires both min_lb_type_dec_exp and min_lb_dec_exp")
                if self.lb is not None or self.min_lb_eq_exp is not None:
                    raise ValueError("DECAY_EXPANDING: lb / min_lb_eq_exp must be None")
                self.min_history = (
                    self.hl * self.min_lb_dec_exp
                    if self.min_lb_type_dec_exp == AbsOrMult.MULTIPLIER
                    else self.min_lb_dec_exp
                )
            case ResidualMode.EQ_EXPANDING:
                if self.min_lb_eq_exp is None:
                    raise ValueError("EQ_EXPANDING requires min_lb_eq_exp")
                if self.lb is not None or self.hl is not None:
                    raise ValueError("EQ_EXPANDING: lb / hl must be None")
                self.min_history = self.min_lb_eq_exp
            case _:  # pragma: no cover - exhaustive
                raise ValueError(f"Unknown mode: {self.mode!r}")

        if self.remove_residual_pcs < 0:
            raise ValueError("remove_residual_pcs must be >= 0.")

    def eff_min_history(self) -> int | None:
        return self.min_history

    @property
    def half_life(self) -> int | None:
        """Compatibility for fitting code (causal_residuals.py reads cfg.half_life
        unprefixed) — no fitting-code changes required."""
        return self.hl if self.mode == ResidualMode.DECAY_EXPANDING else None

    @property
    def lookback(self) -> int | None:
        """Compatibility for fitting code (causal_residuals.py reads cfg.lookback
        unprefixed) — no fitting-code changes required."""
        return self.lb if self.mode == ResidualMode.EQ_ROLLING else None

    @classmethod
    def from_dict(cls, d: dict) -> CausalResidualConfig:
        """
        Reconstruct from a persisted meta.json.

        Requires mode and subtract_risk_free explicitly present — clean
        break, no parsing of the old flat format.
        """
        return cls(
            mode=ResidualMode(d["mode"]),
            subtract_risk_free=d["subtract_risk_free"],
            remove_residual_pcs=d.get("remove_residual_pcs", 0),
            lb=d.get("lb"),
            hl=d.get("hl"),
            min_lb_type_dec_exp=(
                AbsOrMult(d["min_lb_type_dec_exp"]) if d.get("min_lb_type_dec_exp") else None
            ),
            min_lb_dec_exp=d.get("min_lb_dec_exp"),
            min_lb_eq_exp=d.get("min_lb_eq_exp"),
        )

    _KEY_RE_ROLLING = re.compile(
        r"rol_lb(?P<lb>\d+)(?:_mh(?P<mh>\d+))?(?:_pc(?P<pc>\d+))?(?P<rf>_rf)?$"
    )
    _KEY_RE_DECAY_EXPANDING = re.compile(
        r"exp_hl(?P<hl>\d+)_mh(?P<mh>\d+)(?:_pc(?P<pc>\d+))?(?P<rf>_rf)?$"
    )
    _KEY_RE_EQ_EXPANDING = re.compile(
        r"exp_mh(?P<mh>\d+)(?:_pc(?P<pc>\d+))?(?P<rf>_rf)?$"
    )

    @classmethod
    def from_key(cls, key: str) -> "CausalResidualConfig":
        """
        Parse a residual_key string (as produced by .key) back into a
        CausalResidualConfig.

        The round-trip guarantee is on .key only, not full field equality:
        DECAY_EXPANDING always reconstructs with min_lb_type_dec_exp=ABSOLUTE
        (the key format can't distinguish MULTIPLIER from ABSOLUTE, and
        ABSOLUTE with the resolved min_history is always exactly correct
        either way).
        """
        if key.startswith("rol_lb"):
            m = cls._KEY_RE_ROLLING.fullmatch(key)
            if not m:
                raise ValueError(f"Malformed EQ_ROLLING residual key: {key!r}")
            lb = int(m.group("lb"))
            mh = m.group("mh")
            if mh is not None and int(mh) != lb:
                raise ValueError(
                    f"EQ_ROLLING residual key {key!r} has min_history={mh} != lb={lb}; "
                    "this combination cannot arise from any real CausalResidualConfig "
                    "(min_history is always forced equal to lb)."
                )
            return cls(
                mode=ResidualMode.EQ_ROLLING,
                subtract_risk_free=m.group("rf") is not None,
                remove_residual_pcs=int(m.group("pc")) if m.group("pc") else 0,
                lb=lb,
            )
        elif key.startswith("exp_hl"):
            m = cls._KEY_RE_DECAY_EXPANDING.fullmatch(key)
            if not m:
                raise ValueError(f"Malformed DECAY_EXPANDING residual key: {key!r}")
            return cls(
                mode=ResidualMode.DECAY_EXPANDING,
                subtract_risk_free=m.group("rf") is not None,
                remove_residual_pcs=int(m.group("pc")) if m.group("pc") else 0,
                hl=int(m.group("hl")),
                min_lb_type_dec_exp=AbsOrMult.ABSOLUTE,
                min_lb_dec_exp=int(m.group("mh")),
            )
        elif key.startswith("exp_mh"):
            m = cls._KEY_RE_EQ_EXPANDING.fullmatch(key)
            if not m:
                raise ValueError(f"Malformed EQ_EXPANDING residual key: {key!r}")
            return cls(
                mode=ResidualMode.EQ_EXPANDING,
                subtract_risk_free=m.group("rf") is not None,
                remove_residual_pcs=int(m.group("pc")) if m.group("pc") else 0,
                min_lb_eq_exp=int(m.group("mh")),
            )
        else:
            raise ValueError(f"Unrecognized residual key format: {key!r}")

    @property
    def key(self) -> str:
        """
        Canonical short key identifying this residual configuration.

        Used as the matching key between panels and ZScoreConfig, and as
        part of position identity in multi-timescale mode.

        The three residual modes map to distinct key shapes:
            eq_rolling      -> rol_lb21            (rol_lb21_mh40 if min_history != lb)
            eq_expanding    -> exp_mh252
            decay_expanding -> exp_hl504_mh1008    (+ _rf when subtract_risk_free)

        remove_residual_pcs=0 (the always-used production value) omits the
        _pc fragment entirely, so the decay_expanding key stays byte-identical
        to the historical exp_hl504_mh1008_rf. Any nonzero value now correctly
        produces a distinct key — closes a real collision gap where two
        configs differing only in remove_residual_pcs previously produced
        the same key.
        """
        rf = "_rf" if self.subtract_risk_free else ""
        pc = f"_pc{self.remove_residual_pcs}" if self.remove_residual_pcs else ""
        match self.mode:
            case ResidualMode.EQ_ROLLING:
                mh = f"_mh{self.min_history}" if self.min_history != self.lb else ""
                return f"rol_lb{self.lb}{mh}{pc}{rf}"
            case ResidualMode.EQ_EXPANDING:
                return f"exp_mh{self.min_history}{pc}{rf}"
            case ResidualMode.DECAY_EXPANDING:
                return f"exp_hl{self.hl}_mh{self.min_history}{pc}{rf}"


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


# ── Long-form persistence ──────────────────────────────────────────────────
#
# Replaces the old `{stem}_residual_params.pkl` (a pickled
# dict[fit_date, FittedCausalResidualModel]) with a flat parquet: one row
# per factor loading per ticker per fit_date. Pure storage-format swap --
# fit logic (fit_causal_residual_model / apply_causal_residual_model) is
# untouched.
#
# B_proxy's two rows are the proxy's own loadings on [const, bench]; B_stock's
# three rows are each member's loadings on [const, bench, proxy_resid].
# pc_components (when PC removal is on) contributes one row per (pc, member).
# The proxy ticker is distinguished on read by having no "proxy_resid_beta"
# row -- it is never regressed on its own residual, so this is structural,
# not a stored flag. bench_name never appears as a row subject (it is only
# ever a regressor), so it is carried as a constant column alongside every
# row instead.

_RESIDUAL_PARAMS_FACTOR_CONST = "const"
_RESIDUAL_PARAMS_FACTOR_BENCH = "bench_beta"
_RESIDUAL_PARAMS_FACTOR_PROXY_RESID = "proxy_resid_beta"


def _pc_factor_name(k: int) -> str:
    return f"pc{k}_weight"


def save_residual_params(
    fitted_params: dict[pd.Timestamp, FittedCausalResidualModel],
    path: str,
) -> None:
    """
    Persist a walkforward dict of fitted residual models in long form.

    One row per (fit_date, ticker, factor). Raises if fitted_params is empty
    or if the resulting (fit_date, ticker, factor) key is not unique.
    """
    if not fitted_params:
        raise ValueError("fitted_params is empty; nothing to persist.")

    rows: list[dict] = []
    for fit_date, model in fitted_params.items():
        fit_date = pd.Timestamp(fit_date)
        common = {
            "fit_date": fit_date,
            "bench_name": model.bench_name,
            "subtract_risk_free": bool(model.subtract_risk_free),
        }

        rows.append({**common, "ticker": model.proxy_name,
                     "factor": _RESIDUAL_PARAMS_FACTOR_CONST,
                     "loading": float(model.B_proxy[0, 0])})
        rows.append({**common, "ticker": model.proxy_name,
                     "factor": _RESIDUAL_PARAMS_FACTOR_BENCH,
                     "loading": float(model.B_proxy[1, 0])})

        for j, ticker in enumerate(model.members):
            rows.append({**common, "ticker": ticker,
                         "factor": _RESIDUAL_PARAMS_FACTOR_CONST,
                         "loading": float(model.B_stock[0, j])})
            rows.append({**common, "ticker": ticker,
                         "factor": _RESIDUAL_PARAMS_FACTOR_BENCH,
                         "loading": float(model.B_stock[1, j])})
            rows.append({**common, "ticker": ticker,
                         "factor": _RESIDUAL_PARAMS_FACTOR_PROXY_RESID,
                         "loading": float(model.B_stock[2, j])})
            if model.pc_components is not None:
                for k in range(model.pc_components.shape[0]):
                    rows.append({**common, "ticker": ticker,
                                 "factor": _pc_factor_name(k),
                                 "loading": float(model.pc_components[k, j])})

    df = pd.DataFrame(rows)
    df["fit_date"] = pd.to_datetime(df["fit_date"])
    df["ticker"] = df["ticker"].astype("category")
    df["factor"] = df["factor"].astype("category")
    df["bench_name"] = df["bench_name"].astype("category")
    df["loading"] = df["loading"].astype("float64")
    df["subtract_risk_free"] = df["subtract_risk_free"].astype(bool)

    dupe_mask = df.duplicated(subset=["fit_date", "ticker", "factor"], keep=False)
    if dupe_mask.any():
        dupes = df.loc[dupe_mask, ["fit_date", "ticker", "factor"]].drop_duplicates()
        raise ValueError(
            f"residual_params has non-unique (fit_date, ticker, factor) rows:\n{dupes}"
        )

    df = df.reset_index(drop=True)
    df.to_parquet(path)


def load_residual_params(path: str) -> dict[pd.Timestamp, FittedCausalResidualModel]:
    """Reconstruct the walkforward dict[fit_date, FittedCausalResidualModel] written by save_residual_params."""
    df = pd.read_parquet(path)

    result: dict[pd.Timestamp, FittedCausalResidualModel] = {}
    for fit_date, g in df.groupby("fit_date", sort=True, observed=True):
        bench_name = str(g["bench_name"].iloc[0])
        subtract_risk_free = bool(g["subtract_risk_free"].iloc[0])

        piv = g.pivot(index="ticker", columns="factor", values="loading")
        piv = piv.rename(index=str)
        piv.columns = [str(c) for c in piv.columns]

        if _RESIDUAL_PARAMS_FACTOR_PROXY_RESID not in piv.columns:
            raise ValueError(
                f"residual_params for fit_date={fit_date} has no "
                f"'{_RESIDUAL_PARAMS_FACTOR_PROXY_RESID}' factor at all."
            )

        proxy_mask = piv[_RESIDUAL_PARAMS_FACTOR_PROXY_RESID].isna()
        proxy_names = piv.index[proxy_mask].tolist()
        if len(proxy_names) != 1:
            raise ValueError(
                f"Expected exactly one proxy ticker (no proxy_resid_beta row) for "
                f"fit_date={fit_date}, got {proxy_names}"
            )
        proxy_name = proxy_names[0]
        members = sorted(t for t in piv.index if t != proxy_name)

        B_proxy = np.array([
            [piv.loc[proxy_name, _RESIDUAL_PARAMS_FACTOR_CONST]],
            [piv.loc[proxy_name, _RESIDUAL_PARAMS_FACTOR_BENCH]],
        ], dtype=float)

        B_stock = piv.loc[members, [
            _RESIDUAL_PARAMS_FACTOR_CONST,
            _RESIDUAL_PARAMS_FACTOR_BENCH,
            _RESIDUAL_PARAMS_FACTOR_PROXY_RESID,
        ]].to_numpy(dtype=float).T

        pc_cols = sorted(
            (c for c in piv.columns if c.startswith("pc") and c.endswith("_weight")),
            key=lambda c: int(c[len("pc"):-len("_weight")]),
        )
        pc_components = None
        if pc_cols:
            pc_components = piv.loc[members, pc_cols].to_numpy(dtype=float).T

        result[pd.Timestamp(fit_date)] = FittedCausalResidualModel(
            members=members,
            proxy_name=proxy_name,
            bench_name=bench_name,
            B_proxy=B_proxy,
            B_stock=B_stock,
            pc_components=pc_components,
            subtract_risk_free=subtract_risk_free,
        )

    return result


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
    cfg: CausalResidualConfig,
) -> FittedCausalResidualModel:
    """
    Fit the causal residualization model using only data up to `date`.

    The fit window is determined by:
    - min_history: minimum total history required before fitting is allowed
    - lookback: estimation window length; None means expanding
    - half_life: optional exponential decay within the fit window

    All returns are converted to excess returns (minus daily RF) before fitting.
    """
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