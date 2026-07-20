from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover
    adfuller = None

from src.candidates.candidate_panel import (
    CandidatePanelResult,
    compute_candidate_diagnostics,
    finalize_candidate_panel,
    make_candidate_id,
    save_candidate_panel_result,
    serialize_weights_for_spread_id,
)
from src.data.returns import GroupReturnBundle
from src.residuals.causal_residuals import (
    CausalResidualConfig,
    FittedCausalResidualModel,
    apply_causal_residual_model,
    fit_causal_residual_model,
)
from src.residuals.spreads import (
    HedgeRatioMethod,
    generate_spread_ids,
    spread_members,
    ols_beta_no_intercept,
)
from src.utils.date_utils import make_ranking_dates

from src.settings import CANDIDATE_PANELS_ROOT


HedgeRatioMethodList = list[HedgeRatioMethod]


@dataclass(frozen=True)
class PairSpreadConfig:
    """
    Configuration for pair spread candidate generation.

    Parameters
    ----------
    hedge_ratio_methods
        Which weight models to run per pair. Each produces a separate
        candidate row.  Default: OLS only.
    tickers
        Optional subset of tickers to consider.  If None, use all
        tickers available in the bundle at each as-of date.
    min_active_legs
        Always 2 for pairs (validated, not configurable).
    min_return_std
        Minimum spread return standard deviation.
    min_level_std
        Minimum spread level standard deviation.
    min_kappa
        Minimum mean-reversion speed.
    max_half_life
        Maximum acceptable half-life in days.
    tiny_weight_threshold
        Below this absolute weight, a leg is considered inactive.
    """

    hedge_ratio_methods: HedgeRatioMethodList = field(
        default_factory=lambda: ["ols"],
    )
    tickers: list[str] | None = None
    min_return_std: float = 1e-8
    min_level_std: float = 1e-8
    min_kappa: float = 1e-6
    max_half_life: float = 126.0
    tiny_weight_threshold: float = 1e-6
    skip_adf: bool = False

    def __post_init__(self) -> None:
        if not self.hedge_ratio_methods:
            raise ValueError("At least one hedge_ratio_method is required.")
        for m in self.hedge_ratio_methods:
            if m not in ("unit", "ols", "pca"):
                raise ValueError(f"Unsupported hedge_ratio_method: {m!r}")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))

    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if hours or days or minutes:
        parts.append(f"{minutes}min")
    if hours + days == 0:
        parts.append(f"{seconds}sec")
    return " ".join(parts)


def _slice_candidate_window(
    bundle: GroupReturnBundle,
    asof_datetime: pd.Timestamp,
    lookback: int | None,
) -> pd.DataFrame:
    aligned = bundle.aligned_returns.loc[:pd.Timestamp(asof_datetime)].copy()

    if lookback is not None:
        aligned = aligned.tail(int(lookback))

    if aligned.empty:
        raise ValueError(f"No aligned returns available up to {asof_datetime}.")

    return aligned


def _pca_spread_weights(
    residual_returns: pd.DataFrame,
    left: str,
    right: str,
) -> pd.Series:
    """
    PCA-based pair weights using the second principal component.

    PC1 = common factor (co-movement), PC2 = spread (divergence).
    The PC2 eigenvector is normalized so the left ticker has weight 1.0,
    with a sign convention ensuring the right ticker's weight is negative
    (i.e. a long-short spread).
    """
    X = residual_returns[[left, right]].to_numpy(dtype=float)

    cov = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns ascending order; PC2 (smallest eigenvalue) is index 0.
    pc2 = eigenvectors[:, 0].copy()

    # Normalize so left weight = 1.0.
    if abs(pc2[0]) < 1e-12:
        raise ValueError(
            f"PCA: degenerate eigenvector for {left}|{right}, "
            f"left component ~0."
        )

    pc2 = pc2 / pc2[0]

    # Sign convention: right weight should be negative (long-short spread).
    if pc2[1] > 0:
        pc2 = -pc2

    return pd.Series({left: float(pc2[0]), right: float(pc2[1])}, dtype=float)


def _compute_pair_weights(
    residual_returns: pd.DataFrame,
    left: str,
    right: str,
    method: HedgeRatioMethod,
) -> pd.Series:
    """
    Compute pair spread weights as a pd.Series indexed by ticker.

    Convention: spread = left - beta * right → weights = {left: 1.0, right: -beta}

    Methods
    -------
    unit : weights = [1.0, -1.0]
    ols  : beta from y=left, x=right OLS without intercept
    pca  : second principal component of 2-asset covariance matrix,
           normalized so left=1.0 and right is negative
    """
    if method == "unit":
        beta = 1.0
    elif method == "ols":
        beta = ols_beta_no_intercept(
            y=residual_returns[left],
            x=residual_returns[right],
        )
    elif method == "pca":
        return _pca_spread_weights(residual_returns, left, right)
    else:
        raise ValueError(f"Unsupported hedge_ratio_method: {method!r}")

    return pd.Series({left: 1.0, right: -beta}, dtype=float)


def _build_pair_candidate_rows_for_date(
    *,
    bundle: GroupReturnBundle,
    asof_datetime: pd.Timestamp,
    residual_cfg: CausalResidualConfig,
    pair_cfg: PairSpreadConfig,
    debug: bool,
    fitted_model: FittedCausalResidualModel | None = None,
) -> list[dict[str, Any]]:
    dt = pd.Timestamp(asof_datetime)

    if fitted_model is None:
        fitted_model = fit_causal_residual_model(
            bundle=bundle,
            date=dt,
            cfg=residual_cfg,
        )

    aligned_window = _slice_candidate_window(
        bundle=bundle,
        asof_datetime=dt,
        lookback=residual_cfg.lookback,
    )

    residual_returns = apply_causal_residual_model(
        model=fitted_model,
        aligned_returns=aligned_window,
        rf_series=bundle.risk_free_returns if fitted_model.subtract_risk_free else None,
    )

    # Optional ticker subset filtering.
    if pair_cfg.tickers is not None:
        available = [t for t in pair_cfg.tickers if t in residual_returns.columns]
        if len(available) < 2:
            return []
        residual_returns = residual_returns[available]

    pair_ids = generate_spread_ids(
        tickers=list(residual_returns.columns),
        n_legs=2,
    )

    # Pre-clean residuals once (avoids repeated copy+dropna per pair)
    rr_clean = residual_returns.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").dropna(axis=1, how="any")
    rr_numpy = rr_clean.to_numpy(dtype=float)
    rr_columns = list(rr_clean.columns)
    col_index = {t: i for i, t in enumerate(rr_columns)}

    rows: list[dict[str, Any]] = []

    for method in pair_cfg.hedge_ratio_methods:
        for sid in pair_ids:
            left, right = spread_members(sid)

            if left not in col_index or right not in col_index:
                continue

            try:
                weights = _compute_pair_weights(
                    residual_returns=rr_clean,
                    left=left,
                    right=right,
                    method=method,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            # Fast path: compute spread directly from pre-cleaned numpy array
            w_left = float(weights[left])
            w_right = float(weights[right])
            spread_return = rr_numpy[:, col_index[left]] * w_left + rr_numpy[:, col_index[right]] * w_right

            diagnostics = _fast_pair_diagnostics(
                spread_return=spread_return,
                w_left=w_left,
                w_right=w_right,
                tiny_weight_threshold=pair_cfg.tiny_weight_threshold,
                min_return_std=pair_cfg.min_return_std,
                min_level_std=pair_cfg.min_level_std,
                min_kappa=pair_cfg.min_kappa,
                max_half_life=pair_cfg.max_half_life,
                skip_adf=pair_cfg.skip_adf,
            )

            candidate_id = make_candidate_id(
                group_id=bundle.group_id,
                candidate_type="pair",
                weight_model=method,
                asof_datetime=dt,
                spread_id=sid,
            )

            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "asof_datetime": pd.Timestamp(dt),
                "asof_date": pd.Timestamp(dt).normalize(),
                "candidate_type": "pair",
                "candidate_subtype": method,
                "weight_model": method,
                "spread_id": sid,
                "group_id": bundle.group_id,
                "n_legs": 2,
                "weights": serialize_weights_for_spread_id(
                    weights=weights,
                    spread_id=sid,
                ),
                "adf_pvalue": diagnostics["adf_pvalue"],
                "mr_score": diagnostics["mr_score"],
                "kappa": diagnostics["kappa"],
                "half_life": diagnostics["half_life"],
                "residual_std": diagnostics["residual_std"],
                "spread_return_std": diagnostics["spread_return_std"],
                "level_std": diagnostics["level_std"],
                "is_valid": bool(diagnostics["is_valid"]),
            }

            if debug:
                row["failure_reason"] = str(diagnostics["failure_reason"])
                row["hedge_beta"] = float(-weights[right])

            rows.append(row)

    return rows


def _fast_pair_diagnostics(
    *,
    spread_return: np.ndarray,
    w_left: float,
    w_right: float,
    tiny_weight_threshold: float,
    min_return_std: float,
    min_level_std: float,
    min_kappa: float,
    max_half_life: float,
    skip_adf: bool,
) -> dict[str, Any]:
    """
    Pair-specific diagnostics operating directly on numpy arrays.

    Avoids the overhead of compute_candidate_diagnostics (DataFrame copy,
    reindex, dropna) since we already have clean spread returns.
    """
    spread_level = np.cumsum(spread_return)

    spread_return_std = float(np.std(spread_return, ddof=1))
    level_std = float(np.std(spread_level, ddof=1))
    n_legs = int(abs(w_left) >= tiny_weight_threshold) + int(abs(w_right) >= tiny_weight_threshold)

    def invalid(reason: str) -> dict[str, Any]:
        return {
            "is_valid": False,
            "failure_reason": reason,
            "adf_pvalue": np.nan,
            "mr_score": np.nan,
            "kappa": np.nan,
            "half_life": np.nan,
            "residual_std": np.nan,
            "spread_return_std": spread_return_std,
            "level_std": level_std,
            "n_legs": n_legs,
        }

    if n_legs < 2:
        return invalid("too_few_active_legs")
    if spread_return_std < min_return_std:
        return invalid("spread_return_std_too_small")
    if level_std < min_level_std:
        return invalid("spread_level_std_too_small")

    x_raw = np.asarray(spread_level, dtype=float)
    x = x_raw - np.mean(x_raw)
    x_lag = x[:-1]
    dx = np.diff(x)

    if len(dx) < 10:
        return invalid(f"dx_length_{len(dx)}_lt_10")

    X = np.column_stack([np.ones_like(x_lag), x_lag])
    beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    intercept, slope = beta

    fitted = X @ beta
    resid = dx - fitted
    residual_std = float(np.std(resid, ddof=1))
    kappa = float(-slope)

    if not np.isfinite(kappa):
        return invalid("kappa_not_finite")
    if not np.isfinite(residual_std):
        return invalid("residual_std_not_finite")
    if residual_std <= 0.0:
        return invalid("residual_std_le_0")
    if kappa < min_kappa:
        return invalid(f"kappa_lt_min:{kappa}")

    half_life = float(np.log(2.0) / kappa)
    if not np.isfinite(half_life):
        return invalid("half_life_not_finite")
    if half_life <= 0.0:
        return invalid(f"half_life_le_0:{half_life}")
    if half_life > max_half_life:
        return invalid(f"half_life_gt_max:{half_life}")

    mr_score = float(kappa / residual_std)

    adf_pvalue = np.nan
    if not skip_adf and adfuller is not None:
        try:
            adf_pvalue = float(adfuller(x_raw, autolag="AIC")[1])
        except Exception:
            adf_pvalue = np.nan

    return {
        "is_valid": True,
        "failure_reason": "",
        "adf_pvalue": adf_pvalue,
        "mr_score": mr_score,
        "kappa": kappa,
        "half_life": half_life,
        "residual_std": residual_std,
        "spread_return_std": spread_return_std,
        "level_std": level_std,
        "n_legs": n_legs,
    }


def create_pair_candidates_for_date(
    bundle: GroupReturnBundle,
    asof_date: str | pd.Timestamp,
    residual_cfg: CausalResidualConfig,
    pair_cfg: PairSpreadConfig,
    *,
    debug: bool = False,
) -> CandidatePanelResult:
    """
    Create pair spread candidates for one exact as-of date.
    """
    dt = pd.Timestamp(asof_date)
    aligned_index = bundle.aligned_returns.index

    if dt not in aligned_index:
        raise ValueError(
            f"asof_date {dt.date()} is not present in bundle.aligned_returns.index."
        )

    min_history = residual_cfg.eff_min_history()
    if min_history is not None:
        valid_datetimes = make_ranking_dates(
            dates=aligned_index,
            frequency="B",
            min_history=min_history,
        )
        if dt not in set(pd.DatetimeIndex(valid_datetimes)):
            raise ValueError(
                f"asof_date {dt.date()} does not satisfy minimum history "
                f"requirement (min_history={min_history})."
            )

    rows = _build_pair_candidate_rows_for_date(
        bundle=bundle,
        asof_datetime=dt,
        residual_cfg=residual_cfg,
        pair_cfg=pair_cfg,
        debug=debug,
    )

    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = finalize_candidate_panel(panel)

    metadata = {
        "artifact_type": "candidate_panel",
        "mode": "single_date",
        "candidate_type": "pair",
        "group_id": bundle.group_id,
        "asof_date": str(dt.normalize().date()),
        "debug": debug,
        "n_rows": int(len(panel)),
        "pair_cfg": asdict(pair_cfg),
        "residual_cfg": asdict(residual_cfg),
    }

    return CandidatePanelResult(panel=panel, metadata=metadata)


def _clear_stem_artifacts(out_dir: Path, stem: str) -> list[Path]:
    """
    Remove a single panel stem's prior artifacts before a fresh write.

    Matches the stage_download convention of deleting stale artifacts to force
    a clean rebuild: a reused stem must not keep its old meta or, worse, a
    residual_params.pkl fitted under a different config.

    Scoped to THIS stem only — sibling panels sharing a batch directory and the
    shared series/ folder are left untouched. (series/ filenames are keyed by
    spread/ticker, not stem, so pruning it safely is a separate, unresolved
    concern and deliberately out of scope here.)

    Returns the paths actually removed.
    """
    removed: list[Path] = []
    for path in (
        out_dir / f"{stem}.panel.parquet",
        out_dir / f"{stem}.meta.json",
        out_dir / f"{stem}_residual_params.pkl",
    ):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def create_pair_candidate_panel(
    bundle: GroupReturnBundle,
    residual_cfg: CausalResidualConfig,
    pair_cfg: PairSpreadConfig,
    frequency: str | None = None,
    *,
    dates: list[str | pd.Timestamp] | None = None,
    debug: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    max_steps: int | None = None,
    progress: bool = True,
    persist_result: bool = False,
    persist_residual_params: bool = False,
    persist_result_dir: str = "",
    persist_result_file_stem: str = "",
) -> CandidatePanelResult:
    """
    Create a pair spread CandidatePanel at the requested as-of dates.

    Same walkforward interface as create_portfolio_candidate_panel.
    """
    aligned_index = bundle.aligned_returns.index

    # ── resolve as-of datetimes ──────────────────────────────────────
    if dates is not None:
        if frequency is not None:
            print("frequency will be ignored because explicit dates were provided")
        if start_date is not None:
            print("start_date will be ignored because explicit dates were provided")
        if end_date is not None:
            print("end_date will be ignored because explicit dates were provided")
        if max_steps is not None:
            print("max_steps will be ignored because explicit dates were provided")

        asof_datetimes = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in dates}))

        if len(asof_datetimes) == 0:
            raise ValueError("dates was provided but no dates remain after parsing.")

        missing_dates = asof_datetimes.difference(aligned_index)
        if len(missing_dates) > 0:
            raise ValueError(
                "Explicit dates contain values not present in bundle.aligned_returns.index: "
                f"{[str(d.date()) for d in missing_dates]}"
            )

        min_history = residual_cfg.eff_min_history()
        if min_history is not None:
            valid_datetimes = make_ranking_dates(
                dates=aligned_index,
                frequency="B",
                min_history=min_history,
            )
            valid_set = set(pd.DatetimeIndex(valid_datetimes))
            too_early = [dt for dt in asof_datetimes if dt not in valid_set]
            if too_early:
                raise ValueError(
                    "Explicit dates do not satisfy minimum history requirement "
                    f"(min_history={min_history}): "
                    f"{[str(pd.Timestamp(d).date()) for d in too_early]}"
                )
    else:
        if frequency is None:
            raise ValueError("frequency must be provided when dates is None.")

        if end_date is not None and max_steps is not None:
            print(f"end_date will be ignored because max_steps={max_steps} was provided")

        asof_datetimes = make_ranking_dates(
            dates=aligned_index,
            frequency=frequency,
            min_history=residual_cfg.eff_min_history(),
        )

        if start_date is not None:
            asof_datetimes = asof_datetimes[asof_datetimes >= pd.Timestamp(start_date)]
        if end_date is not None:
            asof_datetimes = asof_datetimes[asof_datetimes <= pd.Timestamp(end_date)]
        if max_steps is not None:
            asof_datetimes = asof_datetimes[: int(max_steps)]

    if len(asof_datetimes) == 0:
        raise ValueError("No asof datetimes remain after applying filters.")

    # ── resolve daily dates for residual model fitting ──────────────
    # Panel dates (asof_datetimes) are typically W-FRI.
    # Daily dates cover every business day in the same range — the model
    # is fitted daily and persisted for later use by the simulator.
    daily_dates: pd.DatetimeIndex | None = None

    if persist_residual_params:
        min_history = residual_cfg.eff_min_history()
        daily_dates = make_ranking_dates(
            dates=aligned_index,
            frequency="B",
            min_history=min_history,
        )
        if start_date is not None:
            daily_dates = daily_dates[daily_dates >= pd.Timestamp(start_date)]
        # Use the last panel date as upper bound
        daily_dates = daily_dates[daily_dates <= asof_datetimes[-1]]

    # ── walkforward loop ─────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    fitted_params: dict[pd.Timestamp, FittedCausalResidualModel] = {}
    panel_date_set = set(asof_datetimes)
    t0 = time.time()

    walk_dates = daily_dates if daily_dates is not None else asof_datetimes
    n_dates = len(walk_dates)
    n_panel_dates = len(asof_datetimes)
    panel_count = 0

    for i, dt in enumerate(walk_dates, start=1):
        dt = pd.Timestamp(dt)
        is_panel_date = dt in panel_date_set

        if progress and (is_panel_date or i % 50 == 0 or i == 1):
            pct = 100.0 * i / n_dates
            msg = f"step {i}/{n_dates} ({pct:4.1f}%), {dt.strftime('%Y-%m-%d')}"
            if is_panel_date:
                panel_count += 1
                msg += f"  [panel {panel_count}/{n_panel_dates}]"
            if i > 1:
                elapsed = time.time() - t0
                avg_per_step = elapsed / (i - 1)
                remaining_sec = avg_per_step * (n_dates - (i - 1))
                msg += (
                    f" | elapsed: {_format_duration(elapsed)}"
                    f" | remaining: {_format_duration(remaining_sec)}"
                )
            print(msg)

        # Always fit the model (cheap relative to panel creation)
        model = fit_causal_residual_model(
            bundle=bundle,
            date=dt,
            cfg=residual_cfg,
        )

        # Store fitted params for persistence
        if persist_residual_params:
            fitted_params[dt] = model

        # Build candidate rows only on panel dates
        if is_panel_date:
            rows.extend(
                _build_pair_candidate_rows_for_date(
                    bundle=bundle,
                    asof_datetime=dt,
                    residual_cfg=residual_cfg,
                    pair_cfg=pair_cfg,
                    debug=debug,
                    fitted_model=model,
                )
            )

    if progress:
        total = time.time() - t0
        print(f"\nDone. Total time: {_format_duration(total)}")

    # ── assemble panel ───────────────────────────────────────────────
    panel = pd.DataFrame(rows)
    if not panel.empty:
        panel = finalize_candidate_panel(panel)

    methods_str = "+".join(pair_cfg.hedge_ratio_methods)

    metadata = {
        "artifact_type": "candidate_panel",
        "candidate_type": "pair",
        "group_id": bundle.group_id,
        "frequency": frequency,
        "explicit_dates": None if dates is None else [str(d.date()) for d in asof_datetimes],
        "hedge_ratio_methods": list(pair_cfg.hedge_ratio_methods),
        "debug": debug,
        "n_rows": int(len(panel)),
        "n_asof_datetimes": int(len(asof_datetimes)),
        "asof_date_min": (
            None if panel.empty else str(pd.Timestamp(panel["asof_date"].min()).date())
        ),
        "asof_date_max": (
            None if panel.empty else str(pd.Timestamp(panel["asof_date"].max()).date())
        ),
        "pair_cfg": asdict(pair_cfg),
        "residual_cfg": asdict(residual_cfg),
        "residual_config": asdict(residual_cfg),
    }

    result = CandidatePanelResult(panel=panel, metadata=metadata)

    # ── persist ──────────────────────────────────────────────────────
    if persist_result:
        lookback_str = (
            "expanding" if residual_cfg.lookback is None
            else f"lookback{residual_cfg.lookback}"
        )
        halflife_str = (
            "_" if residual_cfg.half_life is None
            else f"_decay{residual_cfg.half_life}d"
        )

        auto_stem = (
            f"{bundle.group_id}_pairs_{methods_str}"
            f"_{frequency if frequency is not None else 'explicit_dates'}"
            f"_{lookback_str}{halflife_str}_"
            + pd.Timestamp.now().strftime("%Y%m%d_%H%M")
        )

        out_dir = Path(
            CANDIDATE_PANELS_ROOT
            if persist_result_dir == ""
            else Path(CANDIDATE_PANELS_ROOT) / persist_result_dir
        )
        stem = auto_stem if persist_result_file_stem == "" else persist_result_file_stem

        result.metadata["artifact_out_dir"] = str(out_dir)

        # Remove this stem's prior artifacts before writing fresh ones, so a
        # reused stem cannot leave a stale meta or a residual_params.pkl from a
        # different config behind. Stem-scoped: sibling panels in a shared batch
        # directory and the shared series/ folder are untouched.
        for removed_path in _clear_stem_artifacts(out_dir, stem):
            print(f"[persist] removed stale artifact before rewrite: {removed_path}")

        save_candidate_panel_result(
            result=result,
            out_dir=out_dir,
            stem=stem,
        )

        # Persist daily fitted residual model parameters
        if fitted_params:
            import pickle

            params_path = out_dir / f"{stem}_residual_params.pkl"
            with open(params_path, "wb") as f:
                pickle.dump(fitted_params, f, protocol=pickle.HIGHEST_PROTOCOL)

            result.metadata["residual_params_path"] = str(params_path)
            result.metadata["residual_params_n_dates"] = len(fitted_params)

            if progress:
                print(f"[persist] Residual params saved: {params_path} ({len(fitted_params)} dates)")

    return result