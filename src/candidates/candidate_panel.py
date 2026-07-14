from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover
    adfuller = None


CandidateType = Literal["pair", "portfolio"]


@dataclass(frozen=True)
class CandidatePanelResult:
    panel: pd.DataFrame
    metadata: dict[str, Any]


def make_candidate_hash(
    group_id: str,
    candidate_type: str,
    weight_model: str,
    asof_datetime: pd.Timestamp,
    spread_id: str,
    n_hex: int = 16,
) -> str:
    """
    Deterministic short hash for a candidate instance.

    Hash input:
    group_id | candidate_type | weight_model | full asof_datetime | spread_id
    """
    ts = pd.Timestamp(asof_datetime)
    key = f"{group_id}|{candidate_type}|{weight_model}|{ts.isoformat()}|{spread_id}"

    digest = hashlib.blake2b(
        key.encode("utf-8"),
        digest_size=16,
    ).hexdigest()

    return digest[:n_hex]


def make_candidate_id(
    group_id: str,
    candidate_type: str,
    weight_model: str,
    asof_datetime: pd.Timestamp,
    spread_id: str,
    n_hex: int = 16,
) -> str:
    return make_candidate_hash(
        group_id=group_id,
        candidate_type=candidate_type,
        weight_model=weight_model,
        asof_datetime=asof_datetime,
        spread_id=spread_id,
        n_hex=n_hex,
    )


def make_spread_id_from_weights(
    weights: pd.Series,
    tiny_weight_threshold: float,
) -> str:
    """
    Build canonical spread_id from active tickers in alphabetical order.
    """
    w = weights.copy()
    active = w[w.abs() >= tiny_weight_threshold]

    tickers = sorted(active.index.tolist())
    if len(tickers) < 2:
        raise ValueError("Need at least 2 active legs to build spread_id.")

    return "|".join(tickers)


def serialize_weights_for_spread_id(
    weights: pd.Series,
    spread_id: str,
) -> str:
    """
    Serialize weights in spread_id ticker order.

    Notes
    -----
    - spread_id defines both the active leg set and the canonical ordering.
    - this is used for both processed weights and raw weights.
    """
    tickers = spread_id.split("|")
    w = weights.reindex(tickers)

    missing = [t for t in tickers if t not in weights.index]
    if missing:
        raise ValueError(f"Missing weights for tickers: {missing}")

    return w.to_json(double_precision=12)


def compute_candidate_diagnostics(
    residual_returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    *,
    tiny_weight_threshold: float,
    min_active_legs: int,
    min_return_std: float,
    min_level_std: float,
    min_kappa: float,
    max_half_life: float,
    skip_adf: bool = False,
) -> dict[str, Any]:
    """
    Compute common diagnostics for a candidate from residual returns and weights.
    """
    rr = residual_returns.copy()
    rr = rr.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any").dropna(axis=1, how="any")

    if rr.empty:
        raise ValueError("Residual return window is empty after dropping NaNs.")

    if isinstance(weights, pd.Series):
        w = weights.reindex(rr.columns).fillna(0.0).to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != rr.shape[1]:
            raise ValueError(
                f"Weight length mismatch: len(weights)={len(w)}, n_cols={rr.shape[1]}."
            )

    spread_return = rr.to_numpy(dtype=float) @ w
    spread_level = np.cumsum(spread_return)

    spread_return_std = float(np.std(spread_return, ddof=1))
    level_std = float(np.std(spread_level, ddof=1))
    n_legs = int(np.sum(np.abs(w) >= tiny_weight_threshold))

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

    if n_legs < min_active_legs:
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
        "intercept": float(intercept),
        "slope": float(slope),
    }


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def finalize_candidate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup and sorting for a candidate panel.

    Canonical output:
    - flat DataFrame
    - no ambiguous index/column duplication
    - asof_date and candidate_id remain normal columns
    """
    if panel.empty:
        return panel.copy()

    out = panel.copy()

    # Remove potentially ambiguous index structure from persisted files.
    if isinstance(out.index, pd.MultiIndex):
        overlap = [name for name in out.index.names if name is not None and name in out.columns]
        if overlap:
            out = out.reset_index(level=overlap, drop=True)
        out = out.reset_index(drop=True)
    elif out.index.name is not None and out.index.name in out.columns:
        out = out.reset_index(drop=True)
    elif out.index.name is not None:
        out = out.reset_index(drop=True)

    required_cols = [
        "candidate_id",
        "asof_datetime",
        "asof_date",
        "candidate_type",
        "candidate_subtype",
        "weight_model",
        "spread_id",
        "group_id",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Candidate panel missing required columns: {missing}")

    out["asof_datetime"] = pd.to_datetime(out["asof_datetime"])
    out["asof_date"] = pd.to_datetime(out["asof_date"])

    out = out.sort_values(
        ["asof_date", "candidate_type", "weight_model", "spread_id"],
        kind="stable",
    ).reset_index(drop=True)

    right_cols = ["candidate_id", "asof_datetime", "asof_date"]
    left_cols = [c for c in out.columns if c not in right_cols]
    out = out[left_cols + right_cols]

    return out


def save_candidate_panel_result(
    result: CandidatePanelResult,
    out_dir: Path,
    stem: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = out_dir / f"{stem}.panel.parquet"
    meta_path = out_dir / f"{stem}.meta.json"

    result.panel.to_parquet(panel_path)

    metadata = dict(result.metadata)
    metadata["artifact_type"] = "candidate_panel"
    metadata["artifact_out_dir"] = str(out_dir)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_default, sort_keys=True)


def load_candidate_panel_result(
    out_dir: Path,
    stem: str,
) -> CandidatePanelResult:
    out_dir = Path(out_dir)

    panel_path = out_dir / f"{stem}.panel.parquet"
    meta_path = out_dir / f"{stem}.meta.json"

    panel = pd.read_parquet(panel_path)
    panel = finalize_candidate_panel(panel)

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return CandidatePanelResult(
        panel=panel,
        metadata=metadata,
    )