from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


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

    asof_date is the date this candidate's weights (and residual model) were
    fitted — an outer refit date, not the daily residual-fit grid. Distinct
    from a residual model's own fit_date (see src.residuals.series), which
    the two used to share under one column name.
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