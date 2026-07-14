from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ALLOWED_REFERENCE_SCOPE_COLS = {
    "group_id",
    "candidate_type",
    "weight_model",
}


@dataclass(frozen=True)
class CandidateSelectionConfig:
    """
    Configuration for filtering and selecting candidates from a CandidatePanel.

    Parameters
    ----------
    allowed_candidate_subtypes
        Candidate subtypes allowed to pass the selection filter.
        Examples: ("knee", "dense_knee"), ("pair",), ("dense",).

    require_is_valid
        If True, only candidates with is_valid == True are kept.

    require_success
        If True, only candidates with success == True are kept.

    adf_pvalue_max
        Optional upper bound for the ADF p-value. If set, only candidates with
        adf_pvalue <= this threshold are kept.

    mr_retention_min
        Optional lower bound for mr_retention_vs_ref. This is computed relative
        to a reference candidate within each reference scope.

    half_life_mult_max
        Optional upper bound for half_life_mult_vs_ref. This is computed
        relative to a reference candidate within each reference scope.

    reference_scope_cols
        Additional columns defining the reference scope used for relative
        comparisons. The full reference scope is:

            [asof_date] + list(reference_scope_cols)

        Within each scope, one reference candidate is chosen:
        - if a dense or dense_knee portfolio candidate exists, it is used
        - otherwise, the candidate with the highest mr_score is used

        Relative metrics are then computed as:
        - mr_retention_vs_ref = mr_score / ref_mr_score
        - half_life_mult_vs_ref = half_life / ref_half_life

        Allowed values are:
        - "group_id"
        - "candidate_type"
        - "weight_model"
    """
    allowed_candidate_subtypes: tuple[str, ...] = ("knee", "dense_knee")
    require_is_valid: bool = True
    require_success: bool = True
    adf_pvalue_max: float | None = None
    mr_retention_min: float | None = None
    half_life_mult_max: float | None = None
    reference_scope_cols: tuple[str, ...] = ("group_id", "candidate_type", "weight_model")
    excluded_tickers: list[str] | None = None

    def __post_init__(self) -> None:
        if self.adf_pvalue_max is not None:
            if not (0.0 < self.adf_pvalue_max <= 1.0):
                raise ValueError("adf_pvalue_max must be in (0, 1].")

        if self.mr_retention_min is not None:
            if not (0.0 < self.mr_retention_min <= 1.0):
                raise ValueError("mr_retention_min must be in (0, 1].")

        if self.half_life_mult_max is not None:
            if self.half_life_mult_max <= 0.0:
                raise ValueError("half_life_mult_max must be positive.")

        scope = tuple(self.reference_scope_cols)
        if len(scope) != len(set(scope)):
            raise ValueError("reference_scope_cols must not contain duplicates.")

        invalid = [c for c in scope if c not in _ALLOWED_REFERENCE_SCOPE_COLS]
        if invalid:
            raise ValueError(
                f"Unsupported reference_scope_cols: {invalid}. "
                f"Allowed values: {sorted(_ALLOWED_REFERENCE_SCOPE_COLS)}"
            )


@dataclass(frozen=True)
class SelectedCandidatePanelResult:
    panel: pd.DataFrame
    metadata: dict[str, Any]


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable.")


def save_selected_candidate_panel_result(
        result: SelectedCandidatePanelResult,
        out_dir: Path,
        stem: str,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_path = out_dir / f"{stem}.panel.parquet"
    meta_path = out_dir / f"{stem}.meta.json"

    result.panel.to_parquet(panel_path)

    metadata = dict(result.metadata)
    metadata["artifact_type"] = "selected_candidate_panel"
    metadata["artifact_out_dir"] = str(out_dir)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_default, sort_keys=True)


def load_selected_candidate_panel_result(
        out_dir: Path,
        stem: str,
) -> SelectedCandidatePanelResult:
    out_dir = Path(out_dir)

    panel_path = out_dir / f"{stem}.panel.parquet"
    meta_path = out_dir / f"{stem}.meta.json"

    panel = pd.read_parquet(panel_path)

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return SelectedCandidatePanelResult(
        panel=panel,
        metadata=metadata,
    )


def _extract_asof_date_series(panel: pd.DataFrame) -> pd.Series:
    if "asof_date" in panel.columns:
        return pd.to_datetime(panel["asof_date"])

    if isinstance(panel.index, pd.MultiIndex) and "asof_date" in panel.index.names:
        vals = panel.index.get_level_values("asof_date")
        return pd.Series(pd.to_datetime(vals), index=panel.index, name="asof_date")

    raise ValueError("CandidatePanel must contain 'asof_date' as column or index level.")


def _required_cols_for_selection() -> list[str]:
    return [
        "candidate_subtype",
        "is_valid",
        "success",
        "adf_pvalue",
        "mr_score",
        "half_life",
        "candidate_type",
    ]


def _choose_reference_row(group: pd.DataFrame) -> pd.Series:
    """
    Reference rule:
    - if dense/dense_knee portfolio candidate exists, use it
    - otherwise use candidate with highest mr_score
    """
    dense_mask = group["candidate_subtype"].isin(["dense", "dense_knee"])
    dense_rows = group.loc[dense_mask].copy()

    if not dense_rows.empty:
        dense_rows = dense_rows.sort_values(
            ["mr_score", "half_life"],
            ascending=[False, True],
            kind="stable",
        )
        return dense_rows.iloc[0]

    valid_mr = group.loc[group["mr_score"].notna()].copy()
    if valid_mr.empty:
        return group.iloc[0]

    valid_mr = valid_mr.sort_values(
        ["mr_score", "half_life"],
        ascending=[False, True],
        kind="stable",
    )
    return valid_mr.iloc[0]


def _annotate_reference_relative_metrics(
        panel: pd.DataFrame,
        cfg: CandidateSelectionConfig,
) -> pd.DataFrame:
    out = panel.copy()

    asof_date = _extract_asof_date_series(out)
    out["_asof_date_tmp"] = asof_date

    scope_cols = ["_asof_date_tmp", *cfg.reference_scope_cols]

    out["mr_retention_vs_ref"] = np.nan
    out["half_life_mult_vs_ref"] = np.nan
    out["is_reference_candidate"] = False

    for _, idx in out.groupby(scope_cols, sort=True).groups.items():
        grp = out.loc[idx].copy()
        ref = _choose_reference_row(grp)

        ref_mr = ref.get("mr_score", np.nan)
        ref_hl = ref.get("half_life", np.nan)

        if np.isfinite(ref_mr) and ref_mr != 0.0:
            out.loc[grp.index, "mr_retention_vs_ref"] = grp["mr_score"] / float(ref_mr)

        if np.isfinite(ref_hl) and ref_hl != 0.0:
            out.loc[grp.index, "half_life_mult_vs_ref"] = grp["half_life"] / float(ref_hl)

        out.loc[ref.name, "is_reference_candidate"] = True

    out = out.drop(columns=["_asof_date_tmp"])
    return out


def _build_candidate_selection_mask(
        panel: pd.DataFrame,
        cfg: CandidateSelectionConfig,
) -> pd.Series:
    mask = pd.Series(True, index=panel.index)

    if cfg.allowed_candidate_subtypes:
        mask &= panel["candidate_subtype"].isin(cfg.allowed_candidate_subtypes)

    if cfg.require_is_valid:
        mask &= panel["is_valid"].fillna(False)

    if cfg.require_success:
        mask &= panel["success"].fillna(False)

    if cfg.adf_pvalue_max is not None:
        mask &= panel["adf_pvalue"].notna()
        mask &= panel["adf_pvalue"] <= cfg.adf_pvalue_max

    if cfg.mr_retention_min is not None:
        mask &= panel["mr_retention_vs_ref"].notna()
        mask &= panel["mr_retention_vs_ref"] >= cfg.mr_retention_min

    if cfg.half_life_mult_max is not None:
        mask &= panel["half_life_mult_vs_ref"].notna()
        mask &= panel["half_life_mult_vs_ref"] <= cfg.half_life_mult_max

    if cfg.excluded_tickers:
        excluded = set(cfg.excluded_tickers)
        mask &= ~panel["spread_id"].apply(
            lambda sid: bool(excluded & set(sid.split("|")))
        )

    return mask


def select_candidates(
        candidate_panel: pd.DataFrame,
        cfg: CandidateSelectionConfig,
        *,
        source_out_dir: str | None = None,
        source_stem: str | None = None,
) -> SelectedCandidatePanelResult:
    if candidate_panel.empty:
        metadata = {
            "artifact_type": "selected_candidate_panel",
            "selection_cfg": asdict(cfg),
            "source_artifact_type": "candidate_panel",
            "source_out_dir": source_out_dir,
            "source_stem": source_stem,
            "n_input_rows": 0,
            "n_selected_rows": 0,
            "n_input_dates": 0,
            "n_selected_dates": 0,
        }
        return SelectedCandidatePanelResult(
            panel=candidate_panel.copy(),
            metadata=metadata,
        )

    # Check for missing required columns and provide defaults where sensible
    enriched = candidate_panel.copy()
    missing = [c for c in _required_cols_for_selection() if c not in enriched.columns]

    if missing:
        # Attempt to fill missing columns with defaults
        if "success" in missing:
            # Default: treat all candidates as successful if column is missing
            enriched["success"] = True
            missing.remove("success")

        # If other required columns are still missing, raise error
        if missing:
            raise ValueError(f"CandidatePanel missing required columns for selection: {missing}")

    needs_reference = (
            cfg.mr_retention_min is not None
            or cfg.half_life_mult_max is not None )

    if needs_reference:
        enriched = _annotate_reference_relative_metrics(panel=enriched, cfg=cfg)

    mask = _build_candidate_selection_mask(
        panel=enriched,
        cfg=cfg,
    )
    selected = enriched.loc[mask].copy()

    input_dates = _extract_asof_date_series(candidate_panel)
    selected_dates = _extract_asof_date_series(selected) if not selected.empty else pd.Series(dtype="datetime64[ns]")

    metadata = {
        "artifact_type": "selected_candidate_panel",
        "selection_cfg": asdict(cfg),
        "source_artifact_type": "candidate_panel",
        "source_out_dir": source_out_dir,
        "source_stem": source_stem,
        "n_input_rows": int(len(candidate_panel)),
        "n_selected_rows": int(len(selected)),
        "n_input_dates": int(pd.Index(input_dates).nunique()),
        "n_selected_dates": int(pd.Index(selected_dates).nunique()),
    }

    return SelectedCandidatePanelResult(
        panel=selected,
        metadata=metadata,
    )