from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd

from src.simulator.types import CandidateRef
from src.simulator.config import ZScoreConfig
from src.candidates.candidate_selector import (
    CandidateSelectionConfig,
    select_candidates,
)


REQUIRED_SELECTED_COLS = {
    "candidate_id",
    "spread_id",
    "group_id",
    "asof_date",
    "weights",
}


@dataclass(slots=True)
class CandidateFilter:
    config: CandidateSelectionConfig
    # Map residual_key → list[ZScoreConfig]. Used to resolve timescale_label at ref construction.
    # For single-timescale runs, leave as empty dict.
    z_score_configs_by_rkey: dict[str, list[ZScoreConfig]] = None

    def __post_init__(self) -> None:
        # slots=True dataclass: use field default via dataclass field instead
        if self.z_score_configs_by_rkey is None:
            # Can't assign directly in frozen; use mutable default workaround
            self.z_score_configs_by_rkey = {}

    def run(self, candidate_panel: pd.DataFrame) -> pd.DataFrame:
        result = select_candidates(
            candidate_panel=candidate_panel,
            cfg=self.config,
        )
        return self._normalize_selected_panel(result.panel)

    def get_selected_on_date(
        self,
        selected_panel: pd.DataFrame,
        date: pd.Timestamp,
    ) -> pd.DataFrame:
        date = pd.Timestamp(date)
        mask = selected_panel["asof_date"] == date
        return selected_panel.loc[mask].copy()

    def build_candidate_refs(self, selected_today: pd.DataFrame) -> list[CandidateRef]:
        refs = []
        for _, row in selected_today.iterrows():
            import json
            weights_raw = row["weights"]
            if isinstance(weights_raw, str):
                w_dict = json.loads(weights_raw)  # {"BKNG": 1.0, "MAR": -3.452695568989}
                members = tuple(w_dict.keys())
                weights = tuple(w_dict.values())
            else:
                members = tuple(row["spread_id"].split("|"))
                weights = tuple(weights_raw)
            rkey = row.get("residual_key", "")
            zcs = self.z_score_configs_by_rkey.get(rkey, []) if rkey else []

            if len(zcs) <= 1:
                # Single-timescale or single zlb per rkey: one ref per row.
                tsl = zcs[0].timescale_label if zcs else ""
                refs.append(CandidateRef(
                    candidate_id=row["candidate_id"],
                    spread_id=row["spread_id"],
                    group_id=row["group_id"],
                    asof_date=pd.Timestamp(row["asof_date"]),
                    members=members,
                    weights=weights,
                    subtype=row.get("subtype"),
                    residual_key=rkey,
                    timescale_label=tsl,
                ))
            else:
                # Multiple zlbs share this residual_key (sweep mode).
                # Fan out: one ref per ZScoreConfig. candidate_id and spread_id
                # remain clean — position uniqueness is carried by timescale_label.
                for zc in zcs:
                    tsl = zc.timescale_label
                    refs.append(CandidateRef(
                        candidate_id=row["candidate_id"],
                        spread_id=row["spread_id"],
                        group_id=row["group_id"],
                        asof_date=pd.Timestamp(row["asof_date"]),
                        members=members,
                        weights=weights,
                        subtype=row.get("subtype"),
                        residual_key=rkey,
                        timescale_label=tsl,
                    ))
        return refs

    def _normalize_selected_panel(self, selected: pd.DataFrame) -> pd.DataFrame:
        df = selected.copy()

        if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
            df = df.reset_index(drop=True)

        missing = REQUIRED_SELECTED_COLS.difference(df.columns)
        if missing:
            raise ValueError(
                f"Selected candidate panel is missing required columns: {sorted(missing)}"
            )

        df["asof_date"] = pd.to_datetime(df["asof_date"])

        if "subtype" not in df.columns:
            df["subtype"] = df["candidate_subtype"] if "candidate_subtype" in df.columns else None

        return df.sort_values(
            ["asof_date", "group_id", "candidate_id"],
            kind="stable",
        ).reset_index(drop=True)

    def _row_to_candidate_ref(self, row: pd.Series) -> CandidateRef:
        members = tuple(str(x) for x in str(row["spread_id"]).split("|"))
        weights = self._extract_weights(row["weights"], members)

        return CandidateRef(
            candidate_id=str(row["candidate_id"]),
            spread_id=str(row["spread_id"]),
            group_id=str(row["group_id"]),
            asof_date=pd.Timestamp(row["asof_date"]),
            members=members,
            weights=weights,
            subtype=self._optional_str(row, "subtype"),
        )

    def _extract_weights(
        self,
        raw_weights: str | dict | pd.Series,
        members: tuple[str, ...],
    ) -> tuple[float, ...]:
        if isinstance(raw_weights, pd.Series):
            s = raw_weights.reindex(list(members))
            return tuple(float(x) for x in s.to_list())

        if isinstance(raw_weights, dict):
            return tuple(float(raw_weights[m]) for m in members)

        if isinstance(raw_weights, str):
            parsed = json.loads(raw_weights)
            return tuple(float(parsed[m]) for m in members)

        raise TypeError(f"Unsupported weights type: {type(raw_weights)}")

    @staticmethod
    def _optional_str(row: pd.Series, col: str) -> str | None:
        if col not in row or pd.isna(row[col]):
            return None
        return str(row[col])