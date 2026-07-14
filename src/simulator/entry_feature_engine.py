"""
entry_feature_engine.py — computes entry features for proposed open trades.

HARD RULE: All feature computation functions must live in spread_primitives.py
and be imported from there. No feature logic in this module.

Pipeline position (Step 1):
    Trader → proposed opens → EntryFeatureEngine → SizingEngine

EntryFeatureEngine receives proposed OpenCandidateActions, reconstructs the
spread level series for each candidate from cached residuals in the signal
generator, computes configured features, and writes results into the
CandidateAnalyticsState.features dict.

The features dict is then available to SizingEngine for multiplier computation,
and is snapshotted into LiveCandidatePosition.entry_features / 
ClosedCandidateTrade.entry_features at trade entry.

Direction correction: all primitive functions in spread_primitives.py return
raw/unsigned values. Direction correction (multiply by sign(entry_z_score))
is applied here, once, centrally, via spread_primitives.SIGNED_FEATURE_NAMES
+ direction_correct() — the same mechanism used by the offline FM
(spread_metrics_feature_matrix.py). Do not reintroduce per-feature sign
params or per-call-site sign multiplication.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field as dc_field
from typing import Any

import numpy as np
import pandas as pd

from src.simulator.actions import OpenCandidateAction
from src.simulator.config import EntryFeatureConfig, FeatureSpec
from src.simulator.types import CandidateAnalyticsState, CandidateRef


# Registry of available feature functions — all must live in spread_primitives.
# Key: fn name as used in FeatureSpec.fn
# Value: imported callable (populated lazily on first use)
_FN_REGISTRY: dict[str, Any] = {}

_PRIMITIVES_MODULE = "src.analytics.spread_primitives"

# Registry of feature functions — all must live in spread_primitives.
#   key   = feature name (= FeatureSpec.feature = FM column name in
#           spread_metrics_feature_matrix.py = key checked against
#           SIGNED_FEATURE_NAMES for direction correction = key written into
#           CandidateAnalyticsState.features = ef.*/efsc.* column suffix in
#           RunLoader.closed_trades — ONE name, used identically everywhere)
#   value = primitive function name in spread_primitives.py
#
# Many-to-one is expected: e.g. the four entry_delta_z_ewm{N} variants all
# call compute_delta_z_ewm with a different halflife in FeatureSpec.params.
#
# KNOWN GAP (backlogged): feature names do not encode their full parameter
# set (e.g. "x_area_asymmetry_ewm" doesn't pin halflife in its name the way
# "entry_delta_z_ewm21" pins 21). FeatureSpec.params controls the actual
# value computed; nothing here validates name/params consistency, and
# cross-run comparison of the same ef.* column is only valid if the caller
# separately checks each run's config.json.
_FN_NAME_MAP: dict[str, str] = {
    "x_area_asymmetry_ewm":  "compute_x_area_asymmetry_ewm",
    "bw_gain_ann_norm":      "compute_bw_gain_ann_norm",
    "mean_ewm%":             "compute_mean_ewm_pct",
    "entry_delta_z_ewm3":    "compute_delta_z_ewm",
    "entry_delta_z_ewm5":    "compute_delta_z_ewm",
    "entry_delta_z_ewm10":   "compute_delta_z_ewm",
    "entry_delta_z_ewm21":   "compute_delta_z_ewm",
}


def _resolve_fn(feature_name: str) -> Any:
    """Resolve a feature function by feature name. Raises loudly if not found."""
    if feature_name in _FN_REGISTRY:
        return _FN_REGISTRY[feature_name]

    primitive_name = _FN_NAME_MAP.get(feature_name)
    if primitive_name is None:
        raise ValueError(
            f"EntryFeatureEngine: unknown feature name {feature_name!r}. "
            f"Not found in _FN_NAME_MAP. Known names: {sorted(_FN_NAME_MAP)}"
        )

    try:
        mod = importlib.import_module(_PRIMITIVES_MODULE)
        fn = getattr(mod, primitive_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"EntryFeatureEngine: cannot resolve feature {feature_name!r} "
            f"(looked for {primitive_name!r} in {_PRIMITIVES_MODULE}). "
            f"All feature functions must live in spread_primitives.py. Error: {e}"
        ) from e

    _FN_REGISTRY[feature_name] = fn
    return fn


@dataclass(slots=True)
class EntryFeatureEngine:
    """
    Computes entry features for proposed open trades.

    Reconstructs the spread level series from cached residuals in the
    signal generator, computes each configured feature, and writes results
    into CandidateAnalyticsState.features.

    Parameters
    ----------
    config : EntryFeatureConfig
        Which features to compute and their parameters. Each FeatureSpec.feature
        is the descriptive feature name (e.g. "x_area_asymmetry_ewm"), used
        identically everywhere — no alias/abbreviation.
    signal_generator : CandidateSignalGenerator
        Used to access the cached residual matrix via get_level_series().
    """
    config: EntryFeatureConfig
    signal_generator: Any   # CandidateSignalGenerator — avoid circular import

    _resolved_fns: dict[str, Any] = dc_field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        # Eagerly resolve all configured functions at construction time
        # so misconfigured features fail loudly before the simulation starts.
        for spec in self.config.features:
            self._resolved_fns[spec.feature] = _resolve_fn(spec.feature)

    def compute(
        self,
        proposed_opens: list[OpenCandidateAction],
        analytics_by_id: dict[str, CandidateAnalyticsState],
        candidate_refs_by_id: dict[str, CandidateRef],
        date: pd.Timestamp,
    ) -> None:
        """
        Compute entry features for each proposed open and write into
        analytics_by_id[cid].features in-place.

        Output keys in the features dict are the full feature names from
        FeatureSpec.feature (e.g. "x_area_asymmetry_ewm"), matching ef.*
        column naming in RunLoader.closed_trades.

        Parameters
        ----------
        proposed_opens
            Actions proposed by the trader (direction only, not yet sized).
        analytics_by_id
            CandidateAnalyticsState keyed by candidate_id. Modified in-place.
        candidate_refs_by_id
            CandidateRef keyed by candidate_id. Needed for weights.
        date
            Current simulation date. Passed to signal_generator for residual lookup.
        """
        if not proposed_opens or not self.config.features:
            return

        from src.analytics.spread_primitives import SIGNED_FEATURE_NAMES, direction_correct

        for action in proposed_opens:
            cid = action.candidate_id
            state = analytics_by_id.get(cid)
            ref = candidate_refs_by_id.get(cid)

            if state is None or ref is None:
                continue
            if not state.is_signal_ready:
                continue

            levels = self.signal_generator.get_level_series(ref=ref, date=date)
            if levels is None or len(levels) < 2:
                continue

            level_arr = levels.to_numpy(dtype=float)

            # sign(entry_z_score) — needed once per candidate for any signed feature
            z = state.z_score
            sign = float(z / abs(z)) if z is not None and z != 0.0 else None

            features: dict[str, float] = {}

            for spec in self.config.features:
                raw_value = self._compute_feature(
                    spec=spec,
                    level_arr=level_arr,
                    state=state,
                )
                if raw_value is None:
                    continue

                if spec.feature in SIGNED_FEATURE_NAMES:
                    if sign is None:
                        continue  # cannot direction-correct without z_score
                    value = direction_correct(raw_value, sign)
                else:
                    value = raw_value

                features[spec.feature] = value

            # Write into state — CandidateAnalyticsState.features is mutable dict
            state.features.update(features)

    def _compute_feature(
        self,
        *,
        spec: FeatureSpec,
        level_arr: np.ndarray,
        state: CandidateAnalyticsState,
    ) -> float | None:
        """
        Call the feature function with level_arr and spec.params.
        Returns the RAW (unsigned) value — direction correction is applied
        by the caller (compute()), not here.

        Some features need auxiliary state values (e.g. roll_std for
        normalization). These are injected automatically if present in
        spec.params as sentinel values.

        Injection conventions (applied before calling fn):
        - params key "roll_std" with value None → replaced with state.roll_std
        - params key "halflife" with value "residual_hl" → replaced with residual hl
          extracted from state.residual_key

        All other params passed through as-is.
        """
        fn = self._resolved_fns[spec.feature]
        params = dict(spec.params)  # copy to avoid mutating config

        # Auto-inject roll_std from state when requested
        if "roll_std" in params and params["roll_std"] is None:
            params["roll_std"] = state.roll_std

        # Auto-inject halflife from residual_key when requested
        if params.get("halflife") == "residual_hl":
            params["halflife"] = self._extract_residual_hl(state.residual_key)

        try:
            return fn(level_arr, **params)
        except Exception as e:
            # Never crash the simulation on a feature computation error.
            # Log and return None — SizingEngine treats missing features as neutral.
            import warnings
            warnings.warn(
                f"EntryFeatureEngine: feature {spec.feature!r} "
                f"failed for candidate {state.candidate_id!r}: {e}",
                stacklevel=2,
            )
            return None

    @staticmethod
    def _extract_residual_hl(residual_key: str) -> int | None:
        """Extract halflife from residual_key e.g. 'exp_hl504_mh1008_rf' → 504."""
        for part in residual_key.split("_"):
            if part.startswith("hl") and part[2:].isdigit():
                return int(part[2:])
        return None