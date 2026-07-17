"""
Named registry of default SimulatorConfig bundles.

These are the SimulatorConfig fields that _build_sim_config used to set from
hardcoded literals with no SweepConfig-derived counterpart: capital,
candidate_selection, activation, diagnostics, execution, performance,
persistence. Extracting them into a version-suffixed bundle lets a sweep
pin which defaults it ran under (SweepConfig.defaults), so the choice
enters config_hash and is recorded per run.

Representation — MappingProxyType keyed by SimulatorConfig field name,
chosen over a frozen @dataclass per bundle. The union target is
SimulatorConfig(**kwargs), so a name-keyed mapping makes the field
correspondence explicit and lets merge_defaults union and overlap-check
against the sweep-derived kwargs as a plain set operation. Tradeoff: keys
are strings, so a mistyped bundle key surfaces at SimulatorConfig(**...)
time as a loud TypeError rather than at definition; a frozen dataclass
would give attribute access and static field checking instead. The values
are already typed, frozen config objects either way.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from src.simulator.config import (
    ActivationConfig,
    CapitalConfig,
    ExecutionConfig,
    MRDiagnosticsConfig,
    PerformanceConfig,
    PersistenceConfig,
)
from src.candidates.candidate_selector import CandidateSelectionConfig


# Exactly the literals _build_sim_config carried before the registry existed.
# Changing any value here is a new bundle (standard_v2, ...), never an edit.
_STANDARD_V1: Mapping[str, Any] = MappingProxyType({
    "capital": CapitalConfig(total_capital=1_000_000.0),
    "candidate_selection": CandidateSelectionConfig(
        allowed_candidate_subtypes=("pca",), require_is_valid=True,
    ),
    "activation": ActivationConfig(
        one_active_per_group=False, switch_only_when_flat=False,
    ),
    "diagnostics": MRDiagnosticsConfig(lookback=252, compute_frequency="off"),
    "execution": ExecutionConfig(
        allow_fractional_shares=False, share_rounding="nearest",
    ),
    "performance": PerformanceConfig(
        enabled=True,
        metrics_table=True,
        report_html=True,
        benchmark_ticker=None,
        annualization_factor=252,
        per_group_breakdown=True,
    ),
    "persistence": PersistenceConfig(enabled=True),
})


DEFAULT_CONFIGS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "standard_v1": _STANDARD_V1,
})


def get_default_bundle(name: str) -> Mapping[str, Any]:
    """Look up a registered bundle, failing loud on an unknown name."""
    try:
        return DEFAULT_CONFIGS[name]
    except KeyError:
        raise KeyError(
            f"Unknown defaults bundle {name!r}. "
            f"Registered: {sorted(DEFAULT_CONFIGS)}."
        ) from None


def merge_defaults(
    sweep_derived: dict[str, Any],
    bundle: Mapping[str, Any],
    bundle_name: str,
) -> dict[str, Any]:
    """
    Union sweep-derived SimulatorConfig kwargs with a defaults bundle.

    A SimulatorConfig field must come from exactly one bucket. If the two
    ever name the same field — e.g. a field migrates from literal to
    sweep-derived and someone forgets to drop it from the bundle — this
    raises rather than letting one silently win via dict-merge precedence.
    """
    overlap = set(sweep_derived) & set(bundle)
    if overlap:
        raise ValueError(
            f"defaults bundle {bundle_name!r} and sweep-derived fields both set "
            f"{sorted(overlap)}. Each SimulatorConfig field must come from exactly "
            f"one bucket — remove it from one."
        )
    return {**sweep_derived, **bundle}
