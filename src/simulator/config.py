from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import pandas as pd

from src.candidates.candidate_selector import CandidateSelectionConfig
from src.residuals.causal_residuals import CausalResidualConfig

_GROUP_ABBREV = {
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

_GROUP_ABBREV_REV = {v: k for k, v in _GROUP_ABBREV.items()}


def group_resolve(name: str) -> str:
    """Resolve a group name (full or abbreviated) to the canonical full group_id."""
    key = name.lower()
    if key in _GROUP_ABBREV:
        return key
    if key in _GROUP_ABBREV_REV:
        return _GROUP_ABBREV_REV[key]
    raise KeyError(f"Unknown group: {name!r}")


def group_abbrev(
    groups: str | list[str],
    to_string: bool = False,
    brackets: bool = False,
) -> str | list[str]:
    """Convert group name(s) to abbreviation(s). Accepts full or abbreviated input."""
    if isinstance(groups, str):
        key = group_resolve(groups)
        result = _GROUP_ABBREV[key]
        return f"[{result}]" if to_string else result

    result = [_GROUP_ABBREV[group_resolve(s)] for s in groups]
    lb = "[" if brackets else ""
    rb = "]" if brackets else ""
    return lb + ",".join(result) + rb if to_string else result


# ── Cross-timescale signal gate ──────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class CrossTimescaleEntryConfig:
    """
    Cross-timescale entry filter applied before individual-rkey entry decisions.

    When configured, for a spread to enter on any timescale, the z-score
    vector across all timescales must satisfy these conditions.

    All conditions are AND-ed. None = disabled (that condition not checked).

    Parameters
    ----------
    min_abs_z_all
        Every timescale's abs(z) must exceed this.
    mean_abs_z_min
        Mean of abs(z) across timescales must exceed this.
    same_sign_required
        When True, all timescales must agree on direction. Mandatory -- no
        default (Track D follow-up). Unused by every call site in the repo
        today -- zero blast radius.
    """
    min_abs_z_all: float | None = None
    mean_abs_z_min: float | None = None
    same_sign_required: bool = field(kw_only=True)


@dataclass(slots=True, frozen=True)
class TimescaleRiskConfig:
    """
    Timescale selection and diversification policy in risk manager.

    Parameters
    ----------
    max_timescales_per_spread
        Maximum concurrent positions for the same spread across timescales.
        None = unlimited.
    selection
        When max_timescales_per_spread limits the set, how to pick:
        "max_abs_z" | "min_hl" | "max_hl" | "balanced"
    max_pct_single_timescale
        Maximum fraction of total open positions sharing the same residual_key.
        None = uncapped.

    selection is mandatory -- no default (Track D follow-up); only takes
    effect once max_timescales_per_spread is also set. Unused by every call
    site in the repo today -- zero blast radius.
    """
    max_timescales_per_spread: int | None = None
    selection: str = field(kw_only=True)
    max_pct_single_timescale: float | None = None

    def __post_init__(self) -> None:
        valid_selections = ("max_abs_z", "min_hl", "max_hl", "balanced")
        if self.selection not in valid_selections:
            raise ValueError(
                f"TimescaleRiskConfig.selection must be one of {valid_selections}, "
                f"got {self.selection!r}"
            )
        if self.max_timescales_per_spread is not None and self.max_timescales_per_spread < 1:
            raise ValueError(
                f"max_timescales_per_spread must be >= 1, got {self.max_timescales_per_spread}"
            )
        if self.max_pct_single_timescale is not None:
            if not (0.0 < self.max_pct_single_timescale <= 1.0):
                raise ValueError(
                    f"max_pct_single_timescale must be in (0, 1], "
                    f"got {self.max_pct_single_timescale}"
                )


@dataclass(frozen=True)
class GroupDataSource:
    """Per-group data triplet: universe config, candidate panel, optional residual params."""
    universe_config_name: str
    candidate_panel_stem: str
    residual_params_stem: str | None = None
    residual_key: str = ""  # from meta.json → CausalResidualConfig.key


def discover_group_data_sources(
        panel_dir: str | Path,
        universe_dir: str | Path,
        selected_groups: list[str] | None = None,
        excluded_groups: list[str] | None = None,
) -> list[GroupDataSource]:
    """
    Auto-discover GroupDataSource entries from a panel directory.

    Accepts both full (consumer_discretionary) and abbreviated (dsc) group
    names in stems, selected_groups, and excluded_groups.
    """
    if selected_groups is not None and excluded_groups is not None:
        raise ValueError("Cannot specify both selected_groups and excluded_groups.")

    selected_resolved = (
        {group_resolve(s) for s in selected_groups}
        if selected_groups is not None else None
    )
    excluded_resolved = (
        {group_resolve(s) for s in excluded_groups}
        if excluded_groups is not None else None
    )

    panel_dir = Path(panel_dir)
    universe_dir = Path(universe_dir)

    # Several panel files can share the same (group_id, residual_key) — e.g. a
    # freshly built panel left alongside an older timestamped one. Keep only the
    # most recently built file per (group_id, residual_key) so discovery yields
    # one source per group/key rather than duplicates. Distinct timescales carry
    # distinct residual_keys and are preserved.
    best: dict[tuple[str, str], tuple[float, str, GroupDataSource]] = {}

    for parquet in sorted(panel_dir.glob("*.panel.parquet")):
        stem = parquet.name.replace(".panel.parquet", "")

        if "_pairs_" not in stem:
            continue
        raw_id = stem.split("_pairs_")[0]

        try:
            group_id = group_resolve(raw_id)
        except KeyError:
            continue

        if selected_resolved is not None and group_id not in selected_resolved:
            continue
        if excluded_resolved is not None and group_id in excluded_resolved:
            continue

        yaml_path = universe_dir / f"universe.{group_id}_only.v1.yaml"
        if not yaml_path.exists():
            print(f"[discover] Warning: no universe YAML for {group_id}, skipping")
            continue

        params_path = panel_dir / f"{stem}_residual_params.pkl"
        residual_stem = stem if params_path.exists() else None

        residual_key = ""
        meta_path = panel_dir / f"{stem}.meta.json"
        if meta_path.exists():
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            residual_cfg_raw = meta.get("residual_cfg")
            if residual_cfg_raw is not None:
                residual_key = CausalResidualConfig.from_dict(residual_cfg_raw).key

        source = GroupDataSource(
            universe_config_name=yaml_path.name,
            candidate_panel_stem=stem,
            residual_params_stem=residual_stem,
            residual_key=residual_key,
        )

        # Tie-break newest-first by file mtime, then lexical stem (timestamp suffix).
        rank = (parquet.stat().st_mtime, stem)
        key = (group_id, residual_key)
        existing = best.get(key)
        if existing is None or rank > (existing[0], existing[1]):
            best[key] = (rank[0], rank[1], source)

    sources = [source for _, source in sorted(
        (key, val[2]) for key, val in best.items()
    )]

    if not sources:
        missing = f" (selected: {selected_groups})" if selected_groups else ""
        missing += f" (excluded: {excluded_groups})" if excluded_groups else ""
        raise FileNotFoundError(f"No matching panels found in {panel_dir}{missing}")

    found = [s.candidate_panel_stem.split("_pairs_")[0] for s in sources]
    print(f"[discover] Found {len(sources)} groups: {found}")
    return sources


@dataclass(slots=True, frozen=True)
class DataConfig:
    """
    Data source configuration.

    Supports single-group (backward compatible) and multi-group modes.
    """
    # Legacy single-group fields (backward compatible)
    universe_config_name: str = ""
    candidate_panel_stem: str = ""

    # Multi-group field
    groups: list[GroupDataSource] | None = None

    selected_groups: list[str] | None = None
    excluded_groups: list[str] | None = None
    candidate_panel_subdir: str = ""
    data_path: str = "data"
    price_field: str = "Close"
    return_method: str = "log"
    force_download: bool = False
    check_for_corruptions: bool = True
    start_after_nan: bool = True

    def resolved_groups(self) -> list[GroupDataSource]:
        """Return group list, auto-discovering from panel directory if needed."""
        if self.groups is not None:
            return self.groups
        if self.universe_config_name and self.candidate_panel_stem:
            return [GroupDataSource(
                universe_config_name=self.universe_config_name,
                candidate_panel_stem=self.candidate_panel_stem,
            )]
        from src.settings import CANDIDATE_PANELS_ROOT, CONFIG_UNIVERSE
        panel_dir = Path(CANDIDATE_PANELS_ROOT)
        if self.candidate_panel_subdir:
            panel_dir = panel_dir / self.candidate_panel_subdir
        return discover_group_data_sources(
            panel_dir=panel_dir,
            universe_dir=Path(CONFIG_UNIVERSE),
            selected_groups=self.selected_groups,
            excluded_groups=self.excluded_groups,
        )


@dataclass(slots=True, frozen=True)
class CapitalConfig:
    """
    Portfolio capital reference.

    total_capital is used as the denominator for portfolio-level risk cap
    checks (gross exposure, ticker concentration). It does not drive per-trade
    sizing — that is owned by SizingConfig.base_pair_notional.

    Group capital deployment is tracked as bookkeeping in the daily state log
    (sum of |units × price| per group) but no fixed quota is allocated per group.
    """
    total_capital: float

    def __post_init__(self) -> None:
        if self.total_capital <= 0.0:
            raise ValueError(f"CapitalConfig.total_capital must be positive, got {self.total_capital}")


@dataclass(slots=True, frozen=True)
class ActivationConfig:
    """
    Trading-rule booleans. Mandatory -- no default (Track D follow-up): every
    real call site already states both, and disagrees with the class default
    that used to backstop them (run_me.py / sweep_defaults.py both set
    False, False; the removed default was True, True).
    """
    one_active_per_group: bool
    switch_only_when_flat: bool


@dataclass
class ZScoreConfig:
    """
    lookback, ddof, and method are mandatory -- no default (Track D
    follow-up). ddof in particular fed the live rolling z-score std
    computation (candidate_signals.py's _rolling_mean_std_last) with no
    caller anywhere in the repo ever stating it, in either direction --
    unlike the other items in this track, it was never even visibly a
    magic number, since nobody had occasion to notice it was defaulted.
    method genuinely branches the computation (rolling vs. ewm mean/std),
    not merely a display choice.
    """
    lookback: int | list[int]
    weights: list[float] | None = None
    min_periods: int | None = None
    ddof: int = field(kw_only=True)
    method: str = field(kw_only=True)  # "rolling" or "ewm"
    residual_key: str = ""

    def resolved_lookbacks(self) -> list[int]:
        return [self.lookback] if isinstance(self.lookback, int) else list(self.lookback)

    def resolved_min_periods(self, lookback: int) -> int:
        if self.min_periods is not None:
            return self.min_periods
        return lookback + self.ddof

    def resolved_weights(self) -> list[float]:
        lbs = self.resolved_lookbacks()
        if self.weights is None:
            return [1.0 / len(lbs)] * len(lbs)
        if len(self.weights) != len(lbs):
            raise ValueError(f"weights length {len(self.weights)} != lookbacks length {len(lbs)}")
        s = sum(self.weights)
        return [w / s for w in self.weights]

    @property
    def timescale_label(self) -> str:
        """
        Label for position identity: residual_key + z-score params.
        Empty string when single-timescale (residual_key not set).
        """
        if not self.residual_key:
            return ""
        lb = self.resolved_lookbacks()[0]
        z_part = f"zhl{lb}" if self.method == "ewm" else f"zlb{lb}"
        return f"{self.residual_key}__{z_part}"

    @property
    def residual_hl(self) -> int | None:
        """Extract halflife from residual_key, e.g. 'exp_hl252_mh504_rf' -> 252."""
        m = re.search(r"exp_hl(\d+)_", self.residual_key)
        return int(m.group(1)) if m else None


@dataclass(slots=True, frozen=True)
class MRDiagnosticsConfig:
    """
    OU / mean-reversion diagnostics parameters.

    compute_frequency
        "daily"  — compute diagnostics every simulation day
        "weekly" — compute only on dates where new panel candidates arrive
        "off"    — never compute; z-score-only fast path

    Both fields mandatory -- no default (Track D follow-up). Every real
    call site already states both explicitly.
    """
    lookback: int
    compute_frequency: str  # "daily" | "weekly" | "off"

    def __post_init__(self) -> None:
        if self.compute_frequency not in ("daily", "weekly", "off"):
            raise ValueError(
                f"MRDiagnosticsConfig.compute_frequency must be 'daily', 'weekly', or 'off', "
                f"got {self.compute_frequency!r}"
            )


@dataclass(slots=True, frozen=True)
class SpreadMomentumConfig:
    """
    Spread momentum entry filter.

    All fields mandatory -- no default (Track D follow-up). Unused by every
    call site in the repo today (opt-in via
    PortfolioMeanReversionConfig.spread_momentum: SpreadMomentumConfig |
    None = None) -- zero blast radius to fix now, before a first caller
    opts in and silently inherits a class default.
    """
    lookback: int
    norm_window: int
    entry_threshold: float

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError(f"SpreadMomentumConfig.lookback must be >= 1, got {self.lookback}")
        if self.norm_window < self.lookback:
            raise ValueError(
                f"SpreadMomentumConfig.norm_window ({self.norm_window}) "
                f"must be >= lookback ({self.lookback})"
            )
        if self.entry_threshold < 0.0:
            return
            raise ValueError(
                f"SpreadMomentumConfig.entry_threshold must be >= 0.0, got {self.entry_threshold}"
            )


@dataclass(slots=True, frozen=True)
class PortfolioMeanReversionConfig:
    """
    exit_z is mandatory -- no default (Track D follow-up), same treatment as
    entry_z (already required from the original Track D pass). Dormant on
    B's baseline -- see tests/test_portfolio_mean_reversion_trader.py.
    """
    entry_z: float
    exit_z: float
    allow_long: bool = True
    allow_short: bool = True
    time_stop_half_life_multiplier: float | None = None
    mr_deterioration_threshold: float | None = None
    pnl_stop_fraction: float | None = None
    spread_momentum: SpreadMomentumConfig | None = None


@dataclass(slots=True, frozen=True)
class PairSpreadTraderConfig:
    """
    Trading rules for pair spread mean reversion.

    Deliberately minimal: entry on z-threshold, exit on z-cross.
    Sizing is handled by SizingEngine. Risk constraints by RiskManager.

    entry_z and exit_z are mandatory -- no default (Track D follow-up).
    This is the live trader on B's baseline; b_baseline_harness.py already
    states entry_z=1.75 against the removed class default of 2.0, evidence
    the default was already stale before this fix.
    """
    entry_z: float
    exit_z: float
    allow_long: bool = True
    allow_short: bool = True
    cross_ts: CrossTimescaleEntryConfig | None = None
    max_holding_days: str | int | dict[str, int] | None = None
    exit_rule: str | dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.exit_rule is not None and self.max_holding_days is not None:
            raise ValueError(
                "Cannot specify both exit_rule and max_holding_days. "
                "Use exit_rule='days_open > N' for a pure time stop."
            )


@dataclass(slots=True, frozen=True)
class KellyConfig:
    """
    Kelly criterion sizing configuration.

    Derives base_pair_notional from expanding-window trade outcome statistics
    instead of using a fixed value. Activates per-group or globally.

    Parameters
    ----------
    half_life
        EWM half-life in number of trades. None = equal-weighted expanding window.
    min_trades
        Minimum closed trades before Kelly estimate activates.
    blend_target
        Trade count at which Kelly fully replaces base_pair_notional.
    fraction
        Kelly fraction (0.5 = half-Kelly).
    floor_multiplier
        Minimum Kelly notional as multiple of base_pair_notional.
    cap_multiplier
        Maximum Kelly notional as multiple of base_pair_notional.
    per_sector
        If True, track stats and compute Kelly per group_id.

    Every field but half_life is mandatory -- no default (Track D
    follow-up). half_life keeps its None default: documented, legitimate
    "equal-weighted expanding window" opt state, not a silent fallback.
    Unused by every call site in the repo today (opt-in via
    SizingConfig.kelly: KellyConfig | None = None) -- zero blast radius.
    """
    half_life: int | None = None
    min_trades: int = field(kw_only=True)
    blend_target: int = field(kw_only=True)
    fraction: float = field(kw_only=True)
    floor_multiplier: float = field(kw_only=True)
    cap_multiplier: float = field(kw_only=True)
    per_sector: bool = field(kw_only=True)


@dataclass(slots=True, frozen=True)
class VolSizingConfig:
    """
    Vol-normalization sizing parameters.

    Scales each pair's notional by (median_roll_std / pair_roll_std) so every
    pair contributes roughly equal portfolio variance. The ratio is clamped to
    [floor_multiplier, cap_multiplier] to prevent extreme sizing.

    Parameters
    ----------
    floor_multiplier
        Minimum vol-norm size multiplier (prevents over-shrinking low-vol pairs).
        Mandatory — no default (Track D).
    cap_multiplier
        Maximum vol-norm size multiplier (prevents over-sizing high-vol pairs).
        Mandatory — no default (Track D).
    """
    floor_multiplier: float
    cap_multiplier: float

    def __post_init__(self) -> None:
        if self.floor_multiplier <= 0.0:
            raise ValueError(f"VolSizingConfig.floor_multiplier must be > 0, got {self.floor_multiplier}")
        if self.cap_multiplier < self.floor_multiplier:
            raise ValueError(
                f"VolSizingConfig.cap_multiplier ({self.cap_multiplier}) must be >= "
                f"floor_multiplier ({self.floor_multiplier})"
            )



# ── Entry feature engine ──────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class FeatureSpec:
    """
    Specification for one entry feature.

    Parameters
    ----------
    feature
        Descriptive feature name (e.g. "x_area_asymmetry_ewm") — resolves via
        src.simulator.entry_feature_engine._FN_NAME_MAP to the primitive
        function in spread_primitives.py. Used directly (no abbreviation) as
        the CandidateAnalyticsState.features key and as the ef.*/efsc.*
        column suffix in RunLoader.closed_trades. Must be unique within
        EntryFeatureConfig.

        Same field name as FeatureIntervalSpec.feature — both must use this
        exact descriptive name.
    params
        Keyword arguments passed to the compute function.
        Special sentinels resolved at runtime by EntryFeatureEngine:
        - roll_std=None  → replaced with CandidateAnalyticsState.roll_std
        - halflife="residual_hl" → replaced with hl from residual_key
        Do NOT pass "sign" here — direction correction is applied
        centrally by EntryFeatureEngine.compute(), not per-feature.
    """
    feature: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.feature:
            raise ValueError("FeatureSpec.feature must not be empty.")
        if "sign" in self.params:
            raise ValueError(
                f"FeatureSpec(feature={self.feature!r}): params must not contain 'sign' — "
                f"direction correction is applied centrally by EntryFeatureEngine "
                f"via SIGNED_FEATURE_NAMES, not per-feature. Remove this key."
            )


@dataclass(slots=True, frozen=True)
class EntryFeatureConfig:
    """
    Configuration for EntryFeatureEngine.

    Parameters
    ----------
    features
        Tuple of FeatureSpec — which features to compute at trade entry.
    """
    features: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        names = [f.feature for f in self.features]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"EntryFeatureConfig: duplicate feature specs: {sorted(set(dupes))}")


# ── Interval scoring ──────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class FeatureIntervalSpec:
    """
    Interval-based weight specification for one feature.

    Parameters
    ----------
    feature
        Descriptive feature name — must match a key in
        CandidateAnalyticsState.features (same name used in
        entry_features.FeatureSpec.feature).
    interval_limits
        Strictly increasing sequence of n+1 boundary values defining n intervals.
        Use float("-inf") and float("inf") for open-ended boundaries.
    interval_weights
        n weights, one per interval.
        Convention: 1.0 = baseline, 2.0 = double size, 0.0 = exclude.
    interval_names
        Optional labels for intervals, for logging and diagnostics.
    missing_weight
        Weight when feature value is missing. Mandatory -- no default
        (Track D follow-up). Unused by every call site in the repo today
        (opt-in via SizingConfig.interval_scoring:
        IntervalScoringConfig | None = None) -- zero blast radius.
    """
    feature: str
    interval_limits: tuple[float, ...]
    interval_weights: tuple[float, ...]
    interval_names: tuple[str, ...] | None = None
    missing_weight: float = field(kw_only=True)

    def __post_init__(self) -> None:
        n_intervals = len(self.interval_limits) - 1
        if n_intervals < 1:
            raise ValueError(
                f"FeatureIntervalSpec({self.feature!r}): interval_limits must have "
                f"at least 2 values (got {len(self.interval_limits)})."
            )
        if len(self.interval_weights) != n_intervals:
            raise ValueError(
                f"FeatureIntervalSpec({self.feature!r}): interval_weights length "
                f"{len(self.interval_weights)} != n_intervals {n_intervals}."
            )
        if self.interval_names is not None and len(self.interval_names) != n_intervals:
            raise ValueError(
                f"FeatureIntervalSpec({self.feature!r}): interval_names length "
                f"{len(self.interval_names)} != n_intervals {n_intervals}."
            )
        limits = list(self.interval_limits)
        if limits != sorted(limits):
            raise ValueError(
                f"FeatureIntervalSpec({self.feature!r}): interval_limits must be "
                f"strictly increasing, got {limits}."
            )

    def lookup(self, value: float | None) -> float:
        """Return the interval weight for a given feature value."""
        import math
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return self.missing_weight
        limits = self.interval_limits
        for i in range(len(limits) - 1):
            if value < limits[i + 1]:
                return float(self.interval_weights[i])
        return float(self.interval_weights[-1])


@dataclass(slots=True, frozen=True)
class IntervalScoringConfig:
    """
    Interval-based sizing multiplier computation.

    Each feature is mapped to a weight via its FeatureIntervalSpec.
    The final multiplier is the weighted mean of per-feature weights,
    clamped to [floor_multiplier, cap_multiplier].

    A per-feature weight of 0.0 causes the trade to be excluded (dropped
    before RiskManager).

    Parameters
    ----------
    feature_specs
        One FeatureIntervalSpec per feature used in scoring.
    feature_weights
        Relative aggregation weight per feature, keyed by the same name used in
        the corresponding FeatureIntervalSpec.feature. Every spec must have an
        entry and every entry must name a spec — validated here, since a
        missing weight is silently unrecoverable at sizing time: defaulting it
        would promote an intentionally-zeroed feature to full weight and skew
        every trade's size.
    floor_multiplier
        Minimum output multiplier. Mandatory -- no default (Track D
        follow-up).
    cap_multiplier
        Maximum output multiplier. Mandatory -- no default (Track D
        follow-up).

    floor_multiplier, cap_multiplier, and floor_mode are unused by every
    call site in the repo today (opt-in via SizingConfig.interval_scoring:
    IntervalScoringConfig | None = None) -- zero blast radius to fix now.
    """
    feature_specs: tuple[FeatureIntervalSpec, ...]
    feature_weights: dict[str, float] = field(default_factory=dict)
    floor_multiplier: float = field(kw_only=True)
    cap_multiplier: float = field(kw_only=True)
    floor_mode: str = field(kw_only=True)   # "clamp" | "exclude"

    def __post_init__(self) -> None:
        if not self.feature_specs:
            raise ValueError("IntervalScoringConfig.feature_specs must not be empty.")
        if self.floor_multiplier < 0.0:
            raise ValueError(f"floor_multiplier must be >= 0, got {self.floor_multiplier}.")
        if self.cap_multiplier < self.floor_multiplier:
            raise ValueError(
                f"cap_multiplier ({self.cap_multiplier}) must be >= "
                f"floor_multiplier ({self.floor_multiplier})."
            )
        if self.floor_mode not in ("clamp", "exclude"):
            raise ValueError(f"floor_mode must be 'clamp' or 'exclude', got {self.floor_mode!r}.")
        names = [s.feature for s in self.feature_specs]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"IntervalScoringConfig: duplicate feature specs: {sorted(set(dupes))}")

        # feature_weights must correspond exactly to feature_specs, both ways.
        # SizingEngine raises KeyError on a missing weight — correctly, since
        # defaulting one would silently promote an intentionally-zeroed feature
        # to full weight. But that only surfaces at the first trade, minutes
        # into a run, so catch the mismatch at construction instead.
        spec_names = set(names)
        weight_names = set(self.feature_weights)

        unknown = weight_names - spec_names
        if unknown:
            raise ValueError(
                f"IntervalScoringConfig: feature_weights names no such feature spec: "
                f"{sorted(unknown)}. Known feature_specs: {sorted(spec_names)}. "
                f"Keys must match FeatureIntervalSpec.feature exactly."
            )

        missing = spec_names - weight_names
        if missing:
            raise ValueError(
                f"IntervalScoringConfig: feature_specs have no feature_weights entry: "
                f"{sorted(missing)}. Given feature_weights: {sorted(weight_names)}. "
                f"Every spec needs an explicit weight — there is no default."
            )


@dataclass(slots=True, frozen=True)
class SizingConfig:
    """
    Per-trade sizing configuration.

    Owns all sizing logic: base notional, vol normalization, Kelly adaptation.
    The SizingEngine uses this config to compute a concrete pair_notional for
    each approved trade before execution.

    Parameters
    ----------
    base_pair_notional
        Dollar notional per trade before any adjustments. Required — must be
        set explicitly. No silent default to avoid misconfigured runs.
    vol_normalize
        When set, scale each pair's notional by inverse spread volatility
        relative to the cross-sectional median. None = disabled.
    kelly
        When set, derive base_pair_notional adaptively from trade outcome
        statistics. Kelly and vol_normalize are composable: Kelly sets the
        base, vol_normalize scales it per-pair.
    """
    base_pair_notional: float
    vol_normalize: VolSizingConfig | None = None
    kelly: KellyConfig | None = None
    interval_scoring: IntervalScoringConfig | None = None

    def __post_init__(self) -> None:
        if self.base_pair_notional <= 0.0:
            raise ValueError(
                f"SizingConfig.base_pair_notional must be positive, got {self.base_pair_notional}"
            )


@dataclass(slots=True, frozen=True)
class RiskManagerConfig:
    """
    Portfolio-level risk constraints applied to proposed open actions.

    Purely a constraint checker — sizing logic lives in SizingConfig/SizingEngine.

    Parameters
    ----------
    max_gross_exposure
        Maximum total gross notional as a multiple of CapitalConfig.total_capital.
        E.g. 10.0 = allow up to 10× total_capital deployed simultaneously.
        Mandatory — no default (Track D).
    max_ticker_exposure_pct
        Maximum net notional per ticker as a fraction of total_capital.
        E.g. 0.15 = no single ticker can exceed 15% of total_capital as net
        exposure. Mandatory — no default (Track D).
    max_concurrent_positions
        Hard cap on simultaneous open positions. None = uncapped.
    timescale_risk
        Timescale selection and concentration policy. None = approve all.
    """
    max_gross_exposure: float = field(kw_only=True)
    max_ticker_exposure_pct: float = field(kw_only=True)
    max_concurrent_positions: int | None = None
    timescale_risk: TimescaleRiskConfig | None = None


@dataclass(slots=True, frozen=True)
class RunConfig:
    start_date: pd.Timestamp | str | None = None
    end_date: pd.Timestamp | str | None = None
    progress: bool = False
    progress_step: int = 10

    def __post_init__(self) -> None:
        if self.start_date is not None:
            object.__setattr__(self, "start_date", pd.Timestamp(self.start_date))
        if self.end_date is not None:
            object.__setattr__(self, "end_date", pd.Timestamp(self.end_date))
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("run.start_date must be <= run.end_date.")


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    """
    Execution and cost model.

    min_abs_units, and every commission/borrow field below, are mandatory —
    no default. These feed straight into reported PnL and were previously
    inherited silently from a class default that no caller ever stated
    (Track D). Pin them explicitly (or via a versioned bundle) rather than
    reintroducing a magic number.
    """
    allow_fractional_shares: bool = False
    share_rounding: str = "nearest"   # nearest | floor | ceil
    min_abs_units: float = field(kw_only=True)        # ignored if fractional=True

    commission_per_share: float = field(kw_only=True)
    commission_per_order: float = field(kw_only=True)
    min_commission_per_order: float = field(kw_only=True)
    max_commission_per_order: float = field(kw_only=True)
    max_commission_pct_of_trade: float = field(kw_only=True)
    short_borrow_rate_annual_bps: float = field(kw_only=True)


@dataclass(slots=True, frozen=True)
class PerformanceConfig:
    enabled: bool = False
    metrics_table: bool = True
    report_html: bool = True
    report_output_dir: str = ""
    benchmark_ticker: str | None = None
    annualization_factor: int = 252
    per_group_breakdown: bool = False


@dataclass(slots=True, frozen=True)
class PersistenceConfig:
    """
    Simulation run persistence.
    """
    enabled: bool = False
    output_dir: str = ""
    artifacts: tuple[str, ...] = (
        "config",
        "selected_panel",
        "closed_trades",
        "daily_state",
        "daily_portfolio_state",
        "diagnostics",
        "ticker_trade_log",
        "action_log",
        "performance_metrics",
        "performance_report",
    )


TraderConfig = PortfolioMeanReversionConfig | PairSpreadTraderConfig


@dataclass(slots=True, frozen=True)
class SpectrumConfig:
    """
    Configuration for z-score spectrum capture during simulation.
    """
    residual_lookbacks: list[int] | None = None
    zlb_values: list[int] | None = None
    record_exit: bool = True
    identity_check_epsilon: float = 1e-4


@dataclass(slots=True, frozen=True)
class SimulatorConfig:
    data: DataConfig
    candidate_selection: CandidateSelectionConfig
    activation: ActivationConfig
    z_score: ZScoreConfig | list[ZScoreConfig]
    diagnostics: MRDiagnosticsConfig
    trader: TraderConfig
    run: RunConfig
    execution: ExecutionConfig
    capital: CapitalConfig
    sizing: SizingConfig
    # None is deliberate, not a silent fallback (Track D investigation): it
    # means "resolve per-residual_key CausalResidualConfig from the
    # persisted candidate-panel metadata instead" (see
    # simulator_factory._resolve_residual_configs). Sourced from neither
    # SweepConfig nor a DEFAULT_CONFIGS bundle — metadata absence raises
    # loudly (ValueError) rather than defaulting to a value, so the gap in
    # the config surface's three-way partition (sweep-derived / bundle /
    # this field) does not need a guard.
    residual: CausalResidualConfig | None = None
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    risk_manager: RiskManagerConfig | None = None
    spectrum: SpectrumConfig | None = None
    entry_features: EntryFeatureConfig | None = None

    def __post_init__(self) -> None:
        z_configs = self.resolved_z_score_configs()
        max_z_lookback = max(
            max(zc.resolved_lookbacks()) for zc in z_configs
        )
        if max_z_lookback > self.diagnostics.lookback:
            raise ValueError(
                f"z_score max lookback ({max_z_lookback}) must be "
                f"<= diagnostics.lookback ({self.diagnostics.lookback})."
            )
        if self.residual is not None and self.residual.window_mode == "rolling":
            if self.diagnostics.lookback > self.residual.lookback:
                raise ValueError(
                    f"diagnostics.lookback ({self.diagnostics.lookback}) must be "
                    f"<= residual.lookback ({self.residual.lookback})."
                )
            raise ValueError(
                "SimulatorConfig.residual uses an EQ_ROLLING (window_mode='rolling') "
                "residual config. The z-score's EWM history span is inherited from "
                "whichever upstream path supplied the residual series — full history "
                "on the disk-backed path, truncated to the residual lookback on the "
                "recompute path — so the same (spread_id, date) can yield a different "
                "z-score depending on whether a cache existed. The explicit span "
                "parameter needed to make this deterministic is not yet implemented."
            )

        if isinstance(self.z_score, list):
            if len(self.z_score) == 0:
                raise ValueError("z_score list must not be empty.")
            missing = [i for i, zc in enumerate(self.z_score) if not zc.residual_key]
            if missing:
                raise ValueError(
                    f"When z_score is a list, every entry must have residual_key set. "
                    f"Entries at indices {missing} are missing residual_key."
                )
            seen = set()
            for zc in self.z_score:
                key = zc.timescale_label
                if key in seen:
                    raise ValueError(f"Duplicate z_score config timescale_label: {key!r}")
                seen.add(key)

        for rkey, zcs in self.z_score_configs_by_rkey().items():
            if len(zcs) > 1:
                raise ValueError(
                    f"{len(zcs)} z_score configs share residual_key={rkey!r}. "
                    "The trader's open-position dedup key is (spread_id, "
                    "residual_key), which does not include the z-score config, so "
                    "these sleeves would silently contend for the same position "
                    "slot — arrival order would decide which one trades. Run them "
                    "as separate runs for now."
                )

    def resolved_z_score_configs(self) -> list[ZScoreConfig]:
        if isinstance(self.z_score, list):
            return self.z_score
        return [self.z_score]

    def is_multi_timescale(self) -> bool:
        return isinstance(self.z_score, list)

    def z_score_by_residual_key(self) -> dict[str, ZScoreConfig]:
        out: dict[str, ZScoreConfig] = {}
        for zc in self.resolved_z_score_configs():
            if zc.residual_key not in out:
                out[zc.residual_key] = zc
        return out

    def z_score_by_timescale_label(self) -> dict[str, ZScoreConfig]:
        return {zc.timescale_label: zc for zc in self.resolved_z_score_configs()}

    def unique_residual_keys(self) -> set[str]:
        return {zc.residual_key for zc in self.resolved_z_score_configs()}

    def z_score_configs_by_rkey(self) -> dict[str, list[ZScoreConfig]]:
        out: dict[str, list[ZScoreConfig]] = {}
        for zc in self.resolved_z_score_configs():
            out.setdefault(zc.residual_key, []).append(zc)
        return out
