from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass(slots=True, frozen=True)
class CandidateRef:
    candidate_id: str
    spread_id: str
    group_id: str
    asof_date: pd.Timestamp
    members: tuple[str, ...]
    weights: tuple[float, ...]
    subtype: str | None = None
    residual_key: str = ""
    timescale_label: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty.")
        if not self.spread_id:
            raise ValueError("spread_id must not be empty.")
        if not self.group_id:
            raise ValueError("group_id must not be empty.")
        if len(self.members) == 0:
            raise ValueError("members must not be empty.")
        if len(self.members) != len(self.weights):
            raise ValueError(f"members={self.members} and weights={self.weights} must have the same length.")


@dataclass(slots=True, frozen=True)
class CandidateState:
    candidate: CandidateRef
    date: pd.Timestamp
    is_new_arrival: bool
    is_active: bool
    level: float | None
    roll_mean: float | None
    roll_std: float | None
    z_score: float | None
    is_signal_ready: bool


@dataclass(slots=True)
class CandidateMarketSnapshot:
    date: pd.Timestamp
    current_prices: pd.Series
    candidate_states: pd.DataFrame

    def __post_init__(self) -> None:
        required_cols = {
            "spread_id",
            "group_id",
            "is_new_arrival",
            "is_active",
            "level",
            "roll_mean",
            "roll_std",
            "z_score",
            "is_signal_ready",
        }
        missing = required_cols.difference(self.candidate_states.columns)
        if missing:
            raise ValueError(
                f"candidate_states is missing required columns: {sorted(missing)}"
            )
        if self.candidate_states.index.name != "candidate_id":
            self.candidate_states.index.name = "candidate_id"


@dataclass(slots=True)
class CandidateAnalyticsState:
    candidate_id: str
    group_id: str
    spread_id: str
    date: pd.Timestamp

    z_score: float | None
    level: float | None
    roll_mean: float | None
    roll_std: float | None

    adf_pvalue: float | None
    mr_score: float | None
    kappa: float | None
    half_life: float | None

    is_signal_ready: bool

    z_components: tuple[ZScoreComponent, ...] = ()
    residual_key: str = ""
    timescale_label: str = ""
    features: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TickerDelta:
    ticker: str
    delta_units: float


@dataclass(slots=True, frozen=True)
class ExecutionFill:
    ticker: str
    requested_delta_units: float
    filled_delta_units: float
    fill_price: float
    traded_notional: float
    commission: float = 0.0


@dataclass(slots=True, frozen=True)
class ExecutionResult:
    fills: tuple[ExecutionFill, ...]

    @property
    def has_fills(self) -> bool:
        return len(self.fills) > 0


@dataclass(slots=True)
class LiveCandidatePosition:
    trade_id: str
    candidate_id: str
    group_id: str
    spread_id: str

    direction: float           # +1.0 or -1.0
    pair_notional: float       # dollar notional at entry (from SizingEngine)
    gross_value: float         # current mark-to-market gross value

    entry_date: pd.Timestamp
    days_open: int

    entry_z_score: float | None = None
    unrealized_pnl: float = 0.0
    min_unrealized_pnl: float = 0.0

    units_by_ticker: dict[str, int] = field(default_factory=dict)
    entry_prices_by_ticker: dict[str, float] = field(default_factory=dict)

    realized_weights_by_ticker: dict[str, float] = field(default_factory=dict)
    entry_mr_score: float | None = None
    entry_half_life: float | None = None
    entry_adf_pvalue: float | None = None
    entry_z_components: tuple[ZScoreComponent, ...] = ()

    accumulated_transaction_cost: float = 0.0
    accumulated_borrow_cost: float = 0.0

    residual_key: str = ""
    timescale_label: str = ""
    entry_features: dict[str, float] = field(default_factory=dict)
    entry_feature_scores: dict[str, float] = field(default_factory=dict)
    entry_size_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in (1.0, -1.0):
            raise ValueError(f"LiveCandidatePosition.direction must be +1.0 or -1.0, got {self.direction}.")
        if self.pair_notional <= 0.0:
            raise ValueError(f"LiveCandidatePosition.pair_notional must be positive, got {self.pair_notional}.")
        self.entry_date = pd.Timestamp(self.entry_date)


@dataclass(slots=True, frozen=True)
class ClosedCandidateTrade:
    trade_id: str
    candidate_id: str
    group_id: str
    spread_id: str

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    days_open: int

    direction: float           # +1.0 or -1.0
    pair_notional: float       # dollar notional at entry

    entry_z_score: float | None
    exit_z_score: float | None

    realized_pnl_gross: float
    transaction_costs: float
    borrow_costs: float
    realized_pnl_net: float

    entry_z_components: tuple[ZScoreComponent, ...] = ()
    exit_z_components: tuple[ZScoreComponent, ...] = ()

    residual_key: str = ""
    timescale_label: str = ""
    trigger_rhl: int | None = None
    trigger_zlb: int | None = None
    entry_features: dict[str, float] = field(default_factory=dict)
    entry_feature_scores: dict[str, float] = field(default_factory=dict)
    entry_size_multiplier: float = 1.0


@dataclass(slots=True, frozen=True)
class ZScoreComponent:
    lookback: int
    z_score: float | None
    roll_mean: float | None
    roll_std: float | None
