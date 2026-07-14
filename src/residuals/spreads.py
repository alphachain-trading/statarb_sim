from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal, TypeAlias

import pandas as pd

Ticker: TypeAlias = str
SpreadId: TypeAlias = str
HedgeRatioMethod: TypeAlias = Literal["unit", "ols", "pca"]


@dataclass
class SpreadConstructionResult:
    spread_returns: pd.DataFrame      # index=date, columns=spread_id
    spread_metadata: pd.DataFrame     # index=spread_id


def spread_id(tickers: Ticker | list[Ticker]) -> SpreadId:
    """
    Build a canonical spread identifier from one or more distinct tickers.
    """
    if isinstance(tickers, str):
        ticker = tickers.strip()
        if not ticker:
            raise ValueError("Ticker must not be empty.")
        return ticker

    cleaned = [t.strip() for t in tickers]
    if not cleaned:
        raise ValueError("Tickers must not be empty.")
    if any(not t for t in cleaned):
        raise ValueError("Tickers must not contain empty values.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("SpreadId must not contain the same ticker more than once.")

    return "|".join(sorted(cleaned))


def spread_members(spread: SpreadId) -> list[Ticker]:
    """
    Split a canonical spread id into its member tickers.
    """
    members = [part.strip() for part in spread.split("|")]
    if not members or any(not m for m in members):
        raise ValueError(f"Invalid spread_id: {spread!r}")
    if len(set(members)) != len(members):
        raise ValueError(f"Invalid spread_id with duplicate members: {spread!r}")
    return members


def generate_spread_ids(
    tickers: list[Ticker],
    n_legs: int,
) -> list[SpreadId]:
    """
    Generate all canonical spread ids with exactly n_legs distinct members.
    """
    if n_legs < 1:
        raise ValueError("n_legs must be >= 1.")

    cleaned = [t.strip() for t in tickers]
    if any(not t for t in cleaned):
        raise ValueError("Tickers must not contain empty values.")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Input tickers must be unique.")
    if len(cleaned) < n_legs:
        raise ValueError("Not enough tickers for requested n_legs.")

    if n_legs == 1:
        return [spread_id(t) for t in cleaned]

    return [spread_id(list(combo)) for combo in combinations(cleaned, n_legs)]


def ols_beta_no_intercept(
    y: pd.Series,
    x: pd.Series,
    min_denominator: float = 1e-12,
) -> float:
    """
    Estimate beta in y = beta * x + e using OLS without intercept.
    """
    if len(y) != len(x):
        raise ValueError("y and x must have the same length.")

    denom = float((x * x).sum())
    if denom <= min_denominator:
        raise ValueError("OLS beta denominator is too small.")

    numer = float((y * x).sum())
    return numer / denom


def _weights_json_from_pair_weights(w_left: float, w_right: float) -> str:
    return f"[{float(w_left)}, {float(w_right)}]"


def _parse_weights_json(weights_json: str | list[float] | tuple[float, ...]) -> list[float]:
    """
    Parse a JSON-encoded or already-materialized weight vector into a Python list of floats.
    """
    if isinstance(weights_json, str):
        weights = json.loads(weights_json)
    elif isinstance(weights_json, (list, tuple)):
        weights = list(weights_json)
    else:
        raise TypeError(
            "weights_json must be a JSON string, list, or tuple of numeric weights."
        )

    if not isinstance(weights, list):
        raise ValueError("Parsed weights must be a list.")

    if len(weights) < 2:
        raise ValueError("At least two weights are required to extract relative weights.")

    try:
        out = [float(w) for w in weights]
    except (TypeError, ValueError) as e:
        raise ValueError("Weights must be numeric.") from e

    return out


def extract_relative_weights(
    weights_json: str | list[float] | tuple[float, ...],
) -> list[float]:
    """
    Extract relative weights versus the first leg.

    For weights [w1, w2, ..., wn], returns:
        [w2 / w1, w3 / w1, ..., wn / w1]

    Notes
    -----
    - Always returns a vector.
    - For n_legs == 2, the returned vector has length 1.
    - This assumes the first leg is the normalization anchor.
    """
    weights = _parse_weights_json(weights_json)
    w1 = float(weights[0])

    if w1 == 0.0:
        raise ValueError("Cannot extract relative weights when the first weight is zero.")

    return [float(w / w1) for w in weights[1:]]


def add_relative_weights(
    df: pd.DataFrame,
    weights_col: str = "weights_json",
    prefix: str = "rel_w",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add convenience columns with relative weights extracted from weights_json.

    For pair spreads with weights [w1, w2], this adds:
        rel_w_2 = w2 / w1

    For triplets [w1, w2, w3], this adds:
        rel_w_2 = w2 / w1
        rel_w_3 = w3 / w1

    Parameters
    ----------
    df
        Input DataFrame containing a weights column.
    weights_col
        Name of the column containing JSON-encoded or materialized weights.
    prefix
        Prefix for added relative-weight columns.
    inplace
        Whether to modify df in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with added relative-weight columns.
    """
    if weights_col not in df.columns:
        raise ValueError(f"Missing required column: {weights_col!r}")

    out = df if inplace else df.copy()

    rel_series = out[weights_col].apply(extract_relative_weights)
    max_len = int(rel_series.map(len).max()) if len(rel_series) > 0 else 0

    for i in range(max_len):
        col = f"{prefix}_{i + 2}"
        out[col] = rel_series.apply(
            lambda x: float(x[i]) if i < len(x) else pd.NA
        )

    return out


def build_spread_returns(
    residual_returns: pd.DataFrame,
    n_legs: int = 2,
    hedge_ratio_method: HedgeRatioMethod = "unit",
) -> SpreadConstructionResult:
    """
    Build spread return series from residual return columns.

    Currently implemented:
    - n_legs == 2
    - hedge_ratio_method in {"unit", "ols"}

    For each canonical pair A|B:

    - "unit":
        spread = A - B
        weights = [1.0, -1.0]

    - "ols":
        estimate beta from A_t = beta * B_t + e_t using OLS without intercept
        over the provided trailing residual-return window, then construct
        spread = A - beta * B
        weights = [1.0, -beta]

    Notes
    -----
    For "ols", the hedge ratio is estimated once from the full input window
    passed into this function. The returned spread series uses that fixed beta
    for the entire window.
    """
    if residual_returns.empty:
        raise ValueError("residual_returns is empty.")

    if residual_returns.isna().any().any():
        raise ValueError("residual_returns contains NaNs.")

    if residual_returns.columns.has_duplicates:
        raise ValueError("residual_returns columns must be unique tickers.")

    tickers = [str(c).strip() for c in residual_returns.columns]
    if any(not t for t in tickers):
        raise ValueError("residual_returns contains invalid column names.")

    if len(set(tickers)) != len(tickers):
        raise ValueError("residual_returns columns must be unique after normalization.")

    if list(residual_returns.columns) != tickers:
        residual_returns = residual_returns.copy()
        residual_returns.columns = tickers

    if n_legs != 2:
        raise NotImplementedError("Only n_legs=2 is implemented for now.")

    if hedge_ratio_method not in ["unit", "ols"]:
        raise NotImplementedError("Only hedge_ratio_methods unit and ols are implemented for now.")

    pair_ids = generate_spread_ids(tickers=tickers, n_legs=2)
    if not pair_ids:
        raise ValueError("No pair spreads can be generated from the provided columns.")

    data: dict[SpreadId, pd.Series] = {}
    meta_rows: list[dict[str, Any]] = []

    for sid in pair_ids:
        left, right = spread_members(sid)

        left_series = residual_returns[left]
        right_series = residual_returns[right]

        if hedge_ratio_method == "unit":
            beta = 1.0
        elif hedge_ratio_method == "ols":
            beta = ols_beta_no_intercept(
                y=left_series,
                x=right_series,
            )
        else:
            raise NotImplementedError(
                f"Unsupported hedge_ratio_method={hedge_ratio_method!r}"
            )

        weights_json = _weights_json_from_pair_weights(1.0, -beta)

        data[sid] = left_series - beta * right_series

        meta_rows.append(
            {
                "spread_id": sid,
                "n_legs": 2,
                "hedge_ratio_method": hedge_ratio_method,
                "weights_json": weights_json,
                "relative_weights": extract_relative_weights(weights_json),
            }
        )

    spread_returns = pd.DataFrame(data, index=residual_returns.index)

    spread_metadata = (
        pd.DataFrame(meta_rows)
        .set_index("spread_id")
        .sort_index()
    )

    return SpreadConstructionResult(
        spread_returns=spread_returns,
        spread_metadata=spread_metadata,
    )