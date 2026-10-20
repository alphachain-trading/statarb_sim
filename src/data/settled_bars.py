"""
Guard against unsettled (still-forming) daily bars.

A download or snapshot mint during market hours receives the current session's
provisional bar from yfinance. Persisting it would freeze a bar that later
changes. The rule is deliberately simple: only bars dated strictly before today
in America/New_York are kept; same-day bars are always dropped (no buffer, no
exchange calendar).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")


def require_daily_interval(interval: str) -> None:
    """Raise unless ``interval`` is ``"1d"``; the guard only understands daily bars."""
    if interval != "1d":
        raise ValueError(
            f"Unsettled-bar guard supports interval='1d' only, got {interval!r}. "
            "Intraday data is not used anywhere in this repo."
        )


def drop_unsettled_bars(
    df: pd.DataFrame, now: datetime | None = None, label: str = ""
) -> pd.DataFrame:
    """
    Return ``df`` without rows dated today (America/New_York) or later.

    ``now`` is injectable for tests; a naive value is interpreted as New York
    time. The index must be a DatetimeIndex of session dates; a tz-aware index
    is read in its own timezone. Prints a warning when rows are dropped.
    """
    if now is None:
        now = datetime.now(MARKET_TZ)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=MARKET_TZ)
    today = pd.Timestamp(now.astimezone(MARKET_TZ).date())

    index = pd.DatetimeIndex(df.index)
    dates = index.tz_localize(None) if index.tz is not None else index
    keep = dates.normalize() < today
    if keep.all():
        return df

    dropped = [d.date().isoformat() for d in dates[~keep]]
    prefix = f"{label}: " if label else ""
    print(f"Warning: {prefix}dropped unsettled bar(s) dated {', '.join(dropped)}.")
    return df.loc[keep]
