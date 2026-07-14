import pandas as pd


def make_ranking_dates(
    dates: pd.Index,
    frequency: str = "ME",  # B (='business daily'), W-FRI, ME, YE
    min_history: int = 252,
) -> pd.DatetimeIndex:
    """
    :param frequency:   B (='business daily'), W-FRI, ME, YE
    """

    if len(dates) < min_history:
        raise ValueError("Not enough history for requested min_history.")

    s = pd.Series(index=pd.DatetimeIndex(dates), data=1).sort_index()
    rebals = s.resample(frequency).last().dropna().index

    first_allowed = dates[min_history - 1]
    rebals = rebals[rebals >= first_allowed]

    return pd.DatetimeIndex(rebals)
