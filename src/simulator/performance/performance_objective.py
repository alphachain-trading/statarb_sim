from __future__ import annotations

import numpy as np
import pandas as pd


def sortino_ratio(
    returns: pd.Series,
    annualization_factor: int = 252,
    mar: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio.

    Parameters
    ----------
    returns
        Daily returns series (net, as fractions e.g. 0.01 = 1%).
    annualization_factor
        Trading days per year. Default 252.
    mar
        Minimum acceptable return (daily). Default 0.0.

    Returns
    -------
    float
        Annualized Sortino ratio. Returns 0.0 if downside deviation is zero
        or fewer than 2 observations.
    """
    r = np.asarray(returns.dropna(), dtype=float)
    if len(r) < 2:
        return 0.0

    mean_excess = float(np.mean(r - mar)) * annualization_factor

    downside = r - mar
    downside_sq = downside[downside < 0.0] ** 2
    if len(downside_sq) == 0:
        return 0.0 if mean_excess <= 0.0 else float("inf")

    downside_dev = float(np.sqrt(np.mean(downside_sq))) * float(np.sqrt(annualization_factor))
    if downside_dev <= 0.0:
        return 0.0

    return mean_excess / downside_dev