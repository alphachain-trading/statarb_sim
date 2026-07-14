from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.data.universe_marketdata import UniverseMarketData


@dataclass
class GroupReturnBundle:
    group_id: str
    member_returns: pd.DataFrame      # cols = stock tickers
    proxy_returns: pd.Series          # name = proxy ticker
    benchmark_returns: pd.Series      # name = benchmark ticker
    aligned_returns: pd.DataFrame     # cols = [members..., proxy, benchmark]
    risk_free_returns: pd.Series      # name = RF ticker; raw annualized yield in % (e.g. 5.2 = 5.2%)

    @property
    def members(self) -> list[str]:
        return list(self.member_returns.columns)

    @property
    def proxy(self) -> str:
        return str(self.proxy_returns.name)

    @property
    def benchmark(self) -> str:
        return str(self.benchmark_returns.name)

    @property
    def risk_free(self) -> str:
        return str(self.risk_free_returns.name)


def compute_returns(
    prices: pd.DataFrame,
    method: str = "log",
) -> pd.DataFrame:
    """
    Compute returns from a price matrix with columns=tickers.

    Parameters
    ----------
    prices
        DataFrame of prices, index=date, columns=tickers.
    method
        'log' or 'pct'
    """
    if prices.empty:
        raise ValueError("prices is empty")

    prices = prices.sort_index().astype("float64")

    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "pct":
        rets = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method!r}")

    return rets


def build_group_return_bundle(
    umd: UniverseMarketData,
    group_id: str,
    field: str = "Close",
    return_method: str = "log",
    dropna: str = "any",   # "any", "all", "none"
) -> GroupReturnBundle:
    """
    Build aligned member/proxy/benchmark return series for one group.

    Parameters
    ----------
    umd
        Loaded UniverseMarketData
    group_id
        Group to extract, e.g. "utilities"
    field
        Usually "Close"
    return_method
        "log" or "pct"
    dropna
        "any"  -> keep only dates with no missing values anywhere
        "all"  -> drop only rows where all values are NaN
        "none" -> keep all rows
    """
    member_prices = umd.price_matrix(group_id=group_id, field=field)
    proxy_prices = umd.prices_for_proxy(group_id=group_id, field=field)
    benchmark_prices = umd.prices_for_benchmark(group_id=group_id, field=field)
    rf_prices = umd.prices_for_risk_free(group_id=group_id, field=field)

    if member_prices.empty:
        raise ValueError(f"No member prices found for group {group_id!r}")
    if proxy_prices.empty:
        raise ValueError(f"No proxy prices found for group {group_id!r}")
    if benchmark_prices.empty:
        raise ValueError(f"No benchmark prices found for group {group_id!r}")
    if rf_prices.empty:
        raise ValueError(f"No risk-free prices found for group {group_id!r}")

    member_returns = compute_returns(member_prices, method=return_method)
    proxy_returns_df = compute_returns(proxy_prices, method=return_method)
    benchmark_returns_df = compute_returns(benchmark_prices, method=return_method)
    # ^IRX is an annualized yield in %, not a price — store raw, forward-filled.
    # Conversion to daily return happens in causal_residuals at fit/apply time.
    rf_returns = rf_prices.iloc[:, 0].rename(rf_prices.columns[0]).ffill()

    proxy_returns = proxy_returns_df.iloc[:, 0].rename(proxy_returns_df.columns[0])
    benchmark_returns = benchmark_returns_df.iloc[:, 0].rename(benchmark_returns_df.columns[0])

    aligned = pd.concat(
        [member_returns, proxy_returns, benchmark_returns],
        axis=1,
    ).sort_index()

    if dropna == "any":
        aligned = aligned.dropna(how="any")
    elif dropna == "all":
        aligned = aligned.dropna(how="all")
    elif dropna == "none":
        pass
    else:
        raise ValueError(f"Unknown dropna mode: {dropna!r}")

    member_cols = list(member_returns.columns)
    proxy_col = str(proxy_returns.name)
    benchmark_col = str(benchmark_returns.name)

    return GroupReturnBundle(
        group_id=group_id,
        member_returns=aligned[member_cols].copy(),
        proxy_returns=aligned[proxy_col].copy(),
        benchmark_returns=aligned[benchmark_col].copy(),
        aligned_returns=aligned.copy(),
        risk_free_returns=rf_returns.reindex(aligned.index).ffill(),
    )