from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.qc import QCWarning, check_prices_for_corruption


@dataclass
class UniverseMarketData:
    prices: pd.DataFrame          # columns: MultiIndex[ticker, field]
    ticker_info: pd.DataFrame     # index: ticker
    group_info: pd.DataFrame      # index: group_id
    membership: pd.DataFrame      # columns: group_id, ticker
    _close_cache: pd.DataFrame | None = None

    def price_matrix(self, group_id: str | None = None, field='Close') -> pd.DataFrame:
        if field == 'Close' and group_id is None and self._close_cache is not None:
            return self._close_cache.copy()
        df = self.prices.xs(field, axis=1, level=1)
        if field == 'Close' and group_id is None:
            self._close_cache = df
        if group_id is None:
            return df.copy()
        cols = [c for c in self.members(group_id) if c in df.columns]
        return df.loc[:, cols].copy()

    def tickers(self) -> list[str]:
        return list(self.prices.columns.get_level_values(0).unique())

    def fields(self) -> list[str]:
        return list(self.prices.columns.get_level_values(1).unique())

    def members(self, group_id: str) -> list[str]:
        s = self.membership.loc[self.membership["group_id"] == group_id, "ticker"]
        return s.tolist()

    def _price_matrix_deprecated(self, group_id: str | None = None, field='Close') -> pd.DataFrame:
        df = self.prices.xs(field, axis=1, level=1)
        if group_id is None:
            return df.copy()
        cols = [c for c in self.members(group_id) if c in df.columns]
        return df.loc[:, cols].copy()

    def prices_for_group(self, group_id: str) -> pd.DataFrame:
        members = set(self.members(group_id))
        cols = self.prices.columns.get_level_values(0).isin(members)
        return self.prices.loc[:, cols].copy()

    def prices_for_ticker(self, ticker: str, field: str | None = "Close") -> pd.DataFrame:
        if field is None:
            return self.prices[ticker].copy()
        return self.prices.xs(field, axis=1, level=1)[[ticker]].copy()

    def prices_for_proxy(self, group_id: str, field: str | None = "Close") -> pd.DataFrame:
        proxy_ticker = self.group_info.loc[group_id, "proxy_etf"]
        return self.prices_for_ticker(proxy_ticker, field)

    def prices_for_benchmark(self, group_id: str, field: str | None = "Close") -> pd.DataFrame:
        benchmark_ticker = self.group_info.loc[group_id, "benchmark"]
        return self.prices_for_ticker(benchmark_ticker, field)

    def prices_for_risk_free(self, group_id: str, field: str | None = "Close") -> pd.DataFrame:
        rf_ticker = self.group_info.loc[group_id, "risk_free"]
        return self.prices_for_ticker(rf_ticker, field)

    def qc_warnings(self) -> list[QCWarning]:
        return check_prices_for_corruption(self.prices)

