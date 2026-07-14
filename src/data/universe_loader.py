from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import yfinance as yf

from src.data.universe_config import UniverseConfig
from src.data.universe_marketdata import UniverseMarketData


class UniverseDataLoader:
    PRICE_FIELDS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, config: UniverseConfig,
                 data_path: str | Path,
                 progress=False):
        self.config = config
        self.progress = progress
        self.data_path = Path(data_path)
        self.cache_dir = self.data_path / self.config.universe_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure pool directory exists
        self.pool_dir = self.data_path / "sp500_pool"
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        self.prices_path = self.cache_dir / "prices_daily.parquet"
        self.ticker_info_path = self.cache_dir / "ticker_info.parquet"
        self.group_info_path = self.cache_dir / "group_info.parquet"
        self.membership_path = self.cache_dir / "membership.parquet"
        self.manifest_path = self.cache_dir / "manifest.json"

    def load(self, force_download: bool = False, check_for_corruptions=True, start_after_nan: bool = True) -> UniverseMarketData:
        print(f"Loading universe {self.config.universe_name}...")
        if force_download or not self._cache_exists():
            umd = self._download_and_build()
            self._persist(umd)
        else:
            umd = self._load_from_disk()

            # Synchronize with config
            target_symbols = set(self.config.all_symbols())
            loaded_symbols = set(umd.tickers())

            to_remove = loaded_symbols - target_symbols
            to_add = target_symbols - loaded_symbols

            dirty = False

            if to_remove:
                print(f"Removing {len(to_remove)} tickers from cache that are no longer in config...")
                # Filter prices MultiIndex columns
                cols_to_keep = [c for c in umd.prices.columns if c[0] not in to_remove]
                umd.prices = umd.prices.loc[:, cols_to_keep].copy()
                dirty = True

            if to_add:
                print(f"Downloading {len(to_add)} new tickers found in config: {to_add}")
                try:
                    new_prices_raw = self._download_prices(list(to_add))
                except RuntimeError:
                    # Every to_add ticker failed to download (e.g. delisted symbols
                    # with no data). Warn and proceed with the cached universe rather
                    # than crashing the load — the fresh-build path already tolerates
                    # per-ticker failures the same way. Absent tickers reappear as
                    # to_add on the next load, but never abort it.
                    print(
                        f"Warning: none of the {len(to_add)} new ticker(s) "
                        f"{sorted(to_add)} returned data; keeping cached universe."
                    )
                else:
                    new_prices = self._normalize_prices(new_prices_raw)

                    # Merge new prices with existing prices, aligning on index (Date).
                    # On a partial failure _download_prices returns only the tickers
                    # that succeeded; the rest simply stay absent (no crash).
                    umd.prices = pd.concat([umd.prices, new_prices], axis=1).sort_index(axis=1)
                    dirty = True

            if dirty:
                # Rebuild metadata to match new price set and config
                umd.ticker_info = self._build_ticker_info()
                umd.group_info = self._build_group_info()
                umd.membership = self._build_membership()

                print("Persisting synchronized universe to disk...")
                self._persist(umd)
        if start_after_nan:
            # Remove head rows with any NaN in prices so start row has no NaNs at all
            prices = umd.prices
            # Create a boolean mask per row if any NaN present
            mask = prices.isna().any(axis=1)
            # Locate the first row where mask is False (i.e., no NaN in that row)
            first_valid_index = mask.idxmin() if not mask.all() else None
            if first_valid_index is not None:
                umd = UniverseMarketData(
                    prices=prices.loc[first_valid_index:],
                    ticker_info=umd.ticker_info,
                    group_info=umd.group_info,
                    membership=umd.membership,
                )
        if check_for_corruptions:
            for w in umd.qc_warnings():
                print(w)

        return umd

    def _cache_exists(self) -> bool:
        return (
            self.prices_path.exists()
            and self.ticker_info_path.exists()
            and self.group_info_path.exists()
            and self.membership_path.exists()
            and self.manifest_path.exists()
        )

    def _download_and_build(self) -> UniverseMarketData:
        print("_download_and_build...")
        symbols = self.config.all_symbols()
        raw = self._download_prices(symbols)
        print(raw.tail())
        prices = self._normalize_prices(raw)
        ticker_info = self._build_ticker_info()
        group_info = self._build_group_info()
        membership = self._build_membership()

        return UniverseMarketData(
            prices=prices,
            ticker_info=ticker_info,
            group_info=group_info,
            membership=membership,
        )

    def _download_prices(self, symbols: list[str]) -> pd.DataFrame:
        print("_download_prices...")
        frames = []
        d = self.config.loader_defaults

        for symbol in symbols:
            if self.progress:
                print(f"Downloading {symbol}...", end="", flush=True)
            x = yf.download(
                tickers=symbol,
                start=d.get("start_date"),
                end=d.get("end_date"),
                interval=d.get("interval", "1d"),
                auto_adjust=d.get("auto_adjust", True),
                actions=False,
                repair=d.get("repair", False),
                keepna=d.get("keepna", False),
                threads=False,
                progress=False,
            )
            if self.progress:
                print(" done.")

            if x is None or x.empty:
                print(f"Warning: no data for {symbol}")
                continue

            # if yfinance returns MultiIndex like (field, ticker), flatten to just field
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = x.columns.get_level_values(0)

            x.columns = pd.MultiIndex.from_product([[symbol], list(x.columns)])
            frames.append(x)

        if not frames:
            raise RuntimeError("All ticker downloads failed.")

        return pd.concat(frames, axis=1).sort_index(axis=1)

    def _download_prices_batch(self, symbols: list[str]) -> pd.DataFrame:
        d = self.config.loader_defaults
        df = yf.download(
            tickers=symbols,
            start=d.get("start_date", None),
            end=d.get("end_date", None),
            interval=d.get("interval", "1d"),
            auto_adjust=d.get("auto_adjust", True),
            actions=False,
            repair=d.get("repair", True),
            keepna=d.get("keepna", False),
            threads=d.get("threads", True),
            progress=False,
            group_by="column",
        )
        return df

    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise RuntimeError("yfinance returned an empty DataFrame.")

        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected MultiIndex columns from yfinance download.")

        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        fields = set(self.PRICE_FIELDS)

        if fields.issubset(lvl0):
            # (field, ticker) -> (ticker, field)
            df = df.swaplevel(0, 1, axis=1)
        elif fields.issubset(lvl1):
            # already (ticker, field)
            pass
        else:
            raise ValueError(
                f"Could not identify OHLCV field level. "
                f"level0={sorted(lvl0)}, level1={sorted(lvl1)}"
            )

        keep_cols = []
        for ticker in sorted(df.columns.get_level_values(0).unique()):
            for field in self.PRICE_FIELDS:
                col = (ticker, field)
                if col in df.columns:
                    keep_cols.append(col)

        df = df.loc[:, keep_cols].copy()

        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        df.index = pd.DatetimeIndex(idx).normalize()
        df.index.name = "Date"

        df = df.sort_index().sort_index(axis=1)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["ticker", "field"])
        df = df.astype("float64")

        return df

    def _build_ticker_info(self) -> pd.DataFrame:
        rows = []
        for ticker, meta in self.config.tickers.items():
            row = {"ticker": ticker}
            row.update(meta)
            rows.append(row)
        df = pd.DataFrame(rows).set_index("ticker").sort_index()
        return df

    def _build_group_info(self) -> pd.DataFrame:
        rows = []
        for group_id, meta in self.config.groups.items():
            rows.append(
                {
                    "group_id": group_id,
                    "parent": meta.get("parent"),
                    "type": meta.get("type"),
                    "label": meta.get("label"),
                    "proxy_etf": meta.get("proxy_etf"),
                    "benchmark": meta.get("benchmark"),
                    "risk_free": meta.get("risk_free"),
                }
            )
        df = pd.DataFrame(rows).set_index("group_id").sort_index()
        return df

    def _build_membership(self) -> pd.DataFrame:
        rows = []
        for group_id, meta in self.config.groups.items():
            for ticker in meta.get("members", []):
                rows.append({"group_id": group_id, "ticker": ticker})
        df = pd.DataFrame(rows).sort_values(["group_id", "ticker"]).reset_index(drop=True)
        return df

    def _persist(self, data: UniverseMarketData) -> None:
        print(f"_persist prices to {self.prices_path}...")
        print(f"_persist ticker_info to {self.ticker_info_path}...")
        print(f"_persist group_info to {self.group_info_path}...")
        print(f"_persist membership to {self.membership_path}...")
        data.prices.to_parquet(self.prices_path)
        data.ticker_info.to_parquet(self.ticker_info_path)
        data.group_info.to_parquet(self.group_info_path)
        data.membership.to_parquet(self.membership_path)

        manifest = {
            "universe_name": self.config.universe_name,
            "price_fields": self.PRICE_FIELDS,
            "cache_format": "parquet",
            "note": "Fresh download path always persists first and then reloads from disk.",
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_from_disk(self) -> UniverseMarketData:
        prices = pd.read_parquet(self.prices_path)
        ticker_info = pd.read_parquet(self.ticker_info_path)
        group_info = pd.read_parquet(self.group_info_path)
        membership = pd.read_parquet(self.membership_path)

        # re-apply deterministic normalization after read
        prices = self._normalize_loaded_prices(prices)
        ticker_info = ticker_info.sort_index()
        group_info = group_info.sort_index()
        membership = membership.sort_values(["group_id", "ticker"]).reset_index(drop=True)

        return UniverseMarketData(
            prices=prices,
            ticker_info=ticker_info,
            group_info=group_info,
            membership=membership,
        )

    def _normalize_loaded_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Cached prices must have MultiIndex columns.")

        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = pd.DatetimeIndex(idx).normalize()
        df.index = idx
        df.index.name = "Date"

        # enforce same column names/order
        df = df.sort_index()
        df = df.sort_index(axis=1)

        # rebuild column MultiIndex names consistently
        df.columns = pd.MultiIndex.from_tuples(
            [(str(a), str(b)) for a, b in df.columns],
            names=["ticker", "field"],
        )

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        return df

    def load_tickers_to_pool(self, symbols: list[str], force_download: bool = False) -> None:
        """
        Loads and persists individual tickers into the 'sp500_pool' directory.
        If download fails, it warns and doesn't persist.
        Generates a timestamped summary report of the operation.
        """
        from src.data.qc import check_prices_for_corruption

        # For pool downloads, we ignore config date constraints to get full history
        # but keep other flags like repair/auto_adjust from config if it exists
        if self.config and hasattr(self.config, 'loader_defaults'):
            d = self.config.loader_defaults
        else:
            d = {"auto_adjust": True, "repair": True}

        d = self.config.loader_defaults
        summary = {
            "skipped": [],
            "success": [],
            "failed": [],
            "corrupted": []
        }

        for symbol in symbols:
            target_path = self.pool_dir / f"{symbol}.parquet"

            if not force_download and target_path.exists():
                summary["skipped"].append(symbol)
                if self.progress:
                    print(f"Ticker {symbol} already exists in pool. Skipping.")
                continue

            if self.progress:
                print(f"Downloading {symbol} for pool...", end="", flush=True)

            try:
                # 1. Download single ticker
                df = yf.download(
                    tickers=symbol,
                    start=d.get("start_date"),
                    end=d.get("end_date"),
                    interval=d.get("interval", "1d"),
                    auto_adjust=d.get("auto_adjust", True),
                    actions=False,
                    repair=d.get("repair", False),
                    keepna=d.get("keepna", False),
                    threads=False,
                    progress=False,
                )

                if df is None or df.empty:
                    summary["failed"].append(f"{symbol} (No data returned from yfinance)")
                    print(f"\nWarning: No data found for {symbol}. Not persisting.")
                    continue

                # 2. Format and Normalize (identical to original logic)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Wrap in MultiIndex (Ticker, Field) to satisfy normalization/QC requirements
                df.columns = pd.MultiIndex.from_product([[symbol], list(df.columns)])

                # Use existing normalization to ensure identical data types and index cleaning
                df_normalized = self._normalize_prices(df)

                # 3. Check for Corruptions
                warnings = check_prices_for_corruption(df_normalized)
                if warnings:
                    issues = ", ".join([f"{w.issue}: {w.details}" for w in warnings])
                    summary["corrupted"].append(f"{symbol} ({issues})")

                # 4. Persist
                df_normalized.to_parquet(target_path)
                summary["success"].append(symbol)

                if self.progress:
                    print(" done and persisted.")

            except Exception as e:
                summary["failed"].append(f"{symbol} (Error: {str(e)})")
                print(f"\nError processing {symbol}: {e}")

        # Generate the report
        self._write_pool_summary(summary)

    def _write_pool_summary(self, summary: dict) -> None:
        """
        Writes an orderly summary text file with a unique timestamp to the pool directory.
        """
        now = pd.Timestamp.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        report_path = self.pool_dir / f"_pool_summary_{timestamp_str}.txt"

        lines = [
            "=== SP500 Pool Update Summary ===",
            f"Run Timestamp: {now}",
            f"Total Tickers Requested: {sum(len(v) for v in summary.values())}",
            "",
            f"SUCCESS - Downloaded and Persisted ({len(summary['success'])}):",
            f"  {', '.join(summary['success']) if summary['success'] else 'None'}",
            "",
            f"SKIPPED - File already exists ({len(summary['skipped'])}):",
            f"  {', '.join(summary['skipped']) if summary['skipped'] else 'None'}",
            "",
            f"FAILED - Errors or No Data ({len(summary['failed'])}):",
        ]
        lines.extend([f"  - {f}" for f in summary["failed"]])

        lines.append("")
        lines.append(f"CORRUPTIONS - QC Warnings Detected ({len(summary['corrupted'])}):")
        lines.extend([f"  - {c}" for c in summary["corrupted"]])

        report_path.write_text("\n".join(lines), encoding="utf-8")

        if self.progress:
            print(f"\nPool update complete. Summary report: {report_path.name}")
