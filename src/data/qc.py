from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class QCWarning:
    ticker: str
    issue: str
    details: str


def check_prices_for_corruption(
    prices: pd.DataFrame,
    extreme_return_threshold: float = 0.50,
    ignore_pre_inception: bool = True,
    outside_range_tolerance: float = 0.001,
) -> list[QCWarning]:
    """
    Check canonical prices DataFrame with columns MultiIndex (ticker, field).

    Returns a compact list of warnings, one per detected issue/ticker.
    """
    required_fields = {"Open", "High", "Low", "Close", "Volume"}

    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns: (ticker, field).")

    warnings: list[QCWarning] = []
    tickers = prices.columns.get_level_values(0).unique()

    for ticker in tickers:
        df = prices[ticker].copy()

        missing_fields = required_fields.difference(df.columns)
        if missing_fields:
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="missing_fields",
                    details=f"Missing fields: {sorted(missing_fields)}",
                )
            )
            continue

        if ignore_pre_inception:
            first_valid = df[["Open", "High", "Low", "Close"]].notna().any(axis=1)
            if first_valid.any():
                first_idx = first_valid.idxmax()
                df = df.loc[first_idx:].copy()

        o = df["Open"]
        h = df["High"]
        l = df["Low"]
        c = df["Close"]
        v = df["Volume"]

        if df[["Open", "High", "Low", "Close"]].isna().any(axis=None):
            n = int(df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum())
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="missing_ohlc",
                    details=f"{n} rows with missing OHLC values",
                )
            )

        if v.isna().any():
            n = int(v.isna().sum())
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="missing_volume",
                    details=f"{n} rows with missing Volume",
                )
            )

        bad = (h < l)
        if bad.any():
            n = int(bad.sum())
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="high_below_low",
                    details=f"{n} rows where High < Low",
                )
            )

        bad = (o > h) | (o < l)
        if bad.any():
            n = int(bad.sum())
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="open_outside_range",
                    details=f"{n} rows where Open is outside [Low, High]",
                )
            )

        # Close outside range: collect raw stats, but only report if "real" count > 0
        raw_bad = (c > h) | (c < l)
        if raw_bad.any():
            dev_up = (c / h - 1.0).where(c > h)
            dev_dn = (l / c - 1.0).where(c < l)
            dev = pd.concat([dev_up, dev_dn], axis=0).dropna()

            raw_count = int(raw_bad.sum())
            min_dev = float(dev.min()) if not dev.empty else 0.0
            mean_dev = float(dev.mean()) if not dev.empty else 0.0
            max_dev = float(dev.max()) if not dev.empty else 0.0

            real_bad = (c > h * (1.0 + outside_range_tolerance)) | (
                c < l * (1.0 - outside_range_tolerance)
            )
            real_count = int(real_bad.sum())

            if real_count > 0:
                warnings.append(
                    QCWarning(
                        ticker=ticker,
                        issue="close_outside_range",
                        details=(
                            f"{raw_count} raw rows outside [Low, High]; "
                            f"deviation max = "
                            f"{max_dev}; "
                            f"{real_count} rows exceed tolerance {outside_range_tolerance:.4%}"
                        ),
                    )
                )

        bad = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
        if bad.any():
            n = int(bad.sum())
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="nonpositive_price",
                    details=f"{n} rows with non-positive OHLC values",
                )
            )

        bad = v < 0
        if bad.any():
            n = int(bad.sum())
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="negative_volume",
                    details=f"{n} rows with negative Volume",
                )
            )

        rets = c.pct_change()
        bad = rets.abs() > extreme_return_threshold
        if bad.any():
            n = int(bad.sum())
            max_abs_ret = float(rets.abs().max())
            max_idx = rets.abs().idxmax()
            max_date = pd.Timestamp(max_idx).strftime("%Y.%m.%d")
            warnings.append(
                QCWarning(
                    ticker=ticker,
                    issue="extreme_return",
                    details=(
                        f"{n} rows with abs daily return {max_abs_ret:.1%} > {extreme_return_threshold:.0%} "
                        f"on {max_date}"
                    ),
                )
            )

    return warnings


def print_qc_warnings(warnings: list[QCWarning]) -> None:
    if not warnings:
        print("QC: no corruption warnings detected.")
        return

    print("QC warnings:")
    for w in warnings:
        print(f"- {w.ticker}: {w.issue} — {w.details}")