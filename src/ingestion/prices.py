"""Fetch daily OHLCV price history from yfinance.

PostgreSQL persistence is deferred; this module only fetches and shapes data.
"""

import pandas as pd
import yfinance as yf

from src.ingestion._common import normalize_tickers

_OUTPUT_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
]


def fetch_prices(tickers=None, start_date=None, end_date=None):
    """Download daily OHLCV history and compute daily returns.

    Args:
        tickers: Ticker symbol or list of symbols. Defaults to
            ["AAPL", "MSFT", "NVDA", "SPY"].
        start_date: "YYYY-MM-DD" string, or None to use yfinance's default
            (max available history).
        end_date: "YYYY-MM-DD" string, or None to use yfinance's default
            (up to the most recent session).

    Returns:
        pandas.DataFrame with one row per (ticker, date), columns:
        ticker, date, open, high, low, close, volume, daily_return.
        daily_return is the pct change of *adjusted* close within each
        ticker (NaN on each ticker's first row). Sorted by ticker, date.

    Raises:
        ValueError: tickers is empty, or start_date/end_date isn't a
            parseable date, or start_date is after end_date.
        RuntimeError: no data could be fetched for any ticker.
    """
    tickers = normalize_tickers(tickers)
    _validate_date_range(start_date, end_date)

    frames = []
    for ticker in tickers:
        frame = _fetch_one(ticker, start_date, end_date)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No price data could be fetched for any of the requested tickers."
        )

    result = pd.concat(frames, ignore_index=True)
    result = result[_OUTPUT_COLUMNS]
    result = result.sort_values(["ticker", "date"]).reset_index(drop=True)
    return result


def _validate_date_range(start_date, end_date):
    """Validate that start_date/end_date are parseable and start <= end."""
    parsed_start = None
    parsed_end = None

    if start_date is not None:
        try:
            parsed_start = pd.to_datetime(start_date)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid start_date '{start_date}': {e}") from e

    if end_date is not None:
        try:
            parsed_end = pd.to_datetime(end_date)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid end_date '{end_date}': {e}") from e

    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise ValueError(
            f"start_date ({start_date}) must not be after end_date ({end_date})."
        )


def _fetch_one(ticker, start_date, end_date):
    """Download and shape OHLCV history for a single ticker.

    Returns None (after printing a warning) if the download fails or comes
    back empty, so a single bad ticker doesn't abort the whole batch.
    """
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"Skipping {ticker}: {e}")
        return None

    if raw is None or raw.empty:
        print(f"Skipping {ticker}: no data returned.")
        return None

    # yf.download() can return a MultiIndex column frame even for a single
    # ticker depending on version/args; flatten it to plain column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns.name = None

    df = raw.reset_index()
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df["daily_return"] = df["Adj Close"].pct_change()
    df["ticker"] = ticker

    return df[_OUTPUT_COLUMNS]


if __name__ == "__main__":
    df = fetch_prices(start_date="2024-01-01", end_date="2024-01-31")
    print(df.head(10))
    print(df.tail(10))
    print(df.dtypes)
