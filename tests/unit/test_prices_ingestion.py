"""Unit tests for src.ingestion.prices.

Covers daily_return calculation and input validation. yfinance is mocked --
no live network access required.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.ingestion.prices import fetch_prices


def _fake_yf_download(closes):
    """Build a DataFrame shaped like what yf.download() returns for one
    ticker (auto_adjust=False): a DatetimeIndex named "Date" and Open/High/
    Low/Close/Adj Close/Volume columns."""
    dates = pd.date_range("2024-06-03", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=pd.Index(dates, name="Date"),
    )


def test_fetch_prices_daily_return_is_pct_change_of_adj_close():
    closes = [100.0, 102.0, 99.96, 101.9592]  # +2%, -2%, +2%
    with patch("src.ingestion.prices.yf.download", return_value=_fake_yf_download(closes)):
        df = fetch_prices(tickers="AAPL")

    assert pd.isna(df.iloc[0]["daily_return"])  # first row: no prior close
    assert df.iloc[1]["daily_return"] == pytest.approx(0.02)
    assert df.iloc[2]["daily_return"] == pytest.approx(-0.02)
    assert df.iloc[3]["daily_return"] == pytest.approx(0.02)


def test_fetch_prices_output_columns_and_ticker_label():
    with patch("src.ingestion.prices.yf.download", return_value=_fake_yf_download([100.0, 101.0])):
        df = fetch_prices(tickers="AAPL")

    assert list(df.columns) == [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
    ]
    assert (df["ticker"] == "AAPL").all()


def test_fetch_prices_multiple_tickers_sorted_by_ticker_then_date():
    def side_effect(ticker, **kwargs):
        return _fake_yf_download([100.0, 101.0])

    with patch("src.ingestion.prices.yf.download", side_effect=side_effect):
        df = fetch_prices(tickers=["MSFT", "AAPL"])

    assert list(df["ticker"]) == ["AAPL", "AAPL", "MSFT", "MSFT"]
    for _, group in df.groupby("ticker"):
        assert list(group["date"]) == sorted(group["date"])


def test_fetch_prices_skips_ticker_with_empty_response():
    def side_effect(ticker, **kwargs):
        if ticker == "BADTICKER":
            return pd.DataFrame()
        return _fake_yf_download([100.0, 101.0])

    with patch("src.ingestion.prices.yf.download", side_effect=side_effect):
        df = fetch_prices(tickers=["AAPL", "BADTICKER"])

    assert set(df["ticker"]) == {"AAPL"}


def test_fetch_prices_empty_tickers_raises_value_error():
    with pytest.raises(ValueError):
        fetch_prices(tickers=[])


def test_fetch_prices_invalid_date_raises_value_error():
    with pytest.raises(ValueError):
        fetch_prices(tickers="AAPL", start_date="not-a-date")


def test_fetch_prices_start_after_end_raises_value_error():
    with pytest.raises(ValueError):
        fetch_prices(tickers="AAPL", start_date="2024-06-10", end_date="2024-06-01")


def test_fetch_prices_all_tickers_fail_raises_runtime_error():
    with patch("src.ingestion.prices.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(RuntimeError):
            fetch_prices(tickers="AAPL")
