"""Unit tests for src.analysis.analysis.

All pure pandas/dict logic on synthetic data -- no database or network
required.
"""

import pandas as pd
import pytest

from src.analysis.analysis import (
    MIN_OBSERVATIONS,
    _add_forward_price_features,
    _correlation_by_ticker,
    aggregate_daily_sentiment,
    build_analysis_dataframe,
    forward_volatility_comparison,
    next_trading_day_correlation,
    same_day_correlation,
)


# --- aggregate_daily_sentiment ----------------------------------------------


def test_aggregate_daily_sentiment_groups_by_ticker_and_utc_date():
    articles = pd.DataFrame(
        [
            {"ticker": "AAPL", "published_at": "2024-06-01T10:00:00Z", "sentiment_score": 0.5},
            {"ticker": "AAPL", "published_at": "2024-06-01T20:00:00Z", "sentiment_score": -0.5},
            {"ticker": "AAPL", "published_at": "2024-06-02T10:00:00Z", "sentiment_score": 0.2},
            {"ticker": "MSFT", "published_at": "2024-06-01T10:00:00Z", "sentiment_score": 0.8},
        ]
    )
    result = aggregate_daily_sentiment(articles).set_index(["ticker", "date"])

    aapl_day1 = result.loc[("AAPL", pd.Timestamp("2024-06-01").date())]
    assert aapl_day1["avg_sentiment"] == pytest.approx(0.0)  # mean(0.5, -0.5)
    assert aapl_day1["article_count"] == 2

    aapl_day2 = result.loc[("AAPL", pd.Timestamp("2024-06-02").date())]
    assert aapl_day2["avg_sentiment"] == pytest.approx(0.2)
    assert aapl_day2["article_count"] == 1

    msft_day1 = result.loc[("MSFT", pd.Timestamp("2024-06-01").date())]
    assert msft_day1["avg_sentiment"] == pytest.approx(0.8)


def test_aggregate_daily_sentiment_converts_to_utc_before_bucketing():
    # 11:30pm US Eastern (UTC-4 in June) on June 1st is already June 2nd UTC.
    articles = pd.DataFrame(
        [{"ticker": "AAPL", "published_at": "2024-06-01T23:30:00-04:00", "sentiment_score": 0.1}]
    )
    result = aggregate_daily_sentiment(articles)
    assert result.iloc[0]["date"] == pd.Timestamp("2024-06-02").date()


def test_aggregate_daily_sentiment_empty_input():
    empty = pd.DataFrame(columns=["ticker", "published_at", "sentiment_score"])
    result = aggregate_daily_sentiment(empty)
    assert result.empty
    assert list(result.columns) == ["ticker", "date", "avg_sentiment", "article_count"]


# --- build_analysis_dataframe -----------------------------------------------


def test_build_analysis_dataframe_inner_joins_on_ticker_and_date():
    prices = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": pd.Timestamp("2024-06-01").date(), "close": 100.0, "daily_return": 0.01},
            {"ticker": "AAPL", "date": pd.Timestamp("2024-06-02").date(), "close": 101.0, "daily_return": 0.01},
        ]
    )
    sentiment = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": pd.Timestamp("2024-06-01").date(), "avg_sentiment": 0.5, "article_count": 2},
            # No matching price row for 2024-06-03 -- should be dropped.
            {"ticker": "AAPL", "date": pd.Timestamp("2024-06-03").date(), "avg_sentiment": -0.2, "article_count": 1},
        ]
    )
    result = build_analysis_dataframe(prices, sentiment)

    assert len(result) == 1
    assert result.iloc[0]["date"] == pd.Timestamp("2024-06-01").date()
    assert list(result.columns) == ["ticker", "date", "close", "daily_return", "avg_sentiment", "article_count"]


# --- Correlation gating (MIN_OBSERVATIONS) ----------------------------------


def _analysis_df(n, ticker="AAPL"):
    dates = pd.date_range("2024-06-03", periods=n, freq="B").date
    return pd.DataFrame(
        {
            "ticker": [ticker] * n,
            "date": dates,
            "avg_sentiment": [0.1 * i for i in range(n)],
            "daily_return": [0.01 * i for i in range(n)],
        }
    )


def test_correlation_below_min_observations_returns_none():
    df = _analysis_df(MIN_OBSERVATIONS - 1)
    result = _correlation_by_ticker(df, "avg_sentiment", "daily_return", ["AAPL"])
    assert result["AAPL"]["n"] == MIN_OBSERVATIONS - 1
    assert result["AAPL"]["correlation"] is None


def test_correlation_at_min_observations_is_computed():
    df = _analysis_df(MIN_OBSERVATIONS)
    result = _correlation_by_ticker(df, "avg_sentiment", "daily_return", ["AAPL"])
    assert result["AAPL"]["n"] == MIN_OBSERVATIONS
    # avg_sentiment and daily_return are both perfectly linearly increasing.
    assert result["AAPL"]["correlation"] == pytest.approx(1.0)


def test_correlation_drops_nan_pairs_before_counting():
    df = _analysis_df(MIN_OBSERVATIONS + 1)
    df.loc[0, "daily_return"] = float("nan")
    result = _correlation_by_ticker(df, "avg_sentiment", "daily_return", ["AAPL"])
    assert result["AAPL"]["n"] == MIN_OBSERVATIONS  # one row dropped for NaN


def test_correlation_absent_ticker_returns_zero_observations():
    df = _analysis_df(MIN_OBSERVATIONS)
    result = same_day_correlation(df, tickers=["NOTATICKER"])
    assert result["NOTATICKER"] == {"n": 0, "correlation": None}


# --- next_trading_day_correlation: skips weekends, not calendar days -------


def test_next_trading_day_return_skips_weekend_gap():
    # Fri, (weekend), Mon -- "next trading day" after Friday must be Monday's
    # return, not a calendar-adjacent (nonexistent) Saturday.
    prices = pd.DataFrame(
        [
            {"ticker": "AAPL", "date": pd.Timestamp("2024-06-07").date(), "close": 100.0, "daily_return": 0.0},
            {"ticker": "AAPL", "date": pd.Timestamp("2024-06-10").date(), "close": 105.0, "daily_return": 0.05},
            {"ticker": "AAPL", "date": pd.Timestamp("2024-06-11").date(), "close": 103.95, "daily_return": -0.01},
        ]
    )
    featured = _add_forward_price_features(prices)
    friday_row = featured[featured["date"] == pd.Timestamp("2024-06-07").date()].iloc[0]
    assert friday_row["next_trading_day_return"] == pytest.approx(0.05)


def test_next_trading_day_correlation_excludes_most_recent_date():
    # The last date in the price series has no "next" day yet.
    n = MIN_OBSERVATIONS
    prices = _analysis_df(n)[["ticker", "date"]].copy()
    prices["close"] = 100.0
    prices["daily_return"] = [0.01 * i for i in range(n)]
    analysis_df = _analysis_df(n)

    result = next_trading_day_correlation(analysis_df, prices, tickers=["AAPL"])
    assert result["AAPL"]["n"] == n - 1


# --- forward_volatility_comparison ------------------------------------------
# Regression coverage for a real bug caught during development: an earlier
# implementation used shift(-1).rolling(w), which actually averages in 3 days
# of the *past* plus only 1 day ahead (for w=5) -- not a clean forward
# window. The fix is rolling(w).std().shift(-w). These tests pin the correct
# forward-only behavior.


def test_forward_volatility_uses_only_future_returns():
    returns = [0.0, 0.10, -0.10, 0.10, -0.10, 0.10, 1000.0]  # last value must NOT leak backward
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL"] * len(returns),
            "date": pd.date_range("2024-06-03", periods=len(returns), freq="B").date,
            "daily_return": returns,
        }
    )
    featured = _add_forward_price_features(prices)

    # At row 0, the forward-5 window is rows 1..5 (0.10, -0.10, 0.10, -0.10, 0.10),
    # which must NOT include the huge value at row 6 or row 0's own return.
    row0_vol = featured.iloc[0]["forward_volatility_5d"]
    expected = pd.Series([0.10, -0.10, 0.10, -0.10, 0.10]).std()
    assert row0_vol == pytest.approx(expected)


def test_forward_volatility_comparison_splits_by_sentiment_sign():
    n = MIN_OBSERVATIONS * 2 + 6  # enough rows to have >= MIN_OBSERVATIONS forward-vol rows in each group
    dates = pd.date_range("2024-06-03", periods=n, freq="B").date
    prices = pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "date": dates,
            "daily_return": [0.01 if i % 2 == 0 else -0.01 for i in range(n)],
        }
    )
    # Alternate negative/non-negative sentiment per day.
    sentiment = pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "date": dates,
            "avg_sentiment": [-0.3 if i % 2 == 0 else 0.3 for i in range(n)],
            "article_count": [1] * n,
        }
    )
    analysis_df = build_analysis_dataframe(
        prices[["ticker", "date"]].assign(close=100.0, daily_return=prices["daily_return"]), sentiment
    )

    result = forward_volatility_comparison(analysis_df, prices, tickers=["AAPL"])
    assert result["negative"]["n"] > 0
    assert result["non_negative"]["n"] > 0
    # Every group's volatility should be the same, deterministic value given
    # the perfectly alternating +-0.01 return series.
    if result["negative"]["mean_forward_volatility"] is not None:
        assert result["negative"]["mean_forward_volatility"] > 0
