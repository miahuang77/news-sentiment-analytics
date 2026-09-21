"""Daily sentiment aggregation, and a first-pass sentiment/returns analysis.

The dashboard is deferred. This module has two parts:

1. Daily sentiment aggregation -- aggregates news.sentiment_score into one
   row per (ticker, date) for the sentiment_daily table.

   Date assumption (MVP): the calendar date used for aggregation is each
   article's published_at timestamp converted to UTC and truncated to a
   date (published_at.astimezone(UTC).date()). This does not align to
   market trading sessions -- e.g. a US-market article published at 11pm
   UTC (early evening US time) is grouped under the next UTC calendar
   date, which can differ from the trading day it was actually published
   on. More sophisticated market-session alignment (e.g. using exchange
   hours/timezone) is deferred to later work.

2. A first-pass analysis of whether daily news sentiment is *associated*
   with short-term returns and volatility: same-day correlation,
   next-trading-day correlation, and a forward-volatility comparison. This
   is purely correlational -- see IMPORTANT note below.

IMPORTANT -- correlation, not causation: everything in part 2 reports
association only. A nonzero correlation here does not mean sentiment
predicts or causes returns or volatility; it does not control for
confounders (e.g. news often follows price moves rather than the other
way around), and with the sample sizes available in this MVP dataset,
correlations are noisy. Treat this as descriptive, not predictive.
"""

import pandas as pd

from src.database import (
    read_news_sentiment,
    read_prices,
    read_sentiment_daily,
    upsert_sentiment_daily,
)

_OUTPUT_COLUMNS = ["ticker", "date", "avg_sentiment", "article_count"]

# Tickers covered by the first-pass analysis. SPY is excluded for now: its
# current news dataset is concentrated in only 3 dates, too few for any of
# the comparisons below to be meaningful.
FOCUS_TICKERS = ["AAPL", "MSFT", "NVDA"]

# Below this many observations, a Pearson correlation (or a group mean) is
# unstable enough to be misleading, so we report "insufficient data"
# instead of a number. This MVP dataset only has a few weeks of history per
# ticker, so 5 is a practical floor, not a statistically rigorous one --
# treat every correlation this module reports as illustrative, not
# significant.
MIN_OBSERVATIONS = 5

# Forward realized volatility is computed as the std dev of daily returns
# over this many trading days immediately following a given day.
FORWARD_VOLATILITY_WINDOW = 5


def aggregate_daily_sentiment(articles_df):
    """Aggregate article-level sentiment into one row per (ticker, date).

    Args:
        articles_df: DataFrame with columns ticker, published_at,
            sentiment_score (e.g. from src.database.read_news_sentiment()).

    Returns:
        DataFrame with columns ticker, date, avg_sentiment, article_count,
        one row per (ticker, date). date is the UTC calendar date derived
        from published_at -- see the module docstring for that assumption.
    """
    if articles_df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    df = articles_df.copy()
    df["date"] = pd.to_datetime(df["published_at"], utc=True).dt.date

    grouped = (
        df.groupby(["ticker", "date"])
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("sentiment_score", "count"),
        )
        .reset_index()
    )

    return grouped[_OUTPUT_COLUMNS]


def run_daily_sentiment_aggregation():
    """Read sentiment-scored articles, aggregate by ticker/date, and upsert
    the result into sentiment_daily.

    Returns:
        Number of (ticker, date) rows written.
    """
    articles_df = read_news_sentiment()
    daily_df = aggregate_daily_sentiment(articles_df)
    return upsert_sentiment_daily(daily_df)


# --- Sentiment vs. returns/volatility analysis --------------------------


def build_analysis_dataframe(prices_df, sentiment_df):
    """Match prices and sentiment_daily by (ticker, date).

    Args:
        prices_df: DataFrame with columns ticker, date, close, daily_return
            (e.g. from src.database.read_prices()).
        sentiment_df: DataFrame with columns ticker, date, avg_sentiment,
            article_count (e.g. from src.database.read_sentiment_daily()).

    Returns:
        DataFrame with columns ticker, date, close, daily_return,
        avg_sentiment, article_count. Only (ticker, date) pairs present in
        both inputs are included, so this only ever covers dates where
        sentiment data exists (and where a matching trading day exists).
    """
    merged = prices_df.merge(sentiment_df, on=["ticker", "date"], how="inner")
    return merged[["ticker", "date", "close", "daily_return", "avg_sentiment", "article_count"]]


def _add_forward_price_features(prices_df):
    """Add next_trading_day_return and forward_volatility_Nd columns to a
    full (all tickers, all trading days) prices DataFrame.

    Both are computed per ticker from that ticker's own chronological row
    order in prices_df -- i.e. "next" means the next row in the prices
    table for that ticker, NOT the next calendar day. prices_df should be
    the complete price history (not filtered to sentiment days), so gaps
    from days without news don't distort what "next trading day" means.
    """
    df = prices_df.sort_values(["ticker", "date"]).copy()
    daily_return_by_ticker = df.groupby("ticker")["daily_return"]

    df["next_trading_day_return"] = daily_return_by_ticker.shift(-1)
    # A plain rolling(w).std() at position i is backward-looking: std of the
    # w values ENDING at i. Shifting that trailing series back by w positions
    # realigns it so the value shown at position i is the std of the w
    # values immediately AFTER i, which is what "forward volatility" means
    # here. (An earlier version used shift(-1).rolling(w), which is wrong --
    # it mixes in 3 days of the past plus only 1 day ahead for w=5, not a
    # clean forward window.)
    df[f"forward_volatility_{FORWARD_VOLATILITY_WINDOW}d"] = daily_return_by_ticker.transform(
        lambda s: s.rolling(FORWARD_VOLATILITY_WINDOW).std().shift(-FORWARD_VOLATILITY_WINDOW)
    )
    return df


def _correlation_by_ticker(df, col_a, col_b, tickers):
    """Pearson correlation between col_a and col_b, computed separately per
    ticker, gated by MIN_OBSERVATIONS.

    Rows with a NaN in either column are dropped first (e.g. a ticker's
    first-ever price row has a NaN daily_return, since there's no prior
    close to compute a return from) -- Series.corr() would silently drop
    such pairs anyway, but only after this dropna is n guaranteed to match
    the number of pairs actually used, rather than overstating it.

    Returns:
        dict {ticker: {"n": int, "correlation": float or None}}. correlation
        is None (rather than a number) when n < MIN_OBSERVATIONS, so a
        misleadingly small-sample correlation is never reported as if it
        were reliable.
    """
    results = {}
    for ticker in tickers:
        subset = df[df["ticker"] == ticker].dropna(subset=[col_a, col_b])
        n = len(subset)
        correlation = subset[col_a].corr(subset[col_b]) if n >= MIN_OBSERVATIONS else None
        results[ticker] = {"n": n, "correlation": correlation}
    return results


def same_day_correlation(analysis_df, tickers=FOCUS_TICKERS):
    """Per-ticker Pearson correlation between avg_sentiment and the SAME
    day's daily_return.

    Returns:
        dict {ticker: {"n": int, "correlation": float or None}} -- see
        _correlation_by_ticker.
    """
    return _correlation_by_ticker(analysis_df, "avg_sentiment", "daily_return", tickers)


def next_trading_day_correlation(analysis_df, prices_df, tickers=FOCUS_TICKERS):
    """Per-ticker Pearson correlation between avg_sentiment and the NEXT
    trading day's return (next row in prices_df for that ticker, not the
    next calendar day).

    Args:
        analysis_df: e.g. from build_analysis_dataframe().
        prices_df: full price history (e.g. from src.database.read_prices()),
            used to determine each date's next trading day.
        tickers: tickers to report on.

    Returns:
        dict {ticker: {"n": int, "correlation": float or None}} -- see
        _correlation_by_ticker. A sentiment day with no next trading day yet
        in prices_df (i.e. the most recent date) is excluded.
    """
    featured = _add_forward_price_features(prices_df)
    merged = analysis_df.merge(
        featured[["ticker", "date", "next_trading_day_return"]],
        on=["ticker", "date"],
        how="inner",
    ).dropna(subset=["next_trading_day_return"])

    return _correlation_by_ticker(merged, "avg_sentiment", "next_trading_day_return", tickers)


def forward_volatility_comparison(analysis_df, prices_df, tickers=FOCUS_TICKERS):
    """Compare average forward realized volatility on negative- vs.
    non-negative-sentiment days, pooled across tickers.

    Forward volatility for a given day is the std dev of daily_return over
    the FORWARD_VOLATILITY_WINDOW trading days immediately following it
    (computed from prices_df's chronological ordering, not calendar days).

    Args:
        analysis_df: e.g. from build_analysis_dataframe().
        prices_df: full price history, used to compute forward volatility.
        tickers: tickers to include in the pooled comparison.

    Returns:
        dict with keys "negative" and "non_negative", each
        {"n": int, "mean_forward_volatility": float or None}. A group's
        mean is None (not a misleading number) when its n < MIN_OBSERVATIONS.
        Days at the end of a ticker's price history without
        FORWARD_VOLATILITY_WINDOW future trading days yet are excluded.
    """
    vol_col = f"forward_volatility_{FORWARD_VOLATILITY_WINDOW}d"
    featured = _add_forward_price_features(prices_df)

    merged = analysis_df.merge(
        featured[["ticker", "date", vol_col]], on=["ticker", "date"], how="inner"
    ).dropna(subset=[vol_col])
    merged = merged[merged["ticker"].isin(tickers)]

    negative = merged.loc[merged["avg_sentiment"] < 0, vol_col]
    non_negative = merged.loc[merged["avg_sentiment"] >= 0, vol_col]

    def _summarize(series):
        n = len(series)
        mean = series.mean() if n >= MIN_OBSERVATIONS else None
        return {"n": n, "mean_forward_volatility": mean}

    return {"negative": _summarize(negative), "non_negative": _summarize(non_negative)}


def run_sentiment_analysis(tickers=FOCUS_TICKERS):
    """Load prices and sentiment_daily from PostgreSQL and run the full
    first-pass analysis: same-day correlation, next-trading-day
    correlation, and forward-volatility comparison.

    Returns:
        dict with keys "analysis_df" (the merged DataFrame),
        "same_day", "next_day", "volatility" (each as returned by the
        corresponding function above).
    """
    prices_df = read_prices()
    sentiment_df = read_sentiment_daily()
    analysis_df = build_analysis_dataframe(prices_df, sentiment_df)

    return {
        "analysis_df": analysis_df,
        "same_day": same_day_correlation(analysis_df, tickers),
        "next_day": next_trading_day_correlation(analysis_df, prices_df, tickers),
        "volatility": forward_volatility_comparison(analysis_df, prices_df, tickers),
    }


# --- Report printing ------------------------------------------------------


def _print_correlation_table(title, results):
    print(title)
    print(f"  {'Ticker':<8}{'n':>5}   {'Correlation'}")
    for ticker, r in results.items():
        if r["correlation"] is None:
            value = f"insufficient data (n={r['n']}, need >= {MIN_OBSERVATIONS})"
        else:
            value = f"{r['correlation']:+.3f}"
        print(f"  {ticker:<8}{r['n']:>5}   {value}")
    print()


def _print_volatility_table(title, results):
    print(title)
    print(f"  {'Group':<16}{'n':>5}   {'Mean forward volatility'}")
    for label, key in [("Negative", "negative"), ("Non-negative", "non_negative")]:
        r = results[key]
        if r["mean_forward_volatility"] is None:
            value = f"insufficient data (n={r['n']}, need >= {MIN_OBSERVATIONS})"
        else:
            value = f"{r['mean_forward_volatility']:.4f}"
        print(f"  {label:<16}{r['n']:>5}   {value}")
    print()


if __name__ == "__main__":
    results = run_sentiment_analysis()

    focus_df = results["analysis_df"][results["analysis_df"]["ticker"].isin(FOCUS_TICKERS)]
    print(f"Tickers analyzed: {', '.join(FOCUS_TICKERS)} (SPY excluded, see module docstring)")
    print(f"Matched (ticker, date) rows with sentiment data: {len(focus_df)}\n")

    _print_correlation_table(
        "Same-day correlation (avg_sentiment vs. same-day daily_return):",
        results["same_day"],
    )
    _print_correlation_table(
        "Next-trading-day correlation (avg_sentiment vs. next trading day's return):",
        results["next_day"],
    )
    _print_volatility_table(
        f"Forward {FORWARD_VOLATILITY_WINDOW}-trading-day realized volatility, "
        "by same-day sentiment (pooled across tickers):",
        results["volatility"],
    )

    print(
        "Note: all results above are correlational only. They do not show "
        "that sentiment predicts or causes returns or volatility, and "
        "sample sizes in this MVP dataset are small -- treat as "
        "descriptive, not statistically significant."
    )
