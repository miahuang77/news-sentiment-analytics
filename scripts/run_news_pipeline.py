"""Historical news ingestion pipeline.

For each tracked ticker: fetches company-specific, relevance-filtered news
over the trailing LOOKBACK_DAYS, scores it with FinBERT, and upserts it into
PostgreSQL. Finishes by refreshing the sentiment_daily aggregation and
printing final row totals.

This is pure orchestration -- it reuses the existing ingestion, sentiment,
database, and analysis functions rather than duplicating any of that logic:
    src.ingestion.news.fetch_news_with_stats  (company query + relevance filter)
    src.sentiment.finbert.score_articles
    src.database.upsert_news, get_engine
    src.analysis.analysis.run_daily_sentiment_aggregation

Run with: python -m scripts.run_news_pipeline
"""

import datetime as dt

from sqlalchemy import text

from src.analysis.analysis import run_daily_sentiment_aggregation
from src.database import get_engine, upsert_news
from src.ingestion.news import fetch_news_with_stats
from src.sentiment.finbert import score_articles

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
LOOKBACK_DAYS = 30
# NewsAPI's free tier caps a single request at 100 results -- ask for the
# max so as many relevant articles as are available survive the existing
# relevance filter, rather than an arbitrary small limit like 5.
PAGE_SIZE = 100


def main():
    end_date = dt.datetime.now(dt.timezone.utc).date()
    start_date = end_date - dt.timedelta(days=LOOKBACK_DAYS)
    print(f"Fetching news from {start_date} to {end_date} for {', '.join(TICKERS)}.\n")

    for ticker in TICKERS:
        _process_ticker(ticker, start_date.isoformat(), end_date.isoformat())

    print("Refreshing sentiment_daily...")
    daily_rows = run_daily_sentiment_aggregation()
    print(f"  sentiment_daily rows upserted: {daily_rows}\n")

    _print_totals()


def _process_ticker(ticker, start_date, end_date):
    """Fetch, score, and store one ticker's news, printing a summary line."""
    print(f"--- {ticker} ---")
    try:
        news_df, stats = fetch_news_with_stats(
            tickers=ticker, start_date=start_date, end_date=end_date, page_size=PAGE_SIZE
        )
    except RuntimeError as e:
        print(f"  skipped: {e}\n")
        return

    scored_df = score_articles(news_df)
    stored = upsert_news(scored_df)

    print(f"  articles fetched:          {stats[ticker]['fetched']}")
    print(f"  retained after filtering:  {stats[ticker]['retained']}")
    print(f"  articles processed/stored: {stored}\n")


def _print_totals():
    """Print final row counts across the news and sentiment_daily tables."""
    engine = get_engine()
    with engine.connect() as conn:
        news_total = conn.execute(text("SELECT COUNT(*) FROM news")).scalar()
        daily_total = conn.execute(text("SELECT COUNT(*) FROM sentiment_daily")).scalar()

    print(f"Total news rows: {news_total}")
    print(f"Total sentiment_daily rows: {daily_total}")


if __name__ == "__main__":
    main()
