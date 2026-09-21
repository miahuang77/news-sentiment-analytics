"""SQLAlchemy engine and price/news/sentiment_daily persistence for
PostgreSQL.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

_engine = None

_PRICE_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
]

_UPSERT_PRICES_SQL = text(
    """
    INSERT INTO prices (ticker, date, open, high, low, close, volume, daily_return)
    VALUES (:ticker, :date, :open, :high, :low, :close, :volume, :daily_return)
    ON CONFLICT (ticker, date) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        daily_return = EXCLUDED.daily_return
    """
)

_NEWS_COLUMNS = [
    "article_id",
    "ticker",
    "title",
    "description",
    "source",
    "published_at",
    "sentiment_score",
]

_UPSERT_NEWS_SQL = text(
    """
    INSERT INTO news (article_id, ticker, title, description, source, published_at, sentiment_score)
    VALUES (:article_id, :ticker, :title, :description, :source, :published_at, :sentiment_score)
    ON CONFLICT (article_id) DO UPDATE SET
        ticker = EXCLUDED.ticker,
        title = EXCLUDED.title,
        description = EXCLUDED.description,
        source = EXCLUDED.source,
        published_at = EXCLUDED.published_at,
        sentiment_score = EXCLUDED.sentiment_score
    """
)

_SENTIMENT_DAILY_COLUMNS = ["ticker", "date", "avg_sentiment", "article_count"]

_UPSERT_SENTIMENT_DAILY_SQL = text(
    """
    INSERT INTO sentiment_daily (ticker, date, avg_sentiment, article_count)
    VALUES (:ticker, :date, :avg_sentiment, :article_count)
    ON CONFLICT (ticker, date) DO UPDATE SET
        avg_sentiment = EXCLUDED.avg_sentiment,
        article_count = EXCLUDED.article_count
    """
)

_READ_NEWS_SENTIMENT_SQL = text(
    "SELECT ticker, published_at, sentiment_score FROM news "
    "WHERE sentiment_score IS NOT NULL"
)

_READ_PRICES_SQL = text(
    "SELECT ticker, date, close, daily_return FROM prices ORDER BY ticker, date"
)

_READ_SENTIMENT_DAILY_SQL = text(
    "SELECT ticker, date, avg_sentiment, article_count FROM sentiment_daily "
    "ORDER BY ticker, date"
)

_READ_NEWS_SQL = text(
    "SELECT ticker, title, source, published_at, sentiment_score FROM news "
    "WHERE (:ticker IS NULL OR ticker = :ticker) "
    "ORDER BY published_at DESC"
)


def get_engine():
    """Return a shared SQLAlchemy engine, creating it on first use.

    Returns:
        sqlalchemy.engine.Engine

    Raises:
        RuntimeError: DATABASE_URL is not set (copy .env.example to .env
            and fill it in).
    """
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
            )
        _engine = create_engine(DATABASE_URL)
    return _engine


def test_connection():
    """Run a trivial query to confirm the database is reachable.

    Returns:
        True if the connection succeeds.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: if the connection fails.
    """
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return True


def upsert_prices(df):
    """Insert or update rows of df into the prices table.

    Existing rows for the same (ticker, date) are overwritten, so re-running
    ingestion for a date range that's already loaded does not create
    duplicate rows.

    Args:
        df: DataFrame with columns ticker, date, open, high, low, close,
            volume, daily_return (e.g. from src.ingestion.prices.fetch_prices).

    Returns:
        Number of rows written.

    Raises:
        ValueError: df is missing one or more required columns.
    """
    missing = [col for col in _PRICE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    if df.empty:
        return 0

    # Swap NaN (e.g. daily_return on a ticker's first row) for None so it's
    # written as SQL NULL rather than the special float value 'NaN'. The
    # cast to object dtype is required: on a float64 column, pandas .where()
    # silently turns None back into NaN.
    prices = df[_PRICE_COLUMNS]
    clean = prices.astype(object).where(prices.notna(), None)
    records = clean.to_dict(orient="records")

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(_UPSERT_PRICES_SQL, records)

    return len(records)


def upsert_news(df):
    """Insert or update rows of df into the news table.

    Existing rows for the same article_id are overwritten, so re-running
    ingestion/scoring for articles already stored does not create duplicate
    rows. Only sentiment_score is stored -- positive_score, neutral_score,
    and negative_score (e.g. from src.sentiment.finbert.score_articles) are
    ignored if present.

    Args:
        df: DataFrame with columns article_id, ticker, title, description,
            source, published_at, sentiment_score (e.g. the output of
            src.sentiment.finbert.score_articles on
            src.ingestion.news.fetch_news article data).

    Returns:
        Number of rows written.

    Raises:
        ValueError: df is missing one or more required columns.
    """
    missing = [col for col in _NEWS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    if df.empty:
        return 0

    # Swap NaN for None so optional fields (description, sentiment_score)
    # are written as SQL NULL rather than the special float value 'NaN'.
    # The cast to object dtype is required: on a float64 column, pandas
    # .where() silently turns None back into NaN.
    news = df[_NEWS_COLUMNS]
    clean = news.astype(object).where(news.notna(), None)
    records = clean.to_dict(orient="records")

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(_UPSERT_NEWS_SQL, records)

    return len(records)


def read_news_sentiment():
    """Read all sentiment-scored articles from the news table.

    Returns:
        pandas.DataFrame with columns ticker, published_at, sentiment_score,
        one row per article that has a non-null sentiment_score.
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(_READ_NEWS_SENTIMENT_SQL, conn)


def upsert_sentiment_daily(df):
    """Insert or update rows of df into the sentiment_daily table.

    Existing rows for the same (ticker, date) are overwritten, so re-running
    the aggregation does not create duplicate rows.

    Args:
        df: DataFrame with columns ticker, date, avg_sentiment,
            article_count (e.g. from
            src.analysis.analysis.aggregate_daily_sentiment).

    Returns:
        Number of rows written.

    Raises:
        ValueError: df is missing one or more required columns.
    """
    missing = [col for col in _SENTIMENT_DAILY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    if df.empty:
        return 0

    daily = df[_SENTIMENT_DAILY_COLUMNS]
    clean = daily.astype(object).where(daily.notna(), None)
    records = clean.to_dict(orient="records")

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(_UPSERT_SENTIMENT_DAILY_SQL, records)

    return len(records)


def read_prices():
    """Read all price rows from the prices table.

    Returns:
        pandas.DataFrame with columns ticker, date, close, daily_return,
        one row per (ticker, date), ordered by ticker then date.
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(_READ_PRICES_SQL, conn)


def read_sentiment_daily():
    """Read all rows from the sentiment_daily table.

    Returns:
        pandas.DataFrame with columns ticker, date, avg_sentiment,
        article_count, one row per (ticker, date), ordered by ticker then
        date.
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(_READ_SENTIMENT_DAILY_SQL, conn)


def read_news(ticker=None):
    """Read article-level rows from the news table, most recent first.

    Args:
        ticker: If given, only articles for this ticker. If None, all
            tickers.

    Returns:
        pandas.DataFrame with columns ticker, title, source, published_at,
        sentiment_score, ordered by published_at descending.
    """
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(_READ_NEWS_SQL, conn, params={"ticker": ticker})


if __name__ == "__main__":
    # Local integration test: connection -> fetch -> upsert -> row count.
    try:
        test_connection()
        print("Database connection succeeded.")
    except Exception as e:
        print(f"Database connection failed: {e}")
    else:
        from src.ingestion.prices import fetch_prices

        df = fetch_prices(tickers="NVDA", start_date="2024-01-01", end_date="2024-01-31")
        rows = upsert_prices(df)
        print(f"Processed {rows} rows for NVDA.")
