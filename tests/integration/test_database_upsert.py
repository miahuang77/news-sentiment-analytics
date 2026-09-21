"""Integration test: upsert idempotency against a real PostgreSQL database.

Requires DATABASE_URL in .env and sql/schema.sql already applied. Skipped
automatically if the database isn't reachable. No NewsAPI/network access
needed -- this only exercises src.database against Postgres.

Uses clearly-synthetic keys (ticker "ZZTEST", article_id prefixed
"unittest-") so it can't collide with real data, and cleans up the rows it
writes afterward.

Run with: pytest tests/integration/test_database_upsert.py
"""

import pandas as pd
import pytest
from sqlalchemy import text

from src.database import get_engine, upsert_news, upsert_prices, upsert_sentiment_daily
from src.database import test_connection as _check_db_connection

TEST_TICKER = "ZZTEST"
TEST_ARTICLE_ID = "unittest-database-upsert-article"


def _database_available():
    try:
        return _check_db_connection()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _database_available(), reason="DATABASE_URL not set or database unreachable"
)


@pytest.fixture(autouse=True)
def cleanup_test_rows():
    """Remove any synthetic rows this test writes, before and after."""

    def _delete():
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM prices WHERE ticker = :t"), {"t": TEST_TICKER})
            conn.execute(text("DELETE FROM sentiment_daily WHERE ticker = :t"), {"t": TEST_TICKER})
            conn.execute(text("DELETE FROM news WHERE article_id = :a"), {"a": TEST_ARTICLE_ID})

    _delete()
    yield
    _delete()


def _row_count(table, where_col, where_val):
    engine = get_engine()
    with engine.connect() as conn:
        return conn.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE {where_col} = :v"), {"v": where_val}
        ).scalar()


def test_upsert_prices_is_idempotent():
    df = pd.DataFrame(
        [
            {
                "ticker": TEST_TICKER,
                "date": pd.Timestamp("2024-01-02").date(),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000,
                "daily_return": None,
            }
        ]
    )

    first = upsert_prices(df)
    second = upsert_prices(df)  # identical rerun

    assert first == 1
    assert second == 1
    assert _row_count("prices", "ticker", TEST_TICKER) == 1  # no duplicate row


def test_upsert_prices_updates_existing_row_on_conflict():
    base = {
        "ticker": TEST_TICKER,
        "date": pd.Timestamp("2024-01-02").date(),
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 1000,
        "daily_return": None,
    }
    upsert_prices(pd.DataFrame([base]))
    updated = dict(base, close=99.0)
    upsert_prices(pd.DataFrame([updated]))

    engine = get_engine()
    with engine.connect() as conn:
        close = conn.execute(
            text("SELECT close FROM prices WHERE ticker = :t"), {"t": TEST_TICKER}
        ).scalar()
    assert close == pytest.approx(99.0)
    assert _row_count("prices", "ticker", TEST_TICKER) == 1


def test_upsert_news_is_idempotent_on_article_id():
    df = pd.DataFrame(
        [
            {
                "article_id": TEST_ARTICLE_ID,
                "ticker": TEST_TICKER,
                "title": "Synthetic test headline",
                "description": None,
                "source": "Unit Test",
                "published_at": pd.Timestamp("2024-01-02T12:00:00Z"),
                "sentiment_score": 0.1,
            }
        ]
    )

    upsert_news(df)
    upsert_news(df)  # rerun should not duplicate

    assert _row_count("news", "article_id", TEST_ARTICLE_ID) == 1


def test_upsert_sentiment_daily_is_idempotent():
    df = pd.DataFrame(
        [
            {
                "ticker": TEST_TICKER,
                "date": pd.Timestamp("2024-01-02").date(),
                "avg_sentiment": 0.25,
                "article_count": 3,
            }
        ]
    )

    upsert_sentiment_daily(df)
    upsert_sentiment_daily(df)

    assert _row_count("sentiment_daily", "ticker", TEST_TICKER) == 1
