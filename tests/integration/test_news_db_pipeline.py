"""Minimal integration test: NewsAPI ingestion -> FinBERT sentiment scoring
-> PostgreSQL persistence.

Exercises src.ingestion.news.fetch_news(), src.sentiment.finbert
.score_articles(), and src.database.upsert_news() together, end to end.

Requires a real NEWS_API_KEY and DATABASE_URL in .env, network access
(NewsAPI, plus Hugging Face to download ProsusAI/finbert on first run), and
sql/schema.sql already applied to the target database.

Run with: python -m tests.integration.test_news_db_pipeline
"""

from src.database import upsert_news
from src.ingestion.news import fetch_news
from src.sentiment.finbert import score_articles

TICKER = "NVDA"
ARTICLE_LIMIT = 5


def main():
    news_df = fetch_news(tickers=TICKER, page_size=ARTICLE_LIMIT)
    scored_df = score_articles(news_df)
    rows = upsert_news(scored_df)
    print(f"Processed {rows} rows for {TICKER}.")


if __name__ == "__main__":
    main()
