"""Minimal integration test: NewsAPI ingestion -> FinBERT sentiment scoring.

Exercises src.ingestion.news.fetch_news() and src.sentiment.finbert
.score_articles() together, end to end, and prints the results. No
PostgreSQL involved.

Requires a real NEWS_API_KEY in .env and network access (NewsAPI, plus
Hugging Face to download ProsusAI/finbert on first run).

Run with: python -m tests.integration.test_news_sentiment_pipeline
"""

from src.ingestion.news import fetch_news
from src.sentiment.finbert import score_articles

TICKER = "NVDA"
ARTICLE_LIMIT = 5


def main():
    news_df = fetch_news(tickers=TICKER, page_size=ARTICLE_LIMIT)
    print(f"Fetched {len(news_df)} article(s) for {TICKER}.\n")

    scored_df = score_articles(news_df)

    for row in scored_df.itertuples():
        print(row.ticker)
        print(row.title)
        print(f"  positive_score:  {row.positive_score:.4f}")
        print(f"  neutral_score:   {row.neutral_score:.4f}")
        print(f"  negative_score:  {row.negative_score:.4f}")
        print(f"  sentiment_score: {row.sentiment_score:+.4f}")
        print()


if __name__ == "__main__":
    main()
