"""Fetch company-related financial news from NewsAPI.

FinBERT sentiment scoring and PostgreSQL persistence are deferred; this
module only fetches and shapes news articles.
"""

import hashlib
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv

from src.ingestion._common import normalize_tickers

load_dotenv()

NEWS_API_URL = "https://newsapi.org/v2/everything"

# Explicit NewsAPI search query per ticker. Never search on the bare ticker
# symbol -- it's ambiguous (e.g. "SPY" is also an ordinary English word) and
# NewsAPI's matching is loose enough that it lets through weakly related
# results.
COMPANY_QUERIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "SPY": '"S&P 500" OR "SPDR S&P 500 ETF"',
}

# For single-company tickers, an article is only kept if this name actually
# appears in its title or description (checked post-retrieval, since
# NewsAPI's own relevance matching isn't strict enough on its own -- see
# _mentions_company). Not applied to SPY, which isn't a single company.
COMPANY_NAME_FILTERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}

_OUTPUT_COLUMNS = [
    "article_id",
    "ticker",
    "title",
    "description",
    "source",
    "published_at",
]


def fetch_news(tickers=None, start_date=None, end_date=None, page_size=20):
    """Search NewsAPI for company-related news for one or more tickers.

    Args:
        tickers: Ticker symbol or list of symbols. Defaults to
            ["AAPL", "MSFT", "NVDA", "SPY"].
        start_date: "YYYY-MM-DD" string, or None for NewsAPI's default.
            Note NewsAPI's free tier only returns articles from roughly the
            last month regardless of this value.
        end_date: "YYYY-MM-DD" string, or None for NewsAPI's default (now).
        page_size: Max articles returned per ticker. For single-company
            tickers, more than this may be requested from NewsAPI internally
            so enough are left after the relevance filter (NewsAPI caps a
            single request at 100).

    Returns:
        pandas.DataFrame with columns:
        article_id, ticker, title, description, source, published_at.
        article_id is a sha256 hash of the article URL, so the same article
        always gets the same id and duplicate inserts can be avoided.
        For single-company tickers (see COMPANY_NAME_FILTERS), articles are
        also required to mention the company name in their title or
        description, to filter out NewsAPI matches that are only loosely
        related. Deduplicated by article_id; sorted by ticker, published_at.

    Raises:
        ValueError: tickers is empty.
        RuntimeError: NEWS_API_KEY is not set, or no articles could be
            fetched for any ticker.
    """
    df, _stats = _fetch_news_with_stats(tickers, start_date, end_date, page_size)
    return df


def fetch_news_with_stats(tickers=None, start_date=None, end_date=None, page_size=20):
    """Same as fetch_news(), but also returns per-ticker fetch/filter counts.

    Useful for pipeline-style callers that want to report, per ticker, how
    many articles NewsAPI returned versus how many survived filtering.

    Returns:
        (df, stats) where df is exactly what fetch_news() returns, and
        stats is a dict {ticker: {"fetched": int, "retained": int}}.
        fetched is the raw article count NewsAPI returned for that ticker,
        before required-field parsing or the relevance filter. retained is
        how many of those made it into df.

    Raises/Args: see fetch_news().
    """
    return _fetch_news_with_stats(tickers, start_date, end_date, page_size)


def _fetch_news_with_stats(tickers, start_date, end_date, page_size):
    """Shared implementation behind fetch_news() and fetch_news_with_stats()."""
    tickers = normalize_tickers(tickers)
    api_key = _get_api_key()

    rows = []
    stats = {}
    for ticker in tickers:
        ticker_rows, fetched_count = _fetch_one(ticker, api_key, start_date, end_date, page_size)
        rows.extend(ticker_rows)
        stats[ticker] = {"fetched": fetched_count, "retained": len(ticker_rows)}

    if not rows:
        raise RuntimeError(
            "No news articles could be fetched for any of the requested tickers."
        )

    df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    df = df.drop_duplicates(subset="article_id")
    df["published_at"] = pd.to_datetime(df["published_at"])
    df = df.sort_values(["ticker", "published_at"]).reset_index(drop=True)
    return df, stats


def _get_api_key():
    """Read the NewsAPI key from the environment."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEWS_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return api_key


def _fetch_one(ticker, api_key, start_date, end_date, page_size):
    """Fetch and parse articles for a single ticker.

    Returns (rows, fetched_count). rows is empty (after printing a warning)
    on a failed request, an API-reported error, an empty response,
    malformed JSON, or -- for a single-company ticker -- a response where
    no article actually mentions the company, so a problem with one ticker
    doesn't abort the whole batch. fetched_count is the raw number of
    articles NewsAPI returned before any filtering (0 if the request itself
    produced none).
    """
    query = COMPANY_QUERIES.get(ticker, ticker)
    company_name = COMPANY_NAME_FILTERS.get(ticker)

    # When a relevance filter applies, over-fetch and sort by relevancy
    # (rather than plain recency) so there's a large, well-matched candidate
    # pool left after filtering -- otherwise a small, recency-sorted page can
    # easily filter down to nothing. The final result is still sorted by
    # published_at, by fetch_news().
    request_size = min(page_size * 5, 100) if company_name else page_size
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy" if company_name else "publishedAt",
        "pageSize": request_size,
        "apiKey": api_key,
    }
    if start_date:
        params["from"] = start_date
    if end_date:
        params["to"] = end_date

    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Skipping {ticker}: request failed ({e}).")
        return [], 0
    except ValueError as e:
        print(f"Skipping {ticker}: invalid JSON response ({e}).")
        return [], 0

    if payload.get("status") != "ok":
        message = payload.get("message", "unknown error")
        print(f"Skipping {ticker}: NewsAPI error ({message}).")
        return [], 0

    articles = payload.get("articles") or []
    fetched_count = len(articles)
    if not articles:
        print(f"Skipping {ticker}: no articles returned.")
        return [], 0

    rows = [
        row for row in (_parse_article(a, ticker) for a in articles) if row is not None
    ]
    if not rows:
        print(f"Skipping {ticker}: all articles were missing required fields.")
        return [], fetched_count

    if company_name:
        relevant_rows = [r for r in rows if _mentions_company(r, company_name)]
        if not relevant_rows:
            print(
                f"Skipping {ticker}: no articles mentioned '{company_name}' "
                "in title or description."
            )
            return [], fetched_count
        rows = relevant_rows[:page_size]

    return rows, fetched_count


def _parse_article(article, ticker):
    """Shape one raw NewsAPI article into an output row.

    Returns None if the article is missing a required field (url, title, or
    publishedAt) rather than raising, so one bad article doesn't drop the
    whole batch for a ticker.
    """
    url = article.get("url")
    title = article.get("title")
    published_at = article.get("publishedAt")

    if not url or not title or not published_at:
        return None

    source = article.get("source") or {}
    source_name = source.get("name") if isinstance(source, dict) else None

    return {
        "article_id": _make_article_id(url),
        "ticker": ticker,
        "title": title,
        "description": article.get("description"),
        "source": source_name,
        "published_at": published_at,
    }


def _mentions_company(row, company_name):
    """Whether company_name appears as a whole word in the row's title or
    description (case-insensitive), e.g. "NVIDIA" matches "Nvidia" but not
    "Nvidian"."""
    haystack = f"{row['title']} {row['description'] or ''}"
    pattern = rf"\b{re.escape(company_name)}\b"
    return re.search(pattern, haystack, re.IGNORECASE) is not None


def _make_article_id(url):
    """Derive a deterministic article id from the article URL (sha256 hex)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    df = fetch_news(tickers=["AAPL"], page_size=5)
    print(df)
    print(df.dtypes)
