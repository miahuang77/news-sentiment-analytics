"""Unit tests for src.ingestion.news.

Covers deterministic article_id generation, company relevance filtering, and
article parsing. All NewsAPI HTTP calls are mocked -- no live network access
or NEWS_API_KEY required.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("NEWS_API_KEY", "unit-test-dummy-key")

from src.ingestion.news import (  # noqa: E402  (env var must be set first)
    _make_article_id,
    _mentions_company,
    _parse_article,
    fetch_news,
)


# --- Deterministic article_id --------------------------------------------


def test_make_article_id_is_deterministic():
    url = "https://example.com/some-article"
    assert _make_article_id(url) == _make_article_id(url)


def test_make_article_id_differs_by_url():
    assert _make_article_id("https://example.com/a") != _make_article_id("https://example.com/b")


def test_make_article_id_is_sha256_hex():
    article_id = _make_article_id("https://example.com/a")
    assert len(article_id) == 64
    int(article_id, 16)  # raises ValueError if not valid hex


# --- Company relevance filtering (_mentions_company) ----------------------


@pytest.mark.parametrize(
    "title,description,company,expected",
    [
        ("NVIDIA stock soars", "Shares jumped.", "NVIDIA", True),
        ("Nvidia unveils new chip", None, "NVIDIA", True),
        ("Some headline", "Analysts discuss Nvidia earnings.", "NVIDIA", True),
        ("Bank of Nova Scotia Sees Commercial Growth", "Digital strategy gains traction.", "NVIDIA", False),
        ("VideoProc Converter AI 8.12", "A video conversion tool update.", "NVIDIA", False),
        # Word-boundary check: "Nvidian" must not match "NVIDIA".
        ("Nvidian language reforms proposed", None, "NVIDIA", False),
        # Substring false-positive guard: "Apple" must not match inside "Pineapple".
        ("Pineapple sales rise", None, "Apple", False),
        ("Apple unveils new iPhone", None, "Apple", True),
    ],
)
def test_mentions_company(title, description, company, expected):
    row = {"title": title, "description": description}
    assert _mentions_company(row, company) is expected


# --- Article parsing (_parse_article) --------------------------------------


def test_parse_article_valid():
    article = {
        "source": {"name": "Reuters"},
        "title": "Apple unveils new product line",
        "description": "Apple announced several new products today.",
        "url": "https://example.com/1",
        "publishedAt": "2024-06-01T12:00:00Z",
    }
    row = _parse_article(article, "AAPL")
    assert row == {
        "article_id": _make_article_id("https://example.com/1"),
        "ticker": "AAPL",
        "title": "Apple unveils new product line",
        "description": "Apple announced several new products today.",
        "source": "Reuters",
        "published_at": "2024-06-01T12:00:00Z",
    }


@pytest.mark.parametrize(
    "missing_field",
    ["url", "title", "publishedAt"],
)
def test_parse_article_missing_required_field_returns_none(missing_field):
    article = {
        "source": {"name": "Reuters"},
        "title": "Some title",
        "description": "Some description.",
        "url": "https://example.com/1",
        "publishedAt": "2024-06-01T12:00:00Z",
    }
    article[missing_field] = None
    assert _parse_article(article, "AAPL") is None


def test_parse_article_missing_description_is_none():
    article = {
        "source": {"name": "Reuters"},
        "title": "Some title",
        "description": None,
        "url": "https://example.com/1",
        "publishedAt": "2024-06-01T12:00:00Z",
    }
    row = _parse_article(article, "AAPL")
    assert row["description"] is None


def test_parse_article_missing_source_is_none():
    article = {
        "title": "Some title",
        "description": "Some description.",
        "url": "https://example.com/1",
        "publishedAt": "2024-06-01T12:00:00Z",
    }
    row = _parse_article(article, "AAPL")
    assert row["source"] is None


# --- fetch_news, with requests.get mocked ----------------------------------


def _make_response(json_body):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_body
    return response


def test_fetch_news_deduplicates_same_article_across_tickers():
    """The same article_id, returned under two different ticker searches,
    should collapse to one row (article_id is the news table's primary key)."""
    article = {
        "source": {"name": "Reuters"},
        "title": "Apple and Microsoft announce joint venture",
        "description": "A deal involving Apple and Microsoft.",
        "url": "https://example.com/shared",
        "publishedAt": "2024-06-01T12:00:00Z",
    }
    with patch("src.ingestion.news.requests.get") as mock_get:
        mock_get.return_value = _make_response(
            {"status": "ok", "totalResults": 1, "articles": [article]}
        )
        df = fetch_news(tickers=["AAPL", "MSFT"])

    assert len(df) == 1
    assert df.iloc[0]["article_id"] == _make_article_id("https://example.com/shared")


def test_fetch_news_raises_when_no_ticker_yields_articles():
    with patch("src.ingestion.news.requests.get") as mock_get:
        mock_get.return_value = _make_response({"status": "ok", "totalResults": 0, "articles": []})
        with pytest.raises(RuntimeError):
            fetch_news(tickers="AAPL")
