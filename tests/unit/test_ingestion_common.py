"""Unit tests for src.ingestion._common (shared by news.py and prices.py).

Pure logic, no network or database required.
"""

import pytest

from src.ingestion._common import DEFAULT_TICKERS, normalize_tickers


def test_normalize_tickers_none_returns_default():
    assert normalize_tickers(None) == DEFAULT_TICKERS


def test_normalize_tickers_none_returns_a_copy():
    # Callers shouldn't be able to mutate the shared default list.
    result = normalize_tickers(None)
    result.append("ZZZZ")
    assert normalize_tickers(None) == DEFAULT_TICKERS


def test_normalize_tickers_single_string_becomes_list():
    assert normalize_tickers("AAPL") == ["AAPL"]


def test_normalize_tickers_list_passthrough():
    assert normalize_tickers(["AAPL", "MSFT"]) == ["AAPL", "MSFT"]


def test_normalize_tickers_empty_list_raises():
    with pytest.raises(ValueError):
        normalize_tickers([])


def test_normalize_tickers_single_empty_string_does_not_raise():
    # A bare "" becomes the one-element list [""] -- non-empty as a list,
    # even though the ticker value itself is meaningless. The emptiness
    # check is list-length-based, not content-based; document that boundary
    # rather than let it silently change.
    assert normalize_tickers("") == [""]
