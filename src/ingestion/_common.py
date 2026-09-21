"""Shared helpers for the ingestion modules (news, prices).

Kept here rather than duplicated: both src.ingestion.news.fetch_news() and
src.ingestion.prices.fetch_prices() accept the same tickers argument shape
and normalize it identically.
"""

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]


def normalize_tickers(tickers):
    """Normalize a tickers argument into a non-empty list of symbols.

    Args:
        tickers: None (-> DEFAULT_TICKERS), a single ticker string (-> a
            one-element list), or an iterable of ticker strings.

    Returns:
        A list of ticker symbols.

    Raises:
        ValueError: the resulting list is empty.
    """
    if tickers is None:
        return list(DEFAULT_TICKERS)
    if isinstance(tickers, str):
        tickers = [tickers]
    else:
        tickers = list(tickers)

    if not tickers:
        raise ValueError("tickers must not be empty.")
    return tickers
