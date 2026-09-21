"""Historical price ingestion pipeline.

For each tracked ticker, fetches daily OHLCV history for the fixed analysis
window (2026-08-21 through 2026-09-20 -- the same period backing the
historical news dataset, see scripts/run_news_pipeline.py) and upserts it
into PostgreSQL.

Pure orchestration -- reuses the existing ingestion and database functions
rather than duplicating any yfinance or database logic:
    src.ingestion.prices.fetch_prices
    src.database.upsert_prices, get_engine

Run with: python -m scripts.run_prices_pipeline
"""

from sqlalchemy import text

from src.database import get_engine, upsert_prices
from src.ingestion.prices import fetch_prices

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
START_DATE = "2026-08-21"
END_DATE = "2026-09-20"


def main():
    print(f"Fetching prices from {START_DATE} to {END_DATE} for {', '.join(TICKERS)}.\n")

    for ticker in TICKERS:
        _process_ticker(ticker)

    _print_total()


def _process_ticker(ticker):
    """Fetch and store one ticker's price history, printing a summary line."""
    print(f"--- {ticker} ---")
    try:
        df = fetch_prices(tickers=ticker, start_date=START_DATE, end_date=END_DATE)
    except RuntimeError as e:
        print(f"  skipped: {e}\n")
        return

    stored = upsert_prices(df)

    print(f"  rows fetched:          {len(df)}")
    print(f"  rows processed/stored: {stored}\n")


def _print_total():
    """Print the final row count across the prices table."""
    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM prices")).scalar()

    print(f"Total rows in prices: {total}")


if __name__ == "__main__":
    main()
