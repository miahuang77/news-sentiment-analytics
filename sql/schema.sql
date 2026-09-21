-- PostgreSQL schema for Financial News Sentiment Analytics (MVP).
--
-- Three tables:
--   prices          daily OHLCV bars + daily return, from yfinance
--   news            raw headlines from NewsAPI + FinBERT sentiment_score
--   sentiment_daily daily sentiment aggregated per ticker, for the dashboard

CREATE TABLE IF NOT EXISTS prices (
    ticker        VARCHAR(10)      NOT NULL,
    date          DATE             NOT NULL,
    open          DOUBLE PRECISION NOT NULL,
    high          DOUBLE PRECISION NOT NULL,
    low           DOUBLE PRECISION NOT NULL,
    close         DOUBLE PRECISION NOT NULL,
    volume        BIGINT           NOT NULL,
    daily_return  DOUBLE PRECISION,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS news (
    article_id       TEXT             NOT NULL,
    ticker           VARCHAR(10)      NOT NULL,
    title            TEXT             NOT NULL,
    description      TEXT,
    source           TEXT,
    published_at     TIMESTAMPTZ      NOT NULL,
    sentiment_score  DOUBLE PRECISION,
    PRIMARY KEY (article_id)
);

CREATE TABLE IF NOT EXISTS sentiment_daily (
    ticker         VARCHAR(10)      NOT NULL,
    date           DATE             NOT NULL,
    avg_sentiment  DOUBLE PRECISION NOT NULL,
    article_count  INTEGER          NOT NULL,
    PRIMARY KEY (ticker, date)
);
