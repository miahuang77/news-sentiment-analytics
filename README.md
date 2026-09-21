# Financial News Sentiment Analytics

## ✨Inspiration

Financial markets always react continuously to new information, such as market value changes, product launches, and breaking news. The information contained in financial news is usually unstructured. In this project, I want to explore whether or not **financial news sentiment provide useful information about short-term stock returns and market volatility**. 


## 🔍Overview

An end-to-end analytics pipeline that combines market data and financial news, applies FinBERT sentiment analysis, stores structured results in PostgreSQL, and explores relationships between news sentiment, short-term stock returns, and realized volatility.  

How it works:

1️⃣ Collects financial news from **NewsAPI** and historical market data from **yfinance**
2️⃣ Filters articles to retain news relevant to each company
3️⃣ Uses **FinBERT** to convert each article into a sentiment score
4️⃣ Stores market, news, and sentiment data in **PostgreSQL**
5️⃣ Aggregates article-level sentiment into daily ticker-level features
6️⃣ Examines how sentiment relates to **same-day returns, next-trading-day returns, and forward realized volatility**
7️⃣ Presents the results in an interactive **Streamlit + Plotly dashboard**


## 💡Key Features


- **Financial news collection** — Pulls company-specific news from NewsAPI and filters out less relevant articles.
- **Market data pipeline** — Collects historical price and return data with yfinance.
- **FinBERT sentiment analysis** — Turns each news article into a financial sentiment score from negative to positive.
- **PostgreSQL storage** — Keeps price, news, and sentiment data organized and makes pipeline runs repeatable with idempotent upserts.
- **Market analysis** — Explores how news sentiment relates to same-day returns, next-trading-day returns, and short-term volatility.
- **Interactive dashboard** — Lets users explore different tickers, time periods, sentiment trends, market movements, and recent news in Streamlit.
- **Light & dark mode** — Takes into consideration accessibility needs.


## 📈Exploratory Findings

The current MVP uses a limited historical sample, so these results may not be statistically significant or predictive.

### Sentiment and Same-Day Returns

| Ticker | Observations | Pearson Correlation |
|---|---:|---:|
| AAPL | 11 | +0.016 |
| MSFT | 18 | +0.075 |
| NVDA | 14 | +0.370 |

AAPL and MSFT showed little contemporaneous linear relationship between daily sentiment and returns in this sample, while NVDA showed a moderately positive association.

### Sentiment and Next-Trading-Day Returns

| Ticker | Observations | Pearson Correlation |
|---|---:|---:|
| AAPL | 12 | +0.385 |
| MSFT | 18 | +0.159 |
| NVDA | 14 | -0.017 |

The relationships varied substantially across securities. Given the small sample sizes, these values are treated as exploratory observations rather than evidence of forecasting ability.

### Sentiment and Forward Volatility

Across AAPL, MSFT, and NVDA:

| Sentiment Group | Observations | Mean 5-Day Forward Volatility |
|---|---:|---:|
| Negative | 15 | 1.66% |
| Non-negative | 17 | 2.22% |

In this sample, negative-sentiment days were followed by lower average realized volatility. This result does not support a simple assumption that negative news is necessarily followed by greater short-term volatility and illustrates why empirical hypotheses should be tested rather than assumed.

## Data Model

The PostgreSQL database contains three primary tables.

### `prices`

Stores daily market data.

```text
ticker
date
open
high
low
close
volume
daily_return
```

Primary key:

```text
(ticker, date)
```

### `news`

Stores relevant financial news and FinBERT sentiment scores.

```text
article_id
ticker
title
description
source
published_at
sentiment_score
```

Primary key:

```text
article_id
```

### `sentiment_daily`

Stores aggregated daily sentiment.

```text
ticker
date
avg_sentiment
article_count
```

Primary key:

```text
(ticker, date)
```

## 🔧How I build this

| Area | Technology |
|---|---|
| Language | Python |
| Market Data | yfinance |
| News Data | NewsAPI |
| NLP | ProsusAI/FinBERT, Transformers, PyTorch |
| Data Processing | pandas, NumPy |
| Database | PostgreSQL |
| Database Access | SQLAlchemy, psycopg2 |
| Analytics | pandas / statistical analysis |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Testing | pytest |

## 🎥Project Structure

```text
financial-news-sentiment-analytics/
├── dashboard/
│   └── app.py
│
├── scripts/
│   ├── run_news_pipeline.py
│   └── run_prices_pipeline.py
│
├── sql/
│   └── schema.sql
│
├── src/
│   ├── ingestion/
│   │   ├── _common.py
│   │   ├── news.py
│   │   └── prices.py
│   │
│   ├── sentiment/
│   │   └── finbert.py
│   │
│   ├── analysis/
│   │   └── analysis.py
│   │
│   └── database.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 🧩Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd financial-news-sentiment-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Create a PostgreSQL database for the project and initialize the schema:

```bash
psql -d financial_sentiment -f sql/schema.sql
```

### 5. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then configure your local values:

```text
DATABASE_URL=your_postgresql_connection_string
NEWS_API_KEY=your_newsapi_key
```

Never commit `.env` or API credentials to version control.

## ⏩Running the Pipeline

### Market data

```bash
python -m scripts.run_prices_pipeline
```

### Financial news and sentiment

```bash
python -m scripts.run_news_pipeline
```

### Run the analysis

```bash
python -m src.analysis.analysis
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```


## 📌What I'd do next

Potential extensions include:

- market-session-aware news alignment
- longer historical coverage
- improved ETF and market-index news filtering
