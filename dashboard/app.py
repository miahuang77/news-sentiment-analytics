"""Streamlit dashboard for Financial News Sentiment Analytics.

Run with: streamlit run dashboard/app.py

One page: pick a ticker + date range, see KPI cards, a price/sentiment
trend chart, a first-pass sentiment vs. returns/volatility analysis as
compact cards, and recent scored headlines as news cards. A light/dark
theme toggle (persisted in st.session_state) restyles the page and the
Plotly chart together.

This file is presentation only -- it reuses src.analysis.analysis and
src.database exactly as before; no calculation, pipeline, or schema logic
lives here.
"""

import html
import sys
from pathlib import Path

# `streamlit run dashboard/app.py` only puts this file's own folder
# (dashboard/) on sys.path, not the project root -- so `src` isn't
# importable by default. Add the project root (this file's parent's
# parent) before importing anything from src.*. This must run before the
# src imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis.analysis import (
    FORWARD_VOLATILITY_WINDOW,
    MIN_OBSERVATIONS,
    build_analysis_dataframe,
    forward_volatility_comparison,
    next_trading_day_correlation,
    same_day_correlation,
)
from src.database import read_news, read_prices, read_sentiment_daily

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]

# Sentiment badge thresholds -- a small deadband around zero so scores
# very close to neutral aren't over-classified as positive/negative.
SENTIMENT_POSITIVE_THRESHOLD = 0.05
SENTIMENT_NEGATIVE_THRESHOLD = -0.05

SPY_WARNING = (
    "SPY's current news sample is concentrated in only a few dates. "
    "Treat the correlation and volatility figures below as illustrative, "
    "not reliable. There isn't enough data yet for SPY specifically."
)

# --- Theme palettes -------------------------------------------------------
# Validated light/dark pair (dataviz skill references/palette.md): light
# chart surface #fcfcfb / dark #1a1a19, categorical slot 1 (price) and slot
# 8 (negative pole) for light/dark, diverging blue<->red for sentiment
# polarity. Badge tints are hand-picked light/dark washes of the same hues.
THEMES = {
    "light": {
        "bg": "#f9f9f7",
        "surface": "#ffffff",
        "card_border": "rgba(11,11,11,0.10)",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "gridline": "#e1e0d9",
        "chart_surface": "#fcfcfb",
        "price": "#2a78d6",
        "positive": "#2a78d6",
        "negative": "#e34948",
        "badge_positive_bg": "#e8f1fb",
        "badge_negative_bg": "#fceaea",
        "badge_neutral_bg": "#eeeeec",
    },
    "dark": {
        "bg": "#0d0d0d",
        "surface": "#1f1f1e",
        "card_border": "rgba(255,255,255,0.12)",
        "text_primary": "#f5f5f3",
        "text_secondary": "#c3c2b7",
        "muted": "#93918a",
        "gridline": "#33322f",
        "chart_surface": "#1a1a19",
        "price": "#3987e5",
        "positive": "#3987e5",
        "negative": "#e66767",
        "badge_positive_bg": "#1c3a5e",
        "badge_negative_bg": "#4a2323",
        "badge_neutral_bg": "#2c2c2a",
    },
}

st.set_page_config(page_title="Financial News Sentiment Analytics", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


def inject_theme_css(t):
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {t['bg']};
            --surface: {t['surface']};
            --card-border: {t['card_border']};
            --text-primary: {t['text_primary']};
            --text-secondary: {t['text_secondary']};
            --muted: {t['muted']};
            --positive: {t['positive']};
            --negative: {t['negative']};
            --badge-positive-bg: {t['badge_positive_bg']};
            --badge-negative-bg: {t['badge_negative_bg']};
            --badge-neutral-bg: {t['badge_neutral_bg']};
        }}

        .stApp {{
            background-color: var(--bg) !important;
            color: var(--text-primary);
        }}
        [data-testid="stHeader"] {{
            background-color: var(--bg) !important;
        }}
        [data-testid="stMainBlockContainer"] {{
            padding-top: 2rem;
            max-width: 1180px;
        }}
        [data-testid="stCaptionContainer"] p,
        [data-testid="stMarkdownContainer"] p {{
            color: var(--text-secondary);
        }}

        /* Native widgets: give them explicit theme-aware chrome */
        [data-baseweb="select"] > div {{
            background-color: var(--surface) !important;
            border-color: var(--card-border) !important;
            color: var(--text-primary) !important;
        }}
        [data-testid="stDateInputField"] {{
            background-color: var(--surface) !important;
            border-color: var(--card-border) !important;
            color: var(--text-primary) !important;
        }}
        [data-testid="stExpander"] {{
            background-color: var(--surface);
            border: 1px solid var(--card-border);
            border-radius: 10px;
        }}
        [data-testid="stExpander"] summary {{
            color: var(--text-primary);
        }}
        /* Streamlit hardcodes a light background on an OPEN expander's
        summary bar via an auto-generated class, regardless of theme --
        override it with the stable `details[open]` selector rather than
        that unstable generated class name. */
        [data-testid="stExpander"] details[open] > summary {{
            background-color: var(--surface) !important;
            color: var(--text-primary) !important;
        }}

        /* Header */
        .app-title {{
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0 0 0.15rem 0;
        }}
        .app-subtitle {{
            font-size: 1rem;
            color: var(--text-secondary);
            margin: 0 0 0.2rem 0;
        }}
        .app-description {{
            font-size: 0.82rem;
            color: var(--muted);
            margin: 0;
        }}

        .filter-label {{
            font-size: 0.72rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 0 0 0.25rem 0;
        }}

        .section-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 1.7rem 0 0.7rem 0;
        }}

        /* Cards (KPI / analysis / news) */
        .kpi-card, .analysis-card, .news-card {{
            background-color: var(--surface);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 0.95rem 1.05rem;
            margin-bottom: 0.7rem;
        }}
        .kpi-label {{
            font-size: 0.74rem;
            color: var(--muted);
            margin-bottom: 0.3rem;
        }}
        .kpi-value {{
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.15;
        }}
        .kpi-sub {{
            font-size: 0.78rem;
            color: var(--muted);
            margin-top: 0.4rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.66rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }}
        .badge-positive {{ background-color: var(--badge-positive-bg); color: var(--positive); }}
        .badge-negative {{ background-color: var(--badge-negative-bg); color: var(--negative); }}
        .badge-neutral  {{ background-color: var(--badge-neutral-bg);  color: var(--text-secondary); }}

        .analysis-title {{
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.1rem;
        }}
        .analysis-subtitle {{
            font-size: 0.72rem;
            color: var(--muted);
            margin-bottom: 0.55rem;
        }}
        .analysis-value {{
            font-size: 1.35rem;
            font-weight: 700;
        }}
        .analysis-n {{
            font-size: 0.76rem;
            color: var(--muted);
            margin-top: 0.15rem;
        }}
        .insufficient {{
            font-size: 0.82rem;
            color: var(--muted);
            font-style: italic;
        }}

        .news-title {{
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0.4rem 0 0.3rem 0;
            line-height: 1.35;
        }}
        .news-meta {{
            font-size: 0.75rem;
            color: var(--muted);
        }}
        .news-score {{
            font-size: 0.78rem;
            color: var(--muted);
            margin-left: 0.5rem;
        }}

        .disclaimer {{
            font-size: 0.76rem;
            color: var(--muted);
            margin: 0.3rem 0 0.6rem 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_prices():
    return read_prices()


@st.cache_data(ttl=300)
def load_sentiment_daily():
    return read_sentiment_daily()


@st.cache_data(ttl=300)
def load_news(ticker):
    return read_news(ticker=ticker)


def _date_bounds(*frames_and_cols):
    """Min/max date across one or more (DataFrame, date_column) pairs."""
    mins, maxes = [], []
    for df, col in frames_and_cols:
        if not df.empty:
            mins.append(pd.to_datetime(df[col]).min())
            maxes.append(pd.to_datetime(df[col]).max())
    if not mins:
        return None, None
    return min(mins).date(), max(maxes).date()


def classify_sentiment(score):
    """Positive/negative/neutral label for a sentiment score, using a small
    deadband around zero (SENTIMENT_POSITIVE_THRESHOLD /
    SENTIMENT_NEGATIVE_THRESHOLD)."""
    if pd.isna(score):
        return "neutral"
    if score > SENTIMENT_POSITIVE_THRESHOLD:
        return "positive"
    if score < SENTIMENT_NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def _kpi_card(label, value_html, sub_html=""):
    sub = f'<div class="kpi-sub">{sub_html}</div>' if sub_html else ""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{html.escape(label)}</div>
          <div class="kpi-value">{value_html}</div>
          {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _badge_html(kind):
    label = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}[kind]
    return f'<span class="badge badge-{kind}">{label}</span>'


def _correlation_card(title, subtitle, result):
    st.markdown(
        f"""
        <div class="analysis-card">
          <div class="analysis-title">{html.escape(title)}</div>
          <div class="analysis-subtitle">{html.escape(subtitle)}</div>
          {_correlation_body(result)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _correlation_body(result):
    if result["correlation"] is None:
        return (
            f'<div class="insufficient">Insufficient data '
            f'(n={result["n"]}, need &ge; {MIN_OBSERVATIONS})</div>'
        )
    kind = "positive" if result["correlation"] >= 0 else "negative"
    return (
        f'<div class="analysis-value" style="color:var(--{kind})">{result["correlation"]:+.3f}</div>'
        f'<div class="analysis-n">n = {result["n"]} observations</div>'
    )


def _volatility_group_html(label, result):
    if result["mean_forward_volatility"] is None:
        body = f'<div class="insufficient">Insufficient data (n={result["n"]})</div>'
    else:
        body = (
            f'<div class="analysis-value" style="font-size:1.15rem;">'
            f'{result["mean_forward_volatility"]:.4f}</div>'
            f'<div class="analysis-n">n = {result["n"]}</div>'
        )
    return f'<div style="flex:1;"><div class="kpi-label">{html.escape(label)}</div>{body}</div>'


def _volatility_card(volatility):
    st.markdown(
        f"""
        <div class="analysis-card">
          <div class="analysis-title">Forward {FORWARD_VOLATILITY_WINDOW}-trading-day realized volatility</div>
          <div class="analysis-subtitle">By same-day sentiment</div>
          <div style="display:flex; gap:1.5rem;">
            {_volatility_group_html("Negative sentiment days", volatility["negative"])}
            {_volatility_group_html("Non-negative sentiment days", volatility["non_negative"])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _news_card(row):
    kind = classify_sentiment(row["sentiment_score"])
    score_str = f"{row['sentiment_score']:+.3f}" if pd.notna(row["sentiment_score"]) else "n/a"
    published = row["published_at"].strftime("%Y-%m-%d %H:%M UTC")
    source = html.escape(row["source"]) if pd.notna(row["source"]) and row["source"] else "Unknown source"
    title = html.escape(str(row["title"]))
    st.markdown(
        f"""
        <div class="news-card">
          <div>{_badge_html(kind)}<span class="news-score">{score_str}</span></div>
          <div class="news-title">{title}</div>
          <div class="news-meta">{source} &middot; {published}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    theme = THEMES["dark" if st.session_state.dark_mode else "light"]
    inject_theme_css(theme)

    header_col, toggle_col = st.columns([5, 1])
    with header_col:
        st.markdown(
            """
            <div class="app-title">Financial News Sentiment Analytics</div>
            <div class="app-subtitle">News sentiment (FinBERT) vs. price action</div>
            <div class="app-description">An exploratory analysis of financial news sentiment,
            short-term returns, and market volatility.</div>
            """,
            unsafe_allow_html=True,
        )
    with toggle_col:
        st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
        st.toggle("Dark mode", key="dark_mode")

    try:
        prices = load_prices()
        sentiment_daily = load_sentiment_daily()
    except RuntimeError as e:
        st.error(f"Could not connect to the database: {e}")
        st.stop()

    if prices.empty and sentiment_daily.empty:
        st.info("No data yet. Run the ingestion pipelines under scripts/ first.")
        st.stop()

    # --- Filters ------------------------------------------------------------
    filter_ticker_col, filter_date_col = st.columns([1, 3])
    with filter_ticker_col:
        st.markdown('<div class="filter-label">Ticker</div>', unsafe_allow_html=True)
        ticker = st.selectbox("Ticker", TICKERS, label_visibility="collapsed")

    ticker_prices = prices[prices["ticker"] == ticker].sort_values("date")
    ticker_sentiment = sentiment_daily[sentiment_daily["ticker"] == ticker].sort_values("date")

    min_date, max_date = _date_bounds((ticker_prices, "date"), (ticker_sentiment, "date"))
    if min_date is None:
        st.info(f"No price or sentiment data available yet for {ticker}.")
        st.stop()

    with filter_date_col:
        st.markdown('<div class="filter-label">Date range</div>', unsafe_allow_html=True)
        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            label_visibility="collapsed",
        )
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.stop()  # user is still picking the second date
    start_date, end_date = date_range

    if ticker == "SPY":
        st.warning(SPY_WARNING)

    price_ranged = ticker_prices[
        (ticker_prices["date"] >= start_date) & (ticker_prices["date"] <= end_date)
    ]
    sentiment_ranged = ticker_sentiment[
        (ticker_sentiment["date"] >= start_date) & (ticker_sentiment["date"] <= end_date)
    ]
    news_ranged = load_news(ticker)
    news_ranged = news_ranged[
        (news_ranged["published_at"].dt.date >= start_date)
        & (news_ranged["published_at"].dt.date <= end_date)
    ]

    # --- KPI cards ------------------------------------------------------------
    st.markdown('<div class="section-title">Key stats (selected period)</div>', unsafe_allow_html=True)
    kpi_cols = st.columns(4)

    with kpi_cols[0]:
        if not price_ranged.empty:
            _kpi_card("Latest Price", f"${price_ranged.iloc[-1]['close']:.2f}")
        else:
            _kpi_card("Latest Price", "n/a")

    with kpi_cols[1]:
        if not price_ranged.empty and pd.notna(price_ranged.iloc[-1]["daily_return"]):
            ret = price_ranged.iloc[-1]["daily_return"]
            kind = "positive" if ret >= 0 else "negative"
            _kpi_card("Latest Return", f'<span style="color:var(--{kind})">{ret:+.2%}</span>')
        else:
            _kpi_card("Latest Return", "n/a")

    with kpi_cols[2]:
        if not sentiment_ranged.empty:
            score = sentiment_ranged.iloc[-1]["avg_sentiment"]
            kind = classify_sentiment(score)
            _kpi_card("Latest Sentiment", f"{score:+.3f}", _badge_html(kind))
        else:
            _kpi_card("Latest Sentiment", "n/a")

    with kpi_cols[3]:
        _kpi_card("News Articles", f"{len(news_ranged)}")

    # --- Main chart: price dominant, sentiment secondary, stacked panels ----
    st.markdown('<div class="section-title">Price &amp; News Sentiment</div>', unsafe_allow_html=True)
    if price_ranged.empty and sentiment_ranged.empty:
        st.info("No price or sentiment data in the selected date range.")
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.68, 0.32],
            vertical_spacing=0.08,
            subplot_titles=(f"{ticker} closing price", "Daily average sentiment"),
        )

        fig.add_trace(
            go.Scatter(
                x=price_ranged["date"],
                y=price_ranged["close"],
                mode="lines",
                name="Close price",
                showlegend=False,
                line=dict(color=theme["price"], width=3),
                hovertemplate="%{x|%Y-%m-%d}<br>Close: $%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        negative = sentiment_ranged[sentiment_ranged["avg_sentiment"] < 0]
        non_negative = sentiment_ranged[sentiment_ranged["avg_sentiment"] >= 0]
        fig.add_trace(
            go.Bar(
                x=negative["date"],
                y=negative["avg_sentiment"],
                name="Negative sentiment day",
                marker_color=theme["negative"],
                opacity=0.9,
                hovertemplate="%{x|%Y-%m-%d}<br>Avg sentiment: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=non_negative["date"],
                y=non_negative["avg_sentiment"],
                name="Non-negative sentiment day",
                marker_color=theme["positive"],
                opacity=0.9,
                hovertemplate="%{x|%Y-%m-%d}<br>Avg sentiment: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_width=1, line_color=theme["muted"], row=2, col=1)

        fig.update_yaxes(
            title_text="Close ($)", gridcolor=theme["gridline"], color=theme["text_secondary"], row=1, col=1
        )
        fig.update_yaxes(
            title_text="Avg. sentiment (-1 to +1)",
            gridcolor=theme["gridline"],
            color=theme["text_secondary"],
            row=2,
            col=1,
        )
        fig.update_xaxes(gridcolor=theme["gridline"], color=theme["text_secondary"], row=1, col=1)
        fig.update_xaxes(
            title_text="Date", gridcolor=theme["gridline"], color=theme["text_secondary"], row=2, col=1
        )
        for annotation in fig["layout"]["annotations"]:
            annotation["font"] = dict(color=theme["text_secondary"], size=13)

        fig.update_layout(
            height=580,
            hovermode="x unified",
            plot_bgcolor=theme["chart_surface"],
            paper_bgcolor=theme["chart_surface"],
            font_color=theme["text_primary"],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.06,
                xanchor="left",
                x=0,
                font=dict(color=theme["text_secondary"]),
            ),
            hoverlabel=dict(
                bgcolor=theme["surface"],
                font_color=theme["text_primary"],
                bordercolor=theme["card_border"],
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Analysis: reuses src.analysis.analysis, no logic duplicated here ---
    st.markdown('<div class="section-title">Sentiment vs. Returns &amp; Volatility</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="disclaimer">Descriptive only -- these results show association, '
        "not that sentiment predicts or causes returns or volatility. See "
        "Methodology &amp; Limitations below.</div>",
        unsafe_allow_html=True,
    )

    full_analysis_df = build_analysis_dataframe(prices, sentiment_daily)
    ranged_analysis_df = full_analysis_df[
        (full_analysis_df["date"] >= start_date) & (full_analysis_df["date"] <= end_date)
    ]

    same_day = same_day_correlation(ranged_analysis_df, tickers=[ticker])
    next_day = next_trading_day_correlation(ranged_analysis_df, prices, tickers=[ticker])
    volatility = forward_volatility_comparison(ranged_analysis_df, prices, tickers=[ticker])

    analysis_cols = st.columns(3)
    with analysis_cols[0]:
        _correlation_card(
            "Same-day correlation", "Avg. sentiment vs. same-day return", same_day[ticker]
        )
    with analysis_cols[1]:
        _correlation_card(
            "Next-trading-day correlation",
            "Avg. sentiment vs. next trading day's return",
            next_day[ticker],
        )
    with analysis_cols[2]:
        _volatility_card(volatility)

    # --- Latest news, as cards -------------------------------------------------
    st.markdown('<div class="section-title">Latest News</div>', unsafe_allow_html=True)
    if news_ranged.empty:
        st.info("No articles in the selected date range.")
    else:
        for _, row in news_ranged.head(10).iterrows():
            _news_card(row)

    # --- Methodology / disclaimer --------------------------------------------
    with st.expander("Methodology & limitations", expanded=False):
        st.markdown(
            f"""
- Sentiment is generated with **ProsusAI/FinBERT** on each article's title
  (plus description, when available).
- `sentiment_score = positive probability - negative probability`, so it
  ranges from -1 (very negative) to +1 (very positive). The Positive/
  Neutral/Negative badges use a small deadband around zero
  (&gt; {SENTIMENT_POSITIVE_THRESHOLD:+.2f} / &lt; {SENTIMENT_NEGATIVE_THRESHOLD:+.2f}).
- Correlations shown above are **descriptive, not causal or predictive** --
  a nonzero correlation does not mean sentiment predicts or causes returns
  or volatility, and this MVP dataset does not control for confounders
  (e.g. news often follows price moves rather than the other way around).
- The dataset currently covers a **limited time period** (a few weeks), so
  sample sizes are small and results should be treated as illustrative.
- Daily sentiment is aggregated by **UTC calendar date** (an article's
  `published_at` converted to UTC), not aligned to market trading
  sessions -- see `src/analysis/analysis.py` for that assumption.
- **SPY**'s current news sample is concentrated in only a few dates, so
  its correlation/volatility figures are especially unreliable for now.
            """
        )


if __name__ == "__main__":
    main()
