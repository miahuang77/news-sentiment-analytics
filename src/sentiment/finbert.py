"""Score financial news text with FinBERT (ProsusAI/finbert).

PostgreSQL persistence and daily aggregation are deferred; this module only
scores article text and returns per-article probabilities.
"""

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"

_tokenizer = None
_model = None

_INPUT_COLUMNS = ["article_id", "ticker", "title", "description", "source", "published_at"]
_SCORE_COLUMNS = ["positive_score", "neutral_score", "negative_score", "sentiment_score"]


def load_model():
    """Load and cache the FinBERT tokenizer and model.

    The model is put in eval mode since this module only ever runs
    inference, never training.

    Returns:
        (tokenizer, model) tuple.
    """
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_texts(texts, batch_size=16):
    """Score a list of texts with FinBERT, processed in batches.

    Args:
        texts: List of strings to score.
        batch_size: Number of texts run through the model per forward pass.

    Returns:
        List of dicts, one per input text in the same order, each with keys
        positive_score, neutral_score, negative_score, sentiment_score
        (= positive_score - negative_score, range -1 to +1).
    """
    tokenizer, model = load_model()
    label_names = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    results = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        for row in probs.tolist():
            by_label = dict(zip(label_names, row))
            results.append(
                {
                    "positive_score": by_label["positive"],
                    "neutral_score": by_label["neutral"],
                    "negative_score": by_label["negative"],
                    "sentiment_score": by_label["positive"] - by_label["negative"],
                }
            )

    return results


def score_articles(df, batch_size=16):
    """Score a DataFrame of news articles with FinBERT.

    Args:
        df: DataFrame with columns article_id, ticker, title, description,
            source, published_at (e.g. from src.ingestion.news.fetch_news).
            For each row, title and description are combined as the model
            input; if description is missing, title alone is used.
        batch_size: Number of articles run through the model per forward
            pass.

    Returns:
        A copy of df with four new columns added: positive_score,
        neutral_score, negative_score, sentiment_score.

    Raises:
        ValueError: df is missing one or more required columns.
    """
    missing = [col for col in _INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    texts = [_combine_text(t, d) for t, d in zip(df["title"], df["description"])]
    scores = score_texts(texts, batch_size=batch_size)

    result = df.copy()
    scores_df = pd.DataFrame(scores, index=df.index)
    for col in _SCORE_COLUMNS:
        result[col] = scores_df[col]
    return result


def _combine_text(title, description):
    """Combine title and description into one string for the model input.

    Falls back to title alone when description is missing (None, NaN, or
    blank).
    """
    if description is None or (isinstance(description, float) and pd.isna(description)):
        return str(title)
    description = str(description).strip()
    if not description:
        return str(title)
    return f"{title}. {description}"


if __name__ == "__main__":
    headlines = [
        "Apple reports record quarterly profits, beating analyst expectations.",
        "Company shares plunge after massive layoffs and disappointing earnings report.",
        "The company will hold its annual shareholder meeting next Tuesday.",
    ]

    for headline, scores in zip(headlines, score_texts(headlines)):
        print(headline)
        print(f"  positive_score:  {scores['positive_score']:.4f}")
        print(f"  neutral_score:   {scores['neutral_score']:.4f}")
        print(f"  negative_score:  {scores['negative_score']:.4f}")
        print(f"  sentiment_score: {scores['sentiment_score']:+.4f}")
        print()
