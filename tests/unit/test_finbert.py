"""Unit tests for src.sentiment.finbert.

Covers the sentiment_score formula and title/description combining. The
real FinBERT model (~440MB) is never downloaded here: load_model()'s
module-level cache is monkeypatched with a fake tokenizer/model pair that
returns controlled logits, so score_texts()'s real softmax + label-mapping
+ sentiment_score code path is exercised end to end, fast and offline.
"""

import pytest
import torch

from src.sentiment import finbert
from src.sentiment.finbert import _combine_text, score_texts


class _FakeConfig:
    # Matches the real ProsusAI/finbert checkpoint's label order (confirmed
    # via its model config): id 0 is positive, 1 negative, 2 neutral.
    id2label = {0: "positive", 1: "negative", 2: "neutral"}


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    config = _FakeConfig()

    def __call__(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        # Large, well-separated logits per row so softmax outputs are
        # close to 0/1 and easy to reason about.
        return _FakeOutput(torch.tensor([[5.0, -5.0, 0.0]] * batch_size))


class _FakeTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        n = len(texts)
        return {
            "input_ids": torch.zeros((n, 3), dtype=torch.long),
            "attention_mask": torch.ones((n, 3), dtype=torch.long),
        }


def _install_fake_model(monkeypatch):
    monkeypatch.setattr(finbert, "_tokenizer", _FakeTokenizer())
    monkeypatch.setattr(finbert, "_model", _FakeModel())


def test_score_texts_sentiment_score_formula(monkeypatch):
    _install_fake_model(monkeypatch)

    [result] = score_texts(["some headline"])

    expected = torch.nn.functional.softmax(torch.tensor([5.0, -5.0, 0.0]), dim=-1)
    assert result["positive_score"] == pytest.approx(expected[0].item())
    assert result["negative_score"] == pytest.approx(expected[1].item())
    assert result["neutral_score"] == pytest.approx(expected[2].item())
    # The defining formula: sentiment_score = positive - negative.
    assert result["sentiment_score"] == pytest.approx(
        result["positive_score"] - result["negative_score"]
    )


def test_score_texts_range_is_bounded(monkeypatch):
    _install_fake_model(monkeypatch)

    [result] = score_texts(["some headline"])

    assert -1.0 <= result["sentiment_score"] <= 1.0
    total = result["positive_score"] + result["neutral_score"] + result["negative_score"]
    assert total == pytest.approx(1.0)


def test_score_texts_batches_preserve_order(monkeypatch):
    _install_fake_model(monkeypatch)

    texts = [f"headline {i}" for i in range(5)]
    results = score_texts(texts, batch_size=2)

    assert len(results) == 5
    # All rows use the same fake logits, so every result should be identical
    # regardless of which batch it landed in -- if batching broke ordering
    # this would still pass, but it does confirm no rows are dropped/duped.
    assert all(r == results[0] for r in results)


# --- _combine_text ----------------------------------------------------------


def test_combine_text_with_description():
    assert _combine_text("Title", "Description.") == "Title. Description."


def test_combine_text_none_description_falls_back_to_title():
    assert _combine_text("Title", None) == "Title"


def test_combine_text_nan_description_falls_back_to_title():
    assert _combine_text("Title", float("nan")) == "Title"


def test_combine_text_blank_description_falls_back_to_title():
    assert _combine_text("Title", "   ") == "Title"
