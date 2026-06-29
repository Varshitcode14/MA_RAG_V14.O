"""
Generation metrics for RAG evaluation.

Implemented
-----------
- exact_match        : normalized string equality
- token_f1           : token-level F1 (standard QA metric)
- answer_correctness : EM + F1 combined score

Stubs ready for RAGAS integration
----------------------------------
The functions below define the expected signature.
When RAGAS is available, swap the body for:

    from ragas.metrics import faithfulness, answer_relevancy, ...

- faithfulness         : is the answer grounded in the context?
- answer_relevancy     : does the answer address the question?
- context_precision    : are retrieved docs relevant to the question?
- context_recall       : do retrieved docs cover the gold answer?

All stubs return None so callers can detect when RAGAS is absent.
"""

from __future__ import annotations

import re
import string
from collections import Counter


# ── Normalization ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ── Core metrics ─────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> bool:
    """Normalized exact match between prediction and gold answer."""
    return _normalize(prediction) == _normalize(gold)


def token_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 score (standard SQuAD / HotpotQA metric).

    Returns float in [0, 1].
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_correctness(prediction: str, gold: str) -> dict[str, float]:
    """
    Combined answer correctness: EM + token F1.

    Returns:
        {"exact_match": bool_as_float, "token_f1": float}
    """
    return {
        "exact_match": float(exact_match(prediction, gold)),
        "token_f1": token_f1(prediction, gold),
    }


# ── Batch helpers ────────────────────────────────────────────────────

def compute_generation_metrics(
    predictions: list[str],
    gold_answers: list[str],
) -> dict[str, float]:
    """
    Compute EM and F1 over a list of predictions.

    Args:
        predictions:  One predicted answer per question.
        gold_answers: One gold answer per question.

    Returns:
        {"exact_match": float, "token_f1": float}  — macro-averaged.
    """
    if not predictions:
        return {"exact_match": 0.0, "token_f1": 0.0}

    em_scores = [
        float(exact_match(p, g))
        for p, g in zip(predictions, gold_answers)
    ]
    f1_scores = [
        token_f1(p, g)
        for p, g in zip(predictions, gold_answers)
    ]
    n = len(predictions)

    return {
        "exact_match": sum(em_scores) / n,
        "token_f1": sum(f1_scores) / n,
    }


# ── RAGAS-compatible stubs ───────────────────────────────────────────
# Replace these bodies once RAGAS is installed:
#   pip install ragas

def faithfulness(
    answer: str,
    context: str,
) -> float | None:
    """
    Faithfulness: fraction of answer claims supported by context.

    Stub — returns None until RAGAS is integrated.
    To enable: replace body with RAGAS faithfulness scorer.
    """
    return None


def answer_relevancy(
    answer: str,
    question: str,
) -> float | None:
    """
    Answer Relevancy: how well the answer addresses the question.

    Stub — returns None until RAGAS is integrated.
    """
    return None


def context_precision(
    retrieved_docs: list[dict],
    question: str,
    gold_answer: str,
) -> float | None:
    """
    Context Precision: are the retrieved docs relevant to the question?

    Stub — returns None until RAGAS is integrated.
    """
    return None


def context_recall(
    retrieved_docs: list[dict],
    gold_answer: str,
) -> float | None:
    """
    Context Recall: do retrieved docs cover the gold answer?

    Stub — returns None until RAGAS is integrated.
    """
    return None