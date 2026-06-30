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


# ── Verbosity-robust metrics ─────────────────────────────────────────

def token_recall(prediction: str, gold: str) -> float:
    """
    Token recall: fraction of gold tokens covered by the prediction.

    Unlike token F1 this does not penalize a prediction for being longer
    than the gold answer, so it is fair to systems that produce complete,
    self-contained answers (e.g. multi-hop synthesis).
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)
    if not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    return sum(common.values()) / len(gold_tokens)


# Lazy, cached sentence embedder shared with the retrieval stack.
_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from MA_RAG.retrieval.embedder import Embedder
        _EMBEDDER = Embedder()
    return _EMBEDDER


def semantic_similarity(prediction: str, gold: str) -> float:
    """
    Cosine similarity between sentence embeddings of prediction and gold.

    Verbosity-robust and meaning-aware (BERTScore-style signal): rewards
    answers that convey the same meaning regardless of exact wording or
    length. Returns a float in [0, 1] (negatives clamped to 0).
    """
    if not prediction or not gold:
        return 0.0
    embedder = _get_embedder()
    pred_vec = embedder.encode(prediction)[0]
    gold_vec = embedder.encode(gold)[0]
    # Embeddings are L2-normalized, so dot product == cosine similarity.
    sim = float((pred_vec * gold_vec).sum())
    return max(0.0, min(1.0, sim))


# ── Batch helpers ────────────────────────────────────────────────────

def compute_generation_metrics(
    predictions: list[str],
    gold_answers: list[str],
    include_semantic: bool = True,
) -> dict[str, float]:
    """
    Compute generation metrics over a list of predictions.

    Args:
        predictions:      One predicted answer per question.
        gold_answers:     One gold answer per question.
        include_semantic: Whether to compute embedding-based semantic
                          similarity (loads the sentence-transformer).

    Returns:
        Macro-averaged dict with keys:
            exact_match, token_f1, token_recall, semantic_similarity
    """
    if not predictions:
        base = {"exact_match": 0.0, "token_f1": 0.0, "token_recall": 0.0}
        if include_semantic:
            base["semantic_similarity"] = 0.0
        return base

    n = len(predictions)
    em_scores = [float(exact_match(p, g)) for p, g in zip(predictions, gold_answers)]
    f1_scores = [token_f1(p, g) for p, g in zip(predictions, gold_answers)]
    rec_scores = [token_recall(p, g) for p, g in zip(predictions, gold_answers)]

    metrics = {
        "exact_match": sum(em_scores) / n,
        "token_f1": sum(f1_scores) / n,
        "token_recall": sum(rec_scores) / n,
    }

    if include_semantic:
        sem_scores = [
            semantic_similarity(p, g) for p, g in zip(predictions, gold_answers)
        ]
        metrics["semantic_similarity"] = sum(sem_scores) / n

    return metrics


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