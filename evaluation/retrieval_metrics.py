"""
Retrieval metrics for RAG evaluation.

All functions are pure and operate on lists, making them
easy to unit-test and to integrate with RAGAS later.

Metrics
-------
- precision_at_k   : fraction of retrieved docs that are relevant
- recall_at_k      : fraction of relevant docs that are retrieved
- hit_rate         : fraction of questions with ≥1 correct doc in top-k
- mean_reciprocal_rank (MRR) : average of 1/rank of first correct doc
"""

from __future__ import annotations


def precision_at_k(
    retrieved_titles: list[str],
    relevant_titles: list[str],
    k: int | None = None,
) -> float:
    """
    Precision@K: fraction of top-k retrieved docs that are relevant.

    Args:
        retrieved_titles: Ordered list of retrieved document titles.
        relevant_titles:  Ground-truth relevant document titles.
        k:                Cut-off; defaults to len(retrieved_titles).

    Returns:
        Float in [0, 1].
    """
    if not retrieved_titles:
        return 0.0

    top_k = retrieved_titles[:k] if k else retrieved_titles
    relevant_set = set(relevant_titles)
    hits = sum(1 for t in top_k if t in relevant_set)
    return hits / len(top_k)


def recall_at_k(
    retrieved_titles: list[str],
    relevant_titles: list[str],
    k: int | None = None,
) -> float:
    """
    Recall@K: fraction of relevant docs found in top-k results.

    Args:
        retrieved_titles: Ordered list of retrieved document titles.
        relevant_titles:  Ground-truth relevant document titles.
        k:                Cut-off; defaults to len(retrieved_titles).

    Returns:
        Float in [0, 1].  Returns 0.0 when relevant_titles is empty.
    """
    if not relevant_titles:
        return 0.0

    top_k = retrieved_titles[:k] if k else retrieved_titles
    relevant_set = set(relevant_titles)
    hits = sum(1 for t in top_k if t in relevant_set)
    return hits / len(relevant_set)


def hit_rate(
    retrieved_titles_list: list[list[str]],
    relevant_titles_list: list[list[str]],
    k: int | None = None,
) -> float:
    """
    Hit Rate: fraction of questions that have ≥1 relevant doc in top-k.

    Args:
        retrieved_titles_list: One list of retrieved titles per question.
        relevant_titles_list:  One list of relevant titles per question.
        k:                     Cut-off per question.

    Returns:
        Float in [0, 1].
    """
    if not retrieved_titles_list:
        return 0.0

    hits = 0
    for retrieved, relevant in zip(retrieved_titles_list, relevant_titles_list):
        top_k = retrieved[:k] if k else retrieved
        relevant_set = set(relevant)
        if any(t in relevant_set for t in top_k):
            hits += 1

    return hits / len(retrieved_titles_list)


def reciprocal_rank(
    retrieved_titles: list[str],
    relevant_titles: list[str],
) -> float:
    """
    Reciprocal Rank for a single query: 1/rank of the first relevant doc.

    Returns 0.0 if no relevant document is found.
    """
    relevant_set = set(relevant_titles)
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_titles_list: list[list[str]],
    relevant_titles_list: list[list[str]],
) -> float:
    """
    Mean Reciprocal Rank (MRR) across a set of queries.

    Args:
        retrieved_titles_list: One ordered list per question.
        relevant_titles_list:  One list of relevant titles per question.

    Returns:
        Float in [0, 1].
    """
    if not retrieved_titles_list:
        return 0.0

    rr_sum = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(
            retrieved_titles_list, relevant_titles_list
        )
    )
    return rr_sum / len(retrieved_titles_list)


# ── Batch helper ─────────────────────────────────────────────────────

def compute_retrieval_metrics(
    retrieved_titles_list: list[list[str]],
    relevant_titles_list: list[list[str]],
    k: int | None = None,
) -> dict[str, float]:
    """
    Compute all retrieval metrics in one call.

    Returns:
        Dict with keys: precision_at_k, recall_at_k, hit_rate, mrr.
    """
    precision_scores = [
        precision_at_k(r, rel, k)
        for r, rel in zip(retrieved_titles_list, relevant_titles_list)
    ]
    recall_scores = [
        recall_at_k(r, rel, k)
        for r, rel in zip(retrieved_titles_list, relevant_titles_list)
    ]

    n = len(retrieved_titles_list)

    return {
        "precision_at_k": sum(precision_scores) / n if n else 0.0,
        "recall_at_k": sum(recall_scores) / n if n else 0.0,
        "hit_rate": hit_rate(retrieved_titles_list, relevant_titles_list, k),
        "mrr": mean_reciprocal_rank(retrieved_titles_list, relevant_titles_list),
    }