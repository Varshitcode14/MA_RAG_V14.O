"""
Tests for the evaluation metrics package.

All metric functions are pure — no API calls, no file I/O.
Runs fast and can be used as a smoke-test in CI.

Usage:
    python tests/test_evaluation_metrics.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate,
    reciprocal_rank,
    mean_reciprocal_rank,
    compute_retrieval_metrics,
)
from evaluation.generation_metrics import (
    exact_match,
    token_f1,
    answer_correctness,
    compute_generation_metrics,
)


# ── Retrieval metric tests ────────────────────────────────────────────

def test_precision_at_k_perfect() -> None:
    retrieved = ["A", "B", "C"]
    relevant = ["A", "B", "C"]
    assert precision_at_k(retrieved, relevant) == 1.0
    print("PASS  test_precision_at_k_perfect")


def test_precision_at_k_none() -> None:
    retrieved = ["X", "Y"]
    relevant = ["A", "B"]
    assert precision_at_k(retrieved, relevant) == 0.0
    print("PASS  test_precision_at_k_none")


def test_precision_at_k_cutoff() -> None:
    retrieved = ["A", "X", "B"]
    relevant = ["A", "B"]
    # top-1: only A → 1/1 = 1.0
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    # top-2: A,X → 1/2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    print("PASS  test_precision_at_k_cutoff")


def test_recall_at_k_perfect() -> None:
    retrieved = ["A", "B"]
    relevant = ["A", "B"]
    assert recall_at_k(retrieved, relevant) == 1.0
    print("PASS  test_recall_at_k_perfect")


def test_recall_at_k_partial() -> None:
    retrieved = ["A", "X"]
    relevant = ["A", "B"]
    assert recall_at_k(retrieved, relevant) == 0.5
    print("PASS  test_recall_at_k_partial")


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["A"], []) == 0.0
    print("PASS  test_recall_at_k_empty_relevant")


def test_hit_rate_all_hits() -> None:
    retrieved_list = [["A", "B"], ["C", "D"]]
    relevant_list = [["A"], ["C"]]
    assert hit_rate(retrieved_list, relevant_list) == 1.0
    print("PASS  test_hit_rate_all_hits")


def test_hit_rate_no_hits() -> None:
    retrieved_list = [["X", "Y"], ["Z"]]
    relevant_list = [["A"], ["B"]]
    assert hit_rate(retrieved_list, relevant_list) == 0.0
    print("PASS  test_hit_rate_no_hits")


def test_hit_rate_partial() -> None:
    retrieved_list = [["A"], ["X"]]
    relevant_list = [["A"], ["B"]]
    assert hit_rate(retrieved_list, relevant_list) == 0.5
    print("PASS  test_hit_rate_partial")


def test_reciprocal_rank_first() -> None:
    assert reciprocal_rank(["A", "B", "C"], ["A"]) == 1.0
    print("PASS  test_reciprocal_rank_first")


def test_reciprocal_rank_second() -> None:
    assert reciprocal_rank(["X", "A", "C"], ["A"]) == 0.5
    print("PASS  test_reciprocal_rank_second")


def test_reciprocal_rank_miss() -> None:
    assert reciprocal_rank(["X", "Y"], ["A"]) == 0.0
    print("PASS  test_reciprocal_rank_miss")


def test_mrr() -> None:
    retrieved_list = [["A", "B"], ["X", "A"]]
    relevant_list = [["A"], ["A"]]
    # RR: 1.0, 0.5 → MRR = 0.75
    assert abs(mean_reciprocal_rank(retrieved_list, relevant_list) - 0.75) < 1e-6
    print("PASS  test_mrr")


def test_compute_retrieval_metrics_returns_all_keys() -> None:
    result = compute_retrieval_metrics(
        [["A", "B"]], [["A"]], k=5
    )
    assert set(result.keys()) == {"precision_at_k", "recall_at_k", "hit_rate", "mrr"}
    print("PASS  test_compute_retrieval_metrics_returns_all_keys")


# ── Generation metric tests ───────────────────────────────────────────

def test_exact_match_identical() -> None:
    assert exact_match("Paris", "Paris") is True
    print("PASS  test_exact_match_identical")


def test_exact_match_case_insensitive() -> None:
    assert exact_match("paris", "Paris") is True
    print("PASS  test_exact_match_case_insensitive")


def test_exact_match_punctuation() -> None:
    assert exact_match("Yes!", "yes") is True
    print("PASS  test_exact_match_punctuation")


def test_exact_match_different() -> None:
    assert exact_match("London", "Paris") is False
    print("PASS  test_exact_match_different")


def test_token_f1_perfect() -> None:
    assert token_f1("the cat sat", "the cat sat") == 1.0
    print("PASS  test_token_f1_perfect")


def test_token_f1_partial() -> None:
    score = token_f1("the cat", "the cat sat on the mat")
    assert 0.0 < score < 1.0
    print("PASS  test_token_f1_partial")


def test_token_f1_zero() -> None:
    assert token_f1("apple", "orange") == 0.0
    print("PASS  test_token_f1_zero")


def test_answer_correctness_keys() -> None:
    result = answer_correctness("Paris", "Paris")
    assert "exact_match" in result
    assert "token_f1" in result
    assert result["exact_match"] == 1.0
    assert result["token_f1"] == 1.0
    print("PASS  test_answer_correctness_keys")


def test_compute_generation_metrics() -> None:
    preds = ["Paris", "yes", "Robert Erskine Childers"]
    golds = ["Paris", "no", "Robert Erskine Childers DSC"]
    result = compute_generation_metrics(preds, golds)
    assert "exact_match" in result
    assert "token_f1" in result
    # One exact match out of three
    assert abs(result["exact_match"] - 1 / 3) < 1e-6
    print("PASS  test_compute_generation_metrics")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evaluation Metrics Tests")
    print("=" * 60)

    # Retrieval
    test_precision_at_k_perfect()
    test_precision_at_k_none()
    test_precision_at_k_cutoff()
    test_recall_at_k_perfect()
    test_recall_at_k_partial()
    test_recall_at_k_empty_relevant()
    test_hit_rate_all_hits()
    test_hit_rate_no_hits()
    test_hit_rate_partial()
    test_reciprocal_rank_first()
    test_reciprocal_rank_second()
    test_reciprocal_rank_miss()
    test_mrr()
    test_compute_retrieval_metrics_returns_all_keys()

    # Generation
    test_exact_match_identical()
    test_exact_match_case_insensitive()
    test_exact_match_punctuation()
    test_exact_match_different()
    test_token_f1_perfect()
    test_token_f1_partial()
    test_token_f1_zero()
    test_answer_correctness_keys()
    test_compute_generation_metrics()

    print(f"\nAll tests passed.")