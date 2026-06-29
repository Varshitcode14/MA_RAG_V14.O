"""
Tests for evaluation/runner.py — file I/O and orchestration.

Uses temporary directories and mock pipelines.
No API calls, no FAISS, no real dataset required.

Usage:
    python tests/test_evaluation_runner.py
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root (parent of tests/) is on the path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Fixtures ─────────────────────────────────────────────────────────

_FAKE_DATASET = [
    {
        "id": "q001",
        "question": "Who wrote Harry Potter?",
        "answer": "J. K. Rowling",
        "supporting_facts": {"title": ["J. K. Rowling", "Harry Potter"]},
    },
    {
        "id": "q002",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "supporting_facts": {"title": ["Paris", "France"]},
    },
]


def _fake_pipeline(question: str) -> dict:
    """Minimal pipeline stub that returns the shared schema."""
    return {
        "question": question,
        "answer": "J. K. Rowling" if "Harry" in question else "Paris",
        "retrieved_docs": [
            {"title": "J. K. Rowling", "text": "Author of Harry Potter."}
        ],
        "retrieved_titles": ["J. K. Rowling"],
        "context": "J. K. Rowling is an author.",
        "retrieval_time": 0.1,
        "generation_time": 0.2,
        "total_time": 0.3,
        "history": [],
        "reasoning_steps": 1,
        "pipeline": "test_pipeline",
    }


# ── Tests ─────────────────────────────────────────────────────────────

def test_run_inference_saves_file() -> None:
    """run_inference writes a JSON file with the correct number of rows."""
    from evaluation.runner import run_inference

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "dev.json"
        results_dir = Path(tmpdir) / "results"

        with open(dataset_path, "w") as f:
            json.dump(_FAKE_DATASET, f)

        pred_path = run_inference(
            pipeline_fn=_fake_pipeline,
            pipeline_name="test_pipeline",
            num_samples=2,
            dataset_path=dataset_path,
            results_dir=results_dir,
        )

        assert pred_path.exists(), "Prediction file was not created."

        with open(pred_path) as f:
            preds = json.load(f)

        assert len(preds) == 2
        assert preds[0]["id"] == "q001"
        assert "prediction" in preds[0]
        assert "retrieved_titles" in preds[0]

    print("PASS  test_run_inference_saves_file")


def test_run_inference_row_schema() -> None:
    """Each prediction row contains all required fields."""
    from evaluation.runner import run_inference

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "dev.json"
        results_dir = Path(tmpdir) / "results"

        with open(dataset_path, "w") as f:
            json.dump(_FAKE_DATASET[:1], f)

        pred_path = run_inference(
            pipeline_fn=_fake_pipeline,
            pipeline_name="test_pipeline",
            num_samples=1,
            dataset_path=dataset_path,
            results_dir=results_dir,
        )

        with open(pred_path) as f:
            preds = json.load(f)

        row = preds[0]
        required = {
            "id", "question", "gold", "prediction",
            "retrieved_titles", "supporting_titles",
            "context", "total_time", "reasoning_steps", "pipeline",
        }
        for key in required:
            assert key in row, f"Missing key in row: {key}"

    print("PASS  test_run_inference_row_schema")


def test_evaluate_predictions_metrics() -> None:
    """evaluate_predictions returns correct metric keys and plausible values."""
    from evaluation.runner import run_inference, evaluate_predictions

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "dev.json"
        results_dir = Path(tmpdir) / "results"

        with open(dataset_path, "w") as f:
            json.dump(_FAKE_DATASET, f)

        pred_path = run_inference(
            pipeline_fn=_fake_pipeline,
            pipeline_name="test_pipeline",
            num_samples=2,
            dataset_path=dataset_path,
            results_dir=results_dir,
        )

        metrics = evaluate_predictions(pred_path, k=5, use_llm_judge=False)

        assert "exact_match" in metrics
        assert "token_f1" in metrics
        assert "precision_at_k" in metrics
        assert "recall_at_k" in metrics
        assert "hit_rate" in metrics
        assert "mrr" in metrics

        # The fake pipeline answers both questions correctly
        assert metrics["exact_match"] == 1.0
        assert metrics["token_f1"] == 1.0

    print("PASS  test_evaluate_predictions_metrics")


def test_compare_results_returns_both_pipelines() -> None:
    """compare_results returns a dict with one entry per pipeline."""
    from evaluation.runner import run_inference, compare_results

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "dev.json"
        results_dir = Path(tmpdir) / "results"

        with open(dataset_path, "w") as f:
            json.dump(_FAKE_DATASET, f)

        pred_path_a = run_inference(
            pipeline_fn=_fake_pipeline,
            pipeline_name="pipeline_a",
            num_samples=2,
            dataset_path=dataset_path,
            results_dir=results_dir,
        )

        pred_path_b = run_inference(
            pipeline_fn=_fake_pipeline,
            pipeline_name="pipeline_b",
            num_samples=2,
            dataset_path=dataset_path,
            results_dir=results_dir,
        )

        comparison = compare_results(
            {"Pipeline A": pred_path_a, "Pipeline B": pred_path_b},
            k=5,
        )

        assert "Pipeline A" in comparison
        assert "Pipeline B" in comparison

    print("PASS  test_compare_results_returns_both_pipelines")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evaluation Runner Tests")
    print("=" * 60)

    test_run_inference_saves_file()
    test_run_inference_row_schema()
    test_evaluate_predictions_metrics()
    test_compare_results_returns_both_pipelines()

    print("\nAll tests passed.")