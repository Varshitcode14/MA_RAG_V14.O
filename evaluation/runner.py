"""
Evaluation runner.

Supports two dataset formats:
  - HotpotQA  : {"id", "question", "answer", "supporting_facts": {"title": [...]}}
  - AI/ML custom : {"id", "question", "answer", "supporting_titles": [...]}

Inference and evaluation are decoupled:
  - run_inference()         → saves predictions JSON
  - evaluate_predictions()  → loads predictions JSON, computes metrics
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from evaluation.generation_metrics import compute_generation_metrics
from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.llm_judge import batch_judge

RESULTS_DIR = Path("results")


# ── Dataset loader ────────────────────────────────────────────────────

def load_dataset(dataset_path: Path, num_samples: int) -> list[dict]:
    """
    Load samples from either HotpotQA or AI/ML custom format.
    Normalises to: {id, question, answer, supporting_titles}.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data[:num_samples]
    normalised = []

    for s in samples:
        if "supporting_facts" in s and isinstance(s["supporting_facts"], dict):
            sup = s["supporting_facts"].get("title", [])
        else:
            sup = s.get("supporting_titles", [])

        normalised.append({
            "id": s.get("id", ""),
            "question": s["question"],
            "answer": s["answer"],
            "supporting_titles": sup,
        })

    return normalised


# ── Inference ─────────────────────────────────────────────────────────

def run_inference(
    pipeline_fn: Callable[[str], dict],
    pipeline_name: str,
    num_samples: int = 30,
    dataset_path: Path = Path("datasets/ai_ml/qa_dataset.json"),
    results_dir: Path = RESULTS_DIR,
) -> Path:
    """
    Run a pipeline over N samples and save predictions to JSON.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{pipeline_name}_preds.json"

    print(f"\nLoading dataset: {dataset_path}")
    samples = load_dataset(Path(dataset_path), num_samples)
    print(f"Loaded {len(samples)} samples.\n")

    predictions: list[dict] = []
    start_time = time.time()

    for idx, sample in enumerate(samples, start=1):
        question = sample["question"]
        gold = sample["answer"]

        print(f"[{idx}/{len(samples)}] {question[:80]}")

        try:
            result = pipeline_fn(question)

            row = {
                "id": sample["id"],
                "question": question,
                "gold": gold,
                "prediction": result["answer"],
                "retrieved_titles": result.get("retrieved_titles", []),
                "supporting_titles": sample["supporting_titles"],
                "context": result.get("context", ""),
                "total_time": result.get("total_time", 0.0),
                "reasoning_steps": result.get("reasoning_steps", 1),
                "pipeline": result.get("pipeline", pipeline_name),
            }

            predictions.append(row)
            _save_json(predictions, output_path)

            print(f"  Pred : {row['prediction'][:80]}")
            print(f"  Gold : {gold}")

        except Exception as exc:
            print(f"  ERROR: {exc}")

    elapsed = time.time() - start_time
    print(f"\nDone. {len(predictions)}/{len(samples)} samples in {elapsed:.1f}s")
    print(f"Saved to: {output_path}")
    return output_path


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_predictions(
    predictions_path: Path | str,
    k: int = 5,
    use_llm_judge: bool = False,
    judge_faithfulness: bool = True,
    judge_relevancy: bool = True,
) -> dict:
    """
    Load saved predictions and compute all metrics.
    """
    predictions_path = Path(predictions_path)
    print(f"\nLoading predictions: {predictions_path}")

    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions: list[dict] = json.load(f)

    if not predictions:
        print("No predictions found.")
        return {}

    preds     = [p["prediction"] for p in predictions]
    golds     = [p["gold"]       for p in predictions]
    retrieved = [p.get("retrieved_titles", []) for p in predictions]
    relevant  = [p.get("supporting_titles", []) for p in predictions]

    gen_metrics = compute_generation_metrics(preds, golds)
    ret_metrics = compute_retrieval_metrics(retrieved, relevant, k=k)
    metrics: dict = {**gen_metrics, **ret_metrics}

    # ── Efficiency aggregates (present when the pipeline reports them) ──
    def _avg(key: str) -> float:
        vals = [p[key] for p in predictions if isinstance(p.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    metrics["avg_total_time"]      = _avg("total_time")
    metrics["avg_retrieval_time"]  = _avg("retrieval_time")
    metrics["avg_generation_time"] = _avg("generation_time")
    metrics["avg_reasoning_steps"] = _avg("reasoning_steps")

    if use_llm_judge:
        print("\nRunning LLM-as-a-Judge ...")
        judged = batch_judge(
            predictions,
            run_faithfulness=judge_faithfulness,
            run_relevancy=judge_relevancy,
        )
        for key in [
            "judge_correctness_score",
            "judge_faithfulness_score",
            "judge_relevancy_score",
        ]:
            scores = [r[key] for r in judged if key in r]
            if scores:
                metrics[key] = sum(scores) / len(scores)

        judged_path = predictions_path.with_suffix(".judged.json")
        _save_json(judged, judged_path)
        print(f"Judged results saved: {judged_path}")

    return metrics


# ── Comparison ────────────────────────────────────────────────────────

def compare_results(
    paths: dict[str, Path | str],
    k: int = 5,
) -> dict[str, dict]:
    comparison: dict[str, dict] = {}
    for label, path in paths.items():
        print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")
        comparison[label] = evaluate_predictions(path, k=k)
    return comparison


# ── Reporting ─────────────────────────────────────────────────────────

def print_metrics_report(metrics: dict, title: str = "Evaluation Results") -> None:
    _LABELS = {
        "exact_match":              "Exact Match (EM)",
        "token_f1":                 "Token F1",
        "token_recall":             "Token Recall (gold coverage)",
        "semantic_similarity":      "Semantic Similarity",
        "precision_at_k":           "Precision@K",
        "recall_at_k":              "Recall@K",
        "hit_rate":                 "Hit Rate",
        "mrr":                      "MRR",
        "judge_correctness_score":  "LLM Judge — Correctness",
        "judge_faithfulness_score": "LLM Judge — Faithfulness",
        "judge_relevancy_score":    "LLM Judge — Relevancy",
    }
    _TIME_LABELS = {
        "avg_total_time":      "Avg Latency (s)",
        "avg_retrieval_time":  "Avg Retrieval (s)",
        "avg_generation_time": "Avg Generation (s)",
        "avg_reasoning_steps": "Avg Reasoning Steps",
    }
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    for key, label in _LABELS.items():
        if key in metrics:
            v = metrics[key]
            print(f"  {label:<38} {v:.4f}  ({v*100:.2f}%)")
    for key, label in _TIME_LABELS.items():
        if key in metrics:
            print(f"  {label:<38} {metrics[key]:.3f}")
    print("=" * 60)


def print_comparison_report(comparison: dict[str, dict]) -> None:
    if not comparison:
        return
    all_keys  = sorted({k for m in comparison.values() for k in m})
    pipelines = list(comparison.keys())
    col_w     = 22
    header    = f"{'Metric':<38}" + "".join(f"{p:>{col_w}}" for p in pipelines)
    print(f"\n{'='*70}\nPipeline Comparison\n{'='*70}")
    print(header)
    print("-" * len(header))
    for key in all_keys:
        row = f"{key:<38}"
        for p in pipelines:
            val = comparison[p].get(key, "—")
            row += f"{'%6.4f' % val if isinstance(val, float) else str(val):>{col_w}}"
        print(row)
    print("=" * 70)


# ── Utility ───────────────────────────────────────────────────────────

def _save_json(data: list | dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)