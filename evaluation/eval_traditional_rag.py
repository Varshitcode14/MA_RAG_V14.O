"""
Evaluate Traditional RAG on the AI/ML QA dataset.

Usage:
    python evaluation/eval_traditional_rag.py
    python evaluation/eval_traditional_rag.py --samples 30 --llm-judge
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Traditional_RAG.pipeline import run as trad_run
from evaluation.runner import (
    run_inference,
    evaluate_predictions,
    print_metrics_report,
)

DATASET = Path("datasets/ai_ml/qa_dataset.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Traditional RAG")
    p.add_argument("--samples",   type=int,  default=30)
    p.add_argument("--k",         type=int,  default=5)
    p.add_argument("--llm-judge", action="store_true")
    p.add_argument("--dataset",   type=Path, default=DATASET)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("Traditional RAG — Evaluation")
    print("=" * 60)
    print(f"Dataset : {args.dataset}")
    print(f"Samples : {args.samples}  |  K : {args.k}")

    pred_path = run_inference(
        pipeline_fn=trad_run,
        pipeline_name="traditional_rag",
        num_samples=args.samples,
        dataset_path=args.dataset,
    )
    metrics = evaluate_predictions(
        pred_path, k=args.k, use_llm_judge=args.llm_judge
    )
    print_metrics_report(metrics, "Traditional RAG — Results")


if __name__ == "__main__":
    main()