"""
Evaluate MA-RAG on the multi-hop QA dataset.

Usage:
    python evaluation/eval_multihop_ma_rag.py
    python evaluation/eval_multihop_ma_rag.py --llm-judge
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from MA_RAG.pipeline import run as ma_run
from evaluation.runner import (
    run_inference,
    evaluate_predictions,
    print_metrics_report,
)

DATASET = Path("datasets/ai_ml/multihop_qa.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate MA-RAG on multi-hop questions"
    )
    p.add_argument("--samples",   type=int,  default=30)
    p.add_argument("--k",         type=int,  default=5)
    p.add_argument("--llm-judge", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("MA-RAG — Multi-Hop Evaluation")
    print("=" * 60)
    print(f"Dataset : {DATASET}")
    print(f"Samples : {args.samples}  |  K : {args.k}")

    pred_path = run_inference(
        pipeline_fn=ma_run,
        pipeline_name="multihop_ma_rag",
        num_samples=args.samples,
        dataset_path=DATASET,
    )
    metrics = evaluate_predictions(
        pred_path,
        k=args.k,
        use_llm_judge=args.llm_judge,
    )
    print_metrics_report(metrics, "MA-RAG — Multi-Hop Results")


if __name__ == "__main__":
    main()