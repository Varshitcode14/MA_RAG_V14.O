"""
Compare Traditional RAG vs MA-RAG.

Shows:
  1. Side-by-side metric table
  2. Per-question correctness breakdown (where each pipeline wins/loses)
  3. Multi-hop vs single-hop performance split

Usage:
    python evaluation/compare_results.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.runner import compare_results, print_comparison_report
from evaluation.generation_metrics import exact_match, token_f1

RESULTS_DIR = Path("results")
DATASET_PATH = Path("datasets/ai_ml/qa_dataset.json")


def load_preds(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset() -> dict[str, dict]:
    """Return {question: {type, supporting_titles}} from the QA dataset."""
    if not DATASET_PATH.exists():
        return {}
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["question"]: d for d in data}


def per_question_breakdown(
    trad_preds: list[dict],
    ma_preds: list[dict],
    dataset: dict,
) -> None:
    """Print a per-question win/loss/tie table."""
    ma_by_q = {p["question"]: p for p in ma_preds}

    trad_win = ma_win = tie = 0
    multi_hop_results = {"traditional_rag": [], "ma_rag": []}
    single_hop_results = {"traditional_rag": [], "ma_rag": []}

    print(f"\n{'='*80}")
    print("Per-Question Breakdown")
    print(f"{'='*80}")
    print(f"{'#':<4} {'Q (truncated)':<45} {'Gold':<25} {'T-RAG':^8} {'MA-RAG':^8} {'Winner'}")
    print("-" * 100)

    for i, t in enumerate(trad_preds, 1):
        q = t["question"]
        gold = t["gold"]
        ma = ma_by_q.get(q, {})

        t_f1 = token_f1(t["prediction"], gold)
        m_f1 = token_f1(ma.get("prediction", ""), gold) if ma else 0.0

        t_win = t_f1 > m_f1 + 0.05
        m_win = m_f1 > t_f1 + 0.05

        if t_win:
            winner = "T-RAG"
            trad_win += 1
        elif m_win:
            winner = "MA-RAG"
            ma_win += 1
        else:
            winner = "Tie"
            tie += 1

        # Classify multi-hop vs single-hop from dataset
        ds_entry = dataset.get(q, {})
        n_supporting = len(ds_entry.get("supporting_titles", []))
        is_multi = n_supporting >= 2

        bucket_key = "multi_hop" if is_multi else "single_hop"
        multi_hop_results["traditional_rag"].append(t_f1) if is_multi else single_hop_results["traditional_rag"].append(t_f1)
        multi_hop_results["ma_rag"].append(m_f1) if is_multi else single_hop_results["ma_rag"].append(m_f1)

        q_short = q[:44]
        gold_short = gold[:24]
        print(
            f"{i:<4} {q_short:<45} {gold_short:<25} "
            f"{t_f1:^8.2f} {m_f1:^8.2f} {winner}"
        )

    print(f"\n{'='*80}")
    print(f"  Traditional RAG wins : {trad_win}")
    print(f"  MA-RAG wins          : {ma_win}")
    print(f"  Ties                 : {tie}")

    # Multi-hop vs single-hop split
    print(f"\n{'='*80}")
    print("Performance Split: Single-hop vs Multi-hop Questions")
    print(f"{'='*80}")

    for label, bucket in [("Single-hop", single_hop_results), ("Multi-hop", multi_hop_results)]:
        t_scores = bucket["traditional_rag"]
        m_scores = bucket["ma_rag"]
        if t_scores:
            t_avg = sum(t_scores) / len(t_scores)
            m_avg = sum(m_scores) / len(m_scores) if m_scores else 0.0
            print(f"\n  {label} ({len(t_scores)} questions):")
            print(f"    Traditional RAG Token F1 : {t_avg:.4f} ({t_avg*100:.1f}%)")
            print(f"    MA-RAG Token F1          : {m_avg:.4f} ({m_avg*100:.1f}%)")
            delta = m_avg - t_avg
            sign = "+" if delta >= 0 else ""
            print(f"    MA-RAG delta             : {sign}{delta:.4f} ({sign}{delta*100:.1f}%)")

    print("=" * 80)


def main() -> None:
    trad_path = RESULTS_DIR / "traditional_rag_preds.json"
    ma_path   = RESULTS_DIR / "ma_rag_preds.json"

    paths: dict[str, Path] = {}
    if trad_path.exists():
        paths["Traditional RAG"] = trad_path
    else:
        print(f"[WARN] Not found: {trad_path} — run eval_traditional_rag.py first.")

    if ma_path.exists():
        paths["MA-RAG"] = ma_path
    else:
        print(f"[WARN] Not found: {ma_path} — run eval_ma_rag.py first.")

    if not paths:
        sys.exit(1)

    # Overall metrics
    comparison = compare_results(paths, k=5)
    print_comparison_report(comparison)

    # Per-question breakdown (only if both exist)
    if trad_path.exists() and ma_path.exists():
        trad_preds = load_preds(trad_path)
        ma_preds   = load_preds(ma_path)
        dataset    = load_dataset()
        per_question_breakdown(trad_preds, ma_preds, dataset)


if __name__ == "__main__":
    main()