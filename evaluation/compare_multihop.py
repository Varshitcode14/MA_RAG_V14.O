"""
Compare Traditional RAG vs MA-RAG on multi-hop questions.

Shows:
  1. Overall metric table
  2. Per-question Token F1 + win/loss/tie
  3. Results split by difficulty tier (multi-hop / hard-multi-hop)
  4. Predictions vs gold for inspection

Usage:
    python evaluation/compare_multihop.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.runner import compare_results, print_comparison_report
from evaluation.generation_metrics import token_f1

RESULTS_DIR  = Path("results")
DATASET_PATH = Path("datasets/ai_ml/multihop_qa.json")


def load_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieval_coverage(retrieved: list[str], supporting: list[str]) -> float:
    if not supporting:
        return 0.0
    return sum(1 for t in supporting if t in retrieved) / len(supporting)


def per_question_breakdown(trad: list[dict], ma: list[dict], dataset: dict) -> None:
    ma_by_q = {p["question"]: p for p in ma}

    tiers: dict[str, dict] = {
        "multi-hop":      {"trad": [], "ma": [], "trad_wins": 0, "ma_wins": 0, "ties": 0},
        "hard-multi-hop": {"trad": [], "ma": [], "trad_wins": 0, "ma_wins": 0, "ties": 0},
    }

    print(f"\n{'='*110}")
    print("Per-Question Breakdown")
    print(f"{'='*110}")
    print(f"{'#':<4}{'Difficulty':<17}{'Question':<48}{'T-RAG':>8}{'MA-RAG':>8}{'Winner':>10}")
    print("-" * 110)

    trad_total = ma_total = tie_total = 0

    for i, t in enumerate(trad, 1):
        q    = t["question"]
        gold = t["gold"]
        m    = ma_by_q.get(q, {})

        t_f1 = token_f1(t["prediction"], gold)
        m_f1 = token_f1(m.get("prediction", ""), gold) if m else 0.0

        difficulty = dataset.get(q, {}).get("difficulty", "multi-hop")
        tier = tiers.get(difficulty, tiers["multi-hop"])
        tier["trad"].append(t_f1)
        tier["ma"].append(m_f1)

        if t_f1 > m_f1 + 0.05:
            winner = "T-RAG"
            tier["trad_wins"] += 1
            trad_total += 1
        elif m_f1 > t_f1 + 0.05:
            winner = "MA-RAG"
            tier["ma_wins"] += 1
            ma_total += 1
        else:
            winner = "Tie"
            tier["ties"] += 1
            tie_total += 1

        diff_label = f"[{difficulty}]"
        print(f"{i:<4}{diff_label:<17}{q[:47]:<48}{t_f1:>8.3f}{m_f1:>8.3f}{winner:>10}")

    print(f"\n{'='*110}")
    print(f"  Total  — T-RAG wins: {trad_total}  |  MA-RAG wins: {ma_total}  |  Ties: {tie_total}")

    # Per-tier summary
    print(f"\n{'='*110}")
    print("Results by Difficulty Tier")
    print(f"{'='*110}")
    for tier_name, data in tiers.items():
        n = len(data["trad"])
        if n == 0:
            continue
        t_avg = sum(data["trad"]) / n
        m_avg = sum(data["ma"])   / n
        delta = m_avg - t_avg
        sign  = "+" if delta >= 0 else ""
        print(f"\n  [{tier_name}]  {n} questions")
        print(f"    Traditional RAG  Token F1 : {t_avg:.4f} ({t_avg*100:.1f}%)")
        print(f"    MA-RAG           Token F1 : {m_avg:.4f} ({m_avg*100:.1f}%)")
        print(f"    MA-RAG delta              : {sign}{delta:.4f} ({sign}{delta*100:.1f}%)")
        print(f"    Question wins    T-RAG / MA-RAG / Tie : "
              f"{data['trad_wins']} / {data['ma_wins']} / {data['ties']}")
    print(f"{'='*110}")


def judge_summary(trad_path: Path, ma_path: Path) -> None:
    """Print LLM judge scores if judged files exist."""
    trad_j = trad_path.with_suffix(".judged.json")
    ma_j   = ma_path.with_suffix(".judged.json")

    has_trad = trad_j.exists()
    has_ma   = ma_j.exists()

    if not has_trad and not has_ma:
        print("\nNo LLM judge results found yet.")
        print("Run with --llm-judge flag to add them:")
        print("  python evaluation/eval_multihop_traditional.py --llm-judge")
        print("  python evaluation/eval_multihop_ma_rag.py --llm-judge")
        return

    print(f"\n{'='*70}")
    print("LLM-as-a-Judge Summary")
    print(f"{'='*70}")

    keys = ["judge_correctness_score", "judge_faithfulness_score", "judge_relevancy_score"]
    labels = {"judge_correctness_score": "Correctness",
              "judge_faithfulness_score": "Faithfulness",
              "judge_relevancy_score":    "Relevancy"}

    def avg(data: list[dict], key: str) -> float | None:
        vals = [d[key] for d in data if key in d]
        return sum(vals) / len(vals) if vals else None

    header = f"{'Metric':<20}{'T-RAG':>14}{'MA-RAG':>14}"
    print(header)
    print("-" * 50)

    t_data = load_json(trad_j) if has_trad else []
    m_data = load_json(ma_j)   if has_ma   else []

    for key in keys:
        t_val = avg(t_data, key)
        m_val = avg(m_data, key)
        t_str = f"{t_val*100:.1f}%" if t_val is not None else "—"
        m_str = f"{m_val*100:.1f}%" if m_val is not None else "—"
        print(f"  {labels[key]:<18}{t_str:>14}{m_str:>14}")

    print(f"{'='*70}")


def predictions_vs_gold(trad: list[dict], ma: list[dict]) -> None:
    ma_by_q = {p["question"]: p for p in ma}
    print(f"\n{'='*110}")
    print("Predictions vs Gold")
    print(f"{'='*110}")
    for i, t in enumerate(trad, 1):
        q  = t["question"]
        m  = ma_by_q.get(q, {})
        print(f"\nQ{i}: {q}")
        print(f"  Gold   : {t['gold'][:120]}")
        print(f"  T-RAG  : {t['prediction'][:120]}")
        print(f"  MA-RAG : {m.get('prediction','N/A')[:120]}")


def main() -> None:
    trad_path = RESULTS_DIR / "multihop_traditional_rag_preds.json"
    ma_path   = RESULTS_DIR / "multihop_ma_rag_preds.json"

    paths: dict[str, Path] = {}
    if trad_path.exists():
        paths["Traditional RAG"] = trad_path
    else:
        print(f"[WARN] Not found: {trad_path}")
        print("       Run: python evaluation/eval_multihop_traditional.py")

    if ma_path.exists():
        paths["MA-RAG"] = ma_path
    else:
        print(f"[WARN] Not found: {ma_path}")
        print("       Run: python evaluation/eval_multihop_ma_rag.py")

    if not paths:
        sys.exit(1)

    # Overall metric table
    comparison = compare_results(paths, k=5)
    print_comparison_report(comparison)

    if trad_path.exists() and ma_path.exists():
        trad_preds = load_json(trad_path)
        ma_preds   = load_json(ma_path)

        # Build question -> difficulty map
        dataset = {}
        if DATASET_PATH.exists():
            for d in load_json(DATASET_PATH):
                dataset[d["question"]] = d

        per_question_breakdown(trad_preds, ma_preds, dataset)
        judge_summary(trad_path, ma_path)
        predictions_vs_gold(trad_preds, ma_preds)


if __name__ == "__main__":
    main()