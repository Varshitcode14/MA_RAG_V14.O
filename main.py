"""
RAG Research Project — Main Entry Point

Provides an interactive menu to:
    1. Run Traditional RAG (single question)
    2. Run Multi-Agent RAG (single question)
    3. Evaluate Traditional RAG (HotpotQA dev set)
    4. Evaluate MA-RAG (HotpotQA dev set)
    5. Compare pipeline results
    6. Exit

This file orchestrates pipelines; business logic lives in the
respective packages (Traditional_RAG/, MA_RAG/, evaluation/).
"""

from __future__ import annotations

import sys
from pathlib import Path


# ── Menu ─────────────────────────────────────────────────────────────

_MENU = """
╔══════════════════════════════════════════╗
║         RAG Research Project             ║
╠══════════════════════════════════════════╣
║  1. Traditional RAG  (single question)   ║
║  2. Multi-Agent RAG  (single question)   ║
║  3. Evaluate Traditional RAG             ║
║  4. Evaluate MA-RAG                      ║
║  5. Compare Results                      ║
║  6. Exit                                 ║
╚══════════════════════════════════════════╝
"""


def _prompt_question() -> str:
    question = input("\nEnter your question: ").strip()
    if not question:
        print("Question cannot be empty.")
        sys.exit(1)
    return question


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [default {default}]: ").strip()
    return int(raw) if raw.isdigit() else default


def _prompt_yes(label: str) -> bool:
    raw = input(f"{label} [y/N]: ").strip().lower()
    return raw in ("y", "yes")


# ── Handlers ─────────────────────────────────────────────────────────

def _run_traditional_rag() -> None:
    from Traditional_RAG.pipeline import run

    question = _prompt_question()

    print("\nRunning Traditional RAG ...")
    result = run(question)

    print("\n" + "=" * 60)
    print("RESULT — Traditional RAG")
    print("=" * 60)
    print(f"Question : {result['question']}")
    print(f"Answer   : {result['answer']}")
    print(f"\nRetrieved Documents ({len(result['retrieved_titles'])}):")
    for i, title in enumerate(result["retrieved_titles"], start=1):
        print(f"  {i}. {title}")
    print(f"\nTiming:")
    print(f"  Retrieval  : {result['retrieval_time']:.3f}s")
    print(f"  Generation : {result['generation_time']:.3f}s")
    print(f"  Total      : {result['total_time']:.3f}s")


def _run_ma_rag() -> None:
    from MA_RAG.pipeline import run

    question = _prompt_question()

    print("\nRunning Multi-Agent RAG ...")
    result = run(question)

    print("\n" + "=" * 60)
    print("RESULT — Multi-Agent RAG")
    print("=" * 60)
    print(f"Question         : {result['question']}")
    print(f"Answer           : {result['answer']}")
    print(f"Reasoning Steps  : {result['reasoning_steps']}")

    if result["history"]:
        print("\nReasoning History:")
        for item in result["history"]:
            print(f"  Step {item['step']}: {item['goal']}")
            print(f"    → {item['answer']}")

    print(f"\nTotal Time : {result['total_time']:.3f}s")


def _evaluate_traditional_rag() -> None:
    from Traditional_RAG.pipeline import run as trad_run
    from evaluation.runner import (
        run_inference,
        evaluate_predictions,
        print_metrics_report,
    )

    n = _prompt_int("Number of samples", 20)
    k = _prompt_int("Retrieval cut-off K", 5)
    llm_judge = _prompt_yes("Run LLM-as-a-Judge? (slow)")

    pred_path = run_inference(
        pipeline_fn=trad_run,
        pipeline_name="traditional_rag",
        num_samples=n,
    )
    metrics = evaluate_predictions(
        predictions_path=pred_path,
        k=k,
        use_llm_judge=llm_judge,
    )
    print_metrics_report(metrics, title="Traditional RAG — Evaluation Results")


def _evaluate_ma_rag() -> None:
    from MA_RAG.pipeline import run as ma_run
    from evaluation.runner import (
        run_inference,
        evaluate_predictions,
        print_metrics_report,
    )

    n = _prompt_int("Number of samples", 20)
    k = _prompt_int("Retrieval cut-off K", 5)
    llm_judge = _prompt_yes("Run LLM-as-a-Judge? (slow)")

    pred_path = run_inference(
        pipeline_fn=ma_run,
        pipeline_name="ma_rag",
        num_samples=n,
    )
    metrics = evaluate_predictions(
        predictions_path=pred_path,
        k=k,
        use_llm_judge=llm_judge,
    )
    print_metrics_report(metrics, title="MA-RAG — Evaluation Results")


def _compare_results() -> None:
    from evaluation.runner import compare_results, print_comparison_report

    results_dir = Path("results")
    k = _prompt_int("Retrieval cut-off K", 5)

    paths: dict[str, Path] = {}

    trad_path = results_dir / "traditional_rag_preds.json"
    if trad_path.exists():
        paths["Traditional RAG"] = trad_path
    else:
        print(f"[WARN] Not found: {trad_path}  — run option 3 first.")

    ma_path = results_dir / "ma_rag_preds.json"
    if ma_path.exists():
        paths["MA-RAG"] = ma_path
    else:
        print(f"[WARN] Not found: {ma_path}  — run option 4 first.")

    if not paths:
        print("No prediction files available. Run evaluations first.")
        return

    comparison = compare_results(paths, k=k)
    print_comparison_report(comparison)


# ── Main loop ─────────────────────────────────────────────────────────

_HANDLERS = {
    "1": _run_traditional_rag,
    "2": _run_ma_rag,
    "3": _evaluate_traditional_rag,
    "4": _evaluate_ma_rag,
    "5": _compare_results,
}


def main() -> None:
    while True:
        print(_MENU)
        choice = input("Select option (1-6): ").strip()

        if choice == "6":
            print("Goodbye.")
            break

        handler = _HANDLERS.get(choice)
        if handler is None:
            print(f"Invalid choice: '{choice}'. Please enter 1–6.")
            continue

        try:
            handler()
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as exc:
            print(f"\n[ERROR] {exc}")
            raise


if __name__ == "__main__":
    main()