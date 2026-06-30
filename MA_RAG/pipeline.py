"""
MA-RAG pipeline entry point.

Wraps the LangGraph graph invocation and normalizes the
raw GraphState into the shared result schema.

retrieved_titles now collects unique titles from ALL reasoning steps
(accumulated in GraphState.all_retrieved_titles by the retriever node).
This is required for retrieval metrics (Precision@K, Recall@K, Hit Rate,
MRR) to be meaningful — in multi-hop RAG, relevant docs are retrieved at
different steps. Previously this used an undeclared state key that
LangGraph silently dropped, so all MA-RAG retrieval metrics were 0.
"""

import time

from MA_RAG.core.graph import graph


def _normalize(state: dict, total_time: float) -> dict:
    """Convert raw GraphState to the shared result schema."""

    # Titles/docs accumulated across every reasoning step.
    all_titles: list[str] = state.get("all_retrieved_titles", [])
    all_docs: list[dict] = state.get("all_retrieved_docs", [])

    # Context is built from every document seen across all steps so the
    # faithfulness judge sees the full evidence MA-RAG actually used.
    context = "\n\n".join(
        f"Title: {d.get('title', '')}\n{d.get('text', '')}"
        for d in all_docs
    )

    retrieval_time = float(state.get("retrieval_time", 0.0))
    # Everything that is not retrieval is dominated by LLM generation
    # across the planner / step-definer / extractor / qa / final agents.
    generation_time = max(total_time - retrieval_time, 0.0)

    return {
        "question":         state.get("question", ""),
        "answer":           state.get("final_answer", ""),
        "retrieved_docs":   all_docs,
        "retrieved_titles": all_titles,
        "context":          context,
        "retrieval_time":   retrieval_time,
        "generation_time":  generation_time,
        "total_time":       total_time,
        "history":          state.get("history", []),
        "reasoning_steps":  len(state.get("plan", [])),
        "pipeline":         "ma_rag",
    }


def run(question: str) -> dict:
    """
    Run the full MA-RAG pipeline for a single question.

    Returns:
        Standardized result dict.
    """
    initial_state = {
        "question":             question,
        "plan":                 [],
        "current_step":         0,
        "current_goal":         "",
        "subquery":             "",
        "retrieved_docs":       [],
        "evidence":             "",
        "step_answers":         [],
        "history":              [],
        "current_answer":       "",
        "final_answer":         "",
        "all_retrieved_titles": [],
        "all_retrieved_docs":   [],
        "retrieval_time":       0.0,
    }

    start = time.perf_counter()
    final_state = graph.invoke(initial_state)
    total_time = time.perf_counter() - start

    return _normalize(final_state, total_time)
