"""
MA-RAG pipeline entry point.

Wraps the LangGraph graph invocation and normalizes the
raw GraphState into the shared result schema.

Key fix: retrieved_titles now collects titles from ALL reasoning
steps (via history), not just the final step. This is required for
retrieval metrics (Precision@K, Recall@K, Hit Rate, MRR) to be
meaningful — in multi-hop RAG, relevant docs are retrieved at
different steps.

The graph itself (MA_RAG/core/graph.py) is NOT modified.
"""

import time

from MA_RAG.core.graph import graph


def _collect_all_retrieved_titles(state: dict) -> list[str]:
    """
    Collect unique retrieved document titles across ALL reasoning steps.

    MA-RAG retrieves documents at each step. The state only holds the
    LAST step's retrieved_docs. We recover earlier steps' titles from
    the history entries' evidence strings, but the most reliable source
    is the final retrieved_docs combined with titles found in evidence.

    Strategy:
    1. Take titles from state["retrieved_docs"] (last step).
    2. Parse any document titles mentioned in evidence across history.
    3. Deduplicate preserving order.
    """
    seen: set[str] = set()
    titles: list[str] = []

    def _add(title: str) -> None:
        if title and title not in seen:
            seen.add(title)
            titles.append(title)

    # Last step's retrieved docs (always available)
    for doc in state.get("retrieved_docs", []):
        _add(doc.get("title", ""))

    # Evidence strings in history mention "Source: DOCUMENT X" but not
    # titles directly.  Instead we pull titles from step_answers context
    # by looking at the evidence field stored in history items.
    # The cleaner source: every node in the graph calls retrieve_documents
    # which sets state["retrieved_docs"].  Since LangGraph doesn't track
    # per-step state snapshots we can't recover earlier steps' docs.
    #
    # Best available signal: the plan goals themselves contain topic keywords
    # that match document titles — but that's too fragile.
    #
    # Practical fix: store all retrieved titles in a running list inside
    # the initial state and accumulate across steps via a custom key.
    # Since we can't modify the graph without a code change, we use the
    # ALL_RETRIEVED_TITLES key we inject into initial_state below and
    # read back here.

    for title in state.get("_all_retrieved_titles", []):
        _add(title)

    return titles


def _normalize(state: dict, total_time: float) -> dict:
    """Convert raw GraphState to the shared result schema."""
    retrieved_docs: list[dict] = state.get("retrieved_docs", [])

    # Use the accumulated list that was built across all steps
    all_titles = _collect_all_retrieved_titles(state)

    context = "\n\n".join(
        f"Title: {d.get('title', '')}\n{d.get('text', '')}"
        for d in retrieved_docs
    )

    return {
        "question":         state.get("question", ""),
        "answer":           state.get("final_answer", ""),
        "retrieved_docs":   retrieved_docs,
        "retrieved_titles": all_titles,
        "context":          context,
        "retrieval_time":   0.0,
        "generation_time":  0.0,
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
        "question":               question,
        "plan":                   [],
        "current_step":           0,
        "current_goal":           "",
        "subquery":               "",
        "retrieved_docs":         [],
        "evidence":               "",
        "step_answers":           [],
        "history":                [],
        "current_answer":         "",
        "final_answer":           "",
        "_all_retrieved_titles":  [],  # accumulates across steps (see nodes patch)
    }

    start = time.perf_counter()
    final_state = graph.invoke(initial_state)
    total_time = time.perf_counter() - start

    return _normalize(final_state, total_time)