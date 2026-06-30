import os
import time

from MA_RAG.core.state_manager import get_current_goal
from MA_RAG.agents.step_definer import step_definer_agent
from MA_RAG.retrieval.retriever import DenseRetriever
from MA_RAG.agents.extractor import extractor_agent
from MA_RAG.agents.qa_agent import qa_agent
from MA_RAG.core.state_manager import add_step_answer, next_step, has_more_steps

# Verbose step-by-step tracing. Off by default so evaluation runs stay clean.
# Enable with:  set MARAG_VERBOSE=1   (Windows)  /  export MARAG_VERBOSE=1
VERBOSE = os.getenv("MARAG_VERBOSE", "0") == "1"


def _log(title: str, body: str = "") -> None:
    if not VERBOSE:
        return
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if body:
        print(body)


retriever = DenseRetriever()


def set_current_goal(state):
    goal = get_current_goal(state)
    if goal is None:
        return state
    state["current_goal"] = goal
    _log("CURRENT GOAL", goal)
    return state


def generate_subquery(state):
    return step_definer_agent(state)


def answer_current_step(state):
    answer = qa_agent(state)
    state["current_answer"] = answer
    return state


def extract_evidence(state):
    return extractor_agent(state)


def update_history(state):
    state = add_step_answer(state, state["current_answer"])
    state["current_answer"] = ""
    if VERBOSE:
        body = ""
        for item in state["history"]:
            body += (
                f"Step {item['step']}\n"
                f"Goal   : {item['goal']}\n"
                f"Answer : {item['answer']}\n\n"
            )
        _log("UPDATED HISTORY", body)
    return state


def move_to_next_step(state):
    state = next_step(state)
    _log("NEXT STEP", f"Current Step : {state['current_step']}")
    return state


def route_next_step(state):
    if has_more_steps(state):
        _log(
            "NEXT ACTION",
            f"Moving to reasoning step {state['current_step'] + 1} "
            f"of {len(state['plan'])}",
        )
        return "set_goal"
    _log("NEXT ACTION", "Reasoning completed. Generating final answer.")
    return "final_answer"


def retrieve_documents(state):
    start = time.perf_counter()
    docs = retriever.search(query=state["subquery"], top_k=5)
    elapsed = time.perf_counter() - start

    state["retrieved_docs"] = docs
    state["retrieval_time"] = state.get("retrieval_time", 0.0) + elapsed

    # ------------------------------------------------------------------
    # Accumulate retrieved titles AND docs across ALL reasoning steps.
    # These keys are declared in GraphState so LangGraph persists them
    # into the final state (required for retrieval metrics to be valid).
    # ------------------------------------------------------------------
    all_titles = state.get("all_retrieved_titles", [])
    all_docs = state.get("all_retrieved_docs", [])
    seen = set(all_titles)

    for doc in docs:
        title = doc.get("title", "")
        all_docs.append(doc)
        if title and title not in seen:
            all_titles.append(title)
            seen.add(title)

    state["all_retrieved_titles"] = all_titles
    state["all_retrieved_docs"] = all_docs

    _log("RETRIEVED DOCUMENTS", f"Retrieved {len(docs)} documents.")
    return state
