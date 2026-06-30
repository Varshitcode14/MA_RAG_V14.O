from typing import TypedDict


class GraphState(TypedDict, total=False):

    question: str

    plan: list[str]

    current_step: int

    current_goal: str

    subquery: str

    retrieved_docs: list

    evidence: str

    current_answer: str

    step_answers: list

    history: list

    final_answer: str

    # ------------------------------------------------------------------
    # Accumulators (declared in the schema so LangGraph persists them
    # across nodes and into the final returned state).
    #
    # NOTE: keys NOT declared here are dropped by LangGraph on output,
    # which previously caused retrieved_titles to come back empty and
    # all MA-RAG retrieval metrics to be 0. Declaring them fixes that.
    # ------------------------------------------------------------------

    # Unique document titles retrieved across ALL reasoning steps.
    all_retrieved_titles: list

    # All document dicts retrieved across ALL reasoning steps.
    all_retrieved_docs: list

    # Accumulated wall-clock time (seconds) spent inside the retriever.
    retrieval_time: float
