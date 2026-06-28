from typing import TypedDict


class GraphState(TypedDict):

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