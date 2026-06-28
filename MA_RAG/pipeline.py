from MA_RAG.core.graph import graph


def run(question):

    state = {

        "question": question,

        "plan": [],

        "current_step": 0,

        "current_goal": "",

        "subquery": "",

        "retrieved_docs": [],

        "evidence": "",

        "step_answers": [],

        "history": [],

        "current_answer": "",

        "final_answer": ""

    }

    return graph.invoke(state)