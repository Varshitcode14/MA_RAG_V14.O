from MA_RAG.retrieval.retriever import DenseRetriever

from MA_RAG.agents.extractor import extractor_agent
from MA_RAG.agents.qa_agent import qa_agent


retriever = DenseRetriever()

query = "What is Retrieval-Augmented Generation?"

docs = retriever.search(query)

state = {

    "question": query,

    "plan": [],

    "current_step": 0,

    "current_goal": query,

    "subquery": [query],

    "retrieved_docs": docs,

    "evidence": "",

    "step_answers": [],

    "history": [],

    "final_answer": ""

}

state = extractor_agent(state)

answer = qa_agent(state)

# -----------------------------
# Graph (simulated)
# -----------------------------

state["step_answers"].append(answer)

state["history"].append({

    "step": state["current_step"] + 1,

    "goal": state["current_goal"],

    "answer": answer

})

print()

print("=" * 60)
print("UPDATED HISTORY")
print("=" * 60)

print(state["history"])