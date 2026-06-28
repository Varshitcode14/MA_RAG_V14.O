from MA_RAG.retrieval.retriever import DenseRetriever
from MA_RAG.agents.extractor import extractor_agent

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

print()

print("=" * 60)
print("FINAL EVIDENCE")
print("=" * 60)

print(state["evidence"])