from MA_RAG.core.state_manager import get_current_goal
from MA_RAG.agents.step_definer import step_definer_agent
from MA_RAG.retrieval.retriever import DenseRetriever
from MA_RAG.agents.extractor import extractor_agent
from MA_RAG.agents.qa_agent import qa_agent
from MA_RAG.core.state_manager import add_step_answer, next_step, has_more_steps

retriever = DenseRetriever()

def set_current_goal(state):
    goal = get_current_goal(state)
    if goal is None:
        return state
    state["current_goal"] = goal
    print("\n" + "=" * 60)
    print("CURRENT GOAL")
    print("=" * 60)
    print(goal)
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
    print("\n" + "=" * 60)
    print("UPDATED HISTORY")
    print("=" * 60)
    for item in state["history"]:
        print(f"Step {item['step']}")
        print(f"Goal   : {item['goal']}")
        print(f"Answer : {item['answer']}")
        print()
    return state

def move_to_next_step(state):
    state = next_step(state)
    print("\n" + "=" * 60)
    print("NEXT STEP")
    print("=" * 60)
    print(f"Current Step : {state['current_step']}")
    return state

def route_next_step(state):
    if has_more_steps(state):
        print("\n" + "=" * 60)
        print("NEXT ACTION")
        print("=" * 60)
        print(f"Moving to reasoning step {state['current_step'] + 1} of {len(state['plan'])}")
        return "set_goal"
    print("\n" + "=" * 60)
    print("NEXT ACTION")
    print("=" * 60)
    print("Reasoning completed. Generating final answer.")
    return "final_answer"

def retrieve_documents(state):
    docs = retriever.search(query=state["subquery"], top_k=5)
    state["retrieved_docs"] = docs

    # Accumulate titles across ALL steps for evaluation metrics
    if "_all_retrieved_titles" not in state:
        state["_all_retrieved_titles"] = []
    existing = set(state["_all_retrieved_titles"])
    for doc in docs:
        title = doc.get("title", "")
        if title and title not in existing:
            state["_all_retrieved_titles"].append(title)
            existing.add(title)

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)
    print(f"Retrieved {len(docs)} documents.")
    return state