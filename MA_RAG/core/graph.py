from langgraph.graph import StateGraph, END

from MA_RAG.core.state import GraphState

from MA_RAG.agents.planner import planner_agent
from MA_RAG.agents.final_answer import final_answer_agent

from MA_RAG.core.nodes import (
    set_current_goal,
    generate_subquery,
    retrieve_documents,
    extract_evidence,
    answer_current_step,
    update_history,
    move_to_next_step,
    route_next_step,
)

builder = StateGraph(GraphState)

# ------------------------
# Register Nodes
# ------------------------

builder.add_node("planner", planner_agent)
builder.add_node("set_goal", set_current_goal)
builder.add_node("step_definer", generate_subquery)
builder.add_node("retriever", retrieve_documents)
builder.add_node("extractor", extract_evidence)
builder.add_node("qa", answer_current_step)
builder.add_node("update_history", update_history)
builder.add_node("next_step", move_to_next_step)
builder.add_node("final_answer", final_answer_agent)

# ------------------------
# Entry
# ------------------------

builder.set_entry_point("planner")

# ------------------------
# Linear Flow
# ------------------------

builder.add_edge("planner", "set_goal")
builder.add_edge("set_goal", "step_definer")
builder.add_edge("step_definer", "retriever")
builder.add_edge("retriever", "extractor")
builder.add_edge("extractor", "qa")
builder.add_edge("qa", "update_history")
builder.add_edge("update_history", "next_step")

# ------------------------
# Loop
# ------------------------

builder.add_conditional_edges(
    "next_step",
    route_next_step,
    {
        "set_goal": "set_goal",
        "final_answer": "final_answer",
    },
)

# ------------------------
# Finish
# ------------------------

builder.add_edge("final_answer", END)

graph = builder.compile()