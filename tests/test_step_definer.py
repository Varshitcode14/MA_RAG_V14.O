from MA_RAG.agents.step_definer import step_definer_agent

state = {

    "question": "Where was the only European Cup Final in which Jupp Heynckes played held?",

    "current_goal": "Find the year in which Jupp Heynckes played a European Cup Final.",

    "history": [],

    "subquery": ""

}

state = step_definer_agent(state)

print()

print(state["subquery"])