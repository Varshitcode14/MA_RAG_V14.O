from MA_RAG.core.state_manager import (
    add_step_answer,
    next_step,
    has_more_steps,
    get_current_goal
)

state = {

    "plan": [

        "Step 1",

        "Step 2"

    ],

    "current_step": 0,

    "current_goal": "Step 1",

    "step_answers": [],

    "history": []

}

print(get_current_goal(state))

state = add_step_answer(
    state,
    "Answer 1"
)

print(state["history"])

state = next_step(state)

print(state["current_step"])

print(has_more_steps(state))

print(get_current_goal(state))