def add_step_answer(state, answer):

    state["step_answers"].append(answer)

    state["history"].append({

        "step": state["current_step"] + 1,

        "goal": state["current_goal"],

        "answer": answer

    })

    return state


def next_step(state):

    state["current_step"] += 1

    # Clear temporary working memory

    state["current_goal"] = ""

    state["subquery"] = ""

    state["retrieved_docs"] = []

    state["evidence"] = ""

    state["current_answer"] = ""

    return state

def has_more_steps(state):

    return state["current_step"] < len(state["plan"])


def get_current_goal(state):

    if state["current_step"] >= len(state["plan"]):
        return None

    return state["plan"][state["current_step"]]