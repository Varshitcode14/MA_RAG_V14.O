import ast

from utils.provider_manager import ProviderManager

from MA_RAG.prompts.planner_prompt import PLANNER_PROMPT


provider = ProviderManager()


def planner_agent(state):

    prompt = PLANNER_PROMPT.format(
        question=state["question"]
    )

    response = provider.generate(prompt)

    try:

        plan = ast.literal_eval(response)

        if not isinstance(plan, list):
            raise ValueError()

    except Exception:

        plan = [state["question"]]

    state["plan"] = plan

    state["current_step"] = 0

    # if plan:
    #     state["current_goal"] = plan[0]
    # else:
    #     state["current_goal"] = ""

    return state