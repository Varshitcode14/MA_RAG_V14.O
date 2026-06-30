import os

from utils.provider_manager import ProviderManager

from MA_RAG.prompts.step_definer_prompt import STEP_DEFINER_PROMPT

provider = ProviderManager()

VERBOSE = os.getenv("MARAG_VERBOSE", "0") == "1"


def step_definer_agent(state):

    history = ""

    if state["history"]:

        history += "PREVIOUS STEPS\n\n"

        for item in state["history"]:

            history += (
                f"Step {item['step']}\n"
                f"Goal: {item['goal']}\n"
                f"Answer: {item['answer']}\n\n"
            )

    prompt = STEP_DEFINER_PROMPT.format(

        question=state["question"],

        step=state["current_goal"],

        history=history

    )

    query = provider.generate(prompt)

    if VERBOSE:
        print("\n" + "=" * 60)
        print("CURRENT SUBQUERY")
        print("=" * 60)
        print(query)

    state["subquery"] = query

    return state