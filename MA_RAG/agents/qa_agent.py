import os

from utils.provider_manager import ProviderManager

from MA_RAG.prompts.qa_prompt import QA_PROMPT

provider = ProviderManager()

VERBOSE = os.getenv("MARAG_VERBOSE", "0") == "1"


def qa_agent(state):

    history = ""

    if state["history"]:

        for item in state["history"]:

            history += (
                f"Step {item['step']}\n"
                f"Goal: {item['goal']}\n"
                f"Answer: {item['answer']}\n\n"
            )

    prompt = QA_PROMPT.format(

        goal=state["current_goal"],

        history=history if history else "None",

        evidence=state["evidence"]

    )

    answer = provider.generate(prompt)

    if VERBOSE:
        print("\n" + "=" * 60)
        print("STEP ANSWER")
        print("=" * 60)
        print(answer)

    return answer