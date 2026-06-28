from utils.provider_manager import ProviderManager

provider = ProviderManager()


FINAL_PROMPT = """
You are the Final Answer Agent in a Multi-Agent Retrieval-Augmented Generation (MA-RAG) system.

You are given:

1. The user's original question.

2. The reasoning history produced by previous agents.

Your task is to synthesize all intermediate reasoning into one clear, accurate and concise final answer.

Rules:

- Use ONLY the reasoning history.
- Do NOT invent facts.
- If the reasoning does not contain enough information, explicitly state that.
- Do not repeat intermediate reasoning.
- Return ONLY the final answer.

Original Question

{question}

Reasoning History

{history}

Final Answer
"""


def final_answer_agent(state):

    history = ""

    for item in state["history"]:

        history += (
            f"Step {item['step']}\n"
            f"Goal: {item['goal']}\n"
            f"Answer: {item['answer']}\n\n"
        )

    prompt = FINAL_PROMPT.format(
        question=state["question"],
        history=history
    )

    answer = provider.generate(prompt)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)

    state["final_answer"] = answer

    return state