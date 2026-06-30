import os

from utils.provider_manager import ProviderManager

provider = ProviderManager()

VERBOSE = os.getenv("MARAG_VERBOSE", "0") == "1"

FINAL_PROMPT = """
You are the Final Answer Agent in a Multi-Agent RAG system.

You are given:
1. The user's original question.
2. The reasoning history (each step's goal and its answer).

Your task: synthesize ONE complete, self-contained answer to the
original question using ONLY the reasoning history.

RULES:
- Address EVERY part of the question. Multi-part questions (e.g. "name X
  and the Y that replaced it", comparisons, multi-hop chains) require an
  answer that covers all parts.
- Match the expected answer format and length:
    * A "what year / who / which" question → give the direct fact.
    * A "how do X and Y differ" or "name X and the component that..."
      question → answer in one or two complete sentences that state
      every required fact.
- Be factual and grounded ONLY in the reasoning history. Do not add
  outside knowledge or speculation.
- Do NOT include preamble such as "The answer is", "Based on the
  history", or "According to". Start directly with the answer.

Examples:

Question: What year did AlexNet achieve a breakthrough?
Answer: 2012

Question: The architecture in 'Attention Is All You Need' eliminated a
sequential mechanism. Name it and the component that replaced it.
Answer: It eliminated recurrence; self-attention replaced it, allowing all
positions to be processed in parallel.

Original Question:
{question}

Reasoning History:
{history}

Final Answer:
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
        history=history,
    )

    answer = provider.generate(prompt, temperature=0)

    # Strip any accidental preamble the LLM adds despite instructions.
    for prefix in [
        "Final Answer:", "final answer:", "Answer:", "answer:",
        "The answer is", "Based on", "According to",
    ]:
        if answer.strip().startswith(prefix):
            answer = answer.strip()[len(prefix):].strip()
            break

    if VERBOSE:
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(answer)

    state["final_answer"] = answer
    return state
