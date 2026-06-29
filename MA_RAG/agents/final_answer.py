from utils.provider_manager import ProviderManager

provider = ProviderManager()

FINAL_PROMPT = """
You are the Final Answer Agent in a Multi-Agent RAG system.

You are given:
1. The user's original question.
2. The reasoning history from previous agents.

Your task: give ONE concise, direct answer.

CRITICAL RULES:
- Return ONLY the answer — no sentences like "The answer is..." or "Based on the history..."
- Match the expected answer format: if the question asks for a name, return just the name.
  If it asks for a year, return just the year. If it asks for a comparison, be brief.
- Maximum 15 words unless the question requires a longer comparison answer.
- Use ONLY information from the reasoning history.
- Do NOT add explanation or preamble.

Examples of CORRECT output:
  Question: What year did AlexNet achieve a breakthrough?
  Answer: 2012

  Question: What models were trained using RLHF?
  Answer: InstructGPT, ChatGPT, and Claude

  Question: How do GPT and BERT differ in training?
  Answer: GPT is trained autoregressively; BERT uses masked language model objective

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

    # Strip any accidental preamble the LLM adds despite instructions
    for prefix in [
        "Final Answer:", "final answer:", "Answer:", "answer:",
        "The answer is", "Based on", "According to",
    ]:
        if answer.strip().startswith(prefix):
            answer = answer.strip()[len(prefix):].strip()
            break

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(answer)

    state["final_answer"] = answer
    return state