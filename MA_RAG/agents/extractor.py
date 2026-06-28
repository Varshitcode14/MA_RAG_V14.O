from utils.provider_manager import ProviderManager
from MA_RAG.prompts.extractor_prompt import EXTRACTOR_PROMPT

provider = ProviderManager()


def extractor_agent(state):

    docs = ""

    for i, doc in enumerate(state["retrieved_docs"], start=1):

        docs += f"""
DOCUMENT {i}

Title:
{doc['title']}

Content:
{doc['text']}

------------------------------------------------------------

"""

    prompt = EXTRACTOR_PROMPT.format(
        goal=state["current_goal"],
        documents=docs
    )

    evidence = provider.generate(prompt)

    print("\n" + "=" * 60)
    print("EXTRACTED EVIDENCE")
    print("=" * 60)
    print(evidence)

    state["evidence"] = evidence

    return state