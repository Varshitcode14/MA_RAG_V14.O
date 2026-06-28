from utils.provider_manager import ProviderManager


class TLLM:

    def __init__(self):

        self.pm = ProviderManager()

    def generate(
        self,
        question,
        context
    ):

        prompt = f"""
You are a question answering system.

Use ONLY the provided context.

If the answer is not present,
say:

I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

        return self.pm.generate(
            prompt=prompt,
            temperature=0
        )