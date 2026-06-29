"""
Generator for Traditional RAG.

Takes a question and retrieved documents, formats context,
calls the shared ProviderManager, returns a plain string answer.
"""

from utils.provider_manager import ProviderManager
from Traditional_RAG.prompts import TRADITIONAL_RAG_PROMPT


class TraditionalGenerator:

    def __init__(self) -> None:
        self._provider = ProviderManager()

    def generate(
        self,
        question: str,
        docs: list[dict],
        temperature: float = 0.0,
    ) -> str:
        """
        Format context from retrieved docs and generate an answer.

        Args:
            question:    The user's question.
            docs:        List of retrieved document dicts
                         (must have 'title' and 'text' keys).
            temperature: Sampling temperature passed to the LLM.

        Returns:
            The LLM's answer as a plain string.
        """
        context = self._build_context(docs)

        prompt = TRADITIONAL_RAG_PROMPT.format(
            context=context,
            question=question,
        )

        return self._provider.generate(
            prompt=prompt,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(docs: list[dict]) -> str:
        """Concatenate retrieved documents into a single context block."""
        parts: list[str] = []
        for i, doc in enumerate(docs, start=1):
            title = doc.get("title", f"Document {i}")
            text = doc.get("text", "")
            parts.append(f"[{i}] {title}\n{text}")
        return "\n\n".join(parts)