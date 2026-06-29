"""
Retriever for Traditional RAG.

Wraps MA_RAG's DenseRetriever so both pipelines share
the same FAISS index, embedder, and metadata.
No retrieval code is duplicated.
"""

from MA_RAG.retrieval.retriever import DenseRetriever
from Traditional_RAG.config import TOP_K


class TraditionalRetriever:
    """
    Thin wrapper around DenseRetriever.

    Exists so Traditional_RAG has its own namespace
    and can be configured independently if needed.
    """

    def __init__(self) -> None:
        self._retriever = DenseRetriever()

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Retrieve top-k documents for a query.

        Returns:
            List of dicts with keys: title, text, score, source.
        """
        return self._retriever.search(query, top_k=top_k)