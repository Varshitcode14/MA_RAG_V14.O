"""
Traditional RAG pipeline.

Workflow:
    1. Query
    2. Retrieve (reuses MA_RAG DenseRetriever via TraditionalRetriever)
    3. Generate
    4. Return standardized result dict

The result schema is kept identical to the one returned by
MA_RAG's pipeline so the evaluation layer can treat both
systems transparently.
"""

import time

from Traditional_RAG.retriever import TraditionalRetriever
from Traditional_RAG.generator import TraditionalGenerator
from Traditional_RAG.config import TOP_K


class TraditionalRAGPipeline:

    def __init__(self) -> None:
        self.retriever = TraditionalRetriever()
        self.generator = TraditionalGenerator()

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Run the full Traditional RAG pipeline for a single question.

        Returns:
            Standardized result dict (see schema below).

        Result schema
        -------------
        {
            "question":         str,
            "answer":           str,
            "retrieved_docs":   list[dict],
            "retrieved_titles": list[str],
            "context":          str,
            "retrieval_time":   float,   # seconds
            "generation_time":  float,   # seconds
            "total_time":       float,   # seconds
            "history":          list,    # always [] for Traditional RAG
            "reasoning_steps":  int,     # always 1 for Traditional RAG
            "pipeline":         str,     # "traditional_rag"
        }
        """
        total_start = time.perf_counter()

        # ── 1. Retrieve ──────────────────────────────────────────────
        retrieval_start = time.perf_counter()
        docs = self.retriever.search(question, top_k=top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        # ── 2. Build context string (also stored in result) ──────────
        context = "\n\n".join(
            f"Title: {d['title']}\n{d['text']}" for d in docs
        )

        # ── 3. Generate ──────────────────────────────────────────────
        generation_start = time.perf_counter()
        answer = self.generator.generate(question=question, docs=docs)
        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - total_start

        return {
            "question": question,
            "answer": answer,
            "retrieved_docs": docs,
            "retrieved_titles": [d["title"] for d in docs],
            "context": context,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
            "history": [],
            "reasoning_steps": 1,
            "pipeline": "traditional_rag",
        }


def run(question: str, top_k: int = TOP_K) -> dict:
    """
    Module-level convenience function.

    Usage:
        from Traditional_RAG.pipeline import run
        result = run("Who wrote Harry Potter?")
    """
    pipeline = TraditionalRAGPipeline()
    return pipeline.answer(question, top_k=top_k)