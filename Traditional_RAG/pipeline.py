"""
Traditional RAG pipeline.

Fix: TraditionalRAGPipeline is now a singleton-friendly class.
The embedder and FAISS store are initialized ONCE when the pipeline
is first created, then reused for all subsequent questions.
This prevents the "paging file too small" Windows error that occurs
when the embedding model is loaded fresh for every question.
"""

import time

from Traditional_RAG.retriever import TraditionalRetriever
from Traditional_RAG.generator import TraditionalGenerator
from Traditional_RAG.config import TOP_K


class TraditionalRAGPipeline:

    _instance = None  # module-level singleton

    def __init__(self) -> None:
        self.retriever  = TraditionalRetriever()
        self.generator  = TraditionalGenerator()

    def answer(self, question: str, top_k: int = TOP_K) -> dict:
        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        docs = self.retriever.search(question, top_k=top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        context = "\n\n".join(
            f"Title: {d['title']}\n{d['text']}" for d in docs
        )

        generation_start = time.perf_counter()
        answer = self.generator.generate(question=question, docs=docs)
        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - total_start

        return {
            "question":         question,
            "answer":           answer,
            "retrieved_docs":   docs,
            "retrieved_titles": [d["title"] for d in docs],
            "context":          context,
            "retrieval_time":   retrieval_time,
            "generation_time":  generation_time,
            "total_time":       total_time,
            "history":          [],
            "reasoning_steps":  1,
            "pipeline":         "traditional_rag",
        }


# Module-level singleton — created once, reused across all questions
_pipeline: TraditionalRAGPipeline | None = None


def run(question: str, top_k: int = TOP_K) -> dict:
    """
    Module-level entry point.

    The pipeline (embedder + FAISS store) is initialized once on
    first call and reused for all subsequent calls in the same process.
    This is critical on Windows where repeated model loading exhausts
    the paging file.
    """
    global _pipeline
    if _pipeline is None:
        print("Initializing Traditional RAG pipeline (once)...")
        _pipeline = TraditionalRAGPipeline()
        print("Pipeline ready.")
    return _pipeline.answer(question, top_k=top_k)