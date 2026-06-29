"""
Configuration for Traditional RAG pipeline.

Intentionally reuses MA_RAG retrieval config so both
pipelines share the same index and embedding model.
"""

from MA_RAG.config.retrieval_config import (
    EMBEDDING_MODEL,
    FAISS_INDEX,
    METADATA,
    TOP_K,
)

__all__ = [
    "EMBEDDING_MODEL",
    "FAISS_INDEX",
    "METADATA",
    "TOP_K",
]