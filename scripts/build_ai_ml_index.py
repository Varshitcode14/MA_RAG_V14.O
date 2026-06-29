"""
Build FAISS index from the AI/ML corpus.

Reads documents from corpus/ai_ml/documents.py,
encodes them with the same embedder used by MA_RAG,
and writes the index + metadata to MA_RAG/indexes/.

Run from repo root:
    python scripts/build_ai_ml_index.py
"""

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus.ai_ml.documents import DOCUMENTS
from MA_RAG.config.retrieval_config import (
    EMBEDDING_MODEL,
    FAISS_INDEX,
    METADATA,
    INDEX_DIR,
    PROCESSED_DIR,
    CHUNKS,
)
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────
BATCH_SIZE = 32


def chunk_document(doc: dict) -> list[dict]:
    """
    Each document is one chunk for now (texts are already short passages).
    Extend this function if you want sentence-level chunking later.
    """
    return [
        {
            "title": doc["title"],
            "text": doc["text"],
            "source": "ai_ml_corpus",
            "tags": doc.get("tags", []),
        }
    ]


def main() -> None:
    print("=" * 60)
    print("Building AI/ML FAISS Index")
    print("=" * 60)

    # ── Build chunks ─────────────────────────────────────────────────
    chunks: list[dict] = []
    for doc in DOCUMENTS:
        chunks.extend(chunk_document(doc))

    print(f"Documents : {len(DOCUMENTS)}")
    print(f"Chunks    : {len(chunks)}")

    # ── Create directories ───────────────────────────────────────────
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Save chunks as JSON ──────────────────────────────────────────
    with open(CHUNKS, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunks saved to : {CHUNKS}")

    # ── Encode ───────────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]

    print("Encoding chunks ...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine via dot product
    ).astype(np.float32)

    print(f"Embedding shape : {embeddings.shape}")

    # ── Build FAISS index (cosine = normalised dot product) ──────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product on L2-normalised = cosine
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX))
    print(f"FAISS index saved to : {FAISS_INDEX}  ({index.ntotal} vectors)")

    # ── Save metadata ────────────────────────────────────────────────
    with open(METADATA, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Metadata saved to    : {METADATA}")

    print("\nDone. Index is ready.")


if __name__ == "__main__":
    main()