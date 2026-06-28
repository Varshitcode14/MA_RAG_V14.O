from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = PROJECT_ROOT / "corpus" / "ai_corpus"

INDEX_DIR = CORPUS_DIR / "indexes"

PROCESSED_DIR = CORPUS_DIR / "processed"

# -----------------------------
# Files
# -----------------------------

FAISS_INDEX = INDEX_DIR / "faiss.index"

METADATA = INDEX_DIR / "metadata.pkl"

CHUNKS = PROCESSED_DIR / "chunks.json"

# -----------------------------
# Models
# -----------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# -----------------------------
# Retrieval
# -----------------------------

TOP_K = 5