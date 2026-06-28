import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CORPUS_PATH = "corpus/hotpot_corpus_small.jsonl"

INDEX_DIR = "T-RAG/indexes"
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = []
texts = []

print("Loading corpus...")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)

        documents.append(doc)
        texts.append(doc["text"])

print(f"Loaded {len(texts)} documents")

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype(np.float32)

dimension = embeddings.shape[1]

print(f"Embedding dimension: {dimension}")

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(
    index,
    f"{INDEX_DIR}/hotpot_small.faiss"
)

with open(f"{INDEX_DIR}/hotpot_small_docs.pkl", "wb") as f:
    pickle.dump(documents, f)

print("\nDone!")
print(f"Indexed docs: {index.ntotal}")