import sys
from pathlib import Path

# Add retrieval folder to Python path
sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
        / "T-RAG"
        / "retrieval"
    )
)

from T_retriever import TRetriever

retriever = TRetriever(
    faiss_path="T-RAG/indexes/hotpot_cosine.faiss",
    metadata_path="T-RAG/indexes/metadata.pkl",
    cosine=True
)

queries = [
    "Who wrote Harry Potter?",
    "J. K. Rowling",
    "Justin Timberlake clothing line",
    "Arthur's Magazine",
    "Capital of France"
]

for query in queries:

    print("\n" + "=" * 100)
    print("QUERY:", query)
    print("=" * 100)

    results = retriever.search(query, k=3)

    for i, r in enumerate(results, start=1):

        print(f"\n[{i}] {r['title']}")
        print(f"Score: {r['score']:.4f}")
        print(r["text"][:300])