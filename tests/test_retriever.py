from MA_RAG.retrieval.retriever import DenseRetriever


retriever = DenseRetriever()

results = retriever.search(
    "What is Retrieval-Augmented Generation?"
)

print()

print("=" * 80)
print("TOP RETRIEVED DOCUMENTS")
print("=" * 80)

for i, doc in enumerate(results, start=1):

    print(f"\nRank {i}")
    print("-" * 80)

    print("Title :", doc["title"])
    print("Score :", round(doc["score"], 4))
    print("Source:", doc["source"])

    print()

    print(doc["text"][:400])