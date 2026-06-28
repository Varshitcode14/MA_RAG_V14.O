from MA_RAG.retrieval.embedder import Embedder

embedder = Embedder()

embedding = embedder.encode(
    "What is Retrieval-Augmented Generation?"
)

print(embedding.shape)

print(embedding.dtype)