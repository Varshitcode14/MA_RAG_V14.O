from MA_RAG.retrieval.embedder import Embedder
from MA_RAG.retrieval.faiss_store import FAISSStore
from MA_RAG.config.retrieval_config import TOP_K


class DenseRetriever:

    def __init__(self):

        print("Creating Embedder...")

        self.embedder = Embedder()

        print("Embedder Ready.")

        print("Creating FAISS Store...")

        self.store = FAISSStore()

        print("FAISS Store Ready.")

    def search(self, query: str, top_k: int = TOP_K):

        embedding = self.embedder.encode(query)

        results = self.store.search(
            embedding=embedding,
            top_k=top_k
        )

        return results