import pickle
import faiss

from MA_RAG.config.retrieval_config import (
    FAISS_INDEX,
    METADATA
)


class FAISSStore:

    def __init__(self):

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(FAISS_INDEX)
        )

        print("Loading metadata...")

        with open(
            METADATA,
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

        print(
            f"Loaded {len(self.metadata)} chunks."
        )

    def search(
        self,
        embedding,
        top_k
    ):

        scores, indices = self.index.search(
            embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            item = self.metadata[idx].copy()

            item["score"] = float(score)

            results.append(item)

        return results