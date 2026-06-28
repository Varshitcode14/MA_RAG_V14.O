import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer


class TRetriever:

    def __init__(
        self,
        faiss_path,
        metadata_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cosine=True
    ):

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading FAISS index...")
        self.index = faiss.read_index(faiss_path)

        print("Loading metadata...")
        with open(metadata_path, "rb") as f:
            self.docs = pickle.load(f)

        self.cosine = cosine

        print(f"Documents Loaded: {len(self.docs)}")

    def search(self, query, k=5):

        query_vec = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype(np.float32)

        if self.cosine:
            faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(
            query_vec,
            k
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "score": float(score),
                "title": self.docs[idx]["title"],
                "text": self.docs[idx]["text"]
            })

        return results