from sentence_transformers import SentenceTransformer
import numpy as np

from MA_RAG.config.retrieval_config import EMBEDDING_MODEL


class Embedder:

    _model = None

    def __init__(self):

        if Embedder._model is None:

            print("Loading embedding model...")

            Embedder._model = SentenceTransformer(
                EMBEDDING_MODEL
            )

            print("Embedding model ready.")

        self.model = Embedder._model

    def encode(self, text: str):

        embedding = self.model.encode(

            [text],

            convert_to_numpy=True,

            normalize_embeddings=True

        ).astype(np.float32)

        return embedding