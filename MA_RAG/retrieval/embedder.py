import os

# sentence-transformers / transformers will try to import a TensorFlow
# backend if TF is present in the environment. With Keras 3 installed that
# import crashes. We only need the PyTorch backend, so disable TF before
# importing. (Must be set BEFORE transformers is first imported.)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

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