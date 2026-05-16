from __future__ import annotations

import numpy as np


class Encoder:
    """
    Lazy-loading wrapper around sentence-transformers.
    Model is downloaded once and cached by the library in ~/.cache/huggingface.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Return float32 array of shape (N, D)."""
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors → dot product == cosine similarity
        )
        return vectors.astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @staticmethod
    def to_blob(vector: np.ndarray) -> bytes:
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def from_blob(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)
