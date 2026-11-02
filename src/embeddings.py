"""Sentence-transformer embeddings for spending-document retrieval."""

from __future__ import annotations

from collections.abc import Iterable

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: Iterable[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

