"""Persistent ChromaDB index for public spending documents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.embeddings import Embedder


COLLECTION_NAME = "federal_spending_docs"


@dataclass(frozen=True)
class SearchResult:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def score(self) -> float:
        return 1.0 - self.distance


def chunk_id(chunk: dict[str, Any]) -> str:
    identity = "|".join(
        [str(chunk["source"]), str(chunk["chunk_index"]), str(chunk["text"])]
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def read_chunks(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class VectorStore:
    def __init__(self, persist_dir: str | Path = "data/chroma") -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))

    def rebuild(
        self,
        chunks: list[dict[str, Any]],
        embedder: Embedder,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = 64,
    ) -> int:
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[chunk_id(chunk) for chunk in batch],
                embeddings=embedder.encode_documents(chunk["text"] for chunk in batch),
                documents=[chunk["text"] for chunk in batch],
                metadatas=[
                    {
                        key: value
                        for key, value in chunk.items()
                        if key not in {"text", "source", "section", "chunk_index"}
                        and isinstance(value, (str, int, float, bool))
                    }
                    | {
                        "source": chunk["source"],
                        "section": chunk["section"],
                        "chunk_index": int(chunk["chunk_index"]),
                    }
                    for chunk in batch
                ],
            )
        return collection.count()

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 3,
        collection_name: str = COLLECTION_NAME,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        try:
            collection = self.client.get_collection(collection_name)
        except KeyError as exc:
            if str(exc).strip("'") == "_type":
                raise RuntimeError(
                    "The Chroma index was created with an incompatible Chroma version. "
                    "Rebuild it with: python -m src.index"
                ) from exc
            raise
        result = collection.query(
            query_embeddings=[embedder.encode_query(query)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(record_id, document, metadata or {}, float(distance))
            for record_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
        ]
