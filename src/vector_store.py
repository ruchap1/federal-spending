"""Persistent ChromaDB index for public spending documents."""

from __future__ import annotations

import hashlib
import json
import math
import re
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
        collection = self.client.get_collection(collection_name)
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

    def search_hybrid(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 3,
        collection_name: str = COLLECTION_NAME,
        dense_k: int = 20,
    ) -> list[SearchResult]:
        """Combine dense cosine search with lightweight BM25 keyword search."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        collection = self.client.get_collection(collection_name)
        total = collection.count()
        if total == 0:
            return []

        dense_count = min(max(dense_k, top_k), total)
        dense = collection.query(
            query_embeddings=[embedder.encode_query(query)],
            n_results=dense_count,
            include=["documents", "metadatas", "distances"],
        )
        dense_results = [
            SearchResult(record_id, document, metadata or {}, float(distance))
            for record_id, document, metadata, distance in zip(
                dense["ids"][0],
                dense["documents"][0],
                dense["metadatas"][0],
                dense["distances"][0],
            )
        ]

        all_records = collection.get(include=["documents", "metadatas"])
        lexical_scores = self._bm25_scores(query, all_records["documents"])
        lexical_order = sorted(
            range(len(lexical_scores)),
            key=lambda index: lexical_scores[index],
            reverse=True,
        )
        candidate_indexes = set(lexical_order[: max(50, top_k)])
        dense_by_id = {result.id: result for result in dense_results}
        id_by_index = {
            index: record_id for index, record_id in enumerate(all_records["ids"])
        }
        candidate_ids = {id_by_index[index] for index in candidate_indexes}
        candidate_ids.update(dense_by_id)

        max_lexical = max(lexical_scores, default=0.0)
        lexical_by_id = {
            id_by_index[index]: (score / max_lexical if max_lexical else 0.0)
            for index, score in enumerate(lexical_scores)
        }
        dense_by_id_score = {
            result.id: max(0.0, result.score) for result in dense_results
        }

        combined: list[SearchResult] = []
        records_by_id = {
            record_id: (document, metadata or {})
            for record_id, document, metadata in zip(
                all_records["ids"],
                all_records["documents"],
                all_records["metadatas"],
            )
        }
        for record_id in candidate_ids:
            document, metadata = records_by_id[record_id]
            dense_score = dense_by_id_score.get(record_id, 0.0)
            lexical_score = lexical_by_id.get(record_id, 0.0)
            hybrid_score = 0.65 * dense_score + 0.35 * lexical_score
            combined_metadata = dict(metadata)
            combined_metadata.update(
                {"dense_score": dense_score, "keyword_score": lexical_score}
            )
            combined.append(
                SearchResult(
                    record_id,
                    document,
                    combined_metadata,
                    1.0 - hybrid_score,
                )
            )
        combined.sort(key=lambda result: result.score, reverse=True)
        return combined[:top_k]

    @staticmethod
    def _bm25_scores(query: str, documents: list[str]) -> list[float]:
        def normalize_token(token: str) -> str:
            if len(token) > 4 and token.endswith("ies"):
                return token[:-3] + "y"
            if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
                return token[:-1]
            return token

        def tokenize(text: str) -> list[str]:
            return [normalize_token(token) for token in re.findall(r"[a-z0-9]+", text.lower())]

        query_terms = set(tokenize(query))
        tokenized = [tokenize(document) for document in documents]
        lengths = [len(tokens) for tokens in tokenized]
        average_length = sum(lengths) / len(lengths) if lengths else 1.0
        document_frequency = {
            term: sum(term in set(tokens) for tokens in tokenized)
            for term in query_terms
        }
        scores: list[float] = []
        for tokens, length in zip(tokenized, lengths):
            term_counts = {term: tokens.count(term) for term in query_terms}
            score = 0.0
            for term, count in term_counts.items():
                frequency = document_frequency[term]
                if not count or not frequency:
                    continue
                inverse_frequency = math.log(
                    1.0 + (len(documents) - frequency + 0.5) / (frequency + 0.5)
                )
                denominator = count + 1.5 * (
                    1.0 - 0.75 + 0.75 * length / average_length
                )
                score += inverse_frequency * count * 2.5 / denominator
            scores.append(score)
        return scores

