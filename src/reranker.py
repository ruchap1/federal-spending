"""Cross-encoder reranking for a small retrieved candidate set."""

from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from src.vector_store import SearchResult


DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name, model_kwargs={"attn_implementation": "eager"})

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        if not candidates:
            return []
        scores = self.model.predict([(query, result.text) for result in candidates])
        ranked: list[SearchResult] = []
        for result, raw_score in zip(candidates, scores):
            if not math.isfinite(float(raw_score)):
                metadata = dict(result.metadata) | {"reranker_status": "nonfinite_score_fallback"}
                ranked.append(SearchResult(result.id, result.text, metadata, result.distance))
                continue
            probability = 1.0 / (1.0 + math.exp(-float(raw_score)))
            metadata = dict(result.metadata)
            metadata["reranker_score"] = probability
            base_score = max(0.0, min(1.0, result.score))
            combined_score = 0.20 * probability + 0.80 * base_score
            metadata["combined_score"] = combined_score
            ranked.append(SearchResult(result.id, result.text, metadata, 1.0 - combined_score))
        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[:top_k]
