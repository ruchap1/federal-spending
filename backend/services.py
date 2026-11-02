"""Application service that adapts the existing RAG pipeline to an HTTP API."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from src.embeddings import Embedder
from src.generator import LocalGenerator
from src.hybrid import HybridPipeline

from .schemas import Citation, ResearchResponse


class ResearchService:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.generator = LocalGenerator()
        self.pipeline = HybridPipeline(self.embedder, self.generator)

    def answer(self, question: str, agency: str = "", fiscal_year: int | None = None) -> ResearchResponse:
        started = perf_counter()
        result = self.pipeline.answer(question, agency, fiscal_year)
        citations = [
            Citation(
                id=match.id,
                title=Path(str(match.metadata.get("source", "Unknown source"))).stem.replace("-", " ").title(),
                section=str(match.metadata.get("section", "Unknown section")),
                source=str(match.metadata.get("source", "Unknown source")),
                excerpt=match.text,
                score=round(match.score, 4),
                url=match.metadata.get("url"),
            )
            for match in result.results
        ]
        structured_data: dict[str, Any] | None = None
        if result.spending:
            structured_data = {
                "agency": result.agency.name if result.agency else agency,
                "fiscal_year": result.spending.fiscal_year,
                "budgetary_resources": result.spending.budgetary_resources,
                "obligations": result.spending.obligations,
                "outlays": result.spending.outlays,
                "source_url": result.spending.source_url,
            }
        return ResearchResponse(
            answer=result.answer,
            mode=result.mode,
            citations=citations,
            structured_data=structured_data,
            metadata={
                "model": self.generator.model_name,
                "retrieved_chunks": len(result.results),
                "latency_ms": round((perf_counter() - started) * 1000),
            },
        )
