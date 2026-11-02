"""Hybrid pipeline: document RAG for explanations and API data for numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.embeddings import Embedder
from src.generator import LocalGenerator
from src.logging_utils import log_interaction
from src.prompts import build_document_prompt, build_numeric_prompt
from src.reranker import Reranker
from src.spending_api import AgencyMatch, AgencySpending, USAspendingClient
from src.vector_store import SearchResult, VectorStore


NUMERIC_TERMS = re.compile(r"\b(how much|spend|spending|obligat|outlay|budgetary|dollar|funding)\b", re.I)


def is_numeric_question(question: str) -> bool:
    """Return whether a question appears to ask for a spending quantity."""
    return bool(NUMERIC_TERMS.search(question))


@dataclass(frozen=True)
class HybridResponse:
    question: str
    mode: str
    answer: str
    results: list[SearchResult]
    agency: AgencyMatch | None = None
    spending: AgencySpending | None = None


class HybridPipeline:
    def __init__(
        self,
        embedder: Embedder,
        generator: LocalGenerator,
        store: VectorStore | None = None,
        spending_client: USAspendingClient | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.store = store or VectorStore()
        self.spending_client = spending_client or USAspendingClient()
        self.reranker = reranker

    def answer(
        self,
        question: str,
        agency_query: str = "",
        fiscal_year: int | None = None,
        log_path: str = "logs/interactions.jsonl",
    ) -> HybridResponse:
        if not question.strip():
            raise ValueError("question must not be empty")

        if is_numeric_question(question) and agency_query.strip() and fiscal_year:
            matches = self.spending_client.find_funding_agencies(agency_query)
            if matches:
                agency = matches[0]
                spending = self.spending_client.agency_budgetary_resources(
                    agency.agency_code, fiscal_year
                )
                data = {
                    "budgetary_resources": spending.budgetary_resources,
                    "obligations": spending.obligations,
                    "outlays": spending.outlays,
                    "source_url": spending.source_url,
                }
                prompt = build_numeric_prompt(question, agency.name, fiscal_year, data)
                answer = self.generator.generate(prompt)
                answer = self._numeric_answer(answer, agency, spending)
                log_interaction(log_path, question, "structured_api", answer, structured_data=data)
                return HybridResponse(question, "structured_api", answer, [], agency, spending)

        candidate_count = 20 if self.reranker else 3
        results = self.store.search_hybrid(question, self.embedder, top_k=candidate_count)
        if self.reranker:
            results = self.reranker.rerank(question, results, top_k=3)
        answer = self.generator.generate(build_document_prompt(question, results))
        if not answer or len(answer.strip()) < 20:
            answer = f"The retrieved evidence is:\n\n{results[0].text[:700]} [1]" if results else "Not answerable from the indexed evidence."
        elif not re.search(r"\[\d+\]", answer):
            answer = f"{answer.rstrip()} [1]"
        log_interaction(log_path, question, "document_rag", answer, results=results)
        return HybridResponse(question, "document_rag", answer, results)

    @staticmethod
    def _numeric_answer(answer: str, agency: AgencyMatch, spending: AgencySpending) -> str:
        # The generated prose is retained, but the authoritative values are
        # appended directly from the API so the model cannot alter numbers.
        prose = answer.strip() or "The structured USAspending lookup returned these values."
        values = (
            f"\n\nAuthoritative USAspending.gov values for {agency.name} (FY {spending.fiscal_year}):\n"
            f"- Budgetary resources: {HybridPipeline._format_amount(spending.budgetary_resources)}\n"
            f"- Obligations: {HybridPipeline._format_amount(spending.obligations)}\n"
            f"- Outlays: {HybridPipeline._format_amount(spending.outlays)}\n"
            f"Source: [USAspending API]({spending.source_url})"
        )
        return f"{prose}\n{values}"

    @staticmethod
    def _format_amount(value: float | None) -> str:
        return "unavailable" if value is None else f"{value:,.2f}"
