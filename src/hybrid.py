"""Hybrid pipeline: document RAG for explanations and API data for numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.embeddings import Embedder
from src.generator import LocalGenerator
from src.logging_utils import log_interaction
from src.prompts import build_document_prompt
from src.reranker import Reranker
from src.spending_api import AgencyMatch, AgencySpending, USAspendingClient
from src.vector_store import SearchResult, VectorStore
from src.validation import ABSTENTION, validate_answer, attribute_answer
from src.evidence import select_evidence, evidence_answer


NUMERIC_TERMS = re.compile(r"\b(how much|spend\w*|obligat\w*|outlay\w*|budgetary|dollars?|funding)\b", re.I)


def is_numeric_question(question: str) -> bool:
    """Return whether a question appears to ask for a spending quantity."""
    if re.search(r"\b(mean|definition|define)\b", question, re.I):
        return False
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

        years = sorted(set(int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", question)))
        numeric = bool(re.search(r"how much|compare|change|difference|total|amount", question, re.I)) and is_numeric_question(question)
        if numeric:
            if not agency_query.strip() or (not years and fiscal_year is None):
                answer = "Please provide an agency name and fiscal year for an exact spending lookup."
                log_interaction(log_path, question, "clarification", answer)
                return HybridResponse(question, "clarification", answer, [])
            if len(years) > 2:
                return HybridResponse(question, "clarification", "Please compare at most two fiscal years.", [])
            matches = self.spending_client.find_funding_agencies(agency_query)
            exact = [a for a in matches if agency_query.strip().lower() in {a.name.lower(), a.abbreviation.lower()}]
            if len(exact) == 1:
                matches = exact
            if len(matches) != 1:
                return HybridResponse(question, "clarification", "Please specify one unambiguous top-level agency name.", [])
            agency = matches[0]
            values = [self.spending_client.agency_budgetary_resources(agency.agency_code, year)
                      for year in (years or [fiscal_year])]
            answer = "\n\n".join(self._numeric_answer("", agency, row) for row in values)
            if len(values) == 2:
                for metric in ("budgetary_resources", "obligations", "outlays"):
                    before, after = (getattr(row, metric) for row in values)
                    if before is not None and after is not None:
                        change = after - before
                        percent = f" ({change / before * 100:+.2f}%)" if before else " (percentage unavailable: zero baseline)"
                        answer += f"\n- {metric.replace('_', ' ').title()} change, FY {values[0].fiscal_year} to FY {values[1].fiscal_year}: ${change:+,.2f}{percent}"
            log_interaction(log_path, question, "structured_api", answer,
                            structured_data={"agency": agency.name, "years": [row.__dict__ for row in values]})
            return HybridResponse(question, "structured_api", answer, [], agency, values[-1])

        candidate_count = 20
        results = self.store.search_hybrid(question, self.embedder, top_k=candidate_count)
        if self.reranker:
            results = self.reranker.rerank(question, results, top_k=10)
        results = select_evidence(question, results)
        if not results:
            log_interaction(log_path, question, "document_rag", ABSTENTION)
            return HybridResponse(question, "document_rag", ABSTENTION, [])
        answer = self.generator.answer_documents(question, results)
        if re.search(r"insufficient evidence|not enough (?:evidence|information)|cannot (?:answer|determine)", answer, re.I):
            answer = ABSTENTION
        else:
            answer = attribute_answer(answer, results)
        if answer == ABSTENTION or validate_answer(answer, results):
            answer = evidence_answer(results)
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
