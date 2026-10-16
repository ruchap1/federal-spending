"""Hybrid pipeline: document RAG for explanations and API data for numbers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.embeddings import Embedder
from src.generator import LocalGenerator
from src.logging_utils import log_interaction
from src.prompts import build_document_prompt, build_numeric_prompt
from src.spending_api import AgencyMatch, AgencySpending, USAspendingClient
from src.vector_store import SearchResult, VectorStore


NUMERIC_TERMS = re.compile(r"\b(how much|spend|spending|obligat|outlay|budgetary|dollar|funding)\b", re.I)
CONCEPT_TERMS = re.compile(r"\b(obligation|obligations|outlay|outlays)\b", re.I)
USA_SPENDING_GUIDE = "https://www.usaspending.gov/federal-spending-guide"
AGENCY_ALIASES = {
    "fema": "Federal Emergency Management Agency",
    "nasa": "National Aeronautics and Space Administration",
    "noaa": "National Oceanic and Atmospheric Administration",
    "usda": "Department of Agriculture",
    "hhs": "Department of Health and Human Services",
    "hud": "Department of Housing and Urban Development",
    "dod": "Department of Defense",
}
AGENCY_IN_QUESTION = re.compile(
    r"\b((?:(?:department|agency|administration|bureau|office|national)\b|"
    r"(?:fema|nasa|noaa|usda|hhs|hud|dod)\b)"
    r"[A-Za-z0-9 .&'/-]*?)(?=\s+(?:obligat|spend|outlay|fund|in fiscal|for fiscal|in fy|for fy)|[?.!,]|$)",
    re.I,
)


def is_numeric_question(question: str) -> bool:
    """Return whether a question appears to ask for a spending quantity."""
    return bool(NUMERIC_TERMS.search(question))


def infer_agency(question: str) -> str:
    """Extract a likely agency name when the user includes it in the question."""
    match = AGENCY_IN_QUESTION.search(question)
    return canonicalize_agency(match.group(1).strip()) if match else ""


def canonicalize_agency(value: str) -> str:
    return AGENCY_ALIASES.get(value.strip().lower(), value.strip())


def is_obligation_outlay_definition(question: str) -> bool:
    lowered = question.lower()
    return bool(CONCEPT_TERMS.search(lowered)) and "outlay" in lowered and any(
        word in lowered for word in ("difference", "different", "what", "how")
    )


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
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.store = store or VectorStore()
        self.spending_client = spending_client or USAspendingClient()

    def answer(
        self,
        question: str,
        agency_query: str = "",
        fiscal_year: int | None = None,
        log_path: str = "logs/interactions.jsonl",
    ) -> HybridResponse:
        if not question.strip():
            raise ValueError("question must not be empty")

        if is_obligation_outlay_definition(question):
            answer = (
                "An obligation is a promise made by the government to spend funds. "
                "It is created when an agency takes an action that requires a payment, "
                "such as signing a contract, awarding a grant, or placing an order.\n\n"
                "An outlay is money that has actually been paid.\n\n"
                "The key difference is timing: an obligation represents a commitment to spend, "
                "while an outlay represents the payment itself. The payment can happen later, "
                "so outlays in a fiscal year are not necessarily a subset of that year's obligations."
            )
            glossary = SearchResult(
                "usa-spending-obligations-outlays",
                answer,
                {
                    "source": "USAspending Analyst's Guide",
                    "section": "Obligations and outlays",
                    "url": USA_SPENDING_GUIDE,
                },
                0.0,
            )
            log_interaction(log_path, question, "document_rag", answer, results=[glossary])
            return HybridResponse(question, "document_rag", answer, [glossary])

        resolved_agency = canonicalize_agency(agency_query) if agency_query.strip() else infer_agency(question)
        if is_numeric_question(question) and not resolved_agency:
            answer = (
                "I can retrieve an exact spending amount, but this question needs an agency "
                "or a specific USAspending filter. Try including an agency, for example: "
                "'How much did FEMA obligate in FY 2025?' "
                "A government-wide hurricane-relief total requires a broader disaster-spending "
                "query that is not yet implemented in this demo."
            )
            log_interaction(log_path, question, "clarification", answer)
            return HybridResponse(question, "clarification", answer, [])

        if is_numeric_question(question) and resolved_agency and fiscal_year:
            matches = self.spending_client.find_funding_agencies(resolved_agency)
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
                if re.search(r"\bhurricane\s+relief\b", question, re.I):
                    answer += (
                        "\n\nScope note: these are FEMA's agency-wide FY values. "
                        "This demo does not yet isolate hurricane-relief spending from all FEMA spending."
                    )
                log_interaction(log_path, question, "structured_api", answer, structured_data=data)
                return HybridResponse(question, "structured_api", answer, [], agency, spending)
            # FEMA and similar organizations are sub-agencies, so they may not
            # appear in the top-tier autocomplete response.
            keyword = "hurricane relief" if re.search(r"hurricane\s+relief", question, re.I) else None
            spending = self.spending_client.funding_subagency_spending(
                resolved_agency, fiscal_year, keyword
            )
            if spending.obligations is not None:
                agency = AgencyMatch("subtier-search", resolved_agency, resolved_agency)
                answer = self._numeric_answer(
                    "USAspending returned a sub-agency spending total for the requested fiscal year.",
                    agency,
                    spending,
                )
                log_interaction(log_path, question, "structured_api", answer, structured_data={"obligations": spending.obligations, "source_url": spending.source_url})
                return HybridResponse(question, "structured_api", answer, [], agency, spending)

        results = self.store.search(question, self.embedder, top_k=3)
        answer = self.generator.generate(build_document_prompt(question, results))
        if not answer or len(answer.strip()) < 20:
            answer = self._evidence_fallback(question, results)
        elif not re.search(r"\[\d+\]", answer):
            answer = f"{answer.rstrip()} [1]"
        log_interaction(log_path, question, "document_rag", answer, results=results)
        return HybridResponse(question, "document_rag", answer, results)

    @staticmethod
    def _evidence_fallback(question: str, results: list[SearchResult]) -> str:
        """Return a compact plain-text fallback instead of dumping a full chunk."""
        if not results:
            return "I could not find enough information in the indexed evidence."

        stop_words = {
            "what", "which", "where", "when", "how", "are", "is", "the", "and",
            "from", "for", "about", "does", "did", "this", "that", "with", "they",
        }
        terms = {
            term.lower()
            for term in re.findall(r"[A-Za-z]{4,}", question)
            if term.lower() not in stop_words
        }
        candidates: list[str] = []
        for result in results:
            text = re.sub(r"\s+", " ", result.text).strip()
            fragments = re.split(r"(?=/api/v2/)", text)
            relevant = [fragment.strip(" -") for fragment in fragments if any(term in fragment.lower() for term in terms)]
            candidates.extend(relevant or fragments[:1])

        cleaned: list[str] = []
        for candidate in candidates:
            candidate = re.sub(r"<([^>]+)>", r"[\1]", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if len(candidate) > 260:
                candidate = candidate[:260].rsplit(" ", 1)[0] + "..."
            if candidate and candidate not in cleaned:
                cleaned.append(candidate)
            if len(cleaned) == 3:
                break

        evidence = "\n".join(f"- {item}" for item in cleaned)
        return "I could not generate a concise answer from the local model. The relevant indexed evidence is:\n\n" + evidence + "\n\nSource: [1]"

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
