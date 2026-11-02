"""Neutral prompt construction for spending-policy answers."""

from __future__ import annotations

from collections.abc import Sequence

from src.vector_store import SearchResult


INSTRUCTIONS = """You are a neutral federal spending research assistant.
Use only the provided evidence.
Distinguish documented facts from interpretation.
Do not invent dollar amounts, dates, agencies, or policy claims.
Cite document claims with [1], [2], or [3].
If the evidence is insufficient, say that clearly."""


def build_document_context(results: Sequence[SearchResult], max_chars: int = 650) -> str:
    blocks: list[str] = []
    for rank, result in enumerate(results, 1):
        content = result.text
        if len(content) > max_chars:
            content = content[:max_chars].rsplit(" ", 1)[0] + " ..."
        blocks.append(
            f"[{rank}] Source: {result.metadata.get('source', 'unknown')}\n"
            f"Section: {result.metadata.get('section', 'unknown')}\n"
            f"Content: {content}"
        )
    return "\n\n---\n\n".join(blocks)


def build_document_prompt(question: str, results: Sequence[SearchResult]) -> str:
    return (
        f"Instructions:\n{INSTRUCTIONS}\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved evidence:\n{build_document_context(results)}\n\n"
        "Answer with citations:"
    )


def build_numeric_prompt(question: str, agency_name: str, fiscal_year: int, data: dict[str, object]) -> str:
    return (
        f"Instructions:\n{INSTRUCTIONS}\n"
        "Use the structured USAspending data below for every numeric value.\n\n"
        f"Question: {question}\n"
        f"Agency: {agency_name}\n"
        f"Fiscal year: {fiscal_year}\n"
        f"Structured data: {data}\n\n"
        "Answer concisely and label the source as [USAspending API]."
    )

