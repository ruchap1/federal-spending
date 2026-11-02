from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = "New research"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    agency: str = ""
    fiscal_year: int | None = Field(default=None, ge=2000, le=2100)


class Citation(BaseModel):
    id: str
    title: str
    section: str
    source: str
    excerpt: str
    score: float
    url: str | None = None


class ResearchResponse(BaseModel):
    answer: str
    mode: Literal["document_rag", "structured_api", "clarification"]
    citations: list[Citation] = []
    structured_data: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}
