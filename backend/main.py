from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import ConversationCreate, QueryRequest, ResearchResponse
from .services import ResearchService
from .store import ConversationStore

app = FastAPI(
    title="FederalScope API",
    version="0.1.0",
    description="Full-stack research API backed by the existing hybrid federal spending RAG.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_research_service() -> ResearchService:
    return ResearchService()


def get_store() -> ConversationStore:
    return ConversationStore()


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "federalscope-api"}


@app.post("/api/v1/conversations")
def create_conversation(request: ConversationCreate) -> dict:
    return get_store().create_conversation(request.title)


@app.get("/api/v1/conversations")
def list_conversations() -> list[dict]:
    return get_store().list_conversations()


@app.get("/api/v1/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    conversation = get_store().get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.post("/api/v1/conversations/{conversation_id}/messages", response_model=ResearchResponse)
def ask_question(conversation_id: str, request: QueryRequest) -> ResearchResponse:
    store = get_store()
    if store.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    store.add_message(conversation_id, "user", request.question)
    try:
        response = get_research_service().answer(request.question, request.agency, request.fiscal_year)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Research request failed: {exc}") from exc
    store.add_message(conversation_id, "assistant", response.answer, response.model_dump())
    return response


@app.post("/api/v1/research", response_model=ResearchResponse)
def research(request: QueryRequest) -> ResearchResponse:
    try:
        return get_research_service().answer(request.question, request.agency, request.fiscal_year)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Research request failed: {exc}") from exc
