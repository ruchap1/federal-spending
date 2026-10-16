# FederalScope

FederalScope is an interview-ready full-stack federal spending research assistant. It combines the existing hybrid RAG implementation with a FastAPI backend, a React/TypeScript interface, persistent conversations, citations, and structured USAspending.gov results.

## What was reused

The `src/` package and processed corpus are copied from the existing `federal-spending-rag` project. The application reuses the existing `all-MiniLM-L6-v2` embedding path, `google/flan-t5-small` generator, document corpus, and hybrid numeric lookup. The vector database must be rebuilt once with the pinned Chroma version because persisted Chroma indexes are version-sensitive.

## Run locally

### Backend

```bash
cd federal-scope
# Python 3.12 or 3.13 is recommended. Python 3.14 may not have a compatible
# prebuilt tokenizers wheel for the pinned Chroma/Transformers versions.
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.index
uvicorn backend.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`, with interactive documentation at `/docs`.

### Frontend

In a second terminal:

```bash
cd federal-scope/frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`.

The default application uses SQLite at `data/federalscope.db` for conversations. The persistence boundary is isolated in `backend/store.py`, making a later PostgreSQL/SQLAlchemy migration straightforward.

## Architecture

```text
React + TypeScript
        │ JSON API
        ▼
FastAPI ── ConversationStore ── SQLite (demo persistence)
   │
   ├── HybridPipeline ── ChromaDB + copied document index
   ├── LocalGenerator ── Flan-T5
   └── USAspendingClient ── authoritative numeric values
```

The frontend receives structured answer data rather than presentation-ready HTML. This lets it render citations, evidence cards, numeric result cards, and future charts independently of the backend.

## API examples

```bash
curl http://localhost:8000/api/v1/health

curl -X POST http://localhost:8000/api/v1/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are federal obligations?"}'
```

## Interview talking points

- ChromaDB is retained for fast local iteration; the vector-store access is behind the existing pipeline boundary so it can later move to pgvector or a managed store.
- PostgreSQL is the production persistence target, while SQLite keeps the portfolio demo self-contained.
- Numeric questions use the authoritative USAspending API; the language model explains those values but does not own them.
- REST is used first because it is easy to test and sufficient for request/response answers. Server-Sent Events can be added later for token streaming.
- RAG quality is evaluated separately from application correctness: retrieval hit rate, citation correctness, API response tests, and frontend end-to-end tests measure different failure modes.

## Next milestones

1. Add PostgreSQL and SQLAlchemy migrations.
2. Add background indexing jobs and corpus versioning.
3. Add answer feedback and retrieval evaluation dashboards.
4. Add an agency comparison chart and export-to-Markdown flow.
5. Add authentication and per-user conversation ownership.
