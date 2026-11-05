# Federal Spending & Budget Research Assistant

A Streamlit research assistant for exploring federal spending documentation
and agency-level fiscal-year data. The application combines semantic search
over public documents with structured lookups from the USAspending.gov API.

## Features

- Ingests Markdown, text, HTML, and PDF source documents
- Cleans documents and splits them into metadata-rich chunks
- Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Combines semantic vector search with BM25-style keyword search
- Stores and searches embeddings with ChromaDB
- Answers document questions with grounded citations
- Retrieves exact agency spending values from USAspending.gov
- Logs questions, answers, sources, scores, and API data as JSONL
- Evaluates retrieval with SourceHitRate@3
- Provides a Dockerfile for deployment

## Architecture

```text
Public documents ──> ingestion ──> chunks ──> embeddings ──> ChromaDB
                                                              │
User question ──> hybrid router ──────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
   Document RAG                  USAspending API
          │                             │
          └──────────────┬──────────────┘
                         ↓
                 Grounded answer
                         ↓
                  Streamlit interface
```

Document questions use hybrid retrieval: semantic similarity finds conceptually
related text, while keyword scoring preserves exact terms such as agency names,
fiscal years, and endpoint paths. Numeric
questions that include an agency and fiscal year use the structured
USAspending API path so authoritative values are not generated from model
memory.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the configured public sources and build the local index:

```bash
python scripts/download_sources.py
python -m src.ingest
python -m src.index
```

The first model execution downloads the embedding and generation models from
Hugging Face and caches them locally.

## Run the application

Broad questions such as “How does the federal budget work?” are supported.
When the model cannot synthesize a reliable response, the app shows relevant
cited source excerpts. See [broad-question evaluation](evaluation/BROAD_QUESTIONS.md).

```bash
python -m streamlit run app.py
```

Ask a documentation question directly, or provide an agency and fiscal year
in the sidebar for an exact spending lookup. The interface displays the query
mode, answer, retrieved sources, similarity scores, and expandable evidence.

## Retrieval query

```bash
python -m scripts.query_index "What are federal obligations?"
```

## Evaluation

Run the 55-question retrieval evaluation:

```bash
python -m scripts.evaluate
```

The current corpus contains 18 unique document bodies and 1,497 chunks. Several
endpoint URLs intentionally resolve to the same canonical documentation page
and are deduplicated during ingestion. The current retrieval baseline with
hybrid search and reranking is:

```text
SourceHitRate@3: 53/55 = 96.4%
```

Category results:

```text
API overview:      10/10  = 100.0%
Endpoints:         14/15  = 93.3%
Agency data:        9/10  = 90.0%
Data methodology:  10/10 = 100.0%
Treasury:          10/10 = 100.0%
```

Detailed evaluation output is written to `evaluation/results.json`. Answer
quality reviews use the classifications Accurate, Partially Accurate,
Incorrect, and Not Answerable. See [EVALUATION.md](EVALUATION.md) for details.

To evaluate the cross-encoder reranker explicitly:

```bash
python -m scripts.evaluate --rerank
```

## Project structure

Answer generation has a separate evaluation runner:

```bash
python -m scripts.evaluate_answers --per-category 2
```

It saves answers, actual retrieved passages, citation checks, and review fields.
See [answer quality results](evaluation/ANSWER_QUALITY.md) for the initial
12-question diagnostic sample and review procedure.

```text
app.py                         Streamlit application
src/chunking.py                Document cleaning and chunking
src/ingest.py                  Chunk-generation CLI
src/embeddings.py              Sentence-transformer embeddings
src/vector_store.py            ChromaDB persistence and search
src/index.py                   Vector-index rebuild CLI
src/prompts.py                 Grounded prompt construction
src/generator.py               Local Hugging Face answer model
src/spending_api.py            USAspending API client
src/hybrid.py                  Document/API routing pipeline
src/reranker.py                Cross-encoder candidate reranking
src/logging_utils.py           JSONL interaction logging
scripts/download_sources.py    Source downloader
scripts/evaluate.py            SourceHitRate@3 evaluation
data/raw/                      Downloaded source documents
data/processed/                Chunk records
data/chroma/                   Persistent ChromaDB index
evaluation/                    Evaluation questions and reviews
```

## Testing

```bash
python -m pytest -q
```

## Deployment

```bash
docker build -t federal-spending-rag .
docker run --rm -p 8501:8501 federal-spending-rag
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for the AWS EC2/ECR deployment path.

## Limitations

- The current corpus is small and concentrated around USAspending
  documentation and Treasury financial pages.
- Some CBO and GAO sources reject automated downloads with HTTP 403 in the
  current environment.
- The local `google/flan-t5-base` model is lightweight and may produce less
  fluent answers than larger instruction-tuned models.
- SourceHitRate@3 measures retrieval quality, not complete answer correctness.
- ChromaDB is currently local and is not configured as a production managed
  vector database.
