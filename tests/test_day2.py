from pathlib import Path

from src.vector_store import VectorStore, chunk_id


class FakeEmbedder:
    def encode_documents(self, texts):
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    def encode_query(self, query):
        return [0.0, 1.0]


def test_chunk_ids_are_stable():
    chunk = {"source": "a.md", "chunk_index": 0, "text": "hello"}
    assert chunk_id(chunk) == chunk_id(chunk)
    assert chunk_id(chunk) != chunk_id({**chunk, "text": "different"})


def test_vector_store_rebuild_and_search(tmp_path: Path):
    store = VectorStore(tmp_path / "chroma")
    chunks = [
        {
            "text": "Budget obligations are recorded by fiscal year.",
            "source": "budget.md",
            "section": "Obligations",
            "chunk_index": 0,
            "title": "budget",
        },
        {
            "text": "Awards can include contracts and grants.",
            "source": "awards.md",
            "section": "Awards",
            "chunk_index": 0,
            "title": "awards",
        },
    ]
    assert store.rebuild(chunks, FakeEmbedder()) == 2
    results = store.search("fiscal year", FakeEmbedder(), top_k=1)
    assert len(results) == 1
    assert results[0].metadata["source"] == "budget.md"


def test_hybrid_search_uses_keyword_signal(tmp_path: Path):
    store = VectorStore(tmp_path / "chroma")
    chunks = [
        {
            "text": "Federal account endpoint returns spending data.",
            "source": "endpoint.md",
            "section": "Federal Account",
            "chunk_index": 0,
            "title": "endpoint",
        },
        {
            "text": "General information about federal programs.",
            "source": "general.md",
            "section": "Overview",
            "chunk_index": 0,
            "title": "general",
        },
    ]
    store.rebuild(chunks, FakeEmbedder())
    results = store.search_hybrid("federal account endpoint", FakeEmbedder(), top_k=1)
    assert results[0].metadata["source"] == "endpoint.md"
