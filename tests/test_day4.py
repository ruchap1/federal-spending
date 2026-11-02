from src.vector_store import SearchResult
from scripts.evaluate import evaluate


class FakeStore:
    def search_hybrid(self, question, embedder, top_k=3):
        return [
            SearchResult("1", "text", {"source": "data/raw/expected.md"}, 0.1),
            SearchResult("2", "text", {"source": "data/raw/other.md"}, 0.2),
        ]


def test_source_hit_rate_at_three():
    report = evaluate(
        [
            {"question": "one", "expected_source": "expected.md"},
            {"question": "two", "expected_source": "missing.md"},
        ],
        FakeStore(),
        object(),
    )
    assert report["hits_at_3"] == 1
    assert report["source_hit_rate_at_3"] == 0.5
