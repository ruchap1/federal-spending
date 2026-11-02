"""Evaluate retrieval with SourceHitRate@3 over a small labeled question set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embeddings import Embedder
from src.reranker import Reranker
from src.vector_store import VectorStore


def load_questions(path: str | Path) -> list[dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(
    questions: list[dict[str, str]],
    store: VectorStore,
    embedder: Embedder,
    reranker: Reranker | None = None,
) -> dict:
    rows = []
    hits = 0
    for item in questions:
        candidate_count = 20 if reranker else 3
        results = store.search_hybrid(item["question"], embedder, top_k=candidate_count)
        if reranker:
            results = reranker.rerank(item["question"], results, top_k=3)
        sources = [Path(result.metadata.get("source", "")).name for result in results]
        expected_sources = item.get("expected_sources") or [item["expected_source"]]
        hit = bool(set(expected_sources) & set(sources))
        hits += int(hit)
        rows.append({**item, "retrieved_sources": sources, "hit_at_3": hit})
    return {
        "question_count": len(questions),
        "hits_at_3": hits,
        "source_hit_rate_at_3": hits / len(questions) if questions else 0.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--output", default="evaluation/results.json")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    reranker = Reranker() if args.rerank else None
    report = evaluate(load_questions(args.questions), VectorStore(), Embedder(), reranker)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"SourceHitRate@3: {report['hits_at_3']}/{report['question_count']} = "
          f"{report['source_hit_rate_at_3']:.1%}")
    for row in report["rows"]:
        status = "HIT" if row["hit_at_3"] else "MISS"
        print(f"{status}: {row['question']}")


if __name__ == "__main__":
    main()
