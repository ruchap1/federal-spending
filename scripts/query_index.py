"""Run hybrid semantic and keyword search over the local spending index."""

from __future__ import annotations

import argparse

from src.embeddings import Embedder
from src.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    for rank, result in enumerate(VectorStore().search_hybrid(args.query, Embedder(), args.top_k), 1):
        print(f"\n[{rank}] score={result.score:.4f}")
        print(f"source={result.metadata.get('source')}")
        print(f"section={result.metadata.get('section')}")
        print(
            f"dense_score={result.metadata.get('dense_score', 'n/a')} "
            f"keyword_score={result.metadata.get('keyword_score', 'n/a')}"
        )
        print(result.text[:700])


if __name__ == "__main__":
    main()

