"""Rebuild the local ChromaDB spending-document index."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.embeddings import Embedder
from src.vector_store import VectorStore, read_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--persist-dir", type=Path, default=Path("data/chroma"))
    args = parser.parse_args()

    chunks = read_chunks(args.chunks)
    count = VectorStore(args.persist_dir).rebuild(chunks, Embedder())
    print(f"Indexed chunks: {count}")
    print(f"Persisted at: {args.persist_dir}")


if __name__ == "__main__":
    main()

