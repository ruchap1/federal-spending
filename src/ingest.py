"""CLI for creating metadata-rich spending-document chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.chunking import chunk_documents, load_documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents, args.max_chars, args.overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.as_record()) + "\n")

    average = sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Average chunk size: {average:.1f} characters")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

