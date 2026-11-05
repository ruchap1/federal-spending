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
    parser.add_argument("--max-tokens", type=int, default=240)
    args = parser.parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents, args.max_chars, args.overlap)
    from transformers import AutoTokenizer
    from src.embeddings import MODEL_NAME
    from src.chunking import Chunk
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if args.max_tokens < 16:
        parser.error("--max-tokens must be at least 16")
    bounded = []
    source_counts = {}
    for chunk in chunks:
        encoded = tokenizer(chunk.text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        for start in range(0, len(offsets), args.max_tokens - 24 if args.max_tokens > 24 else args.max_tokens):
            window = offsets[start:start + args.max_tokens]
            if not window:
                continue
            text = chunk.text[window[0][0]:window[-1][1]]
            index = source_counts.get(chunk.source, 0)
            source_counts[chunk.source] = index + 1
            bounded.append(Chunk(text, chunk.source, chunk.section, index,
                                 chunk.metadata | {"token_count": len(window)}))
            if start + args.max_tokens >= len(offsets):
                break
    chunks = bounded
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

