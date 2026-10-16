"""Append-only JSONL logging for research interactions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.vector_store import SearchResult


def log_interaction(
    path: str | Path,
    question: str,
    mode: str,
    answer: str,
    results: list[SearchResult] | None = None,
    structured_data: dict[str, Any] | None = None,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "mode": mode,
        "answer": answer,
        "retrieved": [
            {
                "id": result.id,
                "source": result.metadata.get("source"),
                "section": result.metadata.get("section"),
                "score": result.score,
            }
            for result in (results or [])
        ],
        "structured_data": structured_data,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

