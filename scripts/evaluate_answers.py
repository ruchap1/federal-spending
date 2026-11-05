"""Capture generated answers and evidence for reproducible quality review."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from time import perf_counter


def citation_checks(answer: str, evidence_count: int) -> dict:
    labels = sorted({int(label) for label in re.findall(r"\[(\d+)\]", answer)})
    return {
        "citation_labels": labels,
        "invalid_labels": [label for label in labels if not 1 <= label <= evidence_count],
        "has_citation": bool(labels),
        "evidence_fallback": answer.startswith("The retrieved evidence is:"),
        "cited_excerpt_response": answer.startswith("Relevant information from the retrieved sources:"),
    }


def select_questions(questions: list[dict], per_category: int) -> list[dict]:
    counts: dict[str, int] = {}
    selected = []
    for item in questions:
        category = item.get("category", "uncategorized")
        if per_category and counts.get(category, 0) >= per_category:
            continue
        selected.append(item)
        counts[category] = counts.get(category, 0) + 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--per-category", type=int, default=0,
                        help="0 runs all questions; 2 selects two per category")
    parser.add_argument("--output-dir", default="logs/answer-evaluation")
    args = parser.parse_args()
    if args.per_category < 0:
        parser.error("--per-category must be nonnegative")
    from src.embeddings import Embedder
    from src.generator import LocalGenerator
    from src.hybrid import HybridPipeline
    from src.reranker import Reranker
    from src.prompts import build_document_prompt

    questions = select_questions(json.loads(Path(args.questions).read_text()), args.per_category)
    questions += [
        {"question": "What is my personal bank account balance?", "category": "unanswerable",
         "expected_behavior": "Explain that personal bank balances are not available in this corpus."},
        {"question": "What will the exact federal deficit be in fiscal year 2099?", "category": "unanswerable",
         "expected_behavior": "State that the exact future value cannot be established from this evidence."},
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = HybridPipeline(Embedder(), LocalGenerator(), reranker=Reranker())
    rows = []
    # Append each result immediately so completed answers survive an interrupted run.
    with (output / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for index, item in enumerate(questions, 1):
            started = perf_counter()
            row = {"id": f"answer-{index:03d}", **item}
            try:
                response = pipeline.answer(item["question"], log_path=str(output / "interactions.jsonl"))
                row.update({"answer": response.answer, "mode": response.mode,
                            "evidence": [asdict(result) for result in response.results],
                            "prompt": pipeline.generator.last_prompt,
                            "raw_answer": pipeline.generator.last_answer,
                            "checks": citation_checks(response.answer, len(response.results))})
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
            row["latency_seconds"] = round(perf_counter() - started, 3)
            row["review"] = {"reviewer": None, "classification": None,
                             "supported_by_evidence": None, "complete": None,
                             "citations_support_claims": None, "appropriate_abstention": None,
                             "notes": None}
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"{index}/{len(questions)}: {item['question']}", flush=True)
    summary = {
        "questions": len(rows), "errors": sum("error" in row for row in rows),
        "evidence_fallbacks": sum(row.get("checks", {}).get("evidence_fallback", False) for row in rows),
        "answers_with_invalid_citations": sum(bool(row.get("checks", {}).get("invalid_labels")) for row in rows),
        "reviewed": 0, "answer_accuracy": None,
        "abstentions": sum(row.get("answer") == "I don't have enough evidence to answer this question." for row in rows),
        "cited_excerpt_responses": sum(row.get("checks", {}).get("cited_excerpt_response", False) for row in rows),
        "generator": pipeline.generator.model_name,
        "embedding_model": pipeline.embedder.model_name,
        "reranker": pipeline.reranker.model_name,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
