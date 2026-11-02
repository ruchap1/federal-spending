# Evaluation

## Dataset

The evaluation set contains 15 questions about the indexed USAspending.gov
documentation. Each question has one labeled expected source file. The labels
test retrieval, not whether the generated answer is perfect.

## Metric

`SourceHitRate@3` is the percentage of questions for which the labeled expected
source appears in the top three retrieved chunks:

```text
SourceHitRate@3 = questions with expected source in top 3 / total questions
```

The labels were created for the original seed corpus. As the corpus expands,
some questions may have multiple valid official sources; a future evaluation
revision should support multiple acceptable source labels.

Run it with:

```bash
python -m scripts.evaluate
```

The script writes detailed rows to `evaluation/results.json`. The results file
is intentionally ignored by Git because it is generated output.

## Answer-quality review

Retrieval success does not guarantee a correct answer. Review interactions in
`logs/interactions.jsonl` and classify each answer as:

- Accurate
- Partially Accurate
- Incorrect
- Not Answerable

Use `evaluation/review_template.jsonl` as the review format and explain whether
the answer is supported by the retrieved evidence. For numeric questions,
verify the values against the structured USAspending API data recorded in the
log. `evaluation/manual_review.jsonl` contains the initial review of the two
end-to-end Day 3 checks.

## Limitations and improvements

- The current corpus contains 11 unique document bodies and 147 chunks, with
  most content still concentrated in USAspending documentation. Several
  endpoint URLs resolve to duplicate documentation and are deduplicated.
- Some public pages reject automated downloads with HTTP 403.
- The local `flan-t5-small` model is lightweight and can produce weak prose.
- SourceHitRate@3 measures retrieval, not factual correctness.

Next improvements would include a larger multi-source corpus, better chunk
overlap, stronger answer models, and a larger labeled evaluation set with
human-reviewed answer quality.
