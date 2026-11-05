# Answer quality baseline

Answer quality is measured separately from source retrieval. A correct source
filename does not establish that the retrieved passage answers a question.

## Run

```bash
# Two questions per category, plus two unanswerable controls (12 total)
python -m scripts.evaluate_answers --per-category 2

# All 55 retrieval questions plus two unanswerable controls (57 total)
python -m scripts.evaluate_answers
```

The runner uses the same hybrid retrieval, reranker, generator, and fallback
behavior as the Streamlit application. It writes to `logs/answer-evaluation/`:

- `answers.jsonl`: answers, complete retrieved passages, constructed prompts,
  latency, automatic checks, errors, and blank review fields.
- `summary.json`: execution counts, fallback counts, citation-label checks,
  and model identifiers.
- `interactions.jsonl`: pipeline interaction logs.

Each run replaces `answers.jsonl` and `summary.json` in its output directory;
use `--output-dir logs/answer-evaluation/run-name` to preserve another run.
These generated files are ignored by Git. Full prompts are saved before the
generator's 512-token input truncation; the model may see less than the saved
prompt.

## Initial sample results

The initial run covered two questions from each of five categories and two
unanswerable controls, using `google/flan-t5-small`:

| Check | Result |
| --- | --- |
| Completed responses | 12/12 |
| Execution errors | 0 |
| Evidence-copy fallbacks | 11/12 |
| Answers with invalid numeric citation labels | 0/12 |
| Unanswerable controls that explicitly declined | 0/2 |

Assistant inspection of the saved answers identified these concrete failures:

- The federal-account endpoint question returned general download advice
  without naming the requested endpoint.
- The API overview response copied a table of contents.
- The Fiscal Service reports question returned a source URL without an
  explanation.
- The personal-bank-balance question copied government account documentation.
- The FY2099 deficit question copied generic Treasury information rather than
  explaining that an exact future value was unavailable.

This is an assistant-inspected diagnostic sample, not an independent human
accuracy study. No overall answer-accuracy percentage is reported. The existing
55/55 source hit result uses revised labels and a development set that has been
used for tuning; it must not be interpreted as held-out performance or answer
correctness.

## Review procedure

For each row in `answers.jsonl`, inspect the question, answer, and actual evidence.
Populate `review.reviewer` and these fields:

- `classification`: Accurate, Partially Accurate, Incorrect, or Not Answerable.
  Use Not Answerable for an appropriate refusal when evidence is insufficient;
  an unsupported attempted answer is Incorrect.
- `supported_by_evidence`: whether every substantive factual claim is supported.
- `complete`: whether the response addresses the question fully.
- `citations_support_claims`: whether cited passages support the associated claims.
- `appropriate_abstention`: whether answering or declining was appropriate.
- `notes`: missing facts, unsupported claims, and specific evidence references.

Automatic citation checks only verify that numeric labels exist within the
retrieved result range. They do not measure citation support or completeness.
The pipeline can append `[1]` automatically, so a valid label alone is weak
evidence of answer quality.

Next priorities from this sample are better abstention behavior, cleaner
passages and context budgeting, and a stronger generator. Compare future
changes on fixed questions and add an independently labeled held-out set.
Structured API correctness needs a separate evaluation of agency resolution,
year selection, values, and scope; this runner currently evaluates document RAG.
