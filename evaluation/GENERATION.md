# Local generation and validation

The default generator is `google/flan-t5-base`. To select another compatible
sequence-to-sequence checkpoint, set `RAG_GENERATOR_MODEL` before launching:

```bash
RAG_GENERATOR_MODEL=google/flan-t5-small python -m streamlit run app.py
```

The base model is downloaded and cached on first use. Restart Streamlit after
changing this setting because model resources are cached.

## Context budgeting

The generation tokenizer measures the complete prompt. Instructions and the
question are reserved first; remaining tokens are shared among the retrieved
passages. The final `Answer:` instruction remains intact. Oversized questions
raise a clear error instead of silently losing content. Direct generation also
rejects prompts exceeding the 512-token input budget. Evaluation now records
the actual prompt and raw generated answer, before validation.

## Validation policy

Exact sentence matching has been removed. Paraphrases can pass when they have
some content-word overlap with cited evidence. Numeric strings must occur in
the cited passages (commas are normalized); invalid labels, empty output,
evidence dumps, and prompt echoes are rejected. Very short citation-only
outputs remain invalid. A missing citation may be attributed to a passage
sharing at least 60% of the answer's content words, provided the answer has at
least five words and three content words.

These are lexical heuristics, not semantic entailment or verified factual
correctness. Attribution can be wrong, and matching numbers can have different
units or meaning. Human review remains necessary. The policy is deliberately
less restrictive than exact-copy validation while retaining structural checks.

## Numerical stability

Generation and reranking use eager attention. Non-finite reranker scores fall
back to the original finite hybrid score rather than contaminating result
ordering with NaN. Decoding blocks repeated four-token phrases.

## Reproduce

```bash
python -m scripts.evaluate_answers --per-category 2 --output-dir logs/answer-evaluation/base-release
```

Initial experiments on the fixed 12-question diagnostic sample showed fewer
refusals than the previous 12/12-refusal baseline, but some responses copied
context or failed to address the question. A response passing validation is
not counted as an accurate answer. See saved raw answers, prompts, evidence,
and review fields before assigning any accuracy score. The sample includes
two unanswerable controls; it is a development sample, not a held-out test.

The final base-model run completed 12/12 without errors: six answers passed
validation and six abstained. Both unanswerable controls abstained. There were
no invalid numeric citation labels. Assistant inspection found a useful answer
about Treasury financial reporting and a partial answer about account data,
but API and recipient questions still included irrelevant or incomplete text.
These six accepted responses are not six verified correct answers. Twenty
automated tests pass, including paraphrase acceptance, fabricated numbers,
invalid citations, token budgets, and non-finite reranker fallback.
