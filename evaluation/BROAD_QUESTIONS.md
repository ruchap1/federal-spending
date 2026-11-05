# Broad document questions

The document pipeline now retrieves 20 candidates, optionally reranks ten,
then selects up to three readable sentences using question-term coverage.
The selected passages are the context and the evidence displayed by the app.
For broad questions the prompt requests an overview or a useful partial answer.

When generation fails citation/support checks or declines despite relevant
passages, the pipeline returns clearly labeled cited quotations. It does not
silently present those excerpts as generated prose. Existing numeric and
citation checks still apply to generated text. The retrieval evaluation labels
are unchanged; answer availability is measured separately from correctness.

Initial diagnostic results:

| Sample | Responses | Refusals | Cited excerpt responses |
| --- | --- | --- | --- |
| Original 10 document questions + 2 controls | 10 | 2 | 5 |
| 6 broad questions + 2 controls | 6 | 2 | 2 |

Both unanswerable controls declined in both runs. No execution errors or invalid
numeric citation labels were observed. These are development samples. Some
returned passages were tangential and some generated answers incomplete; the
response counts are not accuracy scores. The selector uses lexical heuristics,
so it can miss synonyms or choose topical evidence that does not fully answer
the question. Cited excerpts may contain historical amounts and projections.

```bash
python -m scripts.evaluate_answers --questions evaluation/broad_questions.json --output-dir logs/answer-evaluation/broad-questions
```

The private-data and year-presence checks are narrow heuristics, not a complete
answerability classifier. No-match responses still decline. Further progress
requires human relevance judgments and stronger answer synthesis.
