# Corpus, chunking, numeric routing, and validation

## Corpus

Six new official downloads succeeded: USAGov's budget process, Treasury budget
FAQs, Treasury financial position and nation-by-numbers pages, GSA enacted
budgets, and the FY2025 OMB budget hosted by GovInfo. Transportation returned
403. Download counts are not unique-body counts: ingestion now finds 18 unique
documents and produces 1,497 chunks. Dates and fiscal years inside sources must
be interpreted in context; the FY2025 budget proposal is not an enacted budget.

## Chunking and provenance

HTML comments are removed. Splitting prefers line/word boundaries and retains
section headings before token splitting. The ingestion CLI enforces a default
240-token limit using the embedding model's tokenizer, with 24-token overlap
for oversized chunks. Section titles remain in metadata even when a token
window does not contain the heading. PDF page markers become page metadata.
Downloaded metadata sidecars retain URL, publisher hostname, document type,
and retrieval timestamp. Publication dates are not inferred; they need source
specific extraction. PDF extraction is text based: complex tables are not yet
reconstructed or guaranteed to preserve their layout.

## Numeric queries

Supply an agency in the interface and ask, for example:

`Compare Department of Education obligations in 2023 and 2024.`

Explicit question years override the sidebar year. The pipeline queries both
years and calculates absolute and percentage changes in Python. Missing values
remain unavailable; a zero baseline has no percentage change. Ambiguous or
missing agency selections prompt clarification. Numeric responses bypass the
generator. Comparisons cover agency-wide resources, obligations, and outlays;
program-specific and government-wide trend queries remain unsupported. The
structured backend response still exposes the final year's object; both years
and deltas are present in answer text and both API rows are logged.

## Validation

Document answers require valid numeric citations and supporting text in the
cited passages. Empty, citation-only, evidence-dump, and unverified responses
are replaced with an insufficient-evidence response. This is conservative
text matching, not semantic entailment: valid paraphrases can be rejected and
matching text can still be irrelevant. It is not a hallucination guarantee.

## Verification

The expanded index scored 53/55 (96.4%) on the unchanged development labels;
the two misses were HTTP methods and Spending Explorer. Labels were not widened.
The first validated 12-question generation sample abstained on every question,
including both unanswerable controls. This improves refusal behavior but makes
the application too cautious for useful document answers with the current
generator. A stronger generator and context budgeting should follow; do not
interpret abstentions as answer accuracy. Numeric comparison logic is covered
with fake API tests, including missing values and a zero baseline.
