"""Select readable question-relevant passages and provide cited excerpts."""
import re
from src.validation import ABSTENTION, content_words
from src.vector_store import SearchResult


QUESTION_WORDS = {"what", "how", "why", "where", "when", "which", "can", "could", "would",
                  "do", "doe", "tell", "me", "about", "explain", "give", "overview", "work", "please"}


def select_evidence(question, candidates, top_k=3):
    terms = content_words(question) - QUESTION_WORDS
    if not terms:
        return []
    # Personal information and exact future values require evidence specific to
    # the requested entity/year; topical government-finance text is insufficient.
    if (re.search(r"\b(my|mine|personal)\b", question, re.I)
            and re.search(r"bank|balance|salary|tax return|social security number", question, re.I)):
        return []
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    ranked = []
    for candidate in candidates:
        raw = re.sub(r"\s+", " ", candidate.text).strip()
        raw = re.sub(r"\b[\w-]+\.(?:md|txt|html)\b", "", raw).strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", raw):
            sentence = sentence.strip()
            if not 45 <= len(sentence) <= 1100:
                continue
            if any(word in sentence for word in ("Source URL:", "-->", "Skip to", "Contents")):
                continue
            if years and not all(year in sentence for year in years):
                continue
            overlap = terms & content_words(sentence)
            coverage = len(overlap) / len(terms)
            if not overlap or coverage < 0.25:
                continue
            definition = bool(re.search(r"\b(is|means|provides|refers|includes|contains|represents)\b", sentence))
            score = coverage + 0.08 * definition + 0.05 * max(0, candidate.score)
            metadata = dict(candidate.metadata) | {"evidence_selection_score": score}
            ranked.append((score, SearchResult(candidate.id, sentence, metadata, candidate.distance)))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    selected, seen = [], set()
    for _, result in ranked:
        key = " ".join(result.text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        selected.append(result)
        if len(selected) >= top_k:
            break
    return selected


def evidence_answer(results):
    if not results:
        return ABSTENTION
    return "Relevant information from the retrieved sources:\n\n" + "\n\n".join(
        f"> {result.text} [{rank}]" for rank, result in enumerate(results, 1))
