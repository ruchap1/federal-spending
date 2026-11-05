"""Citation, numeric, and lexical support checks; not semantic entailment."""
import re


ABSTENTION = "I don't have enough evidence to answer this question."


def content_words(text):
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in", "for", "with", "by", "it", "that", "from"}
    return {word.rstrip("s") for word in re.findall(r"[a-z]+", text.lower()) if word not in stop}


def attribute_answer(answer, results):
    """Infer a missing citation only when a passage shares substantial content.

    This lexical attribution is a heuristic, not proof of entailment.
    """
    if re.search(r"\[\d+\]", answer) or len(answer.split()) < 5:
        return answer
    terms = content_words(answer)
    if len(terms) < 3:
        return answer
    scores = [len(terms & content_words(result.text)) / len(terms) for result in results]
    if scores and max(scores) >= 0.6:
        return f"{answer.rstrip()} [{scores.index(max(scores)) + 1}]"
    return answer


def validate_answer(answer, results):
    labels = [int(label) for label in re.findall(r"\[(\d+)\]", answer)]
    issues = []
    if len(re.sub(r"\[\d+\]", "", answer).strip()) < 8:
        issues.append("empty_or_short_answer")
    if not labels:
        issues.append("missing_citation")
    if any(label < 1 or label > len(results) for label in labels):
        issues.append("invalid_citation")
    if answer.startswith("The retrieved evidence is:"):
        issues.append("evidence_dump")
    if re.search(r"Evidence\s*\[|Question:|Answer:|Context:", answer):
        issues.append("prompt_echo")
    for claim in re.split(r"(?<=\])\s+|\n", answer):
        claim_labels = [int(n) for n in re.findall(r"\[(\d+)\]", claim)]
        text = re.sub(r"\[\d+\]", "", claim).strip()
        if not text:
            continue
        passages = [results[n - 1].text for n in claim_labels if 1 <= n <= len(results)]
        # Permit paraphrases. Numbers must still occur in the cited evidence.
        numbers = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)
        evidence_numbers = {n.replace(',', '') for p in passages
                            for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", p)}
        if not passages or any(n.replace(',', '') not in evidence_numbers for n in numbers):
            issues.append("unverified_claim")
        terms = content_words(text)
        evidence_terms = content_words(" ".join(passages))
        if terms and len(terms & evidence_terms) / len(terms) < 0.2:
            issues.append("low_evidence_overlap")
    return sorted(set(issues))
