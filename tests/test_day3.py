from src.hybrid import is_numeric_question
from src.prompts import build_document_prompt
from src.vector_store import SearchResult


def test_document_prompt_is_neutral_and_citation_oriented():
    result = SearchResult("id", "Obligations are recorded by fiscal year.", {"source": "a.md", "section": "Budget"}, 0.2)
    prompt = build_document_prompt("What are obligations?", [result])
    assert "neutral federal spending research assistant" in prompt
    assert "[1]" in prompt
    assert "Do not invent dollar amounts" in prompt


def test_numeric_term_detection():
    assert is_numeric_question("How much did the agency obligate?")
    assert not is_numeric_question("What does an obligation mean?")
