import pytest
from src.prompts import build_budgeted_prompt
from src.validation import validate_answer, attribute_answer
from src.vector_store import SearchResult


def test_paraphrase_is_allowed_but_invented_numbers_are_not():
    results = [SearchResult("1", "Agencies submit financial data monthly to USAspending.", {}, 0)]
    assert not validate_answer("USAspending receives agencies' financial submissions every month [1]", results)
    assert validate_answer("Agencies submit 987 records each month [1]", results)
    assert validate_answer("Elephants live in tropical forests [1]", results)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text.encode()) + ([0] if add_special_tokens else [])

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()


def test_budget_preserves_question_and_answer_instruction():
    results = [SearchResult(str(i), "Evidence words. " * 200, {}, 0) for i in range(3)]
    tokenizer = Tokenizer()
    prompt = build_budgeted_prompt("Which endpoint?", results, tokenizer, 512)
    assert len(tokenizer.encode(prompt)) <= 512
    assert "Which endpoint?" in prompt
    assert all(f"[{rank}]" in prompt for rank in (1, 2, 3))
    assert prompt.endswith("Answer:")
    with pytest.raises(ValueError):
        build_budgeted_prompt("x" * 600, results, tokenizer, 512)


def test_missing_citation_is_attributed_only_with_content_support():
    results = [SearchResult("1", "Federal agencies submit financial data every month.", {}, 0)]
    answer = attribute_answer("Federal agencies submit financial data monthly.", results)
    assert answer.endswith("[1]")
    assert attribute_answer("Data Submission and Extraction", results) == "Data Submission and Extraction"


def test_nonfinite_reranker_preserves_finite_hybrid_results():
    from src.reranker import Reranker
    class Model:
        def predict(self, pairs):
            return [float("nan")] * len(pairs)
    reranker = Reranker.__new__(Reranker)
    reranker.model = Model()
    candidates = [SearchResult("1", "text", {}, 0.2)]
    result = reranker.rerank("query", candidates)[0]
    assert result.score == 0.8
    assert result.metadata["reranker_status"] == "nonfinite_score_fallback"
