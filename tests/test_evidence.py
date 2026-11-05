from src.evidence import select_evidence, evidence_answer
from src.vector_store import SearchResult
from src.validation import ABSTENTION


def test_broad_question_gets_readable_cited_overview():
    result = SearchResult("a", "USAspending.gov provides public information about federal spending and grants.", {"source": "official.md"}, 0.2)
    evidence = select_evidence("Tell me about government spending", [result])
    assert evidence
    assert "[1]" in evidence_answer(evidence)
    assert evidence[0].metadata["source"] == "official.md"


def test_unrelated_and_private_questions_do_not_get_financial_excerpt():
    result = SearchResult("a", "Federal account balances describe public government finances and agency spending.", {}, 0.2)
    assert not select_evidence("What is my personal bank account balance?", [result])
    assert not select_evidence("Explain photosynthesis", [result])
    assert not select_evidence("What is the exact federal spending in 2099?", [result])
    assert evidence_answer([]) == ABSTENTION


def test_selects_sentence_after_boilerplate_and_deduplicates():
    text = "Source URL: https://example.gov. USAspending provides public spending information for federal agencies."
    result = SearchResult("a", text, {}, 0.2)
    selected = select_evidence("What is USAspending?", [result, result])
    assert len(selected) == 1
    assert "Source URL" not in selected[0].text


def test_pipeline_uses_cited_evidence_when_generator_declines(tmp_path):
    from src.hybrid import HybridPipeline
    class Store:
        def search_hybrid(self, *args, **kwargs):
            return [SearchResult("a", "Federal spending includes public spending by government agencies on services and grants.", {}, 0.2)]
    class Generator:
        def answer_documents(self, *args):
            return "Insufficient evidence"
    pipeline = HybridPipeline(None, Generator(), store=Store())
    response = pipeline.answer("Explain government spending", log_path=str(tmp_path / "logs.jsonl"))
    assert response.answer.startswith("Relevant information")
    assert "[1]" in response.answer
    assert response.results[0].id == "a"
