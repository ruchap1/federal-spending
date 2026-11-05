from src.chunking import Document, chunk_document, load_documents
from src.hybrid import HybridPipeline
from src.spending_api import AgencyMatch, AgencySpending
from src.validation import validate_answer, ABSTENTION
from src.vector_store import SearchResult


def test_pdf_page_and_url_metadata(tmp_path):
    (tmp_path / "report.md").write_text("# Report\n\nSource URL: https://example.gov/report\n\n## PDF page 2\n\nBudget facts.")
    doc = load_documents(tmp_path)[0]
    chunk = next(c for c in chunk_document(doc) if c.section == "PDF page 2")
    assert chunk.metadata["page"] == 2
    assert chunk.metadata["url"] == "https://example.gov/report"


def test_validation_rejects_invented_amount_and_bad_citation():
    results = [SearchResult("a", "Obligations were $10.", {}, 0)]
    assert not validate_answer("Obligations were $10. [1]", results)
    assert "unverified_claim" in validate_answer("Obligations were $99. [1]", results)
    assert "invalid_citation" in validate_answer("Obligations were $10. [4]", results)
    assert "empty_or_short_answer" in validate_answer("[1]", results)


class Client:
    def find_funding_agencies(self, query):
        return [AgencyMatch("091", "Department of Education", "ED")]

    def agency_budgetary_resources(self, code, year):
        amount = {2023: 0, 2024: 20}[year]
        return AgencySpending(code, year, amount, amount, None, "https://example.gov")


class Generator:
    def generate(self, prompt):
        raise AssertionError("Structured lookups should not invoke generation")


def test_comparison_uses_explicit_years_and_handles_zero_baseline(tmp_path):
    pipeline = HybridPipeline(None, Generator(), store=object(), spending_client=Client())
    response = pipeline.answer("Compare Education obligations in 2023 and 2024", "Department of Education", 2025,
                               log_path=str(tmp_path / "log.jsonl"))
    assert response.mode == "structured_api"
    assert "$+20.00" in response.answer
    assert "zero baseline" in response.answer
    assert "unavailable" in response.answer


def test_numeric_request_requires_agency():
    pipeline = HybridPipeline(None, Generator(), store=object(), spending_client=Client())
    assert pipeline.answer("How much was spent in 2024?").mode == "clarification"
