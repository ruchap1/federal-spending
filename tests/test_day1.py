from pathlib import Path

from src.chunking import Document, chunk_document, load_documents
from src.spending_api import AgencySpending, USAspendingClient


def test_loader_skips_empty_files_and_preserves_metadata(tmp_path: Path):
    (tmp_path / "empty.md").write_text(" ", encoding="utf-8")
    (tmp_path / "budget.md").write_text("# Obligations\nFederal spending data.", encoding="utf-8")
    documents = load_documents(tmp_path)
    assert len(documents) == 1
    assert documents[0].metadata["file_type"] == ".md"


def test_chunk_has_section_and_title_metadata():
    document = Document("budget.md", "budget", "# Outlays\nMoney paid out.")
    chunk = chunk_document(document)[0]
    assert chunk.section == "Outlays"
    assert chunk.metadata["title"] == "budget"


def test_api_client_normalizes_budgetary_resources_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "agency_data_by_year": [
                    {
                        "fiscal_year": 2024,
                        "agency_budgetary_resources": "12.5",
                        "agency_total_obligated": 10,
                        "agency_total_outlayed": None,
                    }
                ]
            }

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("src.spending_api.requests.get", fake_get)
    result = USAspendingClient().agency_budgetary_resources("123", 2024)
    assert isinstance(result, AgencySpending)
    assert result.budgetary_resources == 12.5
    assert result.obligations == 10.0
