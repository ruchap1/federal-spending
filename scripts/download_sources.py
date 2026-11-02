"""Download a curated corpus from authoritative public spending sources."""

from __future__ import annotations

import argparse
import html
import io
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader


SEED_SOURCES = {
    "usaspending-api.md": ("https://api.usaspending.gov/docs/", "html"),
    "usaspending-intro.md": ("https://api.usaspending.gov/docs/intro-tutorial", "html"),
    "usaspending-endpoints.md": ("https://api.usaspending.gov/docs/endpoints", "html"),
    "usaspending-agencies.md": ("https://www.usaspending.gov/federal-spending-guide", "html"),
    "usaspending-home.md": ("https://www.usaspending.gov/", "html"),
    "usaspending-explorer.md": ("https://www.usaspending.gov/explorer", "html"),
    "usaspending-data.md": ("https://www.usaspending.gov/data", "html"),
    "usaspending-data-sources.md": ("https://www.usaspending.gov/data/data-sources-download.pdf", "pdf"),
    "usaspending-entity-relationships.md": ("https://api.usaspending.gov/docs/entity-relationships", "html"),
    "usaspending-agency-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/agency/", "html"),
    "usaspending-federal-account-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/federal_accounts/", "html"),
    "usaspending-award-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/awards/", "html"),
    "usaspending-search-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/search/", "html"),
    "usaspending-autocomplete-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/autocomplete/", "html"),
    "usaspending-reference-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/references/", "html"),
    "usaspending-reporting-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/reporting/", "html"),
    "usaspending-financial-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/financial_spending/", "html"),
    "usaspending-category-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/spending_by_category/", "html"),
    "usaspending-bulk-download-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/bulk_download/", "html"),
    "usaspending-contract-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/contracts/", "html"),
    "usaspending-assistance-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/assistance/", "html"),
    "usaspending-transaction-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/transaction/", "html"),
    "usaspending-awarding-agency-autocomplete.md": ("https://api.usaspending.gov/docs/endpoints/awarding_agency_office/", "html"),
    "usaspending-funding-agency-autocomplete.md": ("https://api.usaspending.gov/docs/endpoints/funding_agency_office/", "html"),
    "usaspending-recipient-autocomplete.md": ("https://api.usaspending.gov/docs/endpoints/recipient/", "html"),
    "usaspending-country-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/countries/", "html"),
    "usaspending-cfda-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/cfda/", "html"),
    "usaspending-naics-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/naics/", "html"),
    "usaspending-psc-endpoints.md": ("https://api.usaspending.gov/docs/endpoints/psc/", "html"),
    "treasury-data-transparency-about.md": ("https://tfx.treasury.gov/data-transparency/about", "html"),
    "treasury-monthly-reporting.md": ("https://tfx.treasury.gov/monthly-reporting-requirements", "html"),
    "treasury-fiscal-data.md": ("https://fiscal.treasury.gov/financial-data", "html"),
    "treasury-reports-statements.md": ("https://fiscal.treasury.gov/reports-statements", "html"),
    "cbo-budget-topic.md": ("https://www.cbo.gov/topics/budget", "html"),
    "cbo-deficits-topic.md": ("https://www.cbo.gov/topics/deficits", "html"),
    "cbo-debt-topic.md": ("https://www.cbo.gov/topics/debt", "html"),
    "cbo-health-topic.md": ("https://www.cbo.gov/topics/health-care", "html"),
    "cbo-social-security-topic.md": ("https://www.cbo.gov/topics/social-security", "html"),
    "gao-budget-spending-topic.md": ("https://www.gao.gov/topics/budget-and-spending", "html"),
    "gao-federal-financial-management.md": ("https://www.gao.gov/federal-financial-management", "html"),
    "cbo-budget-outlook.md": ("https://www.cbo.gov/publication/62105", "html"),
    "cbo-budget-outlook-series.md": ("https://www.cbo.gov/recurring-publication/55126", "html"),
    "treasury-data-transparency.md": ("https://tfx.treasury.gov/data-transparency", "html"),
    "treasury-financial-report.md": ("https://fiscal.treasury.gov/accounting/us-financial-report", "html"),
    "gao-spending-transparency.md": ("https://www.gao.gov/assets/gao-24-106214.pdf", "pdf"),
    "cbo-introduction.md": ("https://www.cbo.gov/system/files/2024-01/59687-Intro-to-CBO.pdf", "pdf"),
}


def html_to_text(raw: str) -> str:
    raw = re.sub(r"<script\b[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style\b[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<(nav|footer|header|aside|form)\b[^>]*>.*?</\1>", " ", raw, flags=re.I | re.S)
    main_match = re.search(r"<main\b[^>]*>(.*?)</main>", raw, flags=re.I | re.S)
    article_match = re.search(r"<article\b[^>]*>(.*?)</article>", raw, flags=re.I | re.S)
    if main_match:
        raw = main_match.group(1)
    elif article_match:
        raw = article_match.group(1)
    raw = re.sub(r"<h([1-6])\b[^>]*>(.*?)</h\1>", r"\n\n# \2\n\n", raw, flags=re.I | re.S)
    raw = re.sub(r"<(p|div|br|li|tr)\b[^>]*>", "\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def pdf_to_text(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, (url, source_type) in SEED_SOURCES.items():
        request = Request(url, headers={"User-Agent": "federal-spending-rag-learning-project"})
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                text = (
                    html_to_text(raw.decode("utf-8", errors="ignore"))
                    if source_type == "html"
                    else pdf_to_text(raw)
                )
        except (HTTPError, URLError) as error:
            print(f"Skipped {url}: {error}")
            continue
        (args.output_dir / filename).write_text(
            f"# {filename}\n\nSource URL: {url}\n\n{text}\n", encoding="utf-8"
        )
        print(f"Downloaded {url}")


if __name__ == "__main__":
    main()
