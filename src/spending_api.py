"""Small client for public USAspending.gov structured spending data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


BASE_URL = "https://api.usaspending.gov/api/v2"


@dataclass(frozen=True)
class AgencySpending:
    agency_code: str
    fiscal_year: int
    budgetary_resources: float | None
    obligations: float | None
    outlays: float | None
    source_url: str


@dataclass(frozen=True)
class AgencyMatch:
    agency_code: str
    name: str
    abbreviation: str


class USAspendingClient:
    def __init__(self, base_url: str = BASE_URL, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def agency_budgetary_resources(self, agency_code: str, fiscal_year: int) -> AgencySpending:
        """Retrieve an agency's fiscal-year budgetary resources and obligations."""

        url = f"{self.base_url}/agency/{agency_code}/budgetary_resources/"
        response = requests.get(
            url,
            params={"fiscal_year": fiscal_year},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        rows = payload.get("agency_data_by_year") or payload.get("results") or payload.get("data") or []
        row = next(
            (candidate for candidate in rows if int(candidate.get("fiscal_year", -1)) == fiscal_year),
            {},
        )
        return AgencySpending(
            agency_code=agency_code,
            fiscal_year=fiscal_year,
            budgetary_resources=_number(
                row.get("agency_budgetary_resources", row.get("budgetary_resources"))
            ),
            obligations=_number(row.get("agency_total_obligated", row.get("obligations"))),
            outlays=_number(row.get("agency_total_outlayed", row.get("outlays"))),
            source_url=url,
        )

    def find_funding_agencies(self, search_text: str) -> list[AgencyMatch]:
        """Resolve a human agency name to USAspending top-tier agency codes."""

        url = f"{self.base_url}/autocomplete/funding_agency/"
        response = requests.post(
            url,
            json={"search_text": search_text},
            timeout=self.timeout,
        )
        response.raise_for_status()
        matches: list[AgencyMatch] = []
        seen: set[str] = set()
        for item in response.json().get("results", []):
            top_tier = item.get("toptier_agency") or {}
            code = str(top_tier.get("toptier_code", ""))
            if item.get("toptier_flag") and code and code not in seen:
                seen.add(code)
                matches.append(
                    AgencyMatch(
                        agency_code=code,
                        name=top_tier.get("name", ""),
                        abbreviation=top_tier.get("abbreviation", ""),
                    )
                )
        return matches

    def funding_subagency_spending(
        self,
        agency_name: str,
        fiscal_year: int,
        keyword: str | None = None,
    ) -> AgencySpending:
        """Aggregate spending for a funding sub-agency and fiscal year."""
        url = f"{self.base_url}/search/spending_by_category/funding_subagency/"
        filters: dict[str, Any] = {
            "time_period": [{
                "start_date": f"{fiscal_year - 1}-10-01",
                "end_date": f"{fiscal_year}-09-30",
            }],
            "agencies": [{"type": "funding", "tier": "subtier", "name": agency_name}],
        }
        if keyword:
            filters["keywords"] = [keyword]
        response = requests.post(
            url,
            json={
                "filters": filters,
                "category": "funding_subagency",
                "limit": 100,
                "page": 1,
                "spending_level": "transactions",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json().get("results") or []
        total = sum((_number(row.get("amount")) or 0.0) for row in rows)
        return AgencySpending(
            agency_code="subtier-search",
            fiscal_year=fiscal_year,
            budgetary_resources=None,
            obligations=total if rows else None,
            outlays=None,
            source_url=url,
        )


def _number(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
