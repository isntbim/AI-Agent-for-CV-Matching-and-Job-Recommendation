"""Unit tests for scripts/nominate_roles.py."""

from __future__ import annotations

from scripts.nominate_roles import clean_title


def test_clean_title_drops_seniority_levels_and_parenthetical_noise() -> None:
    assert clean_title("Senior Software Engineer I") == "software engineer"
    assert clean_title("Data Engineer Level 3") == "data engineer"
    assert clean_title("Business Analyst lll") == "business analyst"
    assert clean_title("Accounting Technician ll") == "accounting technician"
    assert clean_title("MOBILE SIMULATION ELECTRONIC TECHNICIAN III") == "mobile simulation electronic technician"
    assert clean_title("Software Engineer (12-month contract  REMOTE, team based in Mounds View, MN)") == "software engineer"
    assert clean_title("REMOTE - Servicenow Business Analyst") == "servicenow business analyst"
    assert clean_title("Business Analyst (Part time)") == "business analyst"
    assert clean_title("Sr. IT BA (Agile)") == "it ba"
    assert clean_title("W2 positions: (Marketing)") == "marketing"
    assert clean_title("PMLocation: Chicago, ILType: full-time (Hybrid)") == "pm"
    assert clean_title("Jr. Java Developer") == "java developer"
