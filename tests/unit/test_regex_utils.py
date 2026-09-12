"""
test_regex_utils.py — Unit tests for src/parser/regex_utils.py

Tests all contact info extraction functions with realistic Vietnamese CV patterns.
Run with: .venv/Scripts/python.exe -m pytest tests/unit/test_regex_utils.py -v
"""

import pytest
from src.parser.regex_utils import (
    extract_emails,
    extract_vn_phones,
    extract_urls,
    extract_linkedin,
    extract_github,
    extract_all_contacts,
    _normalize_vn_phone,
)


# ---------------------------------------------------------------------------
# Email Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractEmails:
    def test_single_gmail(self):
        assert extract_emails("Email: tridungit2005@gmail.com") == ["tridungit2005@gmail.com"]

    def test_multiple_emails(self):
        text = "john@gmail.com or info@company.io"
        result = extract_emails(text)
        assert "john@gmail.com" in result
        assert "info@company.io" in result
        assert len(result) == 2

    def test_email_in_sentence(self):
        text = "Please reach me at thanhnhanthai334@gmail.com for inquiries."
        assert extract_emails(text) == ["thanhnhanthai334@gmail.com"]

    def test_email_with_plus_alias(self):
        text = "user+tag@domain.co.uk"
        assert extract_emails(text) == ["user+tag@domain.co.uk"]

    def test_email_lowercased(self):
        """Output should always be lowercase."""
        result = extract_emails("JOHN.DOE@EXAMPLE.COM")
        assert result == ["john.doe@example.com"]

    def test_no_emails_in_text(self):
        assert extract_emails("No email here, just text.") == []

    def test_deduplication(self):
        text = "john@gmail.com and JOHN@GMAIL.COM"
        result = extract_emails(text)
        assert result == ["john@gmail.com"]

    def test_edu_domain(self):
        text = "Student at bangndanh2005@fe.edu.vn"
        assert extract_emails(text) == ["bangndanh2005@fe.edu.vn"]


# ---------------------------------------------------------------------------
# Vietnamese Phone Normalization Tests
# ---------------------------------------------------------------------------

class TestNormalizeVNPhone:
    def test_domestic_with_spaces(self):
        assert _normalize_vn_phone("0941 423 545") == "+84941423545"

    def test_domestic_with_dots(self):
        assert _normalize_vn_phone("0355.692.522") == "+84355692522"

    def test_domestic_with_dashes(self):
        assert _normalize_vn_phone("0823-550-315") == "+84823550315"

    def test_international_plus_with_dashes(self):
        assert _normalize_vn_phone("+84-941-423-545") == "+84941423545"

    def test_international_plus_with_spaces(self):
        assert _normalize_vn_phone("+84 355 692 522") == "+84355692522"

    def test_international_without_plus(self):
        """84xxxxxxxxx format (missing +)"""
        assert _normalize_vn_phone("84941423545") == "+84941423545"

    def test_domestic_no_separator(self):
        assert _normalize_vn_phone("0941423545") == "+84941423545"


# ---------------------------------------------------------------------------
# Vietnamese Phone Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractVNPhones:
    def test_extract_03x_prefix(self):
        result = extract_vn_phones("Call me at 0355 692 522")
        assert "+84355692522" in result

    def test_extract_09x_prefix(self):
        result = extract_vn_phones("Phone: 0941 423 545")
        assert "+84941423545" in result

    def test_extract_08x_prefix(self):
        result = extract_vn_phones("0823 550 315")
        assert "+84823550315" in result

    def test_extract_international_format(self):
        result = extract_vn_phones("+84 979 350 707")
        assert "+84979350707" in result

    def test_extract_multiple_phones(self):
        text = "Team: 0941 423 545, 0355 692 522 and 0823 550 315"
        result = extract_vn_phones(text)
        assert len(result) == 3
        assert "+84941423545" in result
        assert "+84355692522" in result
        assert "+84823550315" in result

    def test_no_phone_in_text(self):
        assert extract_vn_phones("No phone number here.") == []

    def test_deduplication(self):
        text = "0941 423 545 and also 0941423545"
        result = extract_vn_phones(text)
        assert result == ["+84941423545"]

    def test_reject_invalid_prefix(self):
        """Numbers starting with 01x (old format) or 06x should not match."""
        result = extract_vn_phones("Old format: 0123456789")
        assert result == []

    def test_phone_with_dot_separator(self):
        result = extract_vn_phones("Contact: 094.3199.979")
        assert "+84943199979" in result


# ---------------------------------------------------------------------------
# URL Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractUrls:
    def test_https_url(self):
        result = extract_urls("Visit https://johndoe.dev for portfolio.")
        assert "https://johndoe.dev" in result

    def test_http_url(self):
        result = extract_urls("Site: http://oldsite.com")
        assert "http://oldsite.com" in result

    def test_multiple_urls(self):
        text = "Portfolio: https://johndoe.dev | Blog: https://medium.com/@johndoe"
        result = extract_urls(text)
        assert len(result) == 2

    def test_no_urls(self):
        assert extract_urls("Just plain text, no links.") == []

    def test_strip_trailing_punctuation(self):
        """URLs followed by period/comma should not include that punctuation."""
        result = extract_urls("See https://example.com.")
        assert "https://example.com" in result
        assert "https://example.com." not in result

    def test_deduplication(self):
        text = "https://same.com and https://same.com"
        assert len(extract_urls(text)) == 1


# ---------------------------------------------------------------------------
# LinkedIn Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractLinkedIn:
    def test_full_https_url(self):
        result = extract_linkedin("https://www.linkedin.com/in/nguyen-van-a")
        assert len(result) == 1
        assert "linkedin.com/in/nguyen-van-a" in result[0]

    def test_short_url_no_https(self):
        result = extract_linkedin("linkedin.com/in/john-doe")
        assert len(result) == 1

    def test_url_in_sentence(self):
        text = "Connect at https://linkedin.com/in/tridung for professional network."
        result = extract_linkedin(text)
        assert len(result) == 1

    def test_no_linkedin(self):
        assert extract_linkedin("No LinkedIn here.") == []

    def test_deduplication(self):
        text = "linkedin.com/in/john linkedin.com/in/john"
        assert len(extract_linkedin(text)) == 1


# ---------------------------------------------------------------------------
# GitHub Extraction Tests
# ---------------------------------------------------------------------------

class TestExtractGitHub:
    def test_profile_url(self):
        result = extract_github("github.com/isntbim")
        assert len(result) == 1
        assert "github.com/isntbim" in result[0]

    def test_full_https_profile(self):
        result = extract_github("https://github.com/isntbim")
        assert len(result) == 1

    def test_no_github(self):
        assert extract_github("No GitHub here.") == []

    def test_deduplication(self):
        text = "github.com/isntbim and github.com/isntbim"
        assert len(extract_github(text)) == 1


# ---------------------------------------------------------------------------
# extract_all_contacts Integration Test
# ---------------------------------------------------------------------------

class TestExtractAllContacts:
    def test_full_cv_header_line(self):
        """Simulate a typical Vietnamese CV header line."""
        text = (
            "Lê Trí Dũng | AI Engineer\n"
            "Email: tridungit2005@gmail.com | Phone: 0941 423 545\n"
            "GitHub: github.com/isntbim | LinkedIn: linkedin.com/in/tri-dung\n"
            "Portfolio: https://tridung.dev"
        )
        result = extract_all_contacts(text)

        assert "tridungit2005@gmail.com" in result["emails"]
        assert "+84941423545" in result["phones"]
        assert any("github.com/isntbim" in g for g in result["github"])
        assert any("linkedin.com/in/tri-dung" in l for l in result["linkedin"])
        assert any("tridung.dev" in u for u in result["urls"])

    def test_empty_text_returns_empty_lists(self):
        result = extract_all_contacts("")
        assert result["emails"] == []
        assert result["phones"] == []
        assert result["urls"] == []
        assert result["linkedin"] == []
        assert result["github"] == []

    def test_returns_correct_typeddict_keys(self):
        result = extract_all_contacts("test@email.com")
        assert "emails" in result
        assert "phones" in result
        assert "urls" in result
        assert "linkedin" in result
        assert "github" in result

    def test_international_cv_english(self):
        """Non-VN CV should return no phone (no VN number present)."""
        text = "John Doe | john@gmail.com | https://johndoe.dev | github.com/johndoe"
        result = extract_all_contacts(text)
        assert "john@gmail.com" in result["emails"]
        assert result["phones"] == []
        assert any("johndoe.dev" in u for u in result["urls"])
