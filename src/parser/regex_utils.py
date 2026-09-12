"""
regex_utils.py — Tier 1 Fast Contact Info Extractor (CPU-only)

Extracts contact information from raw CV text using regular expressions.
This runs BEFORE the LLM call to reduce token usage and improve extraction accuracy.

Supported extractions:
    - Email addresses (RFC 5322 simplified)
    - Vietnamese phone numbers (+84, 03x, 05x, 07x, 08x, 09x formats)
    - Generic URLs (https/http)
    - LinkedIn profile URLs
    - GitHub profile URLs

Phone normalization output: E.164 format (+84xxxxxxxxx)
"""

import re
from typing import TypedDict

# ---------------------------------------------------------------------------
# Compiled Regex Patterns
# ---------------------------------------------------------------------------

# Email: standard RFC 5322 simplified pattern
_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# Vietnamese phone numbers
# Supports formats:
#   +84xxxxxxxxx  | +84-xxx-xxx-xxx | +84 xxx xxx xxx
#   0xxxxxxxxx    | 0xx.xxx.xxxx    | 0xx-xxx-xxxx
#   Prefixes: 03x, 05x, 07x, 08x, 09x (10-digit VN mobile numbers)
#   Also: +84 country code replacing leading 0
_VN_PHONE_RAW_PATTERN = re.compile(
    r"""
    (?:                        # Group for prefix alternatives
        \+84                   # International prefix +84
        |0(?=3|5|7|8|9)        # Domestic prefix 0 followed by valid 2nd digit
    )
    [\s.\-]?                   # Optional separator after prefix
    (?:                        # 9 digits split into 3-3-3 or 4-3-2 etc.
        \d{1,4}[\s.\-]?\d{1,4}[\s.\-]?\d{1,4}
    )
    """,
    re.VERBOSE,
)

# LinkedIn profile URL
_LINKEDIN_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9\-_%]+/?",
    re.IGNORECASE,
)

# GitHub profile URL (only user profile, not repos)
_GITHUB_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9\-]+(?!/[a-zA-Z0-9\-]+/[a-zA-Z0-9\-]+)/?",
    re.IGNORECASE,
)

# Generic URL (http / https, or domain-based website like people.tamu.edu/~bnguyen4656)
_URL_PATTERN = re.compile(
    r"(?:https?://[^\s<>\"')\]]+|(?<![@\w])(?:[a-zA-Z0-9\-]+\.)+(?:edu|com|org|net|vn|dev|io|ai|tech|me|app|site|online)(?:/[^\s<>\"'|)\]]*)?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Phone Normalization
# ---------------------------------------------------------------------------

def _normalize_vn_phone(raw: str) -> str:
    """
    Normalize a raw Vietnamese phone number string to E.164 format (+84xxxxxxxxx).

    Examples:
        "0941 423 545"  -> "+84941423545"
        "+84-941-423-545" -> "+84941423545"
        "84.355.692.522"  -> "+84355692522"
        "0355.692.522"    -> "+84355692522"
    """
    # Strip everything except digits and leading +
    digits_only = re.sub(r"[^\d]", "", raw)

    if raw.lstrip().startswith("+84"):
        # Already has country code; take all digits after 84
        if digits_only.startswith("84") and len(digits_only) >= 11:
            return f"+{digits_only}"
    elif digits_only.startswith("84") and len(digits_only) == 11:
        # Missing + but has country code (e.g. "84941423545")
        return f"+{digits_only}"
    elif digits_only.startswith("0") and len(digits_only) == 10:
        # Domestic format: replace leading 0 with +84
        return f"+84{digits_only[1:]}"

    # Fallback: return cleaned raw if normalization fails
    return f"+{digits_only}" if digits_only else raw.strip()


# ---------------------------------------------------------------------------
# Extraction Functions
# ---------------------------------------------------------------------------

def extract_emails(text: str) -> list[str]:
    """
    Extract all email addresses from text.

    Args:
        text: Raw CV text string.

    Returns:
        Deduplicated list of email addresses (lowercase).

    Examples:
        >>> extract_emails("Contact me at john@gmail.com or info@company.io")
        ['john@gmail.com', 'info@company.io']
    """
    matches = _EMAIL_PATTERN.findall(text)
    seen: set[str] = set()
    result = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            result.append(m.lower())
    return result


def extract_vn_phones(text: str) -> list[str]:
    """
    Extract and normalize Vietnamese phone numbers from text.

    Args:
        text: Raw CV text string.

    Returns:
        Deduplicated list of phone numbers in E.164 format (+84xxxxxxxxx).

    Examples:
        >>> extract_vn_phones("Phone: 0941 423 545 or +84-355-692-522")
        ['+84941423545', '+84355692522']
    """
    raw_matches = _VN_PHONE_RAW_PATTERN.findall(text)
    seen: set[str] = set()
    result = []
    for raw in raw_matches:
        normalized = _normalize_vn_phone(raw)
        digits = re.sub(r"[^\d]", "", normalized)
        # Validate: must be exactly 11 digits starting with 84
        if len(digits) == 11 and digits.startswith("84"):
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
    return result


def extract_urls(text: str) -> list[str]:
    """
    Extract all HTTP/HTTPS URLs from text.

    Args:
        text: Raw CV text string.

    Returns:
        Deduplicated list of URLs (excluding LinkedIn/GitHub — use their dedicated extractors).

    Examples:
        >>> extract_urls("Portfolio: https://johndoe.dev  GitHub: https://github.com/johndoe")
        ['https://johndoe.dev', 'https://github.com/johndoe']
    """
    matches = _URL_PATTERN.findall(text)
    seen: set[str] = set()
    result = []
    for m in matches:
        # Strip trailing punctuation that was accidentally captured
        cleaned = m.rstrip(".,;:!?)")
        # Exclude linkedin and github profiles from generic urls (they have dedicated extractors)
        cleaned_lower = cleaned.lower()
        if "linkedin.com" in cleaned_lower or "github.com" in cleaned_lower:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def extract_linkedin(text: str) -> list[str]:
    """
    Extract LinkedIn profile URLs from text.

    Args:
        text: Raw CV text string.

    Returns:
        Deduplicated list of LinkedIn profile URLs.

    Examples:
        >>> extract_linkedin("linkedin.com/in/john-doe")
        ['linkedin.com/in/john-doe']
    """
    matches = _LINKEDIN_PATTERN.findall(text)
    seen: set[str] = set()
    result = []
    for m in matches:
        cleaned = m.rstrip("/")
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def extract_github(text: str) -> list[str]:
    """
    Extract GitHub profile URLs from text.

    Args:
        text: Raw CV text string.

    Returns:
        Deduplicated list of GitHub profile URLs (profile level only, not repo URLs).

    Examples:
        >>> extract_github("github.com/isntbim")
        ['github.com/isntbim']
    """
    matches = _GITHUB_PATTERN.findall(text)
    seen: set[str] = set()
    result = []
    for m in matches:
        cleaned = m.rstrip("/")
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


class ContactInfo(TypedDict):
    """Structured contact information extracted from CV text."""
    emails: list[str]
    phones: list[str]
    urls: list[str]
    linkedin: list[str]
    github: list[str]


def extract_all_contacts(text: str) -> ContactInfo:
    """
    Run all extractors on text and return a unified ContactInfo dict.

    This is the primary public interface. Run this on raw CV text before
    passing to the LLM extractor to supplement Basics fields.

    Args:
        text: Raw CV text string (output of PDF/DOCX extractor).

    Returns:
        ContactInfo TypedDict with keys: emails, phones, urls, linkedin, github.

    Examples:
        >>> result = extract_all_contacts("John Doe | john@gmail.com | +84 941 423 545 | github.com/johndoe")
        >>> result["emails"]
        ['john@gmail.com']
        >>> result["phones"]
        ['+84941423545']
    """
    return ContactInfo(
        emails=extract_emails(text),
        phones=extract_vn_phones(text),
        urls=extract_urls(text),
        linkedin=extract_linkedin(text),
        github=extract_github(text),
    )
