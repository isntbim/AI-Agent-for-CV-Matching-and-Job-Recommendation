"""Offline comparison with historical silver labels using stored structured facts."""

import re
from datetime import date

from src.parser.job_schemas import JobDescriptionSchema
from src.parser.schemas import ResumeSchema

from .models import Evidence, Facts, JobInput, ResumeInput

ADAPTER_VERSION = "silver-structured-v1"
DEGREES = {
    "doctorate": r"\b(?:ph\.?\s*d\.?|doctorate|doctoral)\b",
    "master": r"\b(?:master(?:s|'s)?|mba|m\.?\s*(?:s|sc|a|res)\.?)\b",
    "bachelor": r"\b(?:bachelor(?:s|'s)?|b\.?\s*(?:a|s|sc|com|ba|tech)\.?)\b",
    "associate": r"\b(?:associate|a\.?\s*s\.?)\b",
    "high_school": r"\b(?:high school|secondary school)\b",
}
INCOMPLETE = re.compile(r"\b(?:candidate|expected|pursuing|toward|in progress|coursework|non.degree)\b", re.IGNORECASE)
OPTIONAL = re.compile(r"\b(?:preferred|advantage|plus|optional|not required)\b", re.IGNORECASE)
EDUCATION_CONTEXT = re.compile(r"\b(?:degree|bachelor|master|doctorate|doctoral|ph\.?\s*d|high school|secondary school)\b", re.IGNORECASE)
ROLE_WORDS = {
    "developer", "engineer", "analyst", "accountant", "manager", "executive",
    "director", "specialist", "officer", "supervisor", "consultant", "intern",
    "trainee", "fresher", "junior", "senior", "middle", "lead", "expert",
}
ROLE_PHRASES = ("nhân viên", "chuyên viên", "kiểm toán viên", "kế toán viên",
                "thực tập sinh", "trưởng phòng", "quản lý")


def degree_levels(text: str) -> list[str]:
    return [level for level, pattern in DEGREES.items() if re.search(pattern, text, re.IGNORECASE)]


def title_like_skill(name: str, normalized_title: str) -> bool:
    """Same title/seniority pollution guard used by the silver annotation script."""
    value = name.strip().casefold()
    title = normalized_title.replace("_", " ").strip().casefold()
    return (not value or bool(title and (value == title or title in value or value in title))
            or bool(set(value.split()) & ROLE_WORDS)
            or any(phrase in value for phrase in ROLE_PHRASES))


def _evidence(facts: Facts, dimension: str, values) -> None:
    quotes = list(dict.fromkeys(str(value) for value in values if value is not None and str(value).strip()))
    if quotes:
        facts.evidence[dimension] = [Evidence(quote=value) for value in quotes]


def _finished(value: str | None, as_of: date) -> bool:
    if not value:
        return False
    try:
        # Year-only graduation dates are interpreted conservatively as year end.
        end = date.fromisoformat(value + "-12-31" if len(value) == 4 else value + "-01" if len(value) == 7 else value)
    except ValueError:
        return False
    return end <= as_of


def resume_from_record(record: dict, as_of: date) -> ResumeInput:
    """Use CV fields only; domain metadata, labels and teacher scores are excluded."""
    doc = ResumeSchema.model_validate(record)
    facts = Facts()
    _evidence(facts, "skills", doc.get_flat_skills())
    titles = [doc.basics.label, *[work.position for work in doc.work]]
    _evidence(facts, "title", titles)
    _evidence(facts, "industry", titles)
    _evidence(facts, "experience", [value for work in doc.work
                                    for value in (work.position, work.start_date, work.end_date)])
    degrees = []
    degree_quotes = []
    for edu in doc.education:
        text = edu.study_type or ""
        if _finished(edu.end_date, as_of) and not INCOMPLETE.search(text):
            levels = degree_levels(text)
            degrees.extend(levels)
            if levels:
                degree_quotes.extend([text, edu.end_date])
    facts.completed_degrees = list(dict.fromkeys(degrees))
    _evidence(facts, "education", degree_quotes)
    if doc.basics.location:
        facts.country = doc.basics.location.country_code
        _evidence(facts, "location", [doc.basics.location.city, facts.country])
    return ResumeInput(document=doc, facts=facts)


def job_from_record(record: dict) -> tuple[JobInput, list[str]]:
    """Read JD requirements and explicit fields without inferring from pair labels."""
    doc = JobDescriptionSchema.model_validate(record)
    facts = Facts()
    removed = []
    for bucket in (doc.skills.required, doc.skills.preferred):
        removed.extend(item.name for item in bucket if title_like_skill(item.name, doc.title_normalized))
        bucket[:] = [item for item in bucket if not title_like_skill(item.name, doc.title_normalized)]
    _evidence(facts, "skills", [item.name for item in [*doc.skills.required, *doc.skills.preferred]])
    _evidence(facts, "title", [doc.title])
    _evidence(facts, "industry", [doc.title])
    # Stored min_years is used as a structured source value, not a newly extracted claim.
    _evidence(facts, "experience", [doc.seniority.min_years])
    requirements = doc.description.requirements
    levels = []
    degree_quotes = []
    for requirement in requirements:
        for sentence in re.split(r"[;\n]|(?<=\.)\s+(?=[A-Z])", requirement):
            # Acronyms such as MS in "MS Office" are not education requirements.
            found = degree_levels(sentence) if EDUCATION_CONTEXT.search(sentence) else []
            if found and not OPTIONAL.search(sentence):
                levels.extend(found)
                degree_quotes.append(sentence)
            if re.search(r"\bno (?:prior |previous |work )?experience (?:is )?required\b", sentence, re.IGNORECASE):
                facts.no_experience_required = True
                _evidence(facts, "experience", [sentence])
            if re.search(r"\bno (?:degree|education) (?:is )?required\b", sentence, re.IGNORECASE):
                facts.no_education_required = True
                degree_quotes.append(sentence)
    if levels:
        # Alternatives such as "Bachelor's or Master's" mean the lowest listed requirement.
        rank = {level: i for i, level in enumerate(reversed(DEGREES))}
        facts.minimum_degree = min(levels, key=rank.get)
    _evidence(facts, "education", degree_quotes)
    raw_location = record.get("location") or {}
    facts.country = raw_location.get("countryCode") or raw_location.get("country_code") or raw_location.get("country")
    facts.remote = doc.location.remote_policy.casefold() == "remote" if doc.location.remote_policy else False
    # Remote alone does not establish worldwide eligibility.
    _evidence(facts, "location", [raw_location.get("city"), facts.country, raw_location.get("remote_policy_raw")])
    raw_salary = record.get("compensation") or {}
    facts.currency = raw_salary.get("currency")
    _evidence(facts, "salary", [raw_salary.get("raw_text")])
    return JobInput(document=doc, facts=facts), removed
