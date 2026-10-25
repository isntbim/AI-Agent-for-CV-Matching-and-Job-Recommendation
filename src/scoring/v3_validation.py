"""Versioned, literal evidence validation for explicit skill depth."""

from copy import deepcopy
import re
from .normalization import normalize
from .v3_aliases import alias_spellings, name_in_quote, skill_name
from .v2_validation import MANDATORY, OPTIONAL, validate_skill_evidence as validate_v2

RANK = {"intermediate": 1, "advanced": 2, "expert": 3}
POLICY_VERSION = "explicit-skill-depth-v3-no-beginner-v1"


def title_requires_review(unit) -> bool:
    """A human may resolve a title choice; title-only depth still needs body evidence."""
    get = lambda obj, key, default=None: obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    if not (get(unit, "quote", "") or "").startswith("Title:"):
        return False
    depth = any(get(detail, "minimum_proficiency", "normal") != "normal"
                for detail in get(unit, "option_details", []))
    choice = len(get(unit, "options", [])) > 1 and get(unit, "review_status", "pending") != "approved"
    return depth or choice


def ambiguous_depth_context(text: str) -> bool:
    """Conservative scope guard; a negative/training clause is not positive evidence."""
    return bool(re.search(
        r"\b(?:not|never|no|without|lack(?:s|ed|ing)?|isn['’]t|aren['’]t|"
        r"wasn['’]t|weren['’]t|don['’]t|doesn['’]t|cannot|can['’]t|"
        r"train(?:ing|ed)?|teach(?:ing)?|taught|instruct(?:ing|ed)?)\b", text, re.I))


def source_context(text: str, start: int, end: int) -> str:
    """Keep the surrounding clause so a short excerpt cannot drop negation."""
    # Single newlines may be PDF line wrapping and cannot reset negative scope.
    boundaries = ".!?;"
    left = max((text.rfind(char, 0, start) for char in boundaries), default=-1) + 1
    right = min((pos for char in boundaries if (pos := text.find(char, end)) >= 0), default=len(text))
    return text[left:right]


def source_depth_supported(quote: str, text: str) -> bool:
    occurrences = list(re.finditer(re.escape(quote), text)) if quote else []
    return bool(occurrences) and all(
        not ambiguous_depth_context(source_context(text, match.start(), match.end()))
        for match in occurrences)


def direct_level(name: str, quote: str) -> str | None:
    """A level must directly qualify this skill; training other people is ambiguous."""
    text = normalize(quote)
    if ambiguous_depth_context(text):
        return None
    names = alias_spellings(name)
    found = set()
    for spelling in names:
        token = re.escape(spelling)
        for raw in (*RANK, "proficient", "beginner"):
            level = "normal" if raw == "beginner" else "intermediate" if raw == "proficient" else raw
            # Bounded phrases avoid inheriting a group label or a title level.
            patterns = (rf"(?<!\w){raw}\s+(?:(?:in|with|knowledge of|proficiency in)\s+)?{token}(?!\w)",
                        rf"(?<!\w){token}\s*[:\-–(]?\s*{raw}(?!\w)")
            if any(re.search(pattern, text) for pattern in patterns):
                found.add(level)
    return next(iter(found)) if len(found) == 1 else None


def beginner_wording_supported(name: str, raw: str | None, quote: str | None) -> bool:
    """An ignored literal Beginner claim remains evidence-bound, unlike Basic."""
    return normalize(raw or "") == "beginner" and direct_level(name, quote or "") == "normal"


def validate_skill_evidence(envelope, text: str, *, strict: bool = True) -> list[dict]:
    # Reuse provenance/importance checks without changing v2's stricter literal
    # vocabulary. The clone is only for legacy validation, never saved as evidence.
    clone = deepcopy(envelope)
    is_job = hasattr(envelope.document.skills, "requirements")
    if not is_job:
        for group in clone.document.skills:
            for detail in group.details:
                if (detail.proficiency != "normal" and direct_level(detail.name, detail.proficiency_quote or "") == detail.proficiency) or beginner_wording_supported(detail.name, detail.raw_proficiency, detail.proficiency_quote):
                    # Disable just legacy vocabulary checks on a clone; validate
                    # the untouched original quote against the real source below.
                    detail.proficiency = "normal"
                    detail.raw_proficiency = None
    issues = validate_v2(clone, text, strict=False)

    # Extend only legacy name linking; retain provenance, source checksum,
    # negation and importance errors. Never rewrite the original quotation.
    links = {}
    if is_job:
        for unit in envelope.document.skills.requirements:
            path = "requirements." + unit.id
            links.setdefault(path, []).append((unit.options, unit.quote))
            links.setdefault(path + ".importance", []).append((unit.options, unit.importance_quote or ""))
    else:
        for group in envelope.document.skills:
            for detail in group.details:
                path = "skills." + detail.name
                links.setdefault(path, []).append(([detail.name], detail.skill_quote or ""))
                links.setdefault(path + ".proficiency", []).append(([detail.name], detail.proficiency_quote or ""))
    issues = [issue for issue in issues if not (
        issue["reason"] == "quote_not_linked_to_skill" and issue["path"] in links
        and all(name_in_quote(name, quote)
                for names, quote in links[issue["path"]] for name in names))]

    def review(path, reason):
        row = {"path": path, "reason": reason, "severity": "review"}
        if row not in issues:
            issues.append(row)

    # Newly approved equivalences participate in duplicate/conflict review.
    if is_job:
        prior = []
        for unit in envelope.document.skills.requirements:
            options = {skill_name(name) for name in unit.options}
            if len(options) != len(unit.options):
                review("requirements." + unit.id, "duplicate_alternatives")
            for previous, names in prior:
                if options & names and options != names and (len(options) > 1 or len(names) > 1):
                    review("requirements." + unit.id, "overlapping_choice_group:" + previous)
            prior.append((unit.id, options))
    else:
        levels = {}
        for group in envelope.document.skills:
            for detail in group.details:
                if detail.proficiency != "normal":
                    canonical = skill_name(detail.name)
                    if canonical in levels and levels[canonical] != detail.proficiency:
                        review("skills." + detail.name, "conflicting_explicit_proficiency")
                    levels[canonical] = detail.proficiency

    def check(name, level, raw, quote, path):
        if level == "normal" and normalize(raw or "") != "beginner":
            return
        if not quote or quote not in text:
            issues.append({"path": path, "reason": "depth_quote_not_in_source", "severity": "error"})
        elif not source_depth_supported(quote, text):
            issues.append({"path": path, "reason": "depth_negated_or_ambiguous_source_scope", "severity": "error"})
        elif not name_in_quote(name, quote) or direct_level(name, quote) != level or normalize(raw or "") not in {level, "beginner" if level == "normal" else "proficient" if level == "intermediate" else level}:
            issues.append({"path": path, "reason": "depth_not_explicit_skill_specific", "severity": "error"})

    if is_job:
        for unit in envelope.document.skills.requirements:
            if title_requires_review(unit):
                issues.append({"path": "requirements." + unit.id, "reason": "title_choice_or_depth_requires_body_review", "severity": "review"})
            for detail in getattr(unit, "option_details", []):
                path = "requirements." + unit.id + ".depth." + detail.name
                check(detail.name, detail.minimum_proficiency, detail.raw_proficiency, detail.proficiency_quote, path)
                for experience in detail.experience:
                    if experience.quote not in text or not name_in_quote(detail.name, experience.quote):
                        issues.append({"path": path, "reason": "experience_quote_not_skill_linked_source", "severity": "error"})
    else:
        for group in envelope.document.skills:
            for detail in group.details:
                check(detail.name, detail.proficiency, detail.raw_proficiency, detail.proficiency_quote, "skills." + detail.name)
    if strict and any(row["severity"] == "error" for row in issues):
        from src.parser.llm_extractor import ExtractionError
        raise ExtractionError("Invalid v3 skill evidence: " + "; ".join(row["path"] + ":" + row["reason"] for row in issues if row["severity"] == "error"))
    return issues
