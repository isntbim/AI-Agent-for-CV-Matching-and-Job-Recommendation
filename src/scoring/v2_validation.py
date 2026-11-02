"""Literal provenance checks and review nominations for skill annotations.

These checks do not confer human approval of the interpretation of a quote.
"""

import hashlib
import re

from .normalization import SKILL_ALIASES, normalize, skill_name

# \s+ in Vietnamese phrases: CareerViet text carries non-breaking spaces ("ưu\xa0tiên")
MANDATORY = re.compile(r"\b(?:must|mandatory|essential|required|need(?:s)? to)\b|bắt\s+buộc|phải", re.I)
OPTIONAL = re.compile(r"\b(?:preferred|optional|nice.to.have|(?:strong )?plus|advantage|desirable)\b|ưu\s+tiên", re.I)
NEGATIVE = re.compile(r"\b(?:no experience (?:in|with)|do not (?:know|use)|without (?:knowledge|experience)|not (?:familiar|proficient))\b", re.I)


def name_in_quote(name: str, quote: str) -> bool:
    canonical = skill_name(name)
    spellings = {canonical, normalize(name)} | {alias for alias, target in SKILL_ALIASES.items() if target == canonical}
    text = normalize(quote)
    return any(re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w])", text) for value in spellings)


def validate_skill_evidence(envelope, text: str, *, strict: bool = True) -> list[dict]:
    """Return errors and review issues, restoring only whitespace-altered quotes."""
    issues = []

    def issue(path, reason, severity="error"):
        issues.append({"path": path, "reason": reason, "severity": severity})

    def quote(obj, field, path, names=()):
        value = getattr(obj, field)
        if not value:
            issue(path, "missing_source_quote")
            return ""
        if value not in text:
            pattern = r"\s+".join(re.escape(part) for part in value.split())
            match = re.search(pattern, text) if pattern else None
            if not match:
                issue(path, "quote_not_in_source")
                return value
            value = match.group(0)
            setattr(obj, field, value)
        if names and not all(name_in_quote(name, value) for name in names):
            issue(path, "quote_not_linked_to_skill")
        if NEGATIVE.search(value):
            issue(path, "negated_skill_claim")
        return value

    metadata = envelope.annotation
    if not metadata:
        issue("annotation", "missing_annotation_metadata")
    elif metadata.text_sha256 and metadata.text_sha256 != hashlib.sha256(text.encode()).hexdigest():
        issue("annotation.text_sha256", "source_checksum_mismatch")
    document = envelope.document
    if hasattr(document.skills, "requirements"):
        units = document.skills.requirements
        if (document.skills.required or document.skills.preferred) and not units:
            issue("requirements", "legacy_skills_without_v2_requirement_units")
        ids = set()
        prior = []
        for unit in units:
            path = "requirements." + unit.id
            if unit.id in ids:
                issue(path, "duplicate_requirement_id")
            ids.add(unit.id)
            value = quote(unit, "quote", path, unit.options)
            if unit.importance == "core":
                importance = quote(unit, "importance_quote", path + ".importance", unit.options)
                if not MANDATORY.search(importance):
                    issue(path, "core_without_mandatory_source_wording")
                if OPTIONAL.search(importance) or re.search(r"\bnot required\b", importance, re.I):
                    issue(path, "mixed_or_negated_core_scope", "review")
            elif unit.importance_quote:
                quote(unit, "importance_quote", path + ".importance", unit.options)
            options = {skill_name(option) for option in unit.options}
            if len(options) != len(unit.options):
                issue(path, "duplicate_alternatives", "review")
            if len(unit.options) > 1 and not re.search(r"\bor\b|hoặc|/", value, re.I):
                issue(path, "choice_scope_unconfirmed", "review")
            for previous, names in prior:
                if options & names and options != names and (len(options) > 1 or len(names) > 1):
                    issue(path, "overlapping_choice_group:" + previous, "review")
            prior.append((unit.id, options))
            if unit.issues and unit.review_status != "approved":
                issue(path, "unresolved_requirement_interpretation", "review")
    else:
        levels = {}
        all_details = [detail for group in document.skills for detail in group.details]
        for group in document.skills:
            for detail in group.details:
                path = "skills." + detail.name
                quote(detail, "skill_quote", path, [detail.name])
                if detail.provenance and metadata:
                    for key in ("source_id", "source_sha256", "text_sha256"):
                        expected, actual = getattr(metadata, key), getattr(detail.provenance, key)
                        if actual and expected and actual != expected:
                            issue(path, "inconsistent_provenance:" + key)
                canonical = skill_name(detail.name)
                if detail.proficiency != "normal":
                    previous = levels.get(canonical)
                    if previous is not None and previous != detail.proficiency:
                        issue(path, "conflicting_explicit_proficiency", "review")
                    levels[canonical] = detail.proficiency
                if detail.proficiency != "normal":
                    value = quote(detail, "proficiency_quote", path + ".proficiency", [detail.name])
                    if normalize(detail.raw_proficiency or "") != detail.proficiency:
                        issue(path, "unmapped_proficiency_wording", "review")
                    if not re.search(r"\b" + re.escape(detail.proficiency) + r"\b", value, re.I):
                        issue(path, "proficiency_label_not_in_quote")
                    other = {skill_name(item.name) for item in all_details
                             if skill_name(item.name) != canonical and name_in_quote(item.name, value)}
                    if other and re.search(r"\band\b|[,;/]", value, re.I):
                        issue(path, "group_proficiency_is_not_individual_evidence", "review")
                elif detail.raw_proficiency and not (
                    (detail.provenance and detail.provenance.review_status == "approved")
                    or (metadata and metadata.review_status == "approved")
                ):
                    issue(path, "unmapped_proficiency_wording", "review")
    if strict and any(row["severity"] == "error" for row in issues):
        from src.parser.llm_extractor import ExtractionError
        raise ExtractionError("Invalid v2 skill evidence: " + "; ".join(row["path"] + ":" + row["reason"]
                                                                      for row in issues if row["severity"] == "error"))
    return issues
