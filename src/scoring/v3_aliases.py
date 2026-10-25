"""D3-31: human-approved equivalences isolated from v1/v2.

Similarity nominates aliases; it never approves them. Raw source evidence is
preserved. Only v3 canonicalization and quote linking use this extension.
"""
from copy import deepcopy
import hashlib
import json
import re
from .normalization import SKILL_ALIASES, normalize, skill_name as legacy_skill_name

ALIAS_POLICY_VERSION = "human-approved-aliases-v3-v1"
APPROVED_ALIAS_RECORDS = (
    {"alias": "golang", "canonical": "go", "decision_id": "D3-31",
     "approver": "user", "approval_relay": "Claude Code",
     "approved_on": "2026-10-12", "review_kind": "direct_human",
     "approval_evidence": "The user approved golang -> go in the question 5 decision.",
     "proposal_evidence": "Claude Code BGE-M3 illustrative cosine Golang~Go 0.624; not an approval threshold.",
     "record": "docs/research/ats_scorer/decisions.md#d3-31"},
)
APPROVED_ALIASES = {row["alias"]: row["canonical"] for row in APPROVED_ALIAS_RECORDS}
SKILL_ALIASES_V3 = {**SKILL_ALIASES, **APPROVED_ALIASES}
APPROVED_ALIASES_SHA256 = hashlib.sha256(json.dumps(
    APPROVED_ALIAS_RECORDS, sort_keys=True, ensure_ascii=False,
    separators=(",", ":")).encode()).hexdigest()


def skill_name(name: str) -> str:
    canonical = legacy_skill_name(name)
    return APPROVED_ALIASES.get(canonical, canonical)


def alias_spellings(name: str) -> set[str]:
    canonical = skill_name(name)
    return {normalize(name), canonical} | {
        alias for alias in SKILL_ALIASES_V3 if skill_name(alias) == canonical}


# PDF text artifacts: zero-width marks, soft hyphens, and "Â" left by UTF-8 decoded as cp1252 ("Â " for a no-break space).
PDF_MARKS = re.compile(r"[​-‍⁠﻿­]|Â(?=[\s -¿​])")
HYPHEN_WRAP = re.compile(r"-[^\S\n]*\n\s*")
WORD_WRAP = re.compile(r"(?<=[^\W\d_])\n(?=[^\W\d_])")


def name_in_quote(name: str, quote: str) -> bool:
    # A line break inside a quote may be a real break, a word-wrap ("toler\nance", "Micro-\nsoft") or a real
    # hyphen ("e-\ncommerce"); accept every reading.
    quote = PDF_MARKS.sub("", quote)
    texts = {normalize(q) for q in (quote, HYPHEN_WRAP.sub("", quote), HYPHEN_WRAP.sub("-", quote), WORD_WRAP.sub("", quote))}
    return any(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text)
               for value in alias_spellings(name) for text in texts)


def alias_provenance(candidate: str, requirement: str) -> dict | None:
    if skill_name(candidate) != skill_name(requirement):
        return None
    for row in APPROVED_ALIAS_RECORDS:
        if row["alias"] in {normalize(candidate), normalize(requirement)}:
            return deepcopy(row)
    return None
