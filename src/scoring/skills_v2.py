"""Evidence-backed ATS skills scoring with deterministic, one-use assignment.

The module is deliberately pure apart from hashing its own code for audit identity.
Aliases are equivalences. Semantic relations are explicit, directional records;
neither an embedding model nor a language model is consulted during scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .normalization import SKILL_ALIASES, normalize, skill_name
from .v2_validation import MANDATORY, OPTIONAL


GROUP_WEIGHTS = {"core": 1.0, "standard": 0.6, "preferred": 0.4}
GROUP_SHARES = {"core": 0.7, "standard": 0.2, "preferred": 0.1}
IMPORTANCE_ORDER = {"core": 0, "standard": 1, "preferred": 2}
FULFILLMENT = {"beginner": 0.5, "normal": 1.0, "intermediate": 1.0, "advanced": 1.0, "expert": 1.0}
BONUS = {"beginner": 0.0, "normal": 0.0, "intermediate": 0.0, "advanced": 0.1, "expert": 0.2}
RELATION_COEFFICIENT = {"near": 0.7, "transfer": 0.4}
VARIANTS = {"equal_groups", "group_weights", "semantic", "proficiency", "full"}


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=1)
def _code_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = ["src/scoring/skills_v2.py", "src/scoring/normalization.py", "src/scoring/models.py",
             "src/scoring/rules.py", "src/scoring/v2_validation.py", "src/parser/skill_v2.py",
             "src/parser/schemas.py", "src/parser/job_schemas.py", "src/parser/match_schemas.py"]
    return _digest({path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths})


def _mentions_skill(quote: str, canonical: str) -> bool:
    """Check a link, not the human interpretation or source authenticity."""
    names = {canonical, *(alias for alias, target in SKILL_ALIASES.items() if target == canonical)}
    normalized_quote = normalize(quote)
    return any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalized_quote) for name in names)


def _identity(resume: Any, job: Any, config: Any) -> dict[str, Any]:
    cv_meta, jd_meta = _get(resume, "annotation"), _get(job, "annotation")
    relations = sorted((_plain(item) for item in _get(config, "relations", [])), key=lambda item: json.dumps(item, sort_keys=True))
    resume_document, job_document = _get(resume, "document"), _get(job, "document")
    annotations = {
        "resume": {"metadata": _plain(cv_meta), "skills": _plain(_get(resume_document, "skills", []))},
        "job": {"metadata": _plain(jd_meta), "skills": _plain(_get(job_document, "skills"))},
    }
    statuses = {"resume": _get(cv_meta, "review_status", "missing"), "job": _get(jd_meta, "review_status", "missing")}
    return {
        "ruleset": _get(config, "version", "ats-en-v2"),
        "schema_version": "ats-skills-v2",
        "variant": _get(config, "variant", "full"),
        "relation_version": _get(config, "relation_version", "empty-v1"),
        "code_sha256": _code_digest(),
        "config_sha256": _digest(config),
        "annotation_sha256": _digest(annotations),
        "relations_sha256": _digest(relations),
        "source_sha256": {"resume": _get(cv_meta, "source_sha256"), "job": _get(jd_meta, "source_sha256")},
        "text_sha256": {"resume": _get(cv_meta, "text_sha256"), "job": _get(jd_meta, "text_sha256")},
        "review_status": statuses,
        "provisional": any(status != "approved" for status in statuses.values()),
    }


@dataclass(frozen=True)
class _Candidate:
    name: str
    original_name: str
    proficiency: str
    skill_quote: str
    proficiency_quote: str | None
    provenance: Any


@dataclass(frozen=True)
class _Requirement:
    id: str
    options: tuple[str, ...]
    original_options: tuple[str, ...]
    importance: str
    quote: str
    importance_quote: str | None
    review_status: str
    issues: tuple[str, ...]
    source_ids: tuple[str, ...]


def _prepare_candidates(resume: Any) -> tuple[list[_Candidate], list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    candidates: dict[str, list[_Candidate]] = {}
    merges: list[dict[str, Any]] = []
    groups = _get(_get(resume, "document"), "skills", []) or []
    all_names = {skill_name(_get(item, "name", "")) for group in groups for item in (_get(group, "details", []) or []) if normalize(_get(item, "name", ""))}
    for group in groups:
        details = _get(group, "details", []) or []
        keyword_names = {skill_name(name) for name in _get(group, "keywords", []) if normalize(name)}
        detail_names = {skill_name(_get(item, "name", "")) for item in details if normalize(_get(item, "name", ""))}
        if keyword_names != detail_names:
            warnings.append("CV keywords and v2 skill details are inconsistent.")
        for item in details:
            name = _get(item, "name", "")
            canonical = skill_name(name)
            level = _get(item, "proficiency", "normal")
            skill_quote = _get(item, "skill_quote") or ""
            level_quote = _get(item, "proficiency_quote")
            if not canonical or level not in FULFILLMENT:
                warnings.append(f"CV skill {name!r} has an invalid name or proficiency.")
                continue
            if not skill_quote or not _mentions_skill(skill_quote, canonical):
                warnings.append(f"CV skill {canonical} lacks a quote linked to that skill.")
            if level != "normal" and (not level_quote or not _mentions_skill(level_quote, canonical)):
                warnings.append(f"CV proficiency for {canonical} lacks a quote linked to that skill.")
            if level != "normal" and level_quote:
                if not re.search(r"(?<!\w)" + re.escape(level) + r"(?!\w)", normalize(level_quote)):
                    warnings.append(f"CV proficiency for {canonical} is not explicitly stated as {level}.")
                raw_level = _get(item, "raw_proficiency")
                if not raw_level or normalize(raw_level) != level:
                    warnings.append(f"CV proficiency for {canonical} uses an unmapped raw expression.")
                if any(other != canonical and _mentions_skill(level_quote, other) for other in all_names):
                    warnings.append(f"CV proficiency for {canonical} is quoted across multiple skills and requires a per-skill excerpt.")
            neutral_wording_reviewed = (
                _get(_get(item, "provenance"), "review_status") == "approved"
                or _get(_get(resume, "annotation"), "review_status") == "approved"
            )
            if level == "normal" and _get(item, "raw_proficiency") and not neutral_wording_reviewed:
                warnings.append(f"CV proficiency for {canonical} has unmapped wording requiring review.")
            candidates.setdefault(canonical, []).append(_Candidate(canonical, name, level, skill_quote, level_quote, _plain(_get(item, "provenance"))))
    result = []
    for name, items in sorted(candidates.items()):
        explicit_levels = {item.proficiency for item in items if item.proficiency != "normal"}
        if len(explicit_levels) > 1:
            warnings.append(f"CV skill {name} has conflicting explicit proficiency levels.")
        # An explicit, supported level survives a duplicate neutral default.
        chosen = sorted(items, key=lambda item: (item.proficiency == "normal", item.proficiency, normalize(item.original_name), item.skill_quote))[0]
        result.append(chosen)
        if len(items) > 1:
            merges.append({"kind": "candidate", "canonical_name": name, "count": len(items), "proficiency": chosen.proficiency})
    return result, warnings, merges


def _prepare_requirements(job: Any) -> tuple[list[_Requirement], list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    merges: list[dict[str, Any]] = []
    grouped: dict[tuple[str, ...], list[_Requirement]] = {}
    seen_ids: dict[str, tuple[str, ...]] = {}
    for item in _get(_get(_get(job, "document"), "skills"), "requirements", []) or []:
        originals = tuple(_get(item, "options", []) or [])
        options = tuple(sorted({skill_name(name) for name in originals if normalize(name)}))
        importance = _get(item, "importance", "standard")
        quote = _get(item, "quote") or ""
        importance_quote = _get(item, "importance_quote")
        identifier = _get(item, "id", "")
        review = _get(item, "review_status", "pending")
        issues = tuple(sorted(_get(item, "issues", []) or []))
        if not options or importance not in GROUP_WEIGHTS or not identifier:
            warnings.append(f"JD requirement {identifier!r} has invalid options, importance, or ID.")
            continue
        if identifier in seen_ids and seen_ids[identifier] != options:
            warnings.append(f"JD requirement ID {identifier} identifies different requirements.")
        seen_ids[identifier] = options
        if not quote or not all(_mentions_skill(quote, option) for option in options):
            warnings.append(f"JD requirement {identifier} lacks evidence linked to every option.")
        if importance == "core":
            if (not importance_quote or not all(_mentions_skill(importance_quote, option) for option in options)
                    or not MANDATORY.search(importance_quote)):
                warnings.append(f"Core requirement {identifier} lacks skill-linked mandatory importance evidence.")
            elif OPTIONAL.search(importance_quote) or re.search(r"\bnot required\b", importance_quote, re.I):
                warnings.append(f"Core requirement {identifier} has mixed or negated importance scope requiring review.")
        if review == "rejected":
            warnings.append(f"JD requirement {identifier} is rejected and must be corrected.")
        if issues and review != "approved":
            warnings.append(f"JD requirement {identifier} has unresolved review issues: {', '.join(issues)}.")
        if any(re.search(r"\s(?:and|or)\s|\|", option, re.I) for option in options):
            warnings.append(f"JD requirement {identifier} contains an unsplit compound option.")
        grouped.setdefault(options, []).append(_Requirement(identifier, options, originals, importance, quote, importance_quote, review, issues, (identifier,)))
    result = []
    for options, items in grouped.items():
        eligible = [item for item in items if item.importance != "core" or item.importance_quote]
        chosen = min(eligible or items, key=lambda item: (IMPORTANCE_ORDER[item.importance], item.id, item.quote))
        result.append(_Requirement(chosen.id, chosen.options, chosen.original_options, chosen.importance, chosen.quote, chosen.importance_quote, chosen.review_status, chosen.issues, tuple(sorted({item.id for item in items}))))
        if len(items) > 1:
            merges.append({"kind": "requirement", "requirement_ids": sorted(item.id for item in items), "options": list(options), "importance": chosen.importance, "reason": "Highest source-backed importance retained; duplicate denominator removed."})
    result.sort(key=lambda item: (item.id, item.options))
    for index, left in enumerate(result):
        for right in result[index + 1:]:
            if set(left.options) & set(right.options) and (left.review_status != "approved" or right.review_status != "approved"):
                warnings.append(f"Overlapping requirements {left.id} and {right.id} require review.")
    return result, warnings, merges


def _relations(config: Any) -> dict[tuple[str, str], tuple[float, str, bool]]:
    result: dict[tuple[str, str], tuple[float, str, bool]] = {}
    allow_provisional = _get(config, "allow_provisional_relations", False)
    for relation in _get(config, "relations", []) or []:
        status = _get(relation, "review_status", "pending")
        if status != "approved" and not (allow_provisional and status == "pending"):
            continue
        kind = _get(relation, "relation")
        if kind not in RELATION_COEFFICIENT:
            continue
        key = (skill_name(_get(relation, "candidate", "")), skill_name(_get(relation, "requirement", "")))
        value = (RELATION_COEFFICIENT[kind], _get(relation, "id", ""), status != "approved")
        current = result.get(key)
        if current is None or (-value[0], value[2], value[1]) < (-current[0], current[2], current[1]):
            result[key] = value
    return result


@dataclass(frozen=True)
class _Match:
    candidate_index: int
    requirement_index: int
    option: str
    coefficient: float
    exact: bool
    factor: float
    match_type: str
    relation_id: str | None = None
    provisional: bool = False


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int
    cost: int


def _assign(candidates: list[_Candidate], requirements: list[_Requirement], config: Any) -> dict[int, _Match]:
    """Maximum lexicographic allocation through integer-cost augmenting paths.

    The four objectives are Core exact fulfillment, all Required exact fulfillment,
    Required weighted fulfillment, and Preferred weighted fulfillment. An integer
    base larger than every possible lower objective makes these truly ordered.
    Sorted names, requirement IDs, edges, and strict relaxations resolve ties.
    """
    variant = _get(config, "variant", "full")
    semantic = variant in {"semantic", "proficiency", "full"}
    proficiency = variant in {"proficiency", "full"}
    relations = _relations(config) if semantic else {}
    source, sink = 0, 1 + len(candidates) + len(requirements)
    graph: list[list[_Edge]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, cost: int) -> _Edge:
        forward = _Edge(end, len(graph[end]), 1, cost)
        backward = _Edge(start, len(graph[start]), 0, -cost)
        graph[start].append(forward)
        graph[end].append(backward)
        return forward

    base = 1000 * max(1, min(len(candidates), len(requirements))) + 1
    edges: list[tuple[_Edge, _Match]] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_node = 1 + candidate_index
        add_edge(source, candidate_node, 0)
        factor = FULFILLMENT[candidate.proficiency] if proficiency else 1.0
        for requirement_index, requirement in enumerate(requirements):
            possible = []
            for option in requirement.options:
                if candidate.name == option:
                    original_option = min((original for original in requirement.original_options if skill_name(original) == option), key=normalize)
                    match_type = "exact" if candidate.original_name == original_option else "normalized_exact" if normalize(candidate.original_name) == normalize(original_option) else "alias"
                    possible.append(_Match(candidate_index, requirement_index, option, 1.0, True, factor, match_type))
                elif (candidate.name, option) in relations:
                    coefficient, relation_id, provisional = relations[candidate.name, option]
                    possible.append(_Match(candidate_index, requirement_index, option, coefficient, False, factor, "near" if coefficient == 0.7 else "transfer", relation_id, provisional))
            if not possible:
                continue
            match = min(possible, key=lambda item: (-item.coefficient, item.option, item.relation_id or ""))
            weight = GROUP_WEIGHTS[requirement.importance]
            exact_fulfillment = int(round(1000 * weight * factor)) if match.exact else 0
            fulfillment = int(round(1000 * weight * factor * match.coefficient))
            objectives = (
                exact_fulfillment if requirement.importance == "core" else 0,
                exact_fulfillment if requirement.importance != "preferred" else 0,
                fulfillment if requirement.importance != "preferred" else 0,
                fulfillment if requirement.importance == "preferred" else 0,
            )
            utility = sum(objective * base ** (3 - index) for index, objective in enumerate(objectives))
            edge = add_edge(candidate_node, 1 + len(candidates) + requirement_index, -utility)
            edges.append((edge, match))
    for requirement_index in range(len(requirements)):
        add_edge(1 + len(candidates) + requirement_index, sink, 0)

    while True:
        distances: list[int | None] = [None] * len(graph)
        parents: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for node, adjacency in enumerate(graph):
                if distances[node] is None:
                    continue
                for edge_index, edge in enumerate(adjacency):
                    if not edge.capacity:
                        continue
                    distance = distances[node] + edge.cost
                    if distances[edge.to] is None or distance < distances[edge.to]:
                        distances[edge.to] = distance
                        parents[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if distances[sink] is None or distances[sink] >= 0:
            break
        node = sink
        while node != source:
            parent, edge_index = parents[node]  # type: ignore[misc]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent
    return {match.requirement_index: match for edge, match in edges if not edge.capacity}


def score_skills_v2(resume: Any, job: Any, config: Any) -> dict[str, Any]:
    """Return an auditable, JSON-serializable result for the opt-in v2 ruleset."""
    variant = _get(config, "variant", "full")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown skills v2 variant: {variant}")
    candidates, cv_warnings, cv_merges = _prepare_candidates(resume)
    requirements, jd_warnings, jd_merges = _prepare_requirements(job)
    warnings = cv_warnings + jd_warnings
    if not candidates and not _get(_get(resume, "facts"), "no_skills", False):
        warnings.append("CV has no supported skills and no explicit no-skills evidence.")
    identity = _identity(resume, job, config)
    core_present = any(item.importance == "core" for item in requirements)
    jd_annotation = _get(job, "annotation")
    known_no_core = (
        not core_present and not jd_warnings and jd_annotation is not None
        and _get(jd_annotation, "skills_status") == "complete"
        and _get(jd_annotation, "review_status") != "rejected"
    )
    for side, item in (("CV", resume), ("JD", job)):
        annotation = _get(item, "annotation")
        if annotation is None or _get(annotation, "skills_status") != "complete":
            warnings.append(f"{side} v2 skill annotation is missing or incomplete.")
        elif _get(annotation, "review_status") == "rejected":
            warnings.append(f"{side} v2 annotation is rejected.")
    result: dict[str, Any] = {
        "score": 50,
        "supported": False,
        "basis": "missing_evidence",
        "rule_id": "skills-v2",
        "matched_skills": [],
        "missing_skills": [],
        "preferred_skill_coverage": None,
        "requirements": [],
        "group_fulfillment": {group: None for group in GROUP_WEIGHTS},
        "group_shares": {group: None for group in GROUP_WEIGHTS},
        "core_gate_fulfillment": None,
        "base_skills": 50.0,
        "bonus": 0.0,
        "skills_before_gate": 50.0,
        "skills_after_gate": 50.0,
        "gate_status": "not_applicable" if known_no_core else "unknown",
        "force_mismatch": False,
        "strong_blocked": False,
        "warnings": sorted(set(warnings)),
        "merges": cv_merges + jd_merges,
        "identity": identity,
        "evidence": [],
        "fulfilled_requirements": [],
        "partial_requirements": [],
        "unmet_requirements": [],
        "unknown_requirements": [],
        "explanation": "Skills evidence is incomplete; the neutral score is 50 and Core failure is not inferred.",
    }
    if not requirements and known_no_core:
        result.update(basis="no_requirements", gate_status="not_applicable", explanation="No supported skill requirements are available; the neutral skills score is 50.")
        return result
    if not requirements:
        return result
    if warnings:
        result["requirements"] = [{"id": item.id, "requirement_id": item.id,
            "source_ids": list(item.source_ids), "options": list(item.options), "importance": item.importance,
            "weight": GROUP_WEIGHTS[item.importance], "candidate_skill": None, "selected_option": None,
            "match_type": "unknown", "coefficient": None, "fulfillment_factor": None, "fulfillment": None,
            "exact": False, "status": "unknown", "review_status": item.review_status,
            "evidence": {"requirement_quote": item.quote, "importance_quote": item.importance_quote}}
            for item in requirements]
        result["unknown_requirements"] = [item.id for item in requirements]
        return result

    matches = _assign(candidates, requirements, config)
    proficiency_enabled = variant in {"proficiency", "full"}
    group_numerator = {group: 0.0 for group in GROUP_WEIGHTS}
    group_denominator = {group: 0.0 for group in GROUP_WEIGHTS}
    gate_numerator = 0.0
    bonus_numerator = 0.0
    required_denominator = 0.0
    for index, requirement in enumerate(requirements):
        match = matches.get(index)
        candidate = candidates[match.candidate_index] if match else None
        coefficient = match.coefficient if match else 0.0
        factor = match.factor if match else 1.0
        exact = bool(match and match.exact)
        fulfillment = coefficient * factor
        weight = GROUP_WEIGHTS[requirement.importance]
        bonus_factor = BONUS[candidate.proficiency] if candidate and proficiency_enabled else 0.0
        status = "full" if fulfillment == 1.0 else "partial" if fulfillment > 0 else "unmet"
        row = {
            "id": requirement.id,
            "requirement_id": requirement.id,
            "source_ids": list(requirement.source_ids),
            "options": list(requirement.options),
            "importance": requirement.importance,
            "weight": weight,
            "candidate_skill": candidate.name if candidate else None,
            "selected_option": match.option if match else None,
            "match_type": match.match_type if match else "none",
            "relation_id": match.relation_id if match else None,
            "relation_provisional": match.provisional if match else False,
            "coefficient": coefficient,
            "fulfillment_factor": factor,
            "fulfillment": fulfillment,
            "exact": exact,
            "proficiency": candidate.proficiency if candidate else None,
            "bonus_factor": bonus_factor if exact and requirement.importance != "preferred" else 0.0,
            "status": status,
            "evidence": {"requirement_quote": requirement.quote, "importance_quote": requirement.importance_quote, "skill_quote": candidate.skill_quote if candidate else None, "proficiency_quote": candidate.proficiency_quote if candidate else None, "provenance": candidate.provenance if candidate else None},
            "review_status": requirement.review_status,
        }
        result["requirements"].append(row)
        result[{"full": "fulfilled_requirements", "partial": "partial_requirements", "unmet": "unmet_requirements"}[status]].append(requirement.id)
        label = " or ".join(requirement.options)
        result["matched_skills" if fulfillment else "missing_skills"].append(label)
        group_numerator[requirement.importance] += weight * fulfillment
        group_denominator[requirement.importance] += weight
        if requirement.importance == "core" and exact:
            gate_numerator += weight * factor
        if requirement.importance != "preferred":
            required_denominator += weight
            if exact:
                bonus_numerator += weight * factor * bonus_factor
        if match and match.provisional:
            result["identity"]["provisional"] = True
        result["evidence"].append(requirement.quote)
        if candidate:
            result["evidence"].append(candidate.skill_quote)

    shares = {group: 1.0 if variant == "equal_groups" else GROUP_SHARES[group] for group in GROUP_WEIGHTS}
    present_share = sum(shares[group] for group, denominator in group_denominator.items() if denominator)
    base_score = 0.0
    for group in GROUP_WEIGHTS:
        denominator = group_denominator[group]
        if not denominator:
            continue
        fulfillment = group_numerator[group] / denominator
        share = shares[group] / present_share
        result["group_fulfillment"][group] = fulfillment
        result["group_shares"][group] = share
        base_score += 100 * share * fulfillment
    bonus = min(5.0, 25 * bonus_numerator / required_denominator) if required_denominator and proficiency_enabled else 0.0
    before_gate = min(100.0, base_score + bonus)
    after_gate = before_gate
    gate_fulfillment = gate_numerator / group_denominator["core"] if core_present else None
    gate_status = "not_applicable" if not core_present else "passed"
    blocked = force_mismatch = False
    if core_present and variant == "full":
        # No rounded percentage or bonus participates in the gate decision.
        if gate_fulfillment < 0.7:
            gate_status = "failed"
            after_gate = min(49.0, before_gate)
            blocked = True
            force_mismatch = gate_fulfillment < 0.5
    elif core_present:
        # The ablation explicitly disables Core gating, rather than claiming pass.
        gate_status = "not_applicable"
    result.update(
        score=int(round(after_gate)),
        supported=True,
        basis="annotated_skills_v2",
        preferred_skill_coverage=100 * result["group_fulfillment"]["preferred"] if result["group_fulfillment"]["preferred"] is not None else None,
        core_gate_fulfillment=gate_fulfillment,
        base_skills=base_score,
        bonus=bonus,
        skills_before_gate=before_gate,
        skills_after_gate=after_gate,
        gate_status=gate_status,
        force_mismatch=force_mismatch,
        strong_blocked=blocked,
        evidence=sorted(set(result["evidence"])),
        explanation=f"Skill fulfillment produces {base_score:.2f} base points and {bonus:.2f} proficiency bonus points. Core gate: {gate_status}. Final skills score: {int(round(after_gate))}.",
    )
    return result
