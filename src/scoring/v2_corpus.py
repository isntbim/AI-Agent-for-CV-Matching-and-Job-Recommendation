"""Frozen source-backed ATS skill corpus and transactional human-review exchange.

Preparation is an assistant proposal, never human approval. Literal support and
conservative phrase rules do not replace review of the source interpretation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date
from pathlib import Path
from functools import lru_cache

from src.parser.skill_v2 import SkillRelationTable
from .models import JobInput, ResumeInput
from .normalization import SKILL_ALIASES, normalize, skill_name
from .silver import job_from_record, resume_from_record, title_like_skill

CORPUS_VERSION = "ats-source-corpus-v2"
REVIEW_VERSION = "ats-review-v2"
PREPARER_VERSION = "literal-source-scan-v2"
TABLES = ("cv_proficiency", "jd_requirements", "choices", "relations", "ambiguities", "documents")
NEGATIVE = re.compile(r"\b(?:no experience (?:with|in)|no knowledge of|not familiar with|without (?:knowledge|experience)|do not (?:know|use))\b", re.I)
NOT_REQUIRED = re.compile(r"\b(?:not (?:required|mandatory)|no requirement (?:for|to))\b", re.I)
MANDATORY = re.compile(r"\b(?:must(?:[- ]have| have| possess| be)?|mandatory|essential|required)\b", re.I)
PREFERRED = re.compile(r"\b(?:preferred|nice[- ]to[- ]have|advantage|(?:a (?:strong )?)?plus|bonus|optional|not required|not mandatory)\b", re.I)
PROFICIENCY = ("beginner", "intermediate", "advanced", "expert")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source(root, source):
    text_path = (root / source["text_path"]).resolve()
    source_path = (root / source["source_path"]).resolve()
    for path in (text_path, source_path):
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Source path escapes repository")
    if digest(source_path.read_bytes()) != source["source_sha256"]:
        raise ValueError("Stale original source checksum: " + source["source_path"])
    text = text_path.read_text(encoding="utf-8")
    if digest(text.encode("utf-8")) != source["text_sha256"]:
        raise ValueError("Stale extracted source checksum: " + source["text_path"])
    return text


@lru_cache(maxsize=10000)
def _spellings(name):
    canonical = skill_name(name)
    return sorted({name, canonical, *(alias for alias, value in SKILL_ALIASES.items() if value == canonical)}, key=lambda x: (-len(x), x))


@lru_cache(maxsize=20000)
def _pattern(name):
    return re.compile(r"(?<![\w])" + r"\s+".join(re.escape(w) for w in name.split()) + r"(?![\w])", re.I)


def _support(name, text):
    found = []
    for spelling in _spellings(name):
        # Short expansions such as 'py' can be initials in a source, not skills.
        if len(spelling) < 3 and skill_name(spelling) != spelling and spelling != name:
            continue
        for match in _pattern(spelling).finditer(text):
            start = text.rfind("\n", 0, match.start()) + 1
            end = text.find("\n", match.end())
            end = len(text) if end < 0 else end
            quote = text[start:end].strip()
            if quote and len(quote) <= 1200:
                found.append((start, quote, match.group()))
    return sorted(set(found), key=lambda item: (item[0], len(item[1]), item[2]))


def _proficiency(name, quote):
    levels = []
    for spelling in _spellings(name):
        token = re.escape(spelling).replace(r"\ ", r"\s+")
        for level in PROFICIENCY:
            patterns = (rf"\b{level}\s+(?:level\s+(?:in\s+)?|(?:in|with)\s+)?{token}(?!\w)",
                        rf"(?<!\w){token}\s*(?:[:\-]|\()?\s*{level}\b")
            if any(re.search(pattern, quote, re.I) for pattern in patterns):
                levels.append(level)
    return sorted(set(levels))


def _local_proficiency(name, quote, other_names):
    """Keep only a literal level phrase tied to one skill, never a list level."""
    found = []
    for spelling in _spellings(name):
        token = re.escape(spelling)
        for level in PROFICIENCY:
            patterns = (rf"\b{level}\s+(?:level\s+(?:in\s+)?|(?:in|with)\s+)?{token}(?!\w)",
                        rf"(?<!\w){token}\s*(?:[:\-]|\()?\s*{level}\b")
            for pattern in patterns:
                for match in re.finditer(pattern, quote, re.I):
                    tail = quote[match.end():]
                    is_list = any(re.match(r"\s*(?:,|/|and\b|or\b)\s*" + re.escape(other), tail, re.I)
                                  for other in other_names if skill_name(other) != skill_name(name))
                    if not is_list:
                        found.append((level, match.group()))
    return found


def _requirement_clause(name, quote):
    """Don't let mandatory text before a clause delimiter govern optional skills."""
    clauses = [part.strip() for part in re.split(r";|\s+[—–]\s+|(?<=[.!?])\s+(?=[A-Z])", quote) if part.strip()]
    linked = [clause for clause in clauses if _support(name, clause)]
    return min(linked, key=lambda value: (len(value), value)) if linked else quote


def _issue(code, detail, *, blocking=True, **extra):
    return {"code": code, "detail": detail, "blocking": blocking, **extra}


def _provenance(row):
    return {"source_id": row["id"], **{key: row[key] for key in ("source_path", "source_sha256", "text_sha256")}, "review_status": "pending"}


def _annotation(row, status):
    return {**_provenance(row), "schema_version": "ats-skills-v2", "skills_status": status,
            "envelope_review_status": "pending", "reviewer": None}


def _vocabulary(envelopes):
    names = {}
    for kind, envelope in envelopes:
        doc = envelope["document"]
        candidates = [name for group in doc.get("skills", []) for name in group.get("keywords", [])] if kind == "resume" else [item["name"] for bucket in ("required", "preferred") for item in doc["skills"][bucket]]
        for name in candidates:
            canonical = skill_name(name)
            if len(canonical) >= 3 and len(canonical) <= 80 and not title_like_skill(name, ""):
                names.setdefault(canonical, name)
    return names


def _skill_regions(text, kind):
    """Only scan omissions inside explicit skills/requirements sections."""
    lines = text.splitlines(keepends=True)
    active = False
    regions = []
    for line in lines:
        stripped = line.strip().rstrip(":").casefold()
        if kind == "resume":
            heading = bool(re.fullmatch(r"(?:(?:technical|computer|professional|key|core|additional) )?skills|(?:technical |core )?competencies|technical expertise", stripped))
            stop = bool(re.fullmatch(r"(?:work |professional )?experience|education|employment(?: history)?|projects|certifications|interests|summary|accomplishments", stripped))
        else:
            heading = bool(re.fullmatch(r"requirements|job requirements|your skills and experience|skills and experience|qualifications|what you (?:need|bring)|must have|nice to have", stripped))
            stop = bool(re.fullmatch(r"description|responsibilities|job description|benefits(?:_detail)?|other_info|education|locations|salary|why you.*", stripped))
        if heading:
            active = True
            continue
        if stop:
            active = False
        if active and line.strip():
            regions.append(line.strip())
    return "\n".join(regions)


def _cv_skills(envelope, row, text, vocabulary):
    groups = envelope["document"].get("skills", [])
    old = [name for group in groups for name in group.get("keywords", [])]
    names = {skill_name(name): name for name in old}
    regions = _skill_regions(text, "resume")
    for canonical, name in vocabulary.items():
        if canonical not in names and _support(name, regions):
            names[canonical] = name
    details, issues, corrections = [], [], []
    for canonical, name in sorted(names.items()):
        supports = _support(name, text)
        positive = [quote for _, quote, _ in supports if not NEGATIVE.search(quote)]
        if not positive:
            corrections.append({"field": "skills", "before": name, "after": None, "review_status": "pending", "reason": "No positive literal/approved-alias support located; removal is a source-review proposal, not proof that the candidate lacks this skill"})
            if supports:
                issues.append(_issue("negative_skill_evidence", name, quote=supports[0][1]))
            continue
        levels = {match for quote in positive for match in _local_proficiency(name, quote, names.values())}
        proficiency = "normal"
        raw = proficiency_quote = None
        distinct = {level for level, _ in levels}
        if len(distinct) > 1:
            issues.append(_issue("conflicting_proficiency", name, values=sorted(distinct), quote=positive[0]))
        elif distinct:
            proficiency, proficiency_quote = min(levels, key=lambda item: (len(item[1]), item[1]))
            raw = proficiency
        # Unmapped phrases are retained for review instead of becoming a bonus.
        if not distinct:
            for quote in positive:
                local = next((match for spelling in _spellings(name)
                    if (match := re.search(r"\b(?:proficient|familiar|knowledge|competent|fluent|excellent|strong)\s+(?:(?:in|with|of)\s+)?" + re.escape(spelling) + r"(?!\w)", quote, re.I))), None)
                if local:
                    raw = local.group()
                    issues.append(_issue("unmapped_proficiency", name, quote=quote))
                    break
        quote = min(positive, key=lambda value: (len(value), value))
        details.append({"name": canonical, "proficiency": proficiency, "raw_proficiency": raw,
                        "skill_quote": quote, "proficiency_quote": proficiency_quote, "provenance": _provenance(row)})
    envelope["document"]["skills"] = [{"name": "Source reviewed skills", "level": None,
        "keywords": [item["name"] for item in details], "details": details}] if details else []
    return issues, corrections


def _jd_skills(envelope, row, text, vocabulary):
    skills = envelope["document"]["skills"]
    original = {skill_name(item["name"]): (item["name"], bucket) for bucket in ("preferred", "required") for item in skills[bucket]}
    names = {canonical: name for canonical, (name, _) in original.items()}
    regions = _skill_regions(text, "job")
    for canonical, name in vocabulary.items():
        if canonical not in names and _support(name, regions):
            names[canonical] = name
    entries, issues, corrections = [], [], []
    for canonical, name in sorted(names.items()):
        supports = _support(name, regions) or _support(name, text)
        supports = [(position, _requirement_clause(name, quote), literal) for position, quote, literal in supports]
        supports = [support for support in supports if not NEGATIVE.search(support[1])]
        if not supports:
            corrections.append({"field": "skills", "before": name, "after": None, "review_status": "pending", "reason": "No positive literal/approved-alias requirement support located; proposed removal awaits human source interpretation"})
            continue
        _, quote, literal = min(supports, key=lambda item: (len(item[1]), item[0], item[2]))
        importance = "standard"
        linked = quote
        # Mere 'required' in a section heading does not make its items Core.
        explicit_core = bool(MANDATORY.search(quote)) and not NEGATIVE.search(quote) and not NOT_REQUIRED.search(quote)
        preferred = bool(PREFERRED.search(quote))
        if explicit_core and preferred:
            issues.append(_issue("conflicting_importance", name, quote=quote))
        elif explicit_core:
            importance = "core"
        elif preferred:
            importance = "preferred"
        elif original.get(canonical, (None, None))[1] == "preferred":
            # Old classifier output is not source evidence for Preferred.
            corrections.append({"field": "importance", "before": "preferred", "after": "standard", "name": canonical, "reason": "No explicit optional wording in source quote"})
        entries.append({"options": [canonical], "importance": importance, "quote": quote,
                        "importance_quote": linked if explicit_core or preferred else None,
                        "review_status": "pending", "issues": []})
    # Merge a simple source-local OR into exactly one requirement. Mixed expressions
    # and overlapping choice scopes must be resolved by a reviewer.
    by_quote = defaultdict(list)
    for entry in entries:
        by_quote[entry["quote"]].append(entry)
    result = []
    for quote, same in sorted(by_quote.items()):
        spans = []
        for entry in same:
            positions = _support(entry["options"][0], quote)
            match = next((_pattern(spelling).search(quote) for spelling in _spellings(entry["options"][0]) if _pattern(spelling).search(quote)), None)
            if match:
                spans.append((match.start(), match.end(), entry))
        spans.sort(key=lambda item: (item[0], item[1]))
        has_or = bool(re.search(r"\bor\b|/", quote, re.I))
        if has_or and len(spans) > 1:
            connectors = [quote[a[1]:b[0]].strip(" ,()") for a, b in zip(spans, spans[1:])]
            simple = all(re.fullmatch(r"(?:or|/)", connector, re.I) for connector in connectors)
            if simple and len({entry["importance"] for entry in same}) == 1:
                combined = deepcopy(same[0])
                combined["options"] = sorted({option for entry in same for option in entry["options"]})
                result.append(combined)
                continue
            issues.append(_issue("complex_choice_scope", "OR/slash scope needs source review", quote=quote, options=[entry["options"][0] for entry in same]))
            for entry in same:
                entry["issues"].append("complex_choice_scope")
        result.extend(same)
    merged = {}
    order = {"preferred": 0, "standard": 1, "core": 2}
    for entry in result:
        key = tuple(entry["options"])
        if key in merged:
            corrections.append({"field": "requirements", "before": entry, "after": "merged", "reason": "Duplicate canonical requirement"})
        if key not in merged or order[entry["importance"]] > order[merged[key]["importance"]]:
            merged[key] = entry
    requirements = []
    for options, entry in sorted(merged.items()):
        entry["id"] = row["id"] + ":req:" + digest(list(options))[:12]
        requirements.append(entry)
    skills["requirements"] = requirements
    skills["required"] = [{"name": option, "proficiency": "required"} for entry in requirements if entry["importance"] != "preferred" for option in entry["options"]]
    skills["preferred"] = [{"name": option, "proficiency": "preferred"} for entry in requirements if entry["importance"] == "preferred" for option in entry["options"]]
    return issues, corrections


def refresh_identity(corpus):
    for row in corpus["documents"]:
        row["checksum_sha256"] = digest(row["envelope"])
    corpus["counts"] = {"documents": len(corpus["documents"]), "kind": dict(Counter(row["kind"] for row in corpus["documents"])),
                        "skills_status": dict(Counter(row["skills_status"] for row in corpus["documents"])),
                        "skills_review_status": dict(Counter(row["skills_review_status"] for row in corpus["documents"])),
                        "full_envelope_review_status": dict(Counter(row["full_envelope_review_status"] for row in corpus["documents"]))}
    corpus["identity_sha256"] = digest({key: value for key, value in corpus.items() if key not in {"identity_sha256", "review_history"}})
    return corpus


def prepare_corpus(root, output, as_of, *, expected_counts=(88, 361)):
    root, output = Path(root), Path(output)
    if not isinstance(as_of, date):
        as_of = date.fromisoformat(str(as_of))
    source_manifest = read_json(root / "output/extraction_pilot/source_manifest.json")
    split_manifest = read_json(root / "output/extraction_pilot/split_manifest.json")
    splits = {row["id"]: row for row in split_manifest["documents"]}
    cvs = read_json(root / "data/processed/structured_cvs/all_cvs.json")
    jobs = read_json(root / "data/processed/structured_jobs/all_jobs_en.json")
    if (len(cvs), len(jobs)) != expected_counts:
        raise ValueError("Frozen corpus counts do not match expected CV/JD population")
    records = {row["cv_id"]: ("resume", row) for row in cvs}
    records.update({row["id"]: ("job", row) for row in jobs})
    if set(records) != {row["id"] for row in source_manifest["documents"]} or set(records) != set(splits):
        raise ValueError("Source, split and structured manifests must cover exactly the same IDs")
    before = {}
    for id, (kind, record) in records.items():
        model = resume_from_record(record, as_of) if kind == "resume" else job_from_record(record)[0]
        before[id] = model.model_dump(mode="json")
    vocabulary = _vocabulary((records[id][0], envelope) for id, envelope in before.items())
    decisions_path = root / "output/extraction_pilot/review_decisions.json"
    inherited_decisions = read_json(decisions_path) if decisions_path.exists() else {}
    documents = []
    for source in sorted(source_manifest["documents"], key=lambda row: row["id"]):
        id, kind = source["id"], source["kind"]
        text = _source(root, source)
        envelope = deepcopy(before[id])
        pilot = root / "output/extraction_pilot/targets" / (id + ".json")
        inherited = {}
        if pilot.exists():
            model = ResumeInput if kind == "resume" else JobInput
            envelope = model.model_validate(read_json(pilot)).model_dump(mode="json")
            inherited = {"target_path": pilot.relative_to(root).as_posix(), "target_sha256": digest(pilot.read_bytes()),
                         "decision": inherited_decisions.get(id), "status": "assistant_reviewed_awaiting_human_approval"}
        non_skill = {key: {"before": before[id]["document"].get(key), "after": value} for key, value in envelope["document"].items() if key not in {"skills", "schema_version"} and value != before[id]["document"].get(key)}
        issues, corrections = (_cv_skills if kind == "resume" else _jd_skills)(envelope, source, text, vocabulary)
        envelope["document"]["schema_version"] = "resume-v2" if kind == "resume" else "job-v2"
        quotes = [detail["skill_quote"] for group in envelope["document"].get("skills", []) for detail in group.get("details", [])] if kind == "resume" else [entry["quote"] for entry in envelope["document"]["skills"]["requirements"]]
        envelope["facts"]["evidence"].pop("skills", None)
        if quotes:
            envelope["facts"]["evidence"]["skills"] = [{"quote": quote} for quote in dict.fromkeys(quotes)]
        status = "partial" if any(issue.get("blocking") for issue in issues) else "complete" if quotes else "missing"
        envelope["annotation"] = _annotation(source, status)
        model = ResumeInput if kind == "resume" else JobInput
        envelope = model.model_validate(envelope).model_dump(mode="json")
        documents.append({"id": id, "kind": kind, "domain": source["domain"], "source_domain": source.get("source_domain"),
            "split": splits[id]["split"], "duplicate_group": splits[id]["duplicate_group"], "source": source,
            "before": before[id], "envelope": envelope, "review_status": "pending", "skills_review_status": "pending",
            "full_envelope_review_status": "pending", "skills_status": status, "issues": issues,
            "skill_corrections": corrections, "non_skill_corrections": non_skill, "inherited_review": inherited,
            "coverage_note": "Complete means machine-scan literal/alias support coverage for proposed skills, not full source review, human approval, or Qwen gold. Every source interpretation and proposed removal awaits human review."})
    corpus = refresh_identity({"version": CORPUS_VERSION, "preparer_version": PREPARER_VERSION, "as_of_date": as_of.isoformat(),
        "source_manifest_sha256": digest(source_manifest), "split_manifest_sha256": digest(split_manifest),
        "documents": documents, "relations": {"version": "ats-relations-v2", "relations": []}, "review_history": []})
    atomic_json(output / "corpus.json", corpus)
    atomic_json(output / "relations.json", corpus["relations"])
    return corpus


def validate_corpus(corpus, root, *, verify_files=True):
    root = Path(root)
    errors = []
    expected_identity = digest({key: value for key, value in corpus.items() if key not in {"identity_sha256", "review_history"}})
    if corpus.get("identity_sha256") != expected_identity:
        errors.append({"id": None, "error": "Corpus identity checksum mismatch"})
    ids = [row["id"] for row in corpus["documents"]]
    if len(ids) != len(set(ids)):
        errors.append({"id": None, "error": "Duplicate document ID"})
    for row in corpus["documents"]:
        try:
            if row["checksum_sha256"] != digest(row["envelope"]):
                raise ValueError("Envelope checksum mismatch")
            model = ResumeInput if row["kind"] == "resume" else JobInput
            envelope = model.model_validate(row["envelope"])
            if row["skills_status"] != envelope.annotation.skills_status:
                raise ValueError("Corpus/annotation skills status differs")
            for key, expected in (("source_id", row["id"]), *((key, row["source"][key]) for key in ("source_path", "source_sha256", "text_sha256"))):
                if getattr(envelope.annotation, key) != expected:
                    raise ValueError("Corpus/annotation provenance differs: " + key)
            if row["skills_review_status"] != envelope.annotation.review_status or row["full_envelope_review_status"] != envelope.annotation.envelope_review_status:
                raise ValueError("Corpus/annotation review statuses differ")
            if verify_files:
                text = _source(root, row["source"])
                from .v2_validation import validate_skill_evidence
                found = validate_skill_evidence(envelope, text, strict=False)
                errors.extend({"id": row["id"], "error": issue} for issue in found if issue["severity"] == "error")
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"id": row["id"], "error": str(exc)})
    try:
        SkillRelationTable.model_validate(corpus["relations"])
    except ValueError as exc:
        errors.append({"id": None, "error": str(exc)})
    return {"valid": not errors, "documents": len(ids), "errors": errors, "identity_sha256": corpus["identity_sha256"]}


def review_items(corpus):
    items = []
    for row in corpus["documents"]:
        base = {"document_id": row["id"], "source_path": row["source"]["source_path"], "text_path": row["source"]["text_path"],
                "source_sha256": row["source"]["source_sha256"], "text_sha256": row["source"]["text_sha256"],
                "document_checksum": row["checksum_sha256"], "split": row["split"]}
        if row["kind"] == "resume":
            for gi, group in enumerate(row["envelope"]["document"]["skills"]):
                for si, detail in enumerate(group["details"]):
                    previous = [{"group": old.get("name"), "group_level": old.get("level"), "keyword": keyword}
                                for old in row["before"]["document"].get("skills", []) for keyword in old.get("keywords", [])
                                if skill_name(keyword) == skill_name(detail["name"])]
                    items.append({**base, "table": "cv_proficiency", "item_id": row["id"] + ":skill:" + digest(skill_name(detail["name"]))[:12],
                                  "path": ["document", "skills", gi, "details", si], "before": previous or None, "proposal": detail})
        else:
            for si, requirement in enumerate(row["envelope"]["document"]["skills"]["requirements"]):
                item = {**base, "table": "jd_requirements", "item_id": requirement["id"],
                        "path": ["document", "skills", "requirements", si],
                        "before": {bucket: [old for old in row["before"]["document"]["skills"][bucket] if skill_name(old["name"]) in {skill_name(option) for option in requirement["options"]}]
                                   for bucket in ("required", "preferred")}, "proposal": requirement}
                items.append(item)
        for index, issue in enumerate(row["issues"]):
            items.append({**base, "table": "ambiguities", "item_id": row["id"] + ":issue:" + digest(issue)[:12],
                          "path": ["issues", index], "proposal": issue})
        for index, correction in enumerate(row.get("skill_corrections", [])):
            items.append({**base, "table": "ambiguities", "item_id": row["id"] + ":correction:" + digest(correction)[:12],
                          "path": ["skill_corrections", index], "proposal": correction})
        items.append({**base, "table": "documents", "item_id": row["id"] + ":skills-review", "path": ["annotation"],
                      "proposal": {"skills_status": row["skills_status"], "skills_review_status": row["skills_review_status"]}})
        items.append({**base, "table": "documents", "item_id": row["id"] + ":full-envelope", "path": ["envelope"], "before": row["before"], "proposal": row["envelope"]})
    for index, relation in enumerate(corpus["relations"]["relations"]):
        items.append({"table": "relations", "item_id": relation["id"], "document_id": None,
                      "source_sha256": digest(relation["source"]), "text_sha256": None, "document_checksum": digest(corpus["relations"]),
                      "path": ["relations", index], "proposal": relation})
    for item in items:
        if item["table"] == "cv_proficiency":
            item["proposal_reason"] = "Neutral default; no unambiguous skill-local normalized level is asserted and legacy group levels are excluded." if item["proposal"]["proficiency"] == "normal" else "Literal skill-local proficiency phrase; interpretation awaits human review."
        elif item["table"] == "jd_requirements":
            item["proposal_reason"] = "Default Standard; no unambiguous skill-local mandatory/optional wording confirmed." if item["proposal"]["importance"] == "standard" else "Importance comes from the linked source clause; interpretation awaits human review."
        else:
            item["proposal_reason"] = item["proposal"].get("reason", "Human source review required; no automatic approval.")
        item["proposal_sha256"] = digest(item["proposal"])
        item["item_sha256"] = digest({key: value for key, value in item.items() if key != "item_sha256"})
    return items


def export_review(corpus, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    items = review_items(corpus)
    package = {"version": REVIEW_VERSION, "corpus_identity_sha256": corpus["identity_sha256"], "items": items,
               "instructions": "Blank/pending is never accepted. Finalized decisions require reviewer and source/proposal checksums. Skills-only approval does not approve the full envelope."}
    atomic_json(output / "review_package.json", package)
    decisions = [{key: item.get(key) for key in ("item_id", "document_id", "source_sha256", "text_sha256", "document_checksum", "proposal_sha256", "item_sha256")} | {"decision": "pending", "reviewer": "", "reason": "", "replacement": None} for item in items]
    atomic_json(output / "decisions.json", {"version": REVIEW_VERSION, "corpus_identity_sha256": corpus["identity_sha256"], "decisions": decisions})
    for table in TABLES:
        selected = [item for item in items if item["table"] == table or (table == "choices" and item["table"] == "jd_requirements" and len(item["proposal"]["options"]) > 1)]
        with (output / (table + ".csv")).open("w", encoding="utf-8-sig", newline="") as stream:
            fields = ["item_id", "document_id", "source_path", "text_path", "source_sha256", "text_sha256", "document_checksum", "proposal_sha256", "item_sha256", "before", "proposal", "decision", "reviewer", "reason", "replacement"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for item in selected:
                writer.writerow({**{key: item.get(key, "") for key in fields[:-6]}, "before": json.dumps(item.get("before"), ensure_ascii=False), "proposal": json.dumps(item["proposal"], ensure_ascii=False), "decision": "pending", "reviewer": "", "reason": "", "replacement": ""})
    return {"items": len(items), "tables": {table: sum(item["table"] == table for item in items) for table in TABLES}, "identity_sha256": corpus["identity_sha256"]}


def _set_path(container, path, value):
    target = container
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _sync_skills(row):
    envelope = row["envelope"]
    if row["kind"] == "resume":
        for group in envelope["document"]["skills"]:
            group["keywords"] = [detail["name"] for detail in group["details"]]
        quotes = [detail.get("skill_quote") for group in envelope["document"]["skills"] for detail in group["details"]]
    else:
        skills = envelope["document"]["skills"]
        skills["required"] = [{"name": option, "proficiency": "required"} for entry in skills["requirements"] if entry["importance"] != "preferred" for option in entry["options"]]
        skills["preferred"] = [{"name": option, "proficiency": "preferred"} for entry in skills["requirements"] if entry["importance"] == "preferred" for option in entry["options"]]
        quotes = [entry.get("quote") for entry in skills["requirements"]]
    envelope["facts"]["evidence"].pop("skills", None)
    if any(quotes):
        envelope["facts"]["evidence"]["skills"] = [{"quote": quote} for quote in dict.fromkeys(quotes) if quote]


def _xlsx_decisions(path):
    """Read our review workbook without relying on an Excel application."""
    import zipfile
    from xml.etree import ElementTree as ET
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
    supported = {"Documents", "CV proficiency", "JD requirements", "Relations", "Ambiguities"}
    decisions, identities = [], set()
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(t.text or "" for t in item.findall(".//m:t", ns)) for item in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        rels = {rel.get("Id"): rel.get("Target") for rel in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        book = ET.fromstring(archive.read("xl/workbook.xml"))
        for sheet in book.findall("m:sheets/m:sheet", ns):
            if sheet.get("name") not in supported:
                continue
            target = rels[sheet.get("{" + ns["r"] + "}id")]
            member = target.lstrip("/") if target.startswith("/") else "xl/" + target
            values = {}
            for row in ET.fromstring(archive.read(member)).findall("m:sheetData/m:row", ns):
                number = int(row.get("r"))
                cells = {}
                for cell in row.findall("m:c", ns):
                    column = re.sub(r"\d+", "", cell.get("r", ""))
                    value = cell.find("m:v", ns)
                    literal = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                    if value is not None:
                        literal = strings[int(value.text)] if cell.get("t") == "s" else value.text or ""
                    cells[column] = literal
                values[number] = cells
            if values.get(2, {}).get("A") != "corpus_identity_sha256" or not values.get(2, {}).get("B"):
                raise ValueError("Review workbook is missing corpus identity: " + sheet.get("name"))
            identities.add(values[2]["B"])
            headers = values.get(5, {})
            if "item_id" not in headers.values():
                raise ValueError("Review workbook is missing item_id header")
            for number, cells in sorted(values.items()):
                if number >= 6:
                    decision = {header: cells.get(column, "") for column, header in headers.items()}
                    if decision.get("item_id"):
                        decisions.append(decision)
    if len(identities) != 1:
        raise ValueError("Review workbook sheets have inconsistent corpus identity")
    return {"corpus_identity_sha256": identities.pop(), "decisions": decisions}


def import_review(corpus_path, decisions_path, root):
    corpus_path = Path(corpus_path)
    corpus = read_json(corpus_path)
    if Path(decisions_path).suffix.lower() == ".xlsx":
        incoming = _xlsx_decisions(decisions_path)
    elif Path(decisions_path).suffix.lower() == ".csv":
        with Path(decisions_path).open(encoding="utf-8-sig", newline="") as stream:
            incoming = {"decisions": list(csv.DictReader(stream))}
    else:
        incoming = read_json(decisions_path)
    decisions = incoming.get("decisions", [])
    transaction = digest(decisions)
    if transaction in {entry["transaction_sha256"] for entry in corpus.get("review_history", [])}:
        return {"applied": 0, "idempotent": True, "identity_sha256": corpus["identity_sha256"]}
    if incoming.get("corpus_identity_sha256", corpus["identity_sha256"]) != corpus["identity_sha256"]:
        raise ValueError("Stale corpus identity")
    indexed = {item["item_id"]: item for item in review_items(corpus)}
    rows = {row["id"]: row for row in corpus["documents"]}
    active = []
    seen = set()
    for decision in decisions:
        action = (decision.get("decision") or "pending").strip().lower()
        if action == "pending":
            continue
        if action not in {"accept", "edit", "reject"}:
            raise ValueError("Unknown review decision: " + action)
        item_id = decision["item_id"]
        if item_id in seen:
            raise ValueError("Duplicate decision item: " + item_id)
        seen.add(item_id)
        if item_id not in indexed:
            raise ValueError("Unknown review item: " + item_id)
        item = indexed[item_id]
        for key in ("source_sha256", "text_sha256", "document_checksum", "proposal_sha256", "item_sha256"):
            if decision.get(key) != item.get(key) and not (decision.get(key) == "" and item.get(key) is None):
                raise ValueError("Stale " + key + ": " + item_id)
        if not str(decision.get("reviewer") or "").strip():
            raise ValueError("Finalized decisions require a reviewer: " + item_id)
        replacement = decision.get("replacement")
        if isinstance(replacement, str) and replacement.strip():
            replacement = json.loads(replacement)
        if action == "edit" and not isinstance(replacement, dict):
            raise ValueError("Edit requires a complete replacement object: " + item_id)
        active.append((item, decision, action, replacement))
    # Validate every decision before changing the in-memory copy or writing.
    proposed = deepcopy(corpus)
    changed = {row["id"]: row for row in proposed["documents"]}
    removals = []
    for item, decision, action, replacement in active:
        table = item["table"]
        value = deepcopy(replacement if action == "edit" else item["proposal"])
        if table == "relations":
            value["review_status"] = "rejected" if action == "reject" else "approved"
            _set_path(proposed["relations"], item["path"], value)
            continue
        row = changed[item["document_id"]]
        if item["item_id"].endswith(":full-envelope"):
            if action == "edit":
                row["envelope"] = value
            row["full_envelope_review_status"] = "rejected" if action == "reject" else "approved"
            row["envelope"]["annotation"]["envelope_review_status"] = row["full_envelope_review_status"]
            if action != "reject":
                row["skills_review_status"] = row["review_status"] = "approved"
                row["envelope"]["annotation"]["review_status"] = "approved"
                row["envelope"]["annotation"]["reviewer"] = decision["reviewer"]
        elif item["item_id"].endswith(":skills-review"):
            if action == "edit":
                if value.get("skills_status") not in {"complete", "partial", "missing"}:
                    raise ValueError("Invalid skills status replacement")
                row["skills_status"] = row["envelope"]["annotation"]["skills_status"] = value["skills_status"]
            row["skills_review_status"] = row["review_status"] = "rejected" if action == "reject" else "approved"
            row["envelope"]["annotation"]["review_status"] = row["skills_review_status"]
            row["envelope"]["annotation"]["reviewer"] = decision["reviewer"]
        elif table == "ambiguities":
            value["review_status"] = "rejected" if action == "reject" else "approved"
            value["reviewer"] = decision["reviewer"]
            if action in {"accept", "edit"}:
                if not decision.get("reason"):
                    raise ValueError("Resolving an ambiguity requires a source interpretation reason")
                value["blocking"] = False
                value["resolution"] = decision["reason"]
            _set_path(row, item["path"], value)
        elif action == "reject":
            removals.append((row, item["path"]))
        else:
            if table == "jd_requirements":
                value["review_status"] = "approved"
            else:
                value["provenance"] = value.get("provenance") or _provenance(row["source"])
                value["provenance"]["review_status"] = "approved"
            _set_path(row["envelope"], item["path"], value)
    for row, path in sorted(removals, key=lambda item: tuple((1, key) if isinstance(key, int) else (0, key) for key in item[1]), reverse=True):
        target = row["envelope"]
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    for id in {item["document_id"] for item, *_ in active if item["document_id"]}:
        row = changed[id]
        _sync_skills(row)
        if row["skills_review_status"] == "approved" and any(issue.get("blocking") for issue in row["issues"]):
            raise ValueError("Cannot approve skills with unresolved blocking ambiguities: " + id)
        if row["full_envelope_review_status"] == "approved" and row["skills_status"] != "complete":
            raise ValueError("Full-envelope approval requires complete skill review: " + id)
    refresh_identity(proposed)
    validation = validate_corpus(proposed, root)
    if not validation["valid"]:
        raise ValueError("Review transaction failed validation: " + json.dumps(validation["errors"][:20]))
    if active:
        proposed["review_history"].append({"transaction_sha256": transaction, "items": [item["item_id"] for item, *_ in active],
                                          "reviewers": sorted({decision["reviewer"] for _, decision, *_ in active})})
        atomic_json(corpus_path, proposed)
        atomic_json(corpus_path.parent / "relations.json", proposed["relations"])
    return {"applied": len(active), "idempotent": False, "identity_sha256": proposed["identity_sha256"]}


def export_training(corpus, output):
    examples = [{"id": row["id"], "kind": row["kind"], "source": row["source"], "envelope": row["envelope"],
                 "checksum_sha256": row["checksum_sha256"]} for row in corpus["documents"]
                if row["split"] == "train" and row["full_envelope_review_status"] == "approved" and row["skills_review_status"] == "approved" and row["skills_status"] == "complete"]
    atomic_json(output, {"version": "ats-training-export-v2", "corpus_identity_sha256": corpus["identity_sha256"], "examples": examples,
                         "excluded": len(corpus["documents"]) - len(examples), "policy": "Human-approved complete envelopes from train partition only; skills-only review is insufficient."})
    return {"exported": len(examples)}


def propose_relations(corpus, *, model_name, revision, device="cpu", top_k=5, embedder=None):
    """Embedding candidates are pending directed review suggestions, never rules."""
    if not revision:
        raise ValueError("Pinned embedding revision is required")
    if not 1 <= top_k <= 5:
        raise ValueError("top_k must be between one and five")
    vocabulary = sorted(_vocabulary((row["kind"], row["envelope"]) for row in corpus["documents"] if row["split"] == "train"))
    if embedder is None:
        from src.embeddings.embedder import BGEEmbedder
        embedder = BGEEmbedder(model_name=model_name, revision=revision, device=device)
    vectors = embedder.embed_batch(vocabulary)
    pairs = []
    for index, candidate in enumerate(vocabulary):
        scores = vectors @ vectors[index]
        neighbors = sorted(((-float(scores[j]), vocabulary[j]) for j in range(len(vocabulary)) if j != index), key=lambda item: (item[0], item[1]))[:top_k]
        for negative_score, requirement in neighbors:
            pairs.append({"id": "rel:" + digest([candidate, requirement])[:16], "candidate": candidate, "requirement": requirement,
                "relation": "near", "reason": "Embedding similarity proposes review; relationship class and applicability require human judgement.",
                "source": f"training vocabulary only; {model_name}@{revision}; cosine={-negative_score:.6f}", "review_status": "pending"})
    return {"version": "ats-relations-v2", "relations": pairs, "proposal_metadata": {"corpus_identity_sha256": corpus["identity_sha256"],
            "model": model_name, "revision": revision, "vocabulary_partition": "train", "vocabulary_sha256": digest(vocabulary), "top_k": top_k, "approved": 0}}
