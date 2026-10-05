"""Reproducible extraction-training sources, duplicate groups and pilot selection.

No model calls. Source quotation validation is separate from assistant review.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re

from .extraction import text_hash, validate_evidence, write_json, _prompts
from .models import ResumeInput, JobInput, Facts
from .silver import degree_levels, INCOMPLETE

PILOT_VERSION = "source-pilot-v2"
DOMAINS = {"accountant": "ACCOUNTANT", "data_scientist_analyst": "Data Scientist Analyst",
           "frontend_web_developer": "FrontendWebDeveloper", "hr": "HR",
           "marketing_executive": "Marketing Executive", "software_engineer": "Software Engineer"}
REVIEW_STATUS = "assistant_reviewed_awaiting_human_approval"


def sha(data: bytes):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def shingles(text):
    words = " ".join(text.split()).casefold().split()
    return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def similarity(a, b):
    return len(a & b) / len(a | b) if a or b else 0.0


def original_job_rows(root):
    """Read original XLSX cells, preserving missing column positions and row links."""
    import zipfile
    from xml.etree import ElementTree as ET
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    found = defaultdict(list)
    for path in sorted((root / "data/raw/jobs").glob("*.xlsx")):
        with zipfile.ZipFile(path) as archive:
            strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = ["".join(t.text or "" for t in si.findall(".//m:t", ns))
                           for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", ns)]
            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            headers = {}
            for row in sheet.findall(".//m:row", ns):
                cells = {}
                for cell in row.findall("m:c", ns):
                    column = re.sub(r"\d+", "", cell.get("r", ""))
                    v = cell.find("m:v", ns)
                    value = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                    if v is not None:
                        value = strings[int(v.text)] if cell.get("t") == "s" else v.text or ""
                    cells[column] = value
                if not headers:
                    headers = cells
                    continue
                values = {headers[c]: v for c, v in cells.items() if c in headers}
                id = values.get("job_id", "").casefold()
                if id:
                    found[id].append({"path": path, "row": int(row.get("r")), "cells": values})
    return found


def job_source(record, root, originals):
    id = record["id"]
    exported = root / "data/raw/jobs" / (id + ".txt")
    if not exported.is_file():
        raise ValueError(f"Missing raw JD export: {id}")
    match = re.search(r"\.([a-z0-9]+)\.html", record["url"], re.I)
    if "careerviet" in record["url"]:
        candidates = originals.get(match[1].casefold() if match else "", [])
        if not candidates:
            raise ValueError(f"Missing original workbook row: {id}")
        # All observed duplicates are multiple category captures of the same posting.
        fields = ("job_title", "company_name", "salary", "salary_other", "locations", "workplace_detail", "experience", "level", "employment_type", "description", "requirements", "other_info", "education", "benefits", "benefits_detail")
        def identity(c):
            return tuple(" ".join(c["cells"].get(f, "").split()) for f in fields)
        if len({identity(c) for c in candidates}) > 1:
            raise ValueError(f"Ambiguous original workbook rows: {id}")
        selected = candidates[0]
        text = "\n\n".join(f"{f}:\n{selected['cells'][f]}" for f in fields if selected["cells"].get(f))
        return selected["path"], text, {"extraction_method": "original-xlsx-cell-text-v2", "source_row": selected["row"],
                                      "source_fields": [f for f in fields if selected["cells"].get(f)],
                                      "original_job_url": record["url"], "export_path": exported.relative_to(root).as_posix(),
                                      "export_sha256": sha(exported.read_bytes())}
    # ITviec originals survive inside the saved export; omit synthesized metadata defaults.
    saved = exported.read_text(encoding="utf-8")
    body = saved.split("--- JOB DESCRIPTION & REQUIREMENTS ---", 1)
    if len(body) != 2:
        raise ValueError(f"Missing original description body: {id}")
    title = next((line for line in saved.splitlines() if line.startswith("Title: ")), "")
    company = next((line for line in saved.splitlines() if line.startswith("Company: ")), "")
    text = title + "\n" + company + "\n" + body[1].lstrip("\n")
    return exported, text, {"extraction_method": "stored-itviec-description-v1", "original_job_url": record["url"],
                            "provenance_note": "Saved description snapshot; regenerated location/category/seniority/compensation headers are excluded from evidence."}


def build_sources(root: Path, output: Path):
    from src.parser.extractors import extract_text_from_file
    from src.scraper.import_careerviet import detect_language
    cvs = read_json(root / "data/processed/structured_cvs/all_cvs.json")
    jobs = read_json(root / "data/processed/structured_jobs/all_jobs_en.json")
    originals = original_job_rows(root)
    if len(cvs) != 88 or len(jobs) != 361:
        raise ValueError("Expected the frozen 88-CV/361-JD primary corpus")
    rows, errors = [], []
    for kind, records in (("resume", cvs), ("job", jobs)):
        for record in records:
            id = record["cv_id"] if kind == "resume" else record["id"]
            source_domain = record["domain"] if kind == "resume" else record["category"]
            domain = {"security_engineer": "software_engineer", "devops_cloud": "software_engineer"}.get(source_domain, source_domain)
            if domain not in DOMAINS:
                errors.append({"id": id, "error": "Unknown domain", "domain": domain})
                continue
            source_info = {}
            source_text = None
            if kind == "resume":
                source = root / "data/raw/cvs" / DOMAINS[domain] / Path(record["original_file"]).with_suffix(".pdf").name
            else:
                try:
                    source, source_text, source_info = job_source(record, root, originals)
                except ValueError as error:
                    errors.append({"id": id, "error": str(error)})
                    continue
            if not source.is_file():
                errors.append({"id": id, "error": "Missing domain-qualified source", "source": str(source)})
                continue
            source_digest = sha(source.read_bytes())
            extracted = output / "sources" / (id + ".txt")
            cache_meta = output / "sources" / (id + ".meta.json")
            method = "src.parser.extractors.extract_text_from_file" if kind == "resume" else source_info["extraction_method"]
            if (cache_meta.exists() and extracted.exists() and
                    read_json(cache_meta).get("source_sha256") == source_digest and
                    read_json(cache_meta).get("extraction_method") == method):
                text = extracted.read_text(encoding="utf-8")
            else:
                text = extract_text_from_file(source) if kind == "resume" else source_text
                extracted.parent.mkdir(parents=True, exist_ok=True)
                extracted.write_text(text, encoding="utf-8")
                write_json(cache_meta, {"source_sha256": source_digest, "extraction_method": method})
            if not text.strip():
                errors.append({"id": id, "error": "Empty extracted source"})
                continue
            rows.append({"id": id, "kind": kind, "domain": domain, "source_domain": source_domain, **source_info,
                         "source_path": source.relative_to(root).as_posix(),
                         "text_path": extracted.relative_to(root).as_posix(),
                         "source_sha256": source_digest, "text_sha256": sha(text.encode("utf-8")),
                         "normalized_sha256": text_hash(text), "extraction_method": method,
                     "language": detect_language(text),
                     "language_method": "src.scraper.import_careerviet.detect_language; heuristic-v1",
                     "upstream_language": "en" if kind == "resume" else (record.get("metadata") or {}).get("language", "en"),
                         "characters": len(text), "words": len(text.split())})
    write_json(output / "source_errors.json", errors)
    if errors:
        raise ValueError(f"{len(errors)} unresolved sources; see source_errors.json")
    rows.sort(key=lambda r: r["id"])
    identities = [r["id"] for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Duplicate stable document ID")
    texts = {r["id"]: (root / r["text_path"]).read_text(encoding="utf-8") for r in rows}
    sets = {id: shingles(t) for id, t in texts.items()}
    near = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["kind"] != b["kind"] or a["normalized_sha256"] == b["normalized_sha256"]:
                continue
            aa, bb = sets[a["id"]], sets[b["id"]]
            if not aa or min(len(aa), len(bb)) / max(len(aa), len(bb)) < .9:
                continue
            value = similarity(aa, bb)
            if value >= .9:
                near.append({"a": a["id"], "b": b["id"], "jaccard": value, "decision": "pending",
                             "review_status": "awaiting_assistant_review", "reason": ""})
    filename_groups = defaultdict(list)
    for record in cvs:
        filename_groups[record["original_file"]].append(record["cv_id"])
    # Include collisions with PDFs from other domains, even if they are not gold records.
    collisions = []
    for record in cvs:
        filename = Path(record["original_file"]).with_suffix(".pdf").name
        matches = sorted((root / "data/raw/cvs").glob("*/" + filename))
        if len(matches) > 1:
            checksums = [sha(p.read_bytes()) for p in matches]
            collisions.append({"id": record["cv_id"], "original_file": record["original_file"],
                               "domain": record["domain"], "candidate_paths": [p.relative_to(root).as_posix() for p in matches],
                               "candidate_sha256": checksums, "different_content": len(set(checksums)) > 1})
    manifest = {"version": PILOT_VERSION, "count": len(rows), "documents": rows,
                "filename_collisions": collisions, "near_duplicate_threshold": .9,
                "identity_sha256": sha(json.dumps(rows, sort_keys=True).encode())}
    write_json(output / "source_manifest.json", manifest)
    write_json(output / "near_duplicate_nominations.json", near)
    return manifest


def freeze_splits(root: Path, output: Path, near_review: Path, public_dir: Path, seed=42):
    manifest = read_json(output / "source_manifest.json")
    rows = manifest["documents"]
    parents = {r["id"]: r["id"] for r in rows}
    def find(id):
        while parents[id] != id:
            parents[id] = parents[parents[id]]
            id = parents[id]
        return id
    def union(a, b):
        aa, bb = sorted([find(a), find(b)])
        parents[bb] = aa
    exact = {}
    for row in rows:
        key = (row["kind"], row["normalized_sha256"])
        if key in exact:
            union(row["id"], exact[key])
        exact[key] = row["id"]
    reviewed = read_json(near_review)
    nominations = read_json(output / "near_duplicate_nominations.json")
    if {(r["a"], r["b"]) for r in reviewed} != {(r["a"], r["b"]) for r in nominations}:
        raise ValueError("Near-duplicate review is incomplete or stale")
    for row in reviewed:
        if row["decision"] != "keep_together" or row["review_status"] != REVIEW_STATUS or not row["reason"]:
            raise ValueError("Near duplicates must be source-reviewed before freezing")
        union(row["a"], row["b"])
    # Preserve the public benchmark assignments; use original texts only to audit overlap.
    public_manifest = read_json(public_dir / "manifest.json")
    public_pairs = read_json(public_dir / "pairs.json")
    public_texts = {"resume": {}, "job": {}}
    for pair in public_pairs:
        for kind, field in (("resume", "resume_text"), ("job", "job_description_text")):
            text = pair[field]
            public_texts[kind][text_hash(text)] = text
    public_sets = {kind: {hash: shingles(t) for hash, t in texts.items()} for kind, texts in public_texts.items()}
    overlaps = []
    for row in rows:
        hash = row["normalized_sha256"]
        if hash in public_texts[row["kind"]]:
            overlaps.append({"id": row["id"], "public_hash": hash, "reason": "exact_overlap"})
            continue
        s = shingles((root / row["text_path"]).read_text(encoding="utf-8"))
        for public_hash, other in public_sets[row["kind"]].items():
            if not s or not other or min(len(s), len(other)) / max(len(s), len(other)) < .9:
                continue
            value = similarity(s, other)
            if value >= .9:
                overlaps.append({"id": row["id"], "public_hash": public_hash, "reason": "near_overlap", "jaccard": value})
    excluded_groups = {find(r["id"]) for r in overlaps}
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row)
    strata = Counter((r["kind"], r["domain"]) for r in rows if find(r["id"]) not in excluded_groups)
    ratios = {"train": .7, "validation": .15, "test": .15}
    totals = {s: Counter() for s in ratios}
    rng = random.Random(seed)
    order = sorted(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))
    assignments = {}
    for group in order:
        if group in excluded_groups:
            assignments[group] = "excluded_public_overlap"
            continue
        counts = Counter((r["kind"], r["domain"]) for r in groups[group])
        # Minimize the change in squared stratified document-count error.
        costs = {}
        for split, ratio in ratios.items():
            costs[split] = sum((totals[split][st] + n - strata[st] * ratio) ** 2 -
                               (totals[split][st] - strata[st] * ratio) ** 2 for st, n in counts.items())
        split = min(ratios, key=lambda s: (costs[s], list(ratios).index(s)))
        assignments[group] = split
        totals[split].update(counts)
    documents = [{**r, "duplicate_group": find(r["id"]), "split": assignments[find(r["id"]) ]} for r in rows]
    report = {"version": PILOT_VERSION, "seed": seed, "ratios": ratios, "documents": documents,
              "source_manifest_sha256": sha((output / "source_manifest.json").read_bytes()),
              "near_review_sha256": sha(near_review.read_bytes()),
              "public_manifest": public_manifest, "public_pairs_sha256": sha((public_dir / "pairs.json").read_bytes()),
              "public_overlaps": overlaps, "duplicate_group_leakage": False,
              "counts": {s: {f"{k}:{d}": n for (k, d), n in sorted(c.items())} for s, c in totals.items()},
              "purpose": "Extraction-training holdouts. Previously inspected silver pairs are diagnostic data, not independent test data."}
    write_json(output / "split_manifest.json", report)
    return report


def select_targets(root: Path, output: Path):
    split = read_json(output / "split_manifest.json")
    pools = defaultdict(list)
    for row in split["documents"]:
        if row["split"] == "train":
            pools[(row["kind"], row["domain"])].append(row)
    chosen = []
    for (kind, domain), rows in sorted(pools.items()):
        rows.sort(key=lambda r: (r["characters"], r["id"]))
        median_length = (rows[(len(rows) - 1) // 2]["characters"] + rows[len(rows) // 2]["characters"]) / 2
        median = min(rows, key=lambda r: (abs(r["characters"] - median_length), r["id"]))
        upper = rows[max(0, int(.9 * (len(rows) - 1))):]
        long = min([r for r in upper if r["id"] != median["id"]], key=lambda r: r["id"])
        chosen.extend([{**median, "selection": "near_median"}, {**long, "selection": "longer_p90_plus"}])
    if len(chosen) != 24:
        raise ValueError("Pilot requires two CVs and two JDs in each of six domains")
    cvs = {r["cv_id"]: r for r in read_json(root / "data/processed/structured_cvs/all_cvs.json")}
    jobs = {r["id"]: r for r in read_json(root / "data/processed/structured_jobs/all_jobs_en.json")}
    for row in chosen:
        record = deepcopy((cvs if row["kind"] == "resume" else jobs)[row["id"]])
        model = ResumeInput if row["kind"] == "resume" else JobInput
        envelope = model(document=record, facts=Facts()).model_dump(by_alias=True, mode="json")
        write_json(output / "drafts" / (row["id"] + ".json"), envelope)
    write_json(output / "selection.json", chosen)
    return chosen


def review_proposals(root: Path, output: Path):
    """Conservative field support suggestions. Assistant approval is a separate edit."""
    from pydantic import BaseModel
    from typing import get_args
    from .models import Evidence
    months = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
              "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
    for row in read_json(output / "selection.json"):
        text = (root / row["text_path"]).read_text(encoding="utf-8")
        model = ResumeInput if row["kind"] == "resume" else JobInput
        original = read_json(output / "drafts" / (row["id"] + ".json"))
        envelope = model.model_validate(original)
        ledger, supported = [], {}
        def quote(value):
            if not isinstance(value, str) or not value.strip():
                return None
            tokens = value.split()
            pattern = r"(?<!\w)" + r"\s*".join(re.escape(t) for t in tokens) + r"(?!\w)"
            m = re.search(pattern, text, re.I)
            if m:
                return m.group()
            if re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", value):
                year, month = int(value[:4]), int(value[5:7])
                names = [m for m, n in months.items() if n == month]
                for name in names:
                    m = re.search(r"\b(?:" + name + "|" + name[:3] + r"\.?)\s*" + str(year) + r"\b", text, re.I)
                    if m:
                        return m.group()
            return None
        def record(path, before, after, status, evidence=None):
            ledger.append({"path": path, "before": before, "after": after, "status": status, "quote": evidence})
            if evidence:
                supported[path] = evidence
        def walk(obj, prefix):
            for name, field in type(obj).model_fields.items():
                value = getattr(obj, name)
                path = prefix + "." + name
                if isinstance(value, BaseModel):
                    walk(value, path)
                    continue
                if isinstance(value, list):
                    kept = []
                    for i, item in enumerate(value):
                        if isinstance(item, BaseModel):
                            walk(item, f"{path}[{len(kept)}]")
                            kept.append(item)
                        else:
                            q = quote(item)
                            record(f"{path}[{i}]", item, item if q else None, "literal_source_support" if q else "unsupported_removed", q)
                            if q:
                                kept.append(item)
                    setattr(obj, name, kept)
                    continue
                if value is None:
                    continue
                if path in {"document.id", "document.source", "document.url", "document.scraped_at", "document.category", "document.title_normalized", "document.metadata.language"}:
                    record(path, value, value, "transport_or_vocabulary_metadata_not_fact")
                    continue
                if isinstance(value, bool):
                    record(path, value, value, "schema_boolean_placeholder_not_evidence")
                    continue
                q = quote(value) if isinstance(value, str) else None
                if name in {"min_years", "max_years"}:
                    m = re.search(r"(?i)(?:(?:at least|minimum|min|over|from)\s*)?\b" + str(value) + r"\s*(?:[-–]\s*\d+\s*)?\+?\s*(?:years?|năm)(?:\s+(?:of\s+)?(?:experience|kinh nghiệm))?", text)
                    q = m.group() if m else None
                if q:
                    record(path, value, value, "literal_source_support" if q.casefold() == str(value).casefold() else "source_normalization", q)
                else:
                    if type(None) in get_args(field.annotation):
                        after = None
                    elif field.is_required():
                        after = ""
                    elif isinstance(value, str):
                        after = field.default if field.default not in (None, "Vietnam", "VN", "VND") else ""
                    else:
                        after = field.default
                    setattr(obj, name, after)
                    record(path, value, after, "unsupported_cleared" if not after else "schema_placeholder_not_source_fact")
        walk(envelope.document, "document")
        facts = Facts()
        for dimension, prefixes in {"skills": ("document.skills",), "title": ("document.basics.label", "document.work", "document.title"),
                                    "experience": ("document.work", "document.seniority.min_years", "document.seniority.max_years"),
                                    "education": ("document.education",), "location": ("document.basics.location", "document.location"),
                                    "salary": ("document.compensation",)}.items():
            quotes = []
            for path, q in supported.items():
                if any(path.startswith(p) for p in prefixes):
                    if dimension == "title" and "document.work" in path and not path.endswith("position"):
                        continue
                    quotes.append(q)
            quotes = list(dict.fromkeys(quotes))
            if quotes:
                facts.evidence[dimension] = [Evidence(quote=q) for q in quotes]
        if row["kind"] == "job":
            document = envelope.document
            # Country codes normalize only explicit countries, never a city or a schema default.
            explicit_country = quote(document.location.country)
            if explicit_country:
                countries = {"vietnam": "VN", "viet nam": "VN", "united states": "US", "india": "IN"}
                facts.country = countries.get(document.location.country.casefold())
                if facts.country:
                    document.location.country_code = facts.country
            if document.compensation.currency and quote(document.compensation.currency):
                facts.currency = document.compensation.currency
            for name, pattern, dimension in [("no_experience_required", r"no (?:prior |previous |work )?experience (?:is )?required", "experience"),
                                              ("no_education_required", r"no (?:degree|education) (?:is )?required", "education"),
                                              ("no_skills", r"no skills (?:are )?required", "skills")]:
                m = re.search(pattern, text, re.I)
                if m:
                    setattr(facts, name, True)
                    facts.evidence.setdefault(dimension, []).append(Evidence(quote=m.group()))
            # Explicit degree clauses; optional/preferred clauses remain unknown until reviewed.
            for sentence in text.splitlines():
                levels = degree_levels(sentence)
                if levels and re.search(r"(?i)degree|bachelor|master|doctorate|ph\.?d", sentence) and not re.search(r"(?i)preferred|advantage|plus|optional", sentence):
                    order = {"high_school": 0, "associate": 1, "bachelor": 2, "master": 3, "doctorate": 4}
                    level = min(levels, key=order.get)
                    if facts.minimum_degree is None or order[level] < order[facts.minimum_degree]:
                        facts.minimum_degree = level
                    facts.evidence.setdefault("education", []).append(Evidence(quote=sentence))
        envelope.facts = facts
        validate_evidence(facts, text)
        target_path = output / "targets" / (row["id"] + ".json")
        write_json(target_path, envelope.model_dump(by_alias=True, mode="json"))
        review = {"document_id": row["id"], "review_status": "awaiting_assistant_review", "target_sha256": sha(target_path.read_bytes()),
                  "source_path": row["source_path"], "source_row": row.get("source_row"), "text_path": row["text_path"],
                  "field_review": ledger, "notes": ["Schema-required names/booleans and transport fields can be placeholders; no evidence-backed claim is inferred from their defaults.",
                                                    "Literal quotation checks do not establish interpretation. Review date associations, degrees, proficiency and required/preferred buckets."],
                  "ambiguities": [r for r in ledger if r["status"] not in {"literal_source_support", "source_normalization"}]}
        write_json(output / "reviews" / (row["id"] + ".json"), review)
        print(json.dumps({"id": row["id"], "populated_fields_reviewed": len(ledger), "cleared_fields": sum(r["before"] != r["after"] for r in ledger)}))


def apply_review(root: Path, output: Path):
    """Apply the assistant's source-reviewed decisions, with a complete final field audit."""
    from pydantic import BaseModel
    from .models import Evidence
    from src.parser.job_schemas import SkillItem
    decisions = read_json(output / "review_decisions.json")
    selections = read_json(output / "selection.json")
    if set(decisions) != {r["id"] for r in selections}:
        raise ValueError("Source review must cover exactly the selected pilot")
    for row in selections:
        id = row["id"]
        decision = decisions[id]
        text = (root / row["text_path"]).read_text(encoding="utf-8")
        def literal(value):
            if not value:
                return None
            m = re.search(r"(?<!\w)" + r"\s*".join(re.escape(w) for w in str(value).split()) + r"(?!\w)", text, re.I)
            return m.group() if m else None
        def require(value):
            q = literal(value)
            if not q:
                raise ValueError(f"{id}: reviewed value lacks quote: {value!r}")
            return q
        model = ResumeInput if row["kind"] == "resume" else JobInput
        target_path = output / "targets" / (id + ".json")
        envelope = model.model_validate(read_json(target_path))
        old = read_json(output / "reviews" / (id + ".json"))
        doc = envelope.document
        facts = Facts()
        if row["kind"] == "job":
            doc.skills.required = [SkillItem(name=name, proficiency="required") for name in decision["required"]]
            doc.skills.preferred = [SkillItem(name=name, proficiency="preferred") for name in decision["preferred"]]
            names = decision["required"] + decision["preferred"]
            if names:
                facts.evidence["skills"] = [Evidence(quote=require(n)) for n in names]
            facts.evidence["title"] = [Evidence(quote=require(doc.title))]
            doc.seniority.min_years, doc.seniority.max_years = decision["min_years"], decision["max_years"]
            if decision["min_years"] is not None:
                n = str(decision["min_years"])
                m = re.search(r"(?i)(?<!\d)" + n + r"\s*(?:[^\d\n]{0,6}\d+\s*)?\+?\s*(?:years?|năm)", text)
                if not m:
                    raise ValueError(f"{id}: experience quote missing")
                facts.evidence["experience"] = [Evidence(quote=m.group())]
            facts.minimum_degree = decision["minimum_degree"]
            if facts.minimum_degree:
                m = re.search(r"(?i)\bbachelor(?:[’']s)?(?:\s+degree)?", text)
                if not m:
                    raise ValueError(f"{id}: degree quote missing")
                facts.evidence["education"] = [Evidence(quote=m.group())]
            doc.location.city = decision["city"]
            doc.location.region = None
            doc.location.country, doc.location.country_code = "", ""
            doc.location.remote_policy = decision.get("remote_policy")
            doc.location.remote_policy_raw = None
            if doc.location.city:
                facts.evidence.setdefault("location", []).append(Evidence(quote=require(doc.location.city)))
            if decision.get("country"):
                doc.location.country, doc.location.country_code = "Vietnam", decision["country"]
                facts.country = decision["country"]
                facts.evidence.setdefault("location", []).append(Evidence(quote=require("Vietnam")))
            if doc.location.remote_policy:
                facts.evidence.setdefault("location", []).append(Evidence(quote=require(doc.location.remote_policy)))
            doc.compensation.currency = decision["currency"]
            for name in ("min_monthly", "max_monthly", "min_annual", "max_annual"):
                setattr(doc.compensation, name, decision.get(name))
            if decision["currency"]:
                facts.currency = decision["currency"]
                facts.evidence["salary"] = [Evidence(quote=require(decision["currency"]))]
                if doc.compensation.raw_text and literal(doc.compensation.raw_text):
                    facts.evidence["salary"].append(Evidence(quote=require(doc.compensation.raw_text)))
            # Competitive/undisclosed pay does not prove negotiation. The schema boolean cannot express unknown.
            doc.compensation.is_negotiable = not bool(decision.get("min_monthly"))
        else:
            # Skill group labels and inferred proficiency are not assertion evidence.
            for skill in doc.skills:
                skill.level = None
            doc.skills = [s for s in doc.skills if s.keywords]
            doc.certificates = [c for c in doc.certificates if c.name not in {"", "Certificate"}]
            doc.projects = [p for p in doc.projects if p.name not in {"", "Project"} or p.highlights or p.keywords]
            original_doc = model.model_validate(read_json(output / "drafts" / (id + ".json"))).document
            # Restore ISO dates only when a numeric source date directly supports them.
            for collection in ("work", "education", "projects"):
                for current, original in zip(getattr(doc, collection), getattr(original_doc, collection)):
                    for name in ("start_date", "end_date"):
                        value = getattr(original, name)
                        if value and re.fullmatch(r"\d{4}-\d{2}", value):
                            year, month = value[:4], int(value[5:7])
                            m = re.search(r"(?<!\d)0?" + str(month) + r"/" + year + r"(?!\d)", text)
                            if m:
                                setattr(current, name, value)
                        if name == "end_date" and value and value.casefold() == "present" and literal("Current"):
                            current.end_date = "Present"
            quotes = list(dict.fromkeys(q for s in doc.skills for k in s.keywords if (q := literal(k))))
            if quotes:
                facts.evidence["skills"] = [Evidence(quote=q) for q in quotes]
            title_quotes = [q for v in [doc.basics.label, *[w.position for w in doc.work]] if (q := literal(v))]
            if title_quotes:
                facts.evidence["title"] = [Evidence(quote=q) for q in dict.fromkeys(title_quotes)]
            # Preserve validated experience quotations from the field-support pass.
            if envelope.facts.has("experience"):
                facts.evidence["experience"] = envelope.facts.evidence["experience"]
            facts.completed_degrees = decision["completed_degrees"]
            if facts.completed_degrees:
                facts.evidence["education"] = [Evidence(quote=require(decision["degree_quote"]))]
                if "master" in facts.completed_degrees:
                    m = re.search(r"(?i)master[’']?s?\s+degree", text)
                    if m:
                        facts.evidence["education"].append(Evidence(quote=m.group()))
            if decision.get("country"):
                facts.country = decision["country"]
                q = require("Saudi Arabia" if facts.country == "SA" else "Viet Nam")
                facts.evidence["location"] = [Evidence(quote=q)]
                if doc.basics.location:
                    doc.basics.location.country_code = facts.country
            elif doc.basics.location:
                doc.basics.location.country_code = None
        envelope.facts = facts
        validate_evidence(facts, text)
        raw = envelope.model_dump(by_alias=True, mode="json")
        # Validate after assignments as well as quotation support; assignment validation is not enabled by legacy schemas.
        model.model_validate(raw)
        write_json(target_path, raw)
        final_fields = []
        def audit(obj, prefix):
            for name in type(obj).model_fields:
                value = getattr(obj, name)
                path = prefix + "." + name
                if isinstance(value, BaseModel):
                    audit(value, path)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, BaseModel):
                            audit(item, f"{path}[{i}]")
                        else:
                            final_fields.append({"path": f"{path}[{i}]", "value": item, "quote": literal(item), "reviewed": True})
                elif isinstance(value, dict):
                    for dimension, evidence in value.items():
                        for i, item in enumerate(evidence):
                            final_fields.append({"path": f"{path}.{dimension}[{i}]", "value": item.quote, "quote": item.quote, "reviewed": True})
                elif value is not None:
                    final_fields.append({"path": path, "value": value, "quote": literal(value) if isinstance(value, str) else None,
                                         "reviewed": True, "note": "Source-normalized dates/amounts, requirement classification, transport metadata and schema-required placeholders are explained in the original correction ledger and document notes."})
        audit(envelope.document, "document")
        audit(facts, "facts")
        review = {**old, "review_status": REVIEW_STATUS, "target_sha256": sha(target_path.read_bytes()),
                  "source_review_decision_sha256": sha((output / "review_decisions.json").read_bytes()),
                  "final_field_review": final_fields, "notes": old["notes"] + decision["notes"],
                  "human_approval": "pending"}
        write_json(output / "reviews" / (id + ".json"), review)
        print(json.dumps({"id": id, "final_fields_reviewed": len(final_fields), "quotation_valid": True}))


def validate_targets(root: Path, output: Path):
    targets, conversations, notes = [], [], []
    for row in read_json(output / "selection.json"):
        path = output / "targets" / (row["id"] + ".json")
        review = read_json(output / "reviews" / (row["id"] + ".json"))
        if review.get("review_status") != REVIEW_STATUS or review.get("document_id") != row["id"]:
            raise ValueError("Pilot requires an assistant-reviewed ledger")
        text = (root / row["text_path"]).read_text(encoding="utf-8")
        model = ResumeInput if row["kind"] == "resume" else JobInput
        raw = read_json(path)
        if set(raw) != {"document", "facts"}:
            raise ValueError("Pilot target requires document + facts")
        target = model.model_validate(raw)
        validate_evidence(target.facts, text)
        if (not review.get("field_review") or not review.get("final_field_review") or
                not all(r.get("reviewed") for r in review["final_field_review"]) or
                review.get("target_sha256") != sha(path.read_bytes())):
            raise ValueError("Missing or stale field review")
        system, user = _prompts(row["kind"], text)
        content = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
        example = {"id": row["id"], "review_status": REVIEW_STATUS, "split": "train",
                   "source_sha256": row["source_sha256"], "text_sha256": row["text_sha256"],
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user},
                                {"role": "assistant", "content": content}]}
        conversations.append(example)
        notes.append(review)
        targets.append({"id": row["id"], "schema_valid": True, "quotation_valid": True,
                        "review_status": REVIEW_STATUS, "source_path": row["source_path"], "text_path": row["text_path"]})
    write_json(output / "validation_report.json", {"count": len(targets), "targets": targets})
    write_json(output / "review_ledger.json", notes)
    (output / "conversations.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in conversations), encoding="utf-8")
    return targets


def token_report(root: Path, output: Path, revision=None, cache_dir=None):
    import numpy as np
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    model = "Qwen/Qwen2.5-7B-Instruct"
    revision = revision or HfApi().model_info(model).sha
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, cache_dir=cache_dir)
    def count(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    def chat_count(messages, generation_prompt=False):
        # Count the actual rendered template. Transformers versions may return
        # BatchEncoding rather than a bare token list when tokenize=True.
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
        return count(rendered)
    def stats(values):
        return {"count": len(values), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "maximum": max(values), "exceeding": {str(n): sum(x > n for x in values) for n in (4096, 8192, 32768)}}
    documents = []
    for row in read_json(output / "source_manifest.json")["documents"]:
        text = (root / row["text_path"]).read_text(encoding="utf-8")
        documents.append({"id": row["id"], "kind": row["kind"], "domain": row["domain"], "source_tokens": count(text)})
    pilots = []
    for line in (output / "conversations.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        messages = ex["messages"]
        prompt_tokens = chat_count(messages[:2], True)
        full_tokens = chat_count(messages)
        source = json.loads(messages[1]["content"])["source_text"]
        empty_system, empty_user = _prompts(json.loads(messages[1]["content"])["document_kind"], "")
        empty_tokens = chat_count([{"role": "system", "content": empty_system}, {"role": "user", "content": empty_user}], True)
        pilots.append({"id": ex["id"], "source_tokens": count(source), "target_json_tokens": count(messages[2]["content"]),
                       "prompt_tokens": prompt_tokens, "prompt_plus_target_tokens": full_tokens,
                       "system_schema_tokens": count(messages[0]["content"]), "empty_source_prompt_tokens": empty_tokens,
                       "chat_template_overhead": prompt_tokens - sum(count(m["content"]) for m in messages[:2])})
    report = {"tokenizer": model, "resolved_revision": revision, "chat_template_sha256": sha(tokenizer.chat_template.encode()),
              "source_manifest_sha256": sha((output / "source_manifest.json").read_bytes()),
              "conversations_sha256": sha((output / "conversations.jsonl").read_bytes()),
              "source": stats([r["source_tokens"] for r in documents]),
              "source_by_kind": {k: stats([r["source_tokens"] for r in documents if r["kind"] == k]) for k in ("resume", "job")},
              "pilot_target_json": stats([r["target_json_tokens"] for r in pilots]),
              "pilot_prompt_plus_target": stats([r["prompt_plus_target_tokens"] for r in pilots]),
              "documents": documents, "pilot": pilots}
    write_json(output / "token_report.json", report)
    return report
