"""Checksum-bound, item-level v3 reviews and conservative rule-derived edits."""
from __future__ import annotations
from copy import deepcopy
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from .v3_aliases import skill_name
from .v2_corpus import atomic_json, digest, read_json
from .v3_validation import POLICY_VERSION, direct_level, validate_skill_evidence, source_depth_supported
from .models import JobInput, ResumeInput

VERSION = "ats-source-corpus-v3"
REVIEW_VERSION = "ats-review-v3"
RULES = {
    "R3-GROUP": "Group proficiency does not establish individual skill depth.",
    "R3-LOCAL": "An explicit preceding or parenthesized level belongs to its directly named skill.",
    "R3-PROFICIENT": "Proficient maps to Intermediate only as a direct skill claim; training others stays pending.",
    "R3-STANDARD": "Unqualified experience and design skill lists have Standard importance.",
    "R3-SPECIALIZATION": "Specialization years are evidence, not Advanced proficiency.",
    "R3-TITLE": "A title slash nominates review rather than establishing AND/OR scope.",
    "R3-SCRUM": "The approved wording Scrum/Agile development methodologies is one Standard choice.",
    "R3-GA4": "Reject Google Analytics 4 to broad analytics without domain context.",
    "R3-NO-BEGINNER": "D3-30: literal Beginner maps to normal, retaining raw wording and its source quote.",
}


def refresh(corpus):
    for document in corpus["documents"]:
        document["checksum_sha256"] = digest(document["envelope"])
    corpus["counts"] = {"documents": len(corpus["documents"]),
        "kind": dict(Counter(d["kind"] for d in corpus["documents"])),
        "skills_status": dict(Counter(d.get("skills_status", d["envelope"].get("annotation", {}).get("skills_status", "missing")) for d in corpus["documents"])),
        "skills_review_status": dict(Counter(d.get("skills_review_status", "pending") for d in corpus["documents"])),
        "full_envelope_review_status": dict(Counter(d.get("full_envelope_review_status", "pending") for d in corpus["documents"]))}
    corpus["identity_sha256"] = digest({k: v for k, v in corpus.items() if k not in {"identity_sha256", "review_history"}})


def reconcile_document(document, text):
    """Recompute evidence completeness without promoting human envelope approval."""
    cls = ResumeInput if document["kind"] == "resume" else JobInput
    envelope = cls.model_validate(document["envelope"])
    issues = validate_skill_evidence(envelope, text, strict=False)
    previous = document.get("v3_validation_issues", [])
    if envelope.annotation.skills_status != "missing":
        if issues:
            envelope.annotation.skills_status = "partial"
        elif previous:
            envelope.annotation.skills_status = "complete"
    document["envelope"] = envelope.model_dump(mode="json")
    document["skills_status"] = envelope.annotation.skills_status
    document["v3_validation_issues"] = issues


def validate_identity(corpus):
    if corpus.get("version") != VERSION:
        raise ValueError("Expected v3 corpus")
    expected = digest({k: v for k, v in corpus.items() if k not in {"identity_sha256", "review_history"}})
    if corpus.get("identity_sha256") != expected:
        raise ValueError("Stale v3 corpus identity")
    if len({d["id"] for d in corpus["documents"]}) != len(corpus["documents"]):
        raise ValueError("Duplicate document IDs")
    for document in corpus["documents"]:
        if document["checksum_sha256"] != digest(document["envelope"]):
            raise ValueError("Stale document checksum: " + document["id"])


def remove_beginner_level(corpus, root: Path, *, decided_on="2026-10-12"):
    """Apply D3-30 idempotently; policy-derived edits do not approve envelopes."""
    from src.parser.skill_v3 import CVSkillDetailV3
    validate_identity(corpus)
    planned = []
    # Verify every source and plan edits before mutating any item.
    for key, (document, container, index, item) in items(corpus).items():
        if not document or document["kind"] != "resume" or item.get("proficiency") != "beginner":
            continue
        source = document["source"]
        original = (root / source["source_path"]).read_bytes()
        # Corpus text hashes use decoded text with universal newline handling.
        # Preserve that established convention; original document hashes are bytes.
        text = (root / source["text_path"]).read_text(encoding="utf-8")
        if hashlib.sha256(original).hexdigest() != source["source_sha256"] or hashlib.sha256(text.encode("utf-8")).hexdigest() != source["text_sha256"]:
            raise ValueError("Beginner migration source checksum mismatch: " + key)
        quote = item.get("proficiency_quote") or ""
        if not quote or quote not in text or direct_level(item["name"], quote) != "normal" or not source_depth_supported(quote, text):
            raise ValueError("Beginner migration requires literal skill-local evidence: " + key)
        after = CVSkillDetailV3.model_validate(item).model_dump(mode="json")
        planned.append((key, document, container, index, deepcopy(item), after, text))
    if not planned and corpus.get("level_policy_version") == POLICY_VERSION:
        return []
    records = []
    for key, document, container, index, before, after, text in planned:
        container[index] = after
        record = {"date": decided_on, "decision_id": "D3-30", "item_id": key,
            "document_id": document["id"], "review_kind": "policy_derived",
            "rule_id": "R3-NO-BEGINNER", "reason": "Direct human policy: Beginner is unspecified in v3; retain raw wording and quote. This is not whole-envelope approval.",
            "approval_source": "User decision relayed by Claude Code: I think we should remove beginner level since no one would ever write that.",
            "source_sha256": document["source"]["source_sha256"], "text_sha256": document["source"]["text_sha256"],
            "quote": before["proficiency_quote"], "before": before, "after": deepcopy(after),
            "before_sha256": digest(before), "after_sha256": digest(after), "policy_version": POLICY_VERSION}
        record["transaction_sha256"] = digest(record)
        records.append(record)
        reconcile_document(document, text)
    corpus["rules"]["R3-NO-BEGINNER"] = RULES["R3-NO-BEGINNER"]
    corpus["level_policy_version"] = POLICY_VERSION
    corpus["changes"].extend(records)
    refresh(corpus)
    return records


def items(corpus):
    result = {}
    for document in corpus["documents"]:
        doc = document["envelope"]["document"]
        if document["kind"] == "resume":
            for gi, group in enumerate(doc["skills"]):
                for di, detail in enumerate(group["details"]):
                    key = f"{document['id']}:skill:{gi}:{di}"
                    result[key] = (document, group["details"], di, detail)
        else:
            for index, unit in enumerate(doc["skills"]["requirements"]):
                result[unit["id"]] = (document, doc["skills"]["requirements"], index, unit)
    for index, relation in enumerate(corpus["relations"]["relations"]):
        result[relation["id"]] = (None, corpus["relations"]["relations"], index, relation)
    return result


def review_manifest(corpus):
    manifest = []
    for key, (document, _, _, value) in items(corpus).items():
        source = document["source"] if document else {}
        manifest.append({"item_id": key, "document_id": document["id"] if document else None,
            "source_sha256": source.get("source_sha256"), "text_sha256": source.get("text_sha256"),
            "item_sha256": digest(value), "before": deepcopy(value), "decision": "pending"})
    return {"version": REVIEW_VERSION, "corpus_sha256": corpus["identity_sha256"], "items": manifest}


def _rename_keyword(document, container, before, after):
    if document and document["kind"] == "resume" and before["name"] != after["name"]:
        group = next(g for g in document["envelope"]["document"]["skills"] if g["details"] is container)
        old = deepcopy(group["keywords"])
        group["keywords"] = list(dict.fromkeys(after["name"] if n == before["name"] else n for n in old))
        return {"keywords_before": old, "keywords_after": deepcopy(group["keywords"])}
    return {}


def import_review(corpus, payload, root: Path):
    """Validate the whole transaction before applying any mutation; repeat is a no-op."""
    validate_identity(corpus)
    if payload.get("version") != REVIEW_VERSION:
        raise ValueError("Expected v3 review exchange")
    transaction = digest(payload)
    if any(row.get("transaction_sha256") == transaction for row in corpus["review_history"]):
        return {"applied": 0, "idempotent": True, "transaction_sha256": transaction}
    if payload.get("corpus_sha256") != corpus["identity_sha256"]:
        raise ValueError("Review corpus checksum is stale")
    reviewed_on = date.fromisoformat(payload.get("reviewed_on", "2026-10-08")).isoformat()
    lookup = items(corpus)
    selected = []
    seen = set()
    for decision in payload.get("items", []):
        if decision.get("decision", "pending") == "pending":
            continue
        key = decision["item_id"]
        if key in seen:
            raise ValueError("Duplicate review item: " + key)
        seen.add(key)
        if key not in lookup:
            raise ValueError("Unknown review item: " + key)
        document, container, index, before = lookup[key]
        if decision.get("item_sha256") != digest(before):
            raise ValueError("Stale item checksum: " + key)
        if not decision.get("reviewer") or not decision.get("reason"):
            raise ValueError("Review requires reviewer and reason")
        action = decision["decision"]
        if action not in {"accept", "edit", "reject"}:
            raise ValueError("Unknown review action")
        if document:
            source = document["source"]
            for field in ("source_sha256", "text_sha256"):
                if decision.get(field) != source[field]:
                    raise ValueError("Stale source checksum: " + key)
            text = (root / source["text_path"]).read_text(encoding="utf-8")
            if digest(text.encode()) != source["text_sha256"]:
                raise ValueError("Source text changed: " + key)
            original = root / source["source_path"]
            if digest(original.read_bytes()) != source["source_sha256"]:
                raise ValueError("Original source changed: " + key)
            quote = decision.get("source_quote", "")
            if not quote or quote not in text:
                raise ValueError("Approval quote is not a literal source span: " + key)
        after = deepcopy(decision.get("replacement", before))
        if document and document["kind"] == "resume":
            after["provenance"] = after.get("provenance") or {}
            after["provenance"]["review_status"] = "approved" if action != "reject" else "rejected"
        else:
            after["review_status"] = "approved" if action != "reject" else "rejected"
        selected.append((decision, document, container, index, deepcopy(before), after))
    # Validate replacements together in a cloned corpus before publishing.
    trial = deepcopy(corpus)
    trial_lookup = items(trial)
    for decision, document, _, _, _, after in selected:
        td, container, index, before = trial_lookup[decision["item_id"]]
        container[index] = deepcopy(after)
        _rename_keyword(td, container, before, after)
        if td:
            cls = ResumeInput if td["kind"] == "resume" else JobInput
            envelope = cls.model_validate(td["envelope"])
            # Existing pending errors remain review work; new replacements must
            # not introduce invalid evidence on the changed item.
            text = (root / td["source"]["text_path"]).read_text(encoding="utf-8")
            old_issues = validate_skill_evidence(cls.model_validate(document["envelope"]), text, strict=False)
            new_issues = validate_skill_evidence(envelope, text, strict=False)
            old_errors = {(i["path"], i["reason"]) for i in old_issues if i["severity"] == "error"}
            if any((i["path"], i["reason"]) not in old_errors for i in new_issues if i["severity"] == "error"):
                raise ValueError("Review introduces unsupported evidence: " + decision["item_id"])
    for decision, document, container, index, before, after in selected:
        container[index] = after
        keyword_change = _rename_keyword(document, container, before, after)
        corpus["changes"].append({"date": reviewed_on, "item_id": decision["item_id"],
            "document_id": document["id"] if document else None, "review_kind": decision.get("review_kind", "direct_human"),
            "rule_id": decision.get("rule_id"), "reason": decision["reason"], "approval_source": decision["reviewer"],
            "source_sha256": decision.get("source_sha256"), "text_sha256": decision.get("text_sha256"),
            "source_quote": decision.get("source_quote"), "before": before, "after": deepcopy(after),
            "before_sha256": digest(before), "after_sha256": digest(after), "transaction_sha256": transaction, **keyword_change})
    if selected:
        corpus["review_history"].append({"transaction_sha256": transaction, "applied": len(selected), "date": reviewed_on})
        refresh(corpus)
    return {"applied": len(selected), "idempotent": False, "transaction_sha256": transaction}


def _literal(text, excerpt):
    match = re.search(r"\s+".join(re.escape(word) for word in excerpt.split()), text, re.I)
    if not match:
        raise ValueError("Approved excerpt not found: " + excerpt)
    return match.group(0)


def _local_claim(name, text):
    # Search only explicit local forms; no title/group/year inference.
    from .v3_aliases import alias_spellings
    names = alias_spellings(name)
    found = []
    for spelling in names:
        n = re.escape(spelling)
        patterns = (rf"\b(beginner|intermediate|advanced|expert|proficient)\s+(?:(?:in|with)\s+)?{n}(?!\w)",
                    rf"(?<!\w){n}\s*\(\s*(beginner|intermediate|advanced|expert)\s*\)")
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.I):
                quote = match.group(0)
                level = direct_level(name, quote)
                if level and source_depth_supported(quote, text):
                    found.append((level, match.group(1), quote))
    if len({entry[0] for entry in found}) == 1:
        return sorted(found, key=lambda entry: (len(entry[2]), entry[2]))[0]
    return None


def sample3_repair_payload(corpus, root: Path):
    """Normalize inherited level-prefixed tokens without shifting historical item IDs."""
    validate_identity(corpus)
    decisions = []
    for key, (document, _, _, item) in items(corpus).items():
        if not document or document["id"] != "cv_003":
            continue
        match = re.fullmatch(r"(?:intermediate|advanced)\s+(word|excel|access)", item["name"], re.I)
        if not match:
            continue
        name = match.group(1).lower()
        level = {"word": "intermediate", "excel": "advanced", "access": "intermediate"}[name]
        text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
        quote = _literal(text, "Intermediate Word Advanced Excel PowerPoint Intermediate Access")
        depth_quote = _literal(quote, level + " " + name)
        after = deepcopy(item)
        after.update(name=name, proficiency=level, raw_proficiency=depth_quote.split()[0],
                     skill_quote=depth_quote, proficiency_quote=depth_quote)
        decisions.append({"item_id": key, "item_sha256": digest(item), "decision": "edit",
            "source_sha256": document["source"]["source_sha256"], "text_sha256": document["source"]["text_sha256"],
            "source_quote": quote, "replacement": after, "reviewer": "derived-from-user-approved-sample-3",
            "review_kind": "rule_derived", "rule_id": "R3-LOCAL",
            "reason": "Apply sample 3's preceding-level interpretation to inherited level-prefixed tokens. Preserve item IDs; canonical duplicates merge once during scoring."})
    return {"version": REVIEW_VERSION, "corpus_sha256": corpus["identity_sha256"], "items": decisions}


def prepare(root: Path, source: Path, target: Path, review_root: Path):
    if target.exists():
        raise ValueError("Refusing to overwrite an existing v3 review corpus")
    original = read_json(source)
    corpus = deepcopy(original)
    corpus.update(version=VERSION, preparer_version="explicit-local-source-scan-v3", parent_v2_identity=original["identity_sha256"],
                  rules=RULES, changes=[], review_history=[], pending_samples=[2, 6])
    by_id = {d["id"]: d for d in corpus["documents"]}
    for document in corpus["documents"]:
        before = deepcopy(document["envelope"])
        document["v2_envelope"] = before
        document["envelope"]["document"]["schema_version"] = "resume-v3" if document["kind"] == "resume" else "job-v3"
        document["envelope"]["annotation"]["schema_version"] = "ats-skills-v3"
        document["full_envelope_review_status"] = "pending"
        document["envelope"]["annotation"]["envelope_review_status"] = "pending"
        if document["kind"] == "job":
            for unit in document["envelope"]["document"]["skills"]["requirements"]:
                unit["option_details"] = [{"name": n, "minimum_proficiency": "normal", "raw_proficiency": None, "proficiency_quote": None, "experience": []} for n in unit["options"]]
        corpus["changes"].append({"date": "2026-10-08", "document_id": document["id"], "review_kind": "schema_migration", "rule_id": "R3-SCHEMA",
            "reason": "Version the additive depth contract; preserve missing depth as unspecified and do not approve envelopes.",
            "before_sha256": digest(before), "after_sha256": digest(document["envelope"]), "before_reference": "v2_envelope", "after_reference": "envelope"})
    # Add source-supported missing keywords as pending proposals, then bind direct
    # approvals to their concrete IDs and original source checksums.
    for doc_id, name, excerpt in (("cv_001", "program management", "Program Management"),):
        document = by_id[doc_id]
        groups = document["envelope"]["document"]["skills"]
        if not any(skill_name(d["name"]) == skill_name(name) for g in groups for d in g["details"]):
            text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
            groups[0]["details"].append({"name": name, "proficiency": "normal", "raw_proficiency": None,
                "skill_quote": _literal(text, excerpt), "proficiency_quote": None,
                "provenance": {k: document["source"].get(k) for k in ("source_id", "source_path", "source_sha256", "text_sha256")}})
            groups[0]["keywords"].append(name)
            corpus["changes"].append({"date": "2026-10-08", "document_id": doc_id, "review_kind": "source_proposal", "rule_id": "R3-GROUP",
                "before": None, "after": deepcopy(groups[0]["details"][-1]), "source_sha256": document["source"]["source_sha256"],
                "text_sha256": document["source"]["text_sha256"], "source_quote": _literal(text, excerpt),
                "reason": "Add the missing explicitly named skill as a neutral proposal before binding the sample-1 approval."})
    refresh(corpus)
    direct = []
    lookup = items(corpus)
    excerpts = {"cv_001": "Accounting; General Accounting; Accounts Payable; Program Management.",
                "cv_003": "Intermediate Word Advanced Excel PowerPoint Intermediate Access",
                "jd_001": "Experience in both architecture and software design implementation: SOLID principles, Design Patterns, Microservices architectures",
                "jd_021": "Experience with Scrum/Agile development methodologies."}
    for key, (document, _, _, item) in lookup.items():
        if document is None:
            if skill_name(item["candidate"]) == skill_name("Google Analytics 4") and skill_name(item["requirement"]) == "analytics":
                direct.append({"item_id": key, "item_sha256": digest(item), "decision": "reject", "reviewer": "user:approved-sample-8",
                    "reason": RULES["R3-GA4"], "rule_id": "R3-GA4", "source_quote": item.get("reason")})
            continue
        doc_id = document["id"]
        replacement = deepcopy(item)
        rule = None
        sample = None
        if doc_id == "cv_001" and skill_name(item["name"]) in {"accounting", "general accounting", "accounts payable", "program management"}:
            replacement.update(proficiency="normal", raw_proficiency=None, proficiency_quote=None)
            rule, sample = "R3-GROUP", 1
        elif doc_id == "cv_003" and skill_name(item["name"]) in {"word", "excel", "access", "powerpoint"}:
            levels = {"word": "intermediate", "excel": "advanced", "access": "intermediate", "powerpoint": "normal"}
            level = levels[skill_name(item["name"])]
            text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
            q = _literal(text, level + " " + item["name"]) if level != "normal" else None
            replacement.update(proficiency=level, raw_proficiency=q.split()[0] if q else None, proficiency_quote=q)
            rule, sample = "R3-LOCAL", 3
        elif doc_id == "jd_001" and any(skill_name(n) in {"solid principles", "design patterns", "microservices"} for n in item["options"]):
            replacement.update(importance="standard", importance_quote=None, issues=[])
            rule, sample = "R3-STANDARD", 4
        elif doc_id == "jd_021" and set(map(skill_name, item["options"])) == {"scrum", "agile"}:
            replacement.update(importance="standard", importance_quote=None, issues=[])
            rule, sample = "R3-SCRUM", 7
        if rule:
            text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
            quote = _literal(text, excerpts[doc_id])
            direct.append({"item_id": key, "source_sha256": document["source"]["source_sha256"], "text_sha256": document["source"]["text_sha256"],
                "item_sha256": digest(item), "decision": "edit", "reviewer": f"user:approved-sample-{sample}", "reason": RULES[rule],
                "rule_id": rule, "source_quote": quote, "replacement": replacement})
    payload = {"version": REVIEW_VERSION, "corpus_sha256": corpus["identity_sha256"], "items": direct}
    receipt = import_review(corpus, payload, root)
    repeat = import_review(corpus, payload, root)
    if not repeat["idempotent"]:
        raise ValueError("Approval import failed idempotency")
    import_review(corpus, sample3_repair_payload(corpus, root), root)
    # Apply approved rules to additional unambiguous claims, keeping their
    # coverage separate from the five directly reviewed samples.
    derived = []
    lookup = items(corpus)
    for key, (document, _, _, item) in lookup.items():
        if not document:
            continue
        replacement = deepcopy(item)
        rule = None
        quote = None
        if document["kind"] == "resume":
            if document["id"] == "cv_002" and skill_name(item["name"]) == "accounting":
                continue
            text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
            claim = _local_claim(item["name"], text)
            if claim and item.get("proficiency") == "normal":
                level, raw, quote = claim
                replacement.update(proficiency=level, raw_proficiency=raw, proficiency_quote=quote)
                rule = "R3-PROFICIENT" if raw.lower() == "proficient" else "R3-LOCAL"
        else:
            if item.get("quote", "").startswith("Title:"):
                if document["id"] == "jd_012" and set(item["options"]) == {"golang", "javascript"}:
                    replacement["issues"] = sorted(set(item["issues"]) | {"title_choice_scope_pending"})
                    rule, quote = "R3-TITLE", item["quote"]
            else:
                for detail in replacement["option_details"]:
                    claim = _local_claim(detail["name"], item["quote"])
                    if claim:
                        level, raw, quote = claim
                        detail.update(minimum_proficiency=level, raw_proficiency=raw, proficiency_quote=quote)
                        rule = "R3-LOCAL"
                if document["id"] == "jd_002" and set(item["options"]) == {"java", "kotlin"}:
                    for detail in replacement["option_details"]:
                        detail["experience"] = [{"min_years": 5, "max_years": 7, "quote": item["quote"]}]
                    rule, quote = "R3-SPECIALIZATION", item["quote"]
        if rule and replacement != item:
            derived.append({"item_id": key, "source_sha256": document["source"]["source_sha256"], "text_sha256": document["source"]["text_sha256"],
                "item_sha256": digest(item), "decision": "edit", "reviewer": "derived-from-user-approved-rule", "review_kind": "rule_derived",
                "reason": RULES[rule], "rule_id": rule, "source_quote": quote, "replacement": replacement})
    # Title ambiguity is logged but never approved by the generic review importer.
    title_pending = [d for d in derived if d["rule_id"] == "R3-TITLE"]
    derived = [d for d in derived if d["rule_id"] != "R3-TITLE"]
    for decision in title_pending:
        document, container, index, before = items(corpus)[decision["item_id"]]
        container[index] = decision["replacement"]
        corpus["changes"].append({**decision, "before": before, "after": decision["replacement"], "review_kind": "pending_nomination"})
    refresh(corpus)
    derived_payload = {"version": REVIEW_VERSION, "corpus_sha256": corpus["identity_sha256"], "items": derived}
    derived_receipt = import_review(corpus, derived_payload, root)
    # Preserve a separate pending Spring Boot requirement with specialization
    # evidence. The user's fixed policy specifies the separate denominator.
    document = by_id["jd_002"]
    units = document["envelope"]["document"]["skills"]["requirements"]
    if not any("spring boot" in unit["options"] for unit in units):
        source_unit = next(unit for unit in units if set(unit["options"]) == {"java", "kotlin"})
        new = {"id": "jd_002:req:v3-spring-boot", "options": ["spring boot"], "importance": "standard", "quote": source_unit["quote"],
            "importance_quote": None, "review_status": "pending", "issues": [], "option_details": [{"name": "spring boot", "minimum_proficiency": "normal",
            "raw_proficiency": None, "proficiency_quote": None, "experience": [{"min_years": 5, "max_years": 7, "quote": source_unit["quote"]}]}]}
        units.append(new)
        corpus["changes"].append({"date": "2026-10-08", "item_id": new["id"], "before": None, "after": new,
            "rule_id": "R3-SPECIALIZATION", "review_kind": "rule_derived", "reason": "Keep Kotlin OR Java and independent Spring Boot; years are evidence only.",
            "source_sha256": document["source"]["source_sha256"], "text_sha256": document["source"]["text_sha256"], "source_quote": new["quote"]})
    # Item reviews do not promote the document/full-envelope review status.
    for document in corpus["documents"]:
        cls = ResumeInput if document["kind"] == "resume" else JobInput
        envelope = cls.model_validate(document["envelope"])
        text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
        document["v3_validation_issues"] = validate_skill_evidence(envelope, text, strict=False)
        if document["v3_validation_issues"]:
            envelope.annotation.skills_status = "partial" if envelope.annotation.skills_status != "missing" else "missing"
        document["envelope"] = envelope.model_dump(mode="json")
        document["skills_status"] = envelope.annotation.skills_status
    refresh(corpus)
    atomic_json(target, corpus)
    review_root.mkdir(parents=True, exist_ok=True)
    atomic_json(review_root / "imported_approvals.json", payload)
    atomic_json(review_root / "derived_rules.json", derived_payload)
    atomic_json(review_root / "import_receipt.json", {"direct": receipt, "repeat": repeat, "derived": derived_receipt})
    atomic_json(review_root / "review_items.json", review_manifest(corpus))
    atomic_json(review_root / "changes.json", corpus["changes"])
    pending = [{"sample": 2, "document_id": "cv_002", "reason": "Training-others wording does not establish accounting proficiency; retain raw wording and normal."},
               {"sample": 6, "document_id": "jd_012", "reason": "Title Golang/JavaScript does not resolve AND/OR scope; body review required."}]
    atomic_json(review_root / "unresolved.json", {"version": REVIEW_VERSION, "corpus_sha256": corpus["identity_sha256"], "samples": pending,
        "other_exceptions": [{"document_id": d["id"], "issues": d["v3_validation_issues"]} for d in corpus["documents"] if d["v3_validation_issues"]]})
    return {"identity": corpus["identity_sha256"], "direct_items": receipt["applied"], "derived_items": derived_receipt["applied"], "pending_samples": [2, 6]}
