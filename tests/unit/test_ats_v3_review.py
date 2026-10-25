"""Checksum binding, atomic review imports, pending preservation and identities."""
from copy import deepcopy
import pytest
from src.scoring.v3_corpus import VERSION, REVIEW_VERSION, refresh, items, review_manifest, import_review
from src.scoring.v2_corpus import digest
from src.scoring.v2_benchmark import require_same_identity, canonical_hash
from src.scoring import ResumeInput


def fixture(tmp_path):
    cv = ResumeInput.model_validate({"document": {"schema_version": "resume-v3", "skills": [{"name": "Technical", "keywords": ["Python"],
        "details": [{"name": "Python", "skill_quote": "Python"}]}]}, "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    text = "beginner Python"
    (tmp_path / "source.txt").write_text(text, encoding="utf-8")
    source = {"source_path": "source.txt", "text_path": "source.txt", "source_sha256": digest(text.encode()), "text_sha256": digest(text.encode())}
    cv.annotation.source_sha256 = source["source_sha256"]
    cv.annotation.text_sha256 = source["text_sha256"]
    document = {"id": "cv", "kind": "resume", "source": source, "envelope": cv.model_dump(mode="json")}
    corpus = {"version": VERSION, "documents": [document], "relations": {"version": "empty", "relations": []}, "changes": [], "review_history": []}
    refresh(corpus)
    payload = review_manifest(corpus)
    item = payload["items"][0]
    replacement = deepcopy(item["before"])
    replacement.update(proficiency="beginner", raw_proficiency="beginner", proficiency_quote=text)
    item.update(decision="edit", reviewer="human", reason="direct claim", source_quote=text, replacement=replacement)
    return corpus, payload


def test_import_is_bound_and_idempotent(tmp_path):
    corpus, payload = fixture(tmp_path)
    result = import_review(corpus, payload, tmp_path)
    assert result["applied"] == 1
    after = deepcopy(corpus)
    assert import_review(corpus, payload, tmp_path)["idempotent"]
    assert corpus == after
    assert corpus["documents"][0]["envelope"]["annotation"]["envelope_review_status"] == "pending"
    assert corpus["changes"][0]["review_kind"] == "direct_human"


@pytest.mark.parametrize("field", ["item_sha256", "source_sha256", "text_sha256"])
def test_stale_binding_rejects_without_mutation(tmp_path, field):
    corpus, payload = fixture(tmp_path)
    payload["items"][0][field] = "stale"
    before = deepcopy(corpus)
    with pytest.raises(ValueError, match="Stale|stale"):
        import_review(corpus, payload, tmp_path)
    assert corpus == before


def test_pending_does_not_create_approval_or_change_identity(tmp_path):
    corpus, payload = fixture(tmp_path)
    payload["items"][0]["decision"] = "pending"
    before = deepcopy(corpus)
    assert import_review(corpus, payload, tmp_path)["applied"] == 0
    assert corpus == before


def test_normal_review_resolves_completeness_but_not_envelope_approval(tmp_path):
    from src.scoring.v3_corpus import reconcile_document
    corpus, payload = fixture(tmp_path)
    doc = corpus["documents"][0]
    detail = doc["envelope"]["document"]["skills"][0]["details"][0]
    detail["raw_proficiency"] = "ambiguous original wording"
    reconcile_document(doc, "beginner Python")
    assert doc["skills_status"] == "partial"
    refresh(corpus)
    payload = review_manifest(corpus)
    payload["items"][0].update(decision="accept", reviewer="human", reason="Keep normal", source_quote="beginner Python")
    import_review(corpus, payload, tmp_path)
    reconcile_document(doc, "beginner Python")
    assert doc["skills_status"] == "complete" and doc["v3_validation_issues"] == []
    assert doc["envelope"]["document"]["skills"][0]["details"][0]["proficiency"] == "normal"
    assert doc["envelope"]["annotation"]["envelope_review_status"] == "pending"
    after = deepcopy(doc)
    reconcile_document(doc, "beginner Python")
    assert doc == after


def test_missing_evidence_is_not_promoted_by_reconciliation(tmp_path):
    from src.scoring.v3_corpus import reconcile_document
    corpus, _ = fixture(tmp_path)
    doc = corpus["documents"][0]
    doc["envelope"]["annotation"]["skills_status"] = "missing"
    doc["v3_validation_issues"] = [{"reason": "old_issue"}]
    reconcile_document(doc, "beginner Python")
    assert doc["skills_status"] == "missing"


def test_unsupported_replacement_is_atomic(tmp_path):
    corpus, payload = fixture(tmp_path)
    payload["items"][0]["replacement"]["proficiency_quote"] = "invented beginner Python"
    before = deepcopy(corpus)
    with pytest.raises(ValueError, match="unsupported evidence"):
        import_review(corpus, payload, tmp_path)
    assert corpus == before


def test_imported_rejection_is_excluded_from_scoring(tmp_path):
    from src.scoring import ScorerConfig, JobInput
    from src.scoring.skills_v3 import score_skills_v3
    corpus, payload = fixture(tmp_path)
    payload["items"][0].update(decision="reject", replacement=payload["items"][0]["before"])
    import_review(corpus, payload, tmp_path)
    cv = ResumeInput.model_validate(corpus["documents"][0]["envelope"])
    jd = JobInput.model_validate({"document": {"id": "jd", "url": "urn:jd", "scraped_at": "none", "title": "Engineer",
        "title_normalized": "engineer", "category": "engineer", "skills": {"requirements": [
            {"id": "python", "options": ["Python"], "quote": "Python is required", "importance": "core",
             "importance_quote": "Python is required", "review_status": "approved"}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    for policy in ("off", "score_only", "full"):
        result = score_skills_v3(cv, jd, ScorerConfig(version="ats-en-v3", depth_policy=policy))
        # Rejecting the only skill leaves missing evidence, not proof of no skills.
        assert not result["supported"] and result["score"] == 50
        assert result["core_gate_fulfillment"] is None and result["gate_status"] == "unknown"
        assert all(row["candidate_skill"] is None for row in result["requirements"])


def test_sample3_repair_preserves_ids_and_logs_keyword_renames(tmp_path):
    from src.scoring.v3_corpus import sample3_repair_payload
    corpus, _ = fixture(tmp_path)
    doc = corpus["documents"][0]
    doc["id"] = "cv_003"
    text = "Intermediate Word Advanced Excel PowerPoint Intermediate Access"
    (tmp_path / "source.txt").write_text(text, encoding="utf-8")
    for target in (doc["source"], doc["envelope"]["annotation"]):
        target.update(source_sha256=digest(text.encode()), text_sha256=digest(text.encode()))
    group = doc["envelope"]["document"]["skills"][0]
    group["keywords"] = ["intermediate word", "advanced excel", "intermediate access", "powerpoint"]
    group["details"] = [{"name": name, "skill_quote": text, "proficiency": "normal"} for name in group["keywords"]]
    refresh(corpus)
    before_ids = set(items(corpus))
    payload = sample3_repair_payload(corpus, tmp_path)
    assert import_review(corpus, payload, tmp_path)["applied"] == 3
    assert import_review(corpus, payload, tmp_path)["idempotent"]
    assert sample3_repair_payload(corpus, tmp_path)["items"] == []
    assert set(items(corpus)) == before_ids
    assert group["keywords"] == ["word", "excel", "access", "powerpoint"]
    assert [d["proficiency"] for d in group["details"]] == ["intermediate", "advanced", "intermediate", "normal"]
    assert all(c["review_kind"] == "rule_derived" and "keywords_before" in c for c in corpus["changes"])
    assert doc["envelope"]["annotation"]["envelope_review_status"] == "pending"


def test_artifact_identity_refuses_different_depth_families():
    def identity(policy):
        family = {"benchmark_version": "ats-v3-silver-diagnostic-v1", "policy": policy}
        return {**family, "family_sha256": canonical_hash(family)}
    off, full = identity("off"), identity("full")
    assert require_same_identity(off, off) == off["family_sha256"]
    with pytest.raises(ValueError, match="different benchmark identities"):
        require_same_identity(off, full)
    stale = {**off, "policy": "full"}
    with pytest.raises(ValueError, match="checksum"):
        require_same_identity(stale)
