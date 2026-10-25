"""Source-bound structural OR review, atomicity, and single-unit fulfillment."""
from copy import deepcopy
import pytest

from scripts.ats_v3_resolve_jd012 import (OPTIONS, REVIEWED_ON, RETIRED_ID, SOURCE_SENTENCE,
                                       TARGET_ID, apply_decision, prepare_decision)
from src.scoring import JobInput, ResumeInput, ScorerConfig
from src.scoring.skills_v3 import GROUP_WEIGHTS, score_skills_v3
from src.scoring.v2_corpus import digest
from src.scoring.v3_corpus import VERSION, items, reconcile_document, refresh, validate_identity


def fixture(tmp_path):
    text = SOURCE_SENTENCE + "\nPython is required."
    (tmp_path / "source.txt").write_text(text, encoding="utf-8")
    source = {"source_path": "source.txt", "text_path": "source.txt",
              "source_sha256": digest((tmp_path / "source.txt").read_bytes()), "text_sha256": digest(text.encode())}
    requirements = [
        {"id": TARGET_ID, "options": ["databases"], "importance": "standard",
         "quote": SOURCE_SENTENCE, "issues": ["complex_choice_scope"]},
        {"id": RETIRED_ID, "options": ["martech"], "importance": "standard",
         "quote": SOURCE_SENTENCE, "issues": ["complex_choice_scope"]},
        {"id": "unrelated-python", "options": ["Python"], "importance": "core",
         "quote": "Python is required.", "importance_quote": "Python is required."},
    ]
    jd = JobInput.model_validate({"document": {"schema_version": "job-v3", "id": "jd_012",
        "url": "urn:jd012", "scraped_at": "none", "title": "Engineer", "title_normalized": "engineer",
        "category": "engineer", "skills": {"requirements": requirements}},
        "facts": {"evidence": {"skills": [{"quote": text}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "partial",
                       **{key: value for key, value in source.items() if key != "text_path"}}})
    doc = {"id": "jd_012", "kind": "job", "source": source, "envelope": jd.model_dump(mode="json"),
           "full_envelope_review_status": "pending", "skills_status": "partial",
           "issues": [{"code": "complex_choice_scope", "quote": SOURCE_SENTENCE}]}
    reconcile_document(doc, text)
    corpus = {"version": VERSION, "documents": [doc], "relations": {"version": "empty", "relations": []},
              "changes": [], "review_history": []}
    refresh(corpus)
    return corpus


def test_jd012_merge_is_dated_bound_idempotent_and_preserves_other_units(tmp_path):
    corpus = fixture(tmp_path)
    before = deepcopy(items(corpus)["unrelated-python"][3])
    payload = prepare_decision(corpus, tmp_path)
    result = apply_decision(corpus, payload, tmp_path)
    assert result["applied"] == 2 and result["merged"] == 1
    validate_identity(corpus)
    lookup = items(corpus)
    assert RETIRED_ID not in lookup and lookup["unrelated-python"][3] == before
    merged = lookup[TARGET_ID][3]
    assert merged["options"] == OPTIONS and merged["importance"] == "standard"
    assert merged["review_status"] == "approved" and merged["issues"] == []
    doc = corpus["documents"][0]
    assert doc["skills_status"] == "complete" and doc["v3_validation_issues"] == []
    assert doc["issues"] == []
    assert doc["envelope"]["annotation"]["envelope_review_status"] == "pending"
    assert doc["full_envelope_review_status"] == "pending"
    approvals = [c for c in corpus["changes"] if c["review_kind"] == "direct_human"]
    assert len(approvals) == 2 and all(c["date"] == REVIEWED_ON for c in approvals)
    retirement = next(c for c in corpus["changes"] if c["review_kind"] == "direct_human_merge")
    assert retirement["item_id"] == RETIRED_ID and retirement["after"] is None
    assert retirement["merged_into"] == TARGET_ID
    assert corpus["review_history"][0]["date"] == REVIEWED_ON
    after = deepcopy(corpus)
    replay = apply_decision(corpus, payload, tmp_path)
    assert replay["idempotent"] and replay["merged"] == 0 and corpus == after


@pytest.mark.parametrize("field", ["item_sha256", "source_sha256", "text_sha256"])
def test_jd012_merge_stale_retired_binding_is_atomic(tmp_path, field):
    corpus = fixture(tmp_path)
    payload = prepare_decision(corpus, tmp_path)
    payload["items"][1][field] = "stale"
    before = deepcopy(corpus)
    with pytest.raises(ValueError, match="Stale|stale"):
        apply_decision(corpus, payload, tmp_path)
    assert corpus == before


def test_jd012_changed_source_is_rejected_without_mutation(tmp_path):
    corpus = fixture(tmp_path)
    payload = prepare_decision(corpus, tmp_path)
    before = deepcopy(corpus)
    (tmp_path / "source.txt").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="Source text changed"):
        apply_decision(corpus, payload, tmp_path)
    assert corpus == before


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("policy", ["off", "score_only", "full"])
def test_each_jd012_alternative_fulfills_one_standard_unit(tmp_path, option, policy):
    corpus = fixture(tmp_path)
    apply_decision(corpus, prepare_decision(corpus, tmp_path), tmp_path)
    jd = JobInput.model_validate(corpus["documents"][0]["envelope"])
    cv = ResumeInput.model_validate({"document": {"schema_version": "resume-v3", "skills": [
        {"name": "Skills", "keywords": [option], "details": [{"name": option, "skill_quote": option}]}]},
        "facts": {"evidence": {"skills": [{"quote": option}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    result = score_skills_v3(cv, jd, ScorerConfig(version="ats-en-v3", depth_policy=policy))
    rows = [r for r in result["requirements"] if r["evidence"]["requirement_quote"] == SOURCE_SENTENCE]
    assert len(rows) == 1 and rows[0]["id"] == TARGET_ID
    assert rows[0]["fulfillment"] == 1.0 and rows[0]["coefficient"] == 1.0
    assert rows[0]["weight"] == GROUP_WEIGHTS["standard"] == 0.6
    assert result["group_fulfillment"]["standard"] == 1.0
    assert rows[0]["required_level"] == "normal"


def test_jd012_merge_keeps_unrelated_issue_partial(tmp_path):
    corpus = fixture(tmp_path)
    items(corpus)["unrelated-python"][3]["issues"] = ["unreviewed_scope"]
    doc = corpus["documents"][0]
    reconcile_document(doc, (tmp_path / "source.txt").read_text(encoding="utf-8"))
    refresh(corpus)
    apply_decision(corpus, prepare_decision(corpus, tmp_path), tmp_path)
    doc = corpus["documents"][0]
    assert doc["skills_status"] == "partial"
    assert len(doc["v3_validation_issues"]) == 1
    assert doc["v3_validation_issues"][0]["path"] == "requirements.unrelated-python"
