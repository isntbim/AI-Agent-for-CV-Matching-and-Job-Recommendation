"""Meaningful provenance, transactional review and partition boundary checks."""

import json
from collections import Counter
from datetime import date

import numpy as np
import pytest

from src.scoring.v2_corpus import (CORPUS_VERSION, _cv_skills, _jd_skills,
    atomic_json, digest, export_review, export_training, import_review,
    propose_relations, refresh_identity, review_items, validate_corpus)
from src.scoring.models import ResumeInput, JobInput


def document(tmp_path, *, id="cv_a", kind="resume", split="train"):
    text = "Skills\nPython: Expert\nSQL\n" if kind == "resume" else "Requirements:\nMust have Python or Java\nSQL is preferred\n"
    (tmp_path / (id + ".txt")).write_text(text, encoding="utf-8")
    source = {"id": id, "source_path": id + ".txt", "text_path": id + ".txt", "source_sha256": digest((tmp_path / (id + ".txt")).read_bytes()), "text_sha256": digest(text.encode())}
    envelope = (ResumeInput.model_validate({"document": {"basics": {"name": "Candidate"}, "skills": [{"name": "Programming", "keywords": ["Python", "SQL"]}]}}).model_dump(mode="json") if kind == "resume" else JobInput.model_validate({"document": {"id": id, "title": "Developer", "url": "https://example.test/job", "scraped_at": "2026-09-30", "title_normalized": "developer", "category": "software_engineer", "skills": {"required": [{"name": "Python"}, {"name": "Java"}], "preferred": [{"name": "SQL"}]}}}).model_dump(mode="json"))
    before = json.loads(json.dumps(envelope))
    issues, _ = (_cv_skills if kind == "resume" else _jd_skills)(envelope, source, text, {"python": "Python", "sql": "SQL", "java": "Java"})
    envelope["document"]["schema_version"] = "resume-v2" if kind == "resume" else "job-v2"
    envelope["annotation"] = {"schema_version": "ats-skills-v2", "source_id": id, "source_path": source["source_path"], "source_sha256": source["source_sha256"], "text_sha256": source["text_sha256"], "review_status": "pending", "envelope_review_status": "pending", "skills_status": "complete"}
    return {"id": id, "kind": kind, "domain": "software_engineer", "split": split, "source": source, "before": before, "envelope": envelope,
            "issues": issues, "review_status": "pending", "skills_review_status": "pending", "full_envelope_review_status": "pending", "skills_status": "complete"}


def corpus(tmp_path, rows=None):
    value = refresh_identity({"version": CORPUS_VERSION, "documents": rows or [document(tmp_path)], "relations": {"version": "ats-relations-v2", "relations": []}, "review_history": []})
    path = tmp_path / "corpus.json"
    atomic_json(path, value)
    return value, path


def decision(item, action="accept", reviewer="teammate", **extra):
    return {**{key: item.get(key) for key in ("item_id", "source_sha256", "text_sha256", "document_checksum", "proposal_sha256", "item_sha256")}, "decision": action, "reviewer": reviewer, **extra}


def import_decisions(tmp_path, path, rows):
    decisions_path = tmp_path / "decisions.json"
    atomic_json(decisions_path, {"decisions": rows})
    return import_review(path, decisions_path, tmp_path)


def test_explicit_skill_local_proficiency_and_alias_source(tmp_path):
    row = document(tmp_path)
    details = row["envelope"]["document"]["skills"][0]["details"]
    assert {item["name"]: item["proficiency"] for item in details} == {"python": "expert", "sql": "normal"}
    assert details[0]["proficiency_quote"] == "Python: Expert"
    assert validate_corpus(corpus(tmp_path)[0], tmp_path)["valid"]


def test_group_level_and_seniority_never_supply_proficiency(tmp_path):
    row = document(tmp_path)
    env = row["before"]
    env["document"]["skills"][0]["level"] = "Expert"
    _cv_skills(env, row["source"], "Senior developer\nSkills\nExpert in Python and SQL", {"python": "Python", "sql": "SQL"})
    assert all(item["proficiency"] == "normal" for item in env["document"]["skills"][0]["details"])


def test_jd_core_quote_simple_or_and_preferred(tmp_path):
    row = document(tmp_path, id="jd_a", kind="job")
    requirements = row["envelope"]["document"]["skills"]["requirements"]
    core = next(item for item in requirements if item["importance"] == "core")
    assert core["options"] == ["java", "python"]
    assert core["importance_quote"] == "Must have Python or Java"
    assert next(item for item in requirements if item["importance"] == "preferred")["options"] == ["sql"]


def test_required_bucket_alone_is_standard(tmp_path):
    row = document(tmp_path, id="jd_a", kind="job")
    _jd_skills(row["envelope"], row["source"], "Requirements:\nPython\nJava\nSQL\n", {})
    assert all(item["importance"] == "standard" for item in row["envelope"]["document"]["skills"]["requirements"])


def test_optional_clause_does_not_inherit_required_word_from_previous_clause(tmp_path):
    row = document(tmp_path, id="jd_a", kind="job")
    _jd_skills(row["envelope"], row["source"], "Requirements:\nPython is required — Java or SQL is a strong plus\n", {})
    requirements = row["envelope"]["document"]["skills"]["requirements"]
    assert next(item for item in requirements if item["options"] == ["python"])["importance"] == "core"
    optional = next(item for item in requirements if "java" in item["options"])
    assert optional["importance"] == "preferred"
    assert "required" not in optional["quote"]


def test_same_clause_with_mixed_importance_needs_review(tmp_path):
    row = document(tmp_path, id="jd_a", kind="job")
    issues, _ = _jd_skills(row["envelope"], row["source"], "Requirements:\nPython is required, Java is preferred\n", {})
    assert any(item["code"] == "conflicting_importance" for item in issues)
    assert all(item["importance"] == "standard" for item in row["envelope"]["document"]["skills"]["requirements"])


def test_xlsx_review_import_preserves_pending_and_checks_identity(tmp_path):
    import zipfile
    from xml.sax.saxutils import escape
    value, path = corpus(tmp_path)
    item = review_items(value)[0]
    headers = ["item_id", "document_id", "decision", "reviewer", "reason", "replacement", "source_sha256", "text_sha256", "document_checksum", "proposal_sha256", "item_sha256"]
    record = decision(item, "pending", reviewer="")
    def cell(column, number, text):
        return f'<c r="{column}{number}" t="inlineStr"><is><t>{escape(str(text or ""))}</t></is></c>'
    sheet = '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
    sheet += '<row r="2">' + cell("A", 2, "corpus_identity_sha256") + cell("B", 2, value["identity_sha256"]) + '</row>'
    sheet += '<row r="5">' + ''.join(cell(chr(65+i),5, header) for i,header in enumerate(headers)) + '</row>'
    sheet += '<row r="6">' + ''.join(cell(chr(65+i),6, record.get(header)) for i,header in enumerate(headers)) + '</row>'
    sheet += '</sheetData></worksheet>'
    workbook = tmp_path / "review.xlsx"
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="CV proficiency" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    assert import_review(path, workbook, tmp_path)["applied"] == 0


def test_blank_pending_and_skills_only_not_training_gold(tmp_path):
    value, path = corpus(tmp_path)
    item = next(item for item in review_items(value) if item["item_id"].endswith(":skills-review"))
    assert import_decisions(tmp_path, path, [decision(item, "", reviewer="")])["applied"] == 0
    assert import_decisions(tmp_path, path, [decision(item)])["applied"] == 1
    updated = json.loads(path.read_text())
    assert updated["documents"][0]["skills_review_status"] == "approved"
    assert updated["documents"][0]["full_envelope_review_status"] == "pending"
    assert export_training(updated, tmp_path / "training.json")["exported"] == 0


def test_import_idempotent_and_stale_transaction_does_not_write(tmp_path):
    value, path = corpus(tmp_path)
    items = review_items(value)
    finalized = decision(items[0])
    assert import_decisions(tmp_path, path, [finalized])["applied"] == 1
    assert import_decisions(tmp_path, path, [finalized])["idempotent"]
    previous = path.read_bytes()
    finalized["source_sha256"] = "stale"
    with pytest.raises(ValueError, match="Stale"):
        import_decisions(tmp_path, path, [finalized])
    assert path.read_bytes() == previous


def test_validation_of_all_decisions_is_atomic_and_reviewer_required(tmp_path):
    value, path = corpus(tmp_path)
    items = review_items(value)
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="reviewer"):
        import_decisions(tmp_path, path, [decision(items[0]), decision(items[1], reviewer="")])
    assert path.read_bytes() == previous
    edit = decision(items[0], "edit", replacement={**items[0]["proposal"], "skill_quote": "Fabricated"})
    with pytest.raises(ValueError, match="validation"):
        import_decisions(tmp_path, path, [edit])
    assert path.read_bytes() == previous


def test_full_approval_only_exports_train_and_checks_source_checksum(tmp_path):
    value, path = corpus(tmp_path, [document(tmp_path), document(tmp_path, id="cv_b", split="validation")])
    items = [item for item in review_items(value) if item["item_id"].endswith(":full-envelope")]
    import_decisions(tmp_path, path, [decision(item) for item in items])
    updated = json.loads(path.read_text())
    assert export_training(updated, tmp_path / "training.json")["exported"] == 1
    (tmp_path / "cv_a.txt").write_text("changed", encoding="utf-8")
    assert not validate_corpus(updated, tmp_path)["valid"]


def test_export_has_stable_identity_and_empty_choices_table(tmp_path):
    value, _ = corpus(tmp_path)
    result = export_review(value, tmp_path / "review")
    package = json.loads((tmp_path / "review/review_package.json").read_text())
    assert package["corpus_identity_sha256"] == value["identity_sha256"]
    assert result["items"] == len(review_items(value))
    assert (tmp_path / "review/choices.csv").is_file()


def test_relation_proposals_training_only_bounded_and_never_approved(tmp_path):
    rows = [document(tmp_path), document(tmp_path, id="cv_b", split="validation")]
    rows[1]["envelope"]["document"]["skills"][0]["keywords"] = ["HeldoutSkill"]
    value, _ = corpus(tmp_path, rows)
    class Embedder:
        def embed_batch(self, names):
            assert "heldoutskill" not in names
            return np.eye(len(names))
    proposed = propose_relations(value, model_name="BAAI/bge-m3", revision="frozen", top_k=5, embedder=Embedder())
    assert proposed["proposal_metadata"]["vocabulary_partition"] == "train"
    assert all(item["review_status"] == "pending" for item in proposed["relations"])
    assert max(Counter(item["candidate"] for item in proposed["relations"]).values()) <= 5
