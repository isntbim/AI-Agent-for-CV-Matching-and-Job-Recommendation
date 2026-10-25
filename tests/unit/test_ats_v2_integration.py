"""Cross-module source, extraction, cache and gate behavior."""

import hashlib
import json
from datetime import date

import pytest
from pydantic import ValidationError

from src.parser.llm_extractor import ExtractionError
from src.scoring import JobInput, ResumeInput, ScorerConfig, score_pair
from src.scoring.evaluation import calibrate, evaluate
from src.scoring.extraction import ExtractionCache, _prompts, extract_document
from src.scoring.v2_validation import validate_skill_evidence

AS_OF = date(2026, 9, 30)
CV_TEXT = "Software Engineer\nPython (Expert)\nSQL"
JD_TEXT = "Software Engineer\nPython is required. SQL optional."


def cv_payload():
    return {"document": {"schema_version": "resume-v2", "basics": {"label": "Software Engineer"},
        "skills": [{"name": "Technical", "keywords": ["Python", "SQL"], "details": [
            {"name": "Python", "proficiency": "expert", "raw_proficiency": "Expert",
             "skill_quote": "Python", "proficiency_quote": "Python (Expert)"},
            {"name": "SQL", "skill_quote": "SQL"}]}]},
        "facts": {"evidence": {"skills": [{"quote": "Python"}], "title": [{"quote": "Software Engineer"}]}}}


def jd_payload():
    return {"document": {"schema_version": "job-v2", "id": "jd-1", "url": "urn:test", "scraped_at": "none",
        "title": "Software Engineer", "title_normalized": "software_engineer", "category": "software",
        "skills": {"required": [{"name": "Python"}], "preferred": [{"name": "SQL", "proficiency": "preferred"}],
            "requirements": [{"id": "python", "options": ["Python"], "importance": "core",
                "quote": "Python is required.", "importance_quote": "Python is required."},
                {"id": "sql", "options": ["SQL"], "importance": "preferred", "quote": "SQL optional."}]}},
        "facts": {"evidence": {"skills": [{"quote": "Python"}], "title": [{"quote": "Software Engineer"}]}}}


class Client:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def generate(self, system_prompt, user_prompt, temperature=0, max_tokens=8192):
        self.calls += 1
        return json.dumps(self.payload)


def annotated(payload, cls, text):
    return cls.model_validate({**payload, "annotation": {"skills_status": "complete", "source_id": "test",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()}})


def test_v1_defaults_and_v2_are_explicit():
    cv, jd = ResumeInput.model_validate(cv_payload()), JobInput.model_validate(jd_payload())
    default = score_pair(cv, jd, as_of_date=AS_OF)
    assert default == score_pair(cv, jd, config=ScorerConfig(version="ats-en-v1"), as_of_date=AS_OF)
    assert default.scores.skills == 100 and default.skills_breakdown is None
    missing = score_pair(cv, jd, config=ScorerConfig(version="ats-en-v2"), as_of_date=AS_OF)
    assert missing.scores.skills == 50 and missing.skills_breakdown["gate_status"] == "unknown"


def test_v2_extraction_marks_provisional_source_backed_metadata():
    cv = extract_document(CV_TEXT, "resume", Client(cv_payload()), ruleset="v2")
    jd = extract_document(JD_TEXT, "job", Client(jd_payload()), ruleset="v2")
    assert cv.annotation.skills_status == jd.annotation.skills_status == "complete"
    assert cv.annotation.review_status == "pending"
    result = score_pair(cv, jd, config=ScorerConfig(version="ats-en-v2"), as_of_date=AS_OF)
    assert result.scores.skills == 100 and result.skills_breakdown["gate_status"] == "passed"
    assert result.skills_breakdown["bonus"] == pytest.approx(5)
    assert result.skills_breakdown["core_gate_fulfillment"] == 1


def test_v2_cache_identity_is_separate_and_literal_provenance_is_checked(tmp_path):
    client = Client(cv_payload())
    old = ExtractionCache(tmp_path, client, "test", "model")
    new = ExtractionCache(tmp_path, client, "test", "model", ruleset="v2")
    assert old.identity != new.identity and old.identity["schema"] == "ats-envelope-v1"
    first = new.get(CV_TEXT, "resume")
    assert new.get(CV_TEXT, "resume") == first and client.calls == 1
    entry = next(path for path in tmp_path.glob("*.json") if not path.name.endswith(".error.json"))
    data = json.loads(entry.read_text())
    data["result"]["document"]["skills"][0]["details"][0]["skill_quote"] = "Ruby"
    entry.write_text(json.dumps(data))
    with pytest.raises(ExtractionError, match="quote_not_in_source|quote_not_linked"):
        new.get(CV_TEXT, "resume")


def test_v1_prompt_does_not_gain_v2_fields_and_v2_prompt_carries_policy():
    old, old_source = _prompts("resume", CV_TEXT)
    assert "CVSkillDetail" not in old and "schema_version" not in old
    assert "extraction_ruleset" not in json.loads(old_source)
    new, source = _prompts("resume", CV_TEXT, "v2")
    assert "CVSkillDetail" in new and "group of skills" in new
    assert json.loads(source)["extraction_ruleset"] == "v2"


def test_keyword_details_and_explicit_levels_are_structural_contracts():
    payload = cv_payload()
    payload["document"]["skills"][0]["keywords"] = ["Python"]
    with pytest.raises(ValidationError, match="keywords"):
        ResumeInput.model_validate(payload)
    payload = cv_payload()
    payload["document"]["skills"][0]["details"][0]["raw_proficiency"] = None
    with pytest.raises(ValidationError, match="Non-normal"):
        ResumeInput.model_validate(payload)


def test_quote_existence_does_not_validate_unrelated_skill_or_core_meaning():
    payload = jd_payload()
    payload["document"]["skills"]["requirements"][0]["importance_quote"] = "Software Engineer"
    job = annotated(payload, JobInput, JD_TEXT)
    with pytest.raises(ExtractionError, match="quote_not_linked|core_without"):
        validate_skill_evidence(job, JD_TEXT)
    resume = annotated(cv_payload(), ResumeInput, CV_TEXT)
    resume.document.skills[0].details[0].skill_quote = "SQL"
    with pytest.raises(ExtractionError, match="quote_not_linked"):
        validate_skill_evidence(resume, CV_TEXT)


def test_unknown_wording_stays_review_and_group_level_cannot_be_applied():
    payload = cv_payload()
    detail = payload["document"]["skills"][0]["details"][0]
    detail.update(proficiency="normal", raw_proficiency="Proficient", proficiency_quote="Proficient in Python")
    text = CV_TEXT + "\nProficient in Python"
    resume = annotated(payload, ResumeInput, text)
    assert any(row["reason"] == "unmapped_proficiency_wording" for row in validate_skill_evidence(resume, text))
    text = "Expert in Python and SQL"
    payload = cv_payload()
    payload["document"]["skills"][0]["details"][0]["proficiency_quote"] = text
    resume = annotated(payload, ResumeInput, text)
    assert any(row["reason"] == "group_proficiency_is_not_individual_evidence"
               for row in validate_skill_evidence(resume, text))


def test_runtime_calibration_evaluation_use_the_same_gate():
    config = ScorerConfig(version="ats-en-v2")
    rows = [{"split": "development", "label_id": label, "pair_id": str(label), "cv_id": str(label),
        "baseline_skill_score": value, "result": {"config": config.model_dump(), "overall_score": value,
            "scores": {"skills": value, "title": value}, "coverage": 1,
            "dimensions": {"skills": {"supported": True}},
            "skills_breakdown": {"force_mismatch": label == 0, "strong_blocked": label == 0}}}
        for label, value in [(0, 95), (1, 60), (2, 90)]]
    calibrated = calibrate(rows)
    assert calibrated.version == "ats-en-v2"
    test = [{**row, "split": "test"} for row in rows]
    report = evaluate(test, calibrated, ScorerConfig(), 0, bootstrap=0)
    assert report["scorer"]["macro_f1"] == 1
    rows[1]["result"]["config"]["version"] = "ats-en-v1"
    with pytest.raises(ValueError, match="different scorer configurations"):
        calibrate(rows)


def test_runtime_rejects_unrelated_core_and_unmapped_individual_wording():
    cv = annotated(cv_payload(), ResumeInput, CV_TEXT)
    jd = annotated(jd_payload(), JobInput, JD_TEXT)
    jd.document.skills.requirements[0].importance_quote = "Must have Software Engineer"
    result = score_pair(cv, jd, config=ScorerConfig(version="ats-en-v2"), as_of_date=AS_OF)
    assert result.scores.skills == 50 and result.skills_breakdown["gate_status"] == "unknown"
    assert all(row["status"] == "unknown" for row in result.skills_breakdown["requirements"])
    assert result.skills_breakdown["unmet_requirements"] == []
    jd = annotated(jd_payload(), JobInput, JD_TEXT)
    detail = cv.document.skills[0].details[0]
    detail.proficiency, detail.raw_proficiency = "normal", "Proficient"
    result = score_pair(cv, jd, config=ScorerConfig(version="ats-en-v2"), as_of_date=AS_OF)
    assert result.skills_breakdown["gate_status"] == "unknown" and result.skills_breakdown["bonus"] == 0


def test_source_validation_accepts_reviewed_neutral_proficiency_wording():
    text = CV_TEXT.replace("Python (Expert)", "Python (Proficient)")
    payload = cv_payload()
    detail = payload["document"]["skills"][0]["details"][0]
    detail.update(proficiency="normal", raw_proficiency="Proficient", proficiency_quote="Python (Proficient)")
    resume = annotated(payload, ResumeInput, text)
    assert any(issue["reason"] == "unmapped_proficiency_wording" for issue in validate_skill_evidence(resume, text))
    resume.annotation.review_status = "approved"
    assert not validate_skill_evidence(resume, text)
    result = score_pair(resume, annotated(jd_payload(), JobInput, JD_TEXT), config=ScorerConfig(version="ats-en-v2"), as_of_date=AS_OF)
    assert result.scores.skills == 100 and result.skills_breakdown["bonus"] == 0


@pytest.mark.parametrize("recorded_newlines", ["raw", "lf"])
def test_one_pair_cli_preserves_crlf_provenance_and_excludes_unapproved_relations(tmp_path, recorded_newlines):
    from scripts.ats_score import main
    cv_text, jd_text = CV_TEXT.replace("\n", "\r\n"), JD_TEXT.replace("\n", "\r\n")
    cv = annotated(cv_payload(), ResumeInput, cv_text)
    jd = annotated(jd_payload(), JobInput, jd_text)
    if recorded_newlines == "lf":
        cv.annotation.text_sha256 = hashlib.sha256(CV_TEXT.encode()).hexdigest()
        jd.annotation.text_sha256 = hashlib.sha256(JD_TEXT.encode()).hexdigest()
    for filename, envelope in (("cv.json", cv), ("jd.json", jd)):
        (tmp_path / filename).write_text(envelope.model_dump_json(), encoding="utf-8")
    (tmp_path / "cv.txt").write_bytes(cv_text.encode())
    (tmp_path / "jd.txt").write_bytes(jd_text.encode())
    (tmp_path / "relations.json").write_text(json.dumps({"version": "draft-test", "relations": [{
        "id": "draft", "candidate": "Python", "requirement": "SQL", "relation": "near",
        "reason": "Unreviewed example", "source": "test", "review_status": "pending"}]}), encoding="utf-8")
    result = main(["--resume", str(tmp_path / "cv.json"), "--job", str(tmp_path / "jd.json"),
        "--ruleset", "v2", "--as-of", AS_OF.isoformat(), "--relations", str(tmp_path / "relations.json"),
        "--resume-source", str(tmp_path / "cv.txt"), "--job-source", str(tmp_path / "jd.txt"),
        "--output", str(tmp_path / "result.json")])
    assert result["scores"]["skills"] == 100 and result["skills_breakdown"]["gate_status"] == "passed"
    assert result["config"]["relations"] == []
