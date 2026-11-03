"""Depth policy contracts, allocation and historical arithmetic compatibility."""
from copy import deepcopy
from datetime import date
import json
import pytest
from src.scoring import ResumeInput, JobInput, ScorerConfig, score_pair
from src.scoring.extraction import ExtractionCache, _prompts
from src.scoring.models import runtime_grade
from src.scoring.skills_v2 import score_skills_v2
from src.scoring.skills_v3 import score_skills_v3
from src.scoring.v3_validation import direct_level, validate_skill_evidence
from src.parser.skill_v2 import SkillRelation
from src.parser.skill_v3 import RequirementOption, RequirementUnitV3
from src.scoring.evaluation import evaluate

DATE = date(2026, 9, 30)
LEVELS = ["intermediate", "advanced", "expert"]


def pair(level="advanced", minimum="expert", importance="core", options=None, details=None):
    options = options or ["Python"]
    names = ["Python"]
    cv = ResumeInput.model_validate({"document": {"schema_version": "resume-v3", "basics": {"label": "Software Engineer"},
        "skills": [{"name": "Technical", "keywords": names, "details": [{"name": "Python", "proficiency": level,
            "raw_proficiency": level if level != "normal" else None, "skill_quote": "Python",
            "proficiency_quote": level + " Python" if level != "normal" else None}]}]},
        "facts": {"evidence": {"skills": [{"quote": "Python"}], "title": [{"quote": "Software Engineer"}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    quote = "Python is required" if len(options) == 1 else "Python or SQL is required"
    jd = JobInput.model_validate({"document": {"schema_version": "job-v3", "id": "jd", "url": "urn:jd", "scraped_at": "none",
        "title": "Software Engineer", "title_normalized": "software_engineer", "category": "software_engineer",
        "skills": {"requirements": [{"id": "r1", "options": options, "importance": importance, "quote": quote,
            "importance_quote": quote if importance == "core" else None, "review_status": "approved",
            "option_details": details if details is not None else [{"name": "Python", "minimum_proficiency": minimum,
                "raw_proficiency": minimum if minimum != "normal" else None,
                "proficiency_quote": minimum + " Python" if minimum != "normal" else None}]}]}},
        "facts": {"evidence": {"skills": [{"quote": "Python"}], "title": [{"quote": "Software Engineer"}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    return cv, jd


def config(policy="full", **kwargs):
    return ScorerConfig(version="ats-en-v3", depth_policy=policy, **kwargs)


@pytest.mark.parametrize("candidate", LEVELS)
@pytest.mark.parametrize("required", LEVELS)
def test_every_explicit_level_gap(candidate, required):
    cv, jd = pair(candidate, required)
    result = score_skills_v3(cv, jd, config())
    assert result["supported"]
    row = result["requirements"][0]
    gap = max(0, LEVELS.index(required) - LEVELS.index(candidate))
    credit = 1 - .25 * gap
    assert row["depth_credit"] == credit
    assert row["fulfillment"] == credit
    assert row["depth_gap"] == gap
    assert row["bonus_eligible"] == (gap == 0 and candidate in {"advanced", "expert"})


def test_unknown_depth_retains_credit_and_gate():
    cv, jd = pair("normal", "expert")
    result = score_skills_v3(cv, jd, config())
    assert result["score"] == 100 and result["core_gate_fulfillment"] == 1
    row = result["requirements"][0]
    assert row["depth_status"] == "unknown_candidate_depth" and row["depth_gap"] is None
    assert not row["bonus_eligible"]


def test_unspecified_minimum_retains_advanced_bonus():
    cv, jd = pair("advanced", "normal", "standard")
    row = score_skills_v3(cv, jd, config())
    assert row["bonus"] == 2.5 and row["requirements"][0]["bonus_eligible"]


def test_policies_use_legacy_gate_and_bonus_only_in_score_only():
    cv, jd = pair("intermediate", "expert")
    full, score_only, off = [score_skills_v3(cv, jd, config(policy)) for policy in ("full", "score_only", "off")]
    assert full["core_gate_fulfillment"] == .5 and full["score"] == 49 and not full["force_mismatch"]
    assert score_only["base_skills"] == 50 and score_only["core_gate_fulfillment"] == 1 and score_only["score"] == 50
    assert off["score"] == 100
    cv, jd = pair("advanced", "expert", "standard")
    assert score_skills_v3(cv, jd, config("score_only"))["bonus"] == 2.5
    assert score_skills_v3(cv, jd, config())["bonus"] == 0


@pytest.mark.parametrize("fulfilled,expected_status,forced", [(4, "failed", True), (5, "failed", False), (6, "failed", False), (7, "passed", False)])
def test_unrounded_core_boundaries(fulfilled, expected_status, forced):
    cv, jd = pair("normal", "normal")
    seed = jd.document.skills.requirements[0].model_dump()
    units = []
    for i in range(10):
        name = f"Skill{i}"
        unit = {**seed, "id": str(i), "options": [name], "quote": name + " is required", "importance_quote": name + " is required", "option_details": []}
        units.append(RequirementUnitV3.model_validate(unit))
    jd.document.skills.requirements = units
    group = cv.document.skills[0]
    group.keywords = [f"Skill{i}" for i in range(fulfilled)]
    from src.parser.skill_v2 import CVSkillDetail
    group.details = [CVSkillDetail(name=name, skill_quote=name) for name in group.keywords]
    result = score_skills_v3(cv, jd, config())
    assert result["core_gate_fulfillment"] == fulfilled / 10
    assert result["gate_status"] == expected_status and result["force_mismatch"] == forced


def test_semantic_matches_never_establish_depth_or_enter_core_gate():
    cv, jd = pair("expert", "expert")
    unit = jd.document.skills.requirements[0]
    unit.options = ["Java"]
    unit.quote = unit.importance_quote = "Java is required"
    unit.option_details = [RequirementOption(name="Java", minimum_proficiency="expert", raw_proficiency="expert", proficiency_quote="expert Java")]
    relation = SkillRelation(id="related", candidate="Python", requirement="Java", relation="near", reason="test", source="test", review_status="approved")
    result = score_skills_v3(cv, jd, config(relations=[relation]))
    row = result["requirements"][0]
    assert row["fulfillment"] == .7 and row["depth_gap"] is None and row["depth_credit"] == 1
    assert row["depth_status"] == "related_skill_depth_unknown"
    assert result["core_gate_fulfillment"] == 0 and result["force_mismatch"] and result["bonus"] == 0


@pytest.mark.parametrize("policy", ["full", "score_only"])
@pytest.mark.parametrize("importance", ["core", "standard", "preferred"])
def test_or_option_selection_uses_allocation_objectives(policy, importance):
    cv, jd = pair("intermediate", "expert", importance, options=["Python", "SQL"])
    relation = SkillRelation(id="related", candidate="Python", requirement="SQL", relation="near",
                             reason="test", source="test", review_status="approved")
    result = score_skills_v3(cv, jd, config(policy, relations=[relation]))
    row = result["requirements"][0]
    assert result["supported"]
    assert row["selected_option"] == ("sql" if importance == "preferred" else "python")
    if importance == "core":
        assert result["core_gate_fulfillment"] == (.5 if policy == "full" else 1)
        assert not result["force_mismatch"]
    jd.document.skills.requirements[0].options.reverse()
    assert score_skills_v3(cv, jd, config(policy, relations=[relation]))["requirements"] == result["requirements"]


def test_depth_off_preserves_legacy_or_selection():
    cv, jd = pair("intermediate", "expert", options=["Python", "SQL"])
    relation = SkillRelation(id="related", candidate="Python", requirement="SQL", relation="near",
                             reason="test", source="test", review_status="approved")
    old = score_skills_v2(cv, jd, ScorerConfig(version="ats-en-v2", relations=[relation]))
    new = score_skills_v3(cv, jd, config("off", relations=[relation]))
    for key in ("score", "base_skills", "core_gate_fulfillment", "force_mismatch"):
        assert old[key] == new[key]


@pytest.mark.parametrize("quote", ["I am not proficient in Python", "Not advanced Python", "I lack advanced Python",
                                    "Python: not advanced", "I am not yet advanced Python", "I am not an expert in Python"])
def test_negative_depth_cannot_support_runtime_or_source(quote):
    assert direct_level("Python", quote) is None
    cv, _ = pair("advanced", "normal")
    cv.document.skills[0].details[0].proficiency_quote = quote
    assert not score_skills_v3(cv, pair()[1], config())["supported"]
    assert any(i["severity"] == "error" for i in validate_skill_evidence(cv, "Python. " + quote, strict=False))


@pytest.mark.parametrize("text", ["I am not advanced Python.", "I lack advanced Python.", "Trained others in advanced Python.",
                                   "I am not\nadvanced Python."])
def test_source_scope_cannot_be_removed_by_short_excerpt(text):
    from src.scoring.v3_corpus import _local_claim
    cv, jd = pair("advanced", "advanced")
    # A newline is potentially a wrapped sentence, not evidence of positive scope.
    assert _local_claim("Python", text) is None
    assert any(i["severity"] == "error" for i in validate_skill_evidence(cv, text, strict=False))
    assert any(i["severity"] == "error" for i in validate_skill_evidence(jd, "Python is required. " + text, strict=False))


def test_positive_depth_sentence_is_independent_of_unrelated_negative_sentence():
    from src.scoring.v3_corpus import _local_claim
    assert _local_claim("Python", "No Java experience. Advanced Python.") == ("advanced", "Advanced", "Advanced Python")


def test_or_alternatives_have_independent_minima():
    details = [{"name": "Python", "minimum_proficiency": "expert", "raw_proficiency": "expert", "proficiency_quote": "expert Python"},
               {"name": "SQL", "minimum_proficiency": "beginner", "raw_proficiency": "beginner", "proficiency_quote": "beginner SQL"}]
    cv, jd = pair("intermediate", "expert", options=["Python", "SQL"], details=details)
    from src.parser.skill_v2 import CVSkillDetail
    cv.document.skills[0].keywords.append("SQL")
    cv.document.skills[0].details.append(CVSkillDetail(name="SQL", skill_quote="SQL", proficiency="advanced", raw_proficiency="advanced", proficiency_quote="advanced SQL"))
    result = score_skills_v3(cv, jd, config())
    assert result["requirements"][0]["selected_option"] == "sql"
    assert result["requirements"][0]["required_level"] == "normal"
    assert result["score"] == 100


def test_allocation_uses_depth_deterministically_and_each_skill_once():
    cv, jd = pair("intermediate", "expert")
    first = jd.document.skills.requirements[0].model_dump()
    other = deepcopy(first)
    other.update(id="r2", importance="standard", importance_quote=None, options=["Python", "SQL"], quote="Python or SQL is required")
    jd.document.skills.requirements.append(RequirementUnitV3.model_validate(other))
    # Overlap is directly approved; the Core lexicographic objective wins.
    a = score_skills_v3(cv, jd, config())
    jd.document.skills.requirements.reverse()
    b = score_skills_v3(cv, jd, config())
    assert a["requirements"] == b["requirements"]
    assert sum(row["candidate_skill"] is not None for row in a["requirements"]) == 1


def test_duplicate_neutral_requirement_cannot_erase_explicit_minimum():
    cv, jd = pair("intermediate", "normal")
    explicit = deepcopy(jd.document.skills.requirements[0].model_dump())
    explicit["id"] = "z-explicit"
    explicit["option_details"][0].update(minimum_proficiency="expert", raw_proficiency="expert", proficiency_quote="expert Python")
    jd.document.skills.requirements.append(RequirementUnitV3.model_validate(explicit))
    result = score_skills_v3(cv, jd, config())
    assert len(result["requirements"]) == 1 and result["requirements"][0]["required_level"] == "expert"
    assert result["requirements"][0]["fulfillment"] == .5


def test_conflicting_duplicate_minima_require_review():
    cv, jd = pair("intermediate", "expert")
    conflict = deepcopy(jd.document.skills.requirements[0].model_dump())
    conflict["id"] = "conflict"
    conflict["option_details"][0].update(minimum_proficiency="advanced", raw_proficiency="advanced", proficiency_quote="advanced Python")
    jd.document.skills.requirements.append(RequirementUnitV3.model_validate(conflict))
    result = score_skills_v3(cv, jd, config())
    assert not result["supported"] and any("conflicting explicit minima" in warning for warning in result["warnings"])


@pytest.mark.parametrize("level", ["normal", *LEVELS])
def test_depth_off_preserves_v2_arithmetic(level):
    cv, jd = pair(level, "expert")
    old = score_skills_v2(cv, jd, ScorerConfig(version="ats-en-v2"))
    new = score_skills_v3(cv, jd, config("off"))
    for key in ("score", "base_skills", "bonus", "core_gate_fulfillment", "force_mismatch", "strong_blocked", "group_fulfillment"):
        assert old[key] == new[key]


def test_direct_wording_and_titles():
    assert direct_level("Python", "proficient in Python") == "intermediate"
    assert direct_level("accounting", "Trained others in proficient accounting") is None
    assert direct_level("Python", "Senior Python Engineer") is None
    cv, jd = pair("advanced", "expert")
    with pytest.raises(Exception, match="depth_quote_not_in_source"):
        validate_skill_evidence(jd, "Python is required")


def test_title_depth_cannot_establish_a_supported_minimum():
    cv, jd = pair("beginner", "expert", "standard")
    jd.document.skills.requirements[0].quote = "Title: Expert Python Engineer"
    result = score_skills_v3(cv, jd, config())
    assert not result["supported"] and result["gate_status"] == "unknown"
    assert result["requirements"][0]["depth_status"] == "unsupported_annotation_depth"


@pytest.mark.parametrize("status", ["approved", "pending", "rejected"])
def test_human_review_resolves_only_the_title_choice(status):
    cv, jd = pair("normal", "normal", "standard", options=["Python", "SQL"])
    unit = jd.document.skills.requirements[0]
    unit.quote = "Title: Engineer (Python/SQL)"
    unit.review_status = status
    result = score_skills_v3(cv, jd, config())
    assert result["supported"] == (status == "approved")
    issues = validate_skill_evidence(jd, unit.quote, strict=False)
    assert any(i["reason"] == "title_choice_or_depth_requires_body_review" for i in issues) == (status != "approved")
    if status == "approved":
        assert result["score"] == 100 and result["bonus"] == 0
        assert len(result["requirements"]) == 1


def test_cache_and_prompt_contract_are_versioned(tmp_path):
    identities = [ExtractionCache(tmp_path, None, "offline", "test", ruleset=ruleset).identity for ruleset in ("v1", "v2", "v3")]
    assert len({i["prompt"] for i in identities}) == len({i["schema"] for i in identities}) == 3
    assert identities[2]["validation_policy"] == "explicit-skill-depth-v3-no-beginner-v1"
    assert all("validation_policy" not in i for i in identities[:2])
    v3, _ = _prompts("job", "expert Python", "v3")
    assert "minimum_proficiency" in v3 and "job-v3" in v3
    v2, _ = _prompts("job", "expert Python", "v2")
    assert "minimum_proficiency" not in v2 and "job-v3" not in v2
    assert ScorerConfig().version == "ats-en-v1" and ScorerConfig(version="ats-en-v3").depth_policy == "full"
    with pytest.raises(ValueError, match="requires ats-en-v3"):
        ScorerConfig(version="ats-en-v2", depth_policy="full")


def test_runtime_grade_honors_v3_gate():
    assert runtime_grade(95, 95, 95, config(), {"force_mismatch": True}) == 0
    assert runtime_grade(95, 95, 95, config(), {"strong_blocked": True}) == 1


def test_runtime_and_evaluation_grades_agree():
    from src.scoring.evaluation import classification_metrics
    rows = []
    cfg = config()
    for label, level in enumerate(("beginner", "intermediate", "expert")):
        cv, jd = pair(level, "expert")
        result = score_pair(cv, jd, config=cfg, as_of_date=DATE).model_dump(mode="json")
        rows.append({"pair_id": str(label), "cv_id": str(label), "split": "test", "label_id": label,
            "baseline_skill_score": 50, "result": result})
    report = evaluate(rows, cfg, ScorerConfig(), 0, bootstrap=0)
    assert report["scorer"] == classification_metrics([0, 1, 2], [r["result"]["grade"] for r in rows])


def test_v3_extraction_and_cache_revalidate_actual_source(tmp_path):
    cv, _ = pair("advanced", "normal")
    payload = cv.model_dump(mode="json", exclude={"annotation"})
    class Client:
        def generate(self, *args, **kwargs):
            return json.dumps(payload)
    cache = ExtractionCache(tmp_path, Client(), "offline", "fixture", ruleset="v3", as_of="2026-10-25")
    text = "Software Engineer\nadvanced Python"
    result = cache.get(text, "resume")
    assert result.document.schema_version == "resume-v3" and result.annotation.schema_version == "ats-skills-v3"
    assert result.annotation.skills_status == "complete"
    assert cache.get(text, "resume").model_dump() == result.model_dump()


@pytest.mark.parametrize("ruleset,policy", [("v2", None), ("v3", "score_only")])
def test_existing_score_cli_supports_source_validation_and_v3(tmp_path, ruleset, policy):
    from scripts.ats_score import main
    cv, jd = pair("advanced", "expert")
    for name, envelope, text in (("resume", cv, "Software Engineer\nadvanced Python"), ("job", jd, "Software Engineer\nPython is required\nexpert Python")):
        (tmp_path / (name + ".json")).write_text(envelope.model_dump_json(), encoding="utf-8")
        (tmp_path / (name + ".txt")).write_text(text, encoding="utf-8")
    args = ["--resume", str(tmp_path / "resume.json"), "--job", str(tmp_path / "job.json"), "--ruleset", ruleset,
        "--resume-source", str(tmp_path / "resume.txt"), "--job-source", str(tmp_path / "job.txt"),
        "--as-of", "2026-09-30", "--output", str(tmp_path / "score.json")]
    if policy:
        args += ["--depth-policy", policy]
    result = main(args)
    assert result["config"]["version"] == "ats-en-" + ruleset
    if policy:
        assert result["config"]["depth_policy"] == policy and result["skills_breakdown"]["core_gate_fulfillment"] == 1


def test_automatic_review_policy_scores_through_review_warnings():
    cv, jd = pair("normal", "normal", "standard")
    cv.document.skills[0].details[0].raw_proficiency = "Proficient"  # unmapped wording: a review-level warning
    cv.annotation.skills_status = "partial"
    strict = score_skills_v3(cv, jd, config())
    auto = score_skills_v3(cv, jd, config(review_policy="automatic"))
    assert strict["basis"] == "missing_evidence" and strict["score"] == 50 and not strict["supported"]
    assert auto["supported"] and auto["score"] == 100 and auto["matched_skills"] == ["python"]
    assert auto["warnings"] == strict["warnings"] and auto["warnings"]  # still reported
    assert auto["identity"]["review_policy"] == "automatic"


def test_automatic_review_policy_still_blocks_rejected_requirements():
    cv, jd = pair("normal", "normal", "standard")
    jd.document.skills.requirements[0].review_status = "rejected"
    result = score_skills_v3(cv, jd, config(review_policy="automatic"))
    assert result["basis"] == "missing_evidence" and not result["supported"]
