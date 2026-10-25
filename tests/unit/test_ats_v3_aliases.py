"""D3-31 approved equivalences, evidence linking and legacy isolation."""
from copy import deepcopy
import hashlib
import json

import pytest
from pydantic import ValidationError

from src.parser.skill_v2 import SkillRelation
from src.parser.skill_v3 import RequirementOption, RequirementUnitV3
from src.scoring import ScorerConfig, score_pair
from src.scoring.extraction import ExtractionCache
from src.scoring.normalization import SKILL_ALIASES, skill_name as legacy_name
from src.scoring.skills_v2 import score_skills_v2
from src.scoring.skills_v3 import score_skills_v3
from src.scoring.v3_aliases import (ALIAS_POLICY_VERSION, APPROVED_ALIAS_RECORDS,
    APPROVED_ALIASES_SHA256, alias_provenance, name_in_quote, skill_name)
from src.scoring.v3_validation import direct_level, validate_skill_evidence
from tests.unit.test_ats_v3_contrast_set import (AS_OF, Candidate, Fixture,
    Requirement, build_fixture, requirement)


def pair(candidate="Golang", target="Go", level="intermediate", minimum="intermediate", importance="core"):
    return build_fixture(Fixture((requirement(target, minimum, importance),),
                                 (Candidate(candidate, level),), {}))


@pytest.mark.parametrize("candidate,target", [("Golang", "Go"), ("Go", "Golang"), (" GOLANG ", "go")])
def test_alias_exact_depth_and_human_provenance(candidate, target):
    cv, jd, config = pair(candidate, target)
    result = score_skills_v3(cv, jd, config)
    row = result["requirements"][0]
    assert result["supported"] and result["score"] == 100
    assert result["core_gate_fulfillment"] == 1 and result["gate_status"] == "passed"
    assert row["exact"] and row["match_type"] == "alias"
    assert row["depth_status"] == "meets_minimum" and row["depth_gap"] == 0
    assert row["alias_approval"]["decision_id"] == "D3-31"
    assert row["alias_approval"]["approver"] == "user"
    assert candidate.strip() in row["evidence"]["proficiency_quote"]
    assert score_pair(cv, jd, config=config, as_of_date=AS_OF).scores.skills == result["score"]


@pytest.mark.parametrize("level,minimum,score,status,bonus,gate", [
    ("normal", "expert", 100, "unknown_candidate_depth", 0, 1),
    ("intermediate", "expert", 49, "explicit_gap", 0, .5),
    ("advanced", "expert", 75, "explicit_gap", 0, .75),
    ("advanced", "advanced", 100, "meets_minimum", 2.5, 1),
    ("expert", "expert", 100, "meets_minimum", 5, 1),
])
def test_alias_obeys_depth_bonus_and_gate(level, minimum, score, status, bonus, gate):
    cv, jd, config = pair(level=level, minimum=minimum)
    result = score_skills_v3(cv, jd, config)
    assert result["score"] == score and result["bonus"] == bonus
    assert result["core_gate_fulfillment"] == gate
    assert result["requirements"][0]["depth_status"] == status


@pytest.mark.parametrize("status", ["pending", "rejected"])
@pytest.mark.parametrize("kind", ["near", "transfer"])
def test_unapproved_relation_has_no_v3_credit_even_with_legacy_flag(status, kind):
    cv, jd, _ = pair("Python", "Java", "expert", "expert")
    relation = SkillRelation(id="proposal", candidate="Python", requirement="Java",
        relation=kind, reason="Illustrative proposal", source="test", review_status=status)
    config = ScorerConfig(version="ats-en-v3", relations=[relation], allow_provisional_relations=True)
    result = score_skills_v3(cv, jd, config)
    assert result["score"] == 0 and result["core_gate_fulfillment"] == 0
    assert result["requirements"][0]["match_type"] == "none"


@pytest.mark.parametrize("kind,coefficient", [("near", .7), ("transfer", .4)])
def test_approved_relation_keeps_base_credit_without_gate_or_depth_or_bonus(kind, coefficient):
    cv, jd, _ = pair("Python", "Java", "expert", "expert")
    relation = SkillRelation(id="approved", candidate="Python", requirement="Java",
        relation=kind, reason="Controlled fixture", source="test", review_status="approved")
    result = score_skills_v3(cv, jd, ScorerConfig(version="ats-en-v3", relations=[relation]))
    row = result["requirements"][0]
    assert result["base_skills"] == 100 * coefficient
    assert result["core_gate_fulfillment"] == 0 and result["force_mismatch"]
    assert row["fulfillment"] == coefficient and not row["exact"]
    assert row["depth_status"] == "related_skill_depth_unknown" and row["depth_gap"] is None
    assert result["bonus"] == 0 and row["alias_approval"] is None


def test_v1_v2_alias_and_pending_relation_behavior_are_unchanged():
    assert "golang" not in SKILL_ALIASES
    assert legacy_name("Golang") == "golang" and legacy_name("Go") == "go"
    cv, jd, _ = pair()
    assert score_skills_v2(cv, jd, ScorerConfig(version="ats-en-v2"))["score"] == 0
    # V1 reads the legacy required list, not v3 requirement units.
    legacy_jd = jd.model_dump(mode="json")
    legacy_jd["document"]["skills"]["required"] = [{"name": "Go", "proficiency": "required"}]
    assert score_pair(cv, type(jd).model_validate(legacy_jd),
        config=ScorerConfig(version="ats-en-v1"), as_of_date=AS_OF).scores.skills == 0
    cv, jd, _ = pair("Python", "Java", "expert", "expert")
    relation = SkillRelation(id="pending", candidate="Python", requirement="Java", relation="near",
        reason="Historical opt-in fixture", source="test", review_status="pending")
    old = score_skills_v2(cv, jd, ScorerConfig(version="ats-en-v2", relations=[relation], allow_provisional_relations=True))
    assert old["base_skills"] == 70 and old["core_gate_fulfillment"] == 0


def test_alias_source_linking_preserves_quote_and_rejects_unrelated_or_missing_source():
    cv, jd, _ = pair(level="advanced", minimum="advanced")
    detail = cv.document.skills[0].details[0]
    detail.name = "Go"
    cv.document.skills[0].keywords = ["Go"]
    unit = jd.document.skills.requirements[0]
    unit.quote = unit.importance_quote = "Advanced Golang required"
    unit.option_details[0].proficiency_quote = "Advanced Golang"
    text = "Golang\nAdvanced Golang required"
    assert validate_skill_evidence(cv, text, strict=False) == []
    assert validate_skill_evidence(jd, text, strict=False) == []
    assert detail.skill_quote == "Golang" and detail.proficiency_quote == "Advanced Golang"
    assert direct_level("Go", "Advanced Golang") == "advanced"
    assert direct_level("Golang", "Advanced Go") == "advanced"
    assert not name_in_quote("Go", "Golf") and not name_in_quote("Golang", "Django")
    assert any(i["reason"] == "quote_not_in_source" for i in validate_skill_evidence(cv, "unrelated", strict=False))
    detail.skill_quote = "Golf"
    assert any(i["reason"] == "quote_not_linked_to_skill" for i in validate_skill_evidence(cv, "Golf\nAdvanced Golang", strict=False))


def test_option_detail_identity_accepts_only_approved_equivalences():
    unit = RequirementUnitV3(id="go", options=["Golang"], quote="Go experience",
        option_details=[RequirementOption(name="Go")])
    assert unit.option_details[0].name == "Go"
    with pytest.raises(ValidationError):
        RequirementUnitV3(id="go", options=["Golang"], quote="Go experience",
            option_details=[RequirementOption(name="Golf")])


def test_duplicate_name_cannot_hide_an_unlinked_quote():
    cv, _, _ = pair(level="normal", minimum="normal")
    invalid = deepcopy(cv.document.skills[0].details[0])
    invalid.skill_quote = "Golf"
    # Both legacy issues share a path; a later valid record must not suppress
    # the earlier unlinked record's error when extending alias recognition.
    cv.document.skills[0].details.insert(0, invalid)
    issues = validate_skill_evidence(cv, "Golf\nGolang", strict=False)
    assert any(i["reason"] == "quote_not_linked_to_skill" for i in issues)


def test_alias_duplicates_use_candidate_once_and_allocate_deterministically():
    fixture = Fixture((requirement("Go", "intermediate"), requirement("Golang", "intermediate")),
                      (Candidate("Go", "intermediate"), Candidate("Golang", "intermediate")), {})
    cv, jd, config = build_fixture(fixture)
    first = score_skills_v3(cv, jd, config)
    assert first["supported"] and len(first["requirements"]) == 1
    assert first["score"] == 100 and len(first["merges"]) == 2
    cv.document.skills[0].details.reverse()
    jd.document.skills.requirements.reverse()
    assert score_skills_v3(cv, jd, config)["requirements"] == first["requirements"]


def test_conflicting_alias_levels_and_duplicate_or_options_require_review():
    cv, jd, _ = pair()
    other = deepcopy(cv.document.skills[0].details[0])
    other.name = "Go"
    other.proficiency = "expert"
    other.raw_proficiency = "Expert"
    other.skill_quote = "Go"
    other.proficiency_quote = "Expert Go"
    cv.document.skills[0].details.append(other)
    cv.document.skills[0].keywords.append("Go")
    assert any(i["reason"] == "conflicting_explicit_proficiency" for i in validate_skill_evidence(
        cv, "Golang\nIntermediate Golang\nGo\nExpert Go", strict=False))
    unit = jd.document.skills.requirements[0]
    unit.options = ["Go", "Golang"]
    unit.quote = unit.importance_quote = "Go or Golang required"
    assert any(i["reason"] == "duplicate_alternatives" for i in validate_skill_evidence(
        jd, "Go or Golang required\nIntermediate Go", strict=False))


def test_alias_identity_and_extraction_cache_are_versioned_only_for_v3(tmp_path):
    canonical = json.dumps(APPROVED_ALIAS_RECORDS, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == APPROVED_ALIASES_SHA256
    cv, jd, config = pair()
    identity = score_skills_v3(cv, jd, config)["identity"]
    assert identity["alias_policy_version"] == ALIAS_POLICY_VERSION
    assert identity["approved_aliases_sha256"] == APPROVED_ALIASES_SHA256
    for ruleset in ("v1", "v2", "v3"):
        cache = ExtractionCache(tmp_path / ruleset, None, "fixture", "fixture", ruleset=ruleset)
        assert ("approved_aliases_sha256" in cache.identity) == (ruleset == "v3")
    assert skill_name("Golang") == skill_name("Go")
    assert skill_name("JavaScript") != skill_name("Java")
    assert alias_provenance("Golang", "Go")["review_kind"] == "direct_human"
    assert alias_provenance("Python", "Java") is None


def test_hyphen_at_line_break_links_both_readings():
    assert name_in_quote("Microsoft Office", "Micro-\nsoft Office")  # PDF word-wrap
    assert name_in_quote("e-commerce", "e-\ncommerce")  # real hyphen
    assert name_in_quote("Microsoft", "Micro- \r\n soft")
    assert not name_in_quote("Microsoft", "Micro- soft")  # no line break: not a wrap
    assert not name_in_quote("Golf", "Gol-\nfer")


def test_pdf_marks_inside_a_name_still_link():
    assert name_in_quote("Microsoft Office", "Proficiency in Microsoft \u00c2\u200b Office")
    assert name_in_quote("Excel", "MS Ex\u00adcel")  # soft hyphen
    assert name_in_quote("fault tolerance", "software testing and fault toler\nance")  # unhyphenated word-wrap
    assert name_in_quote("Project Management", "Project\nManagement")
    assert name_in_quote("Python", "Python\nDjango")  # a real line break between words still links
    assert not name_in_quote("Golf", "Gol\n fer")
