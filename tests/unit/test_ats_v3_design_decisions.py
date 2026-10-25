"""Approved Q2/Q3/Q4/Q6/Q7 contracts, with independent boundary oracles."""
from copy import deepcopy
from datetime import date
import json

import pytest

from src.parser.skill_v2 import SkillRelation
from src.scoring import SalaryPreference, ScorerConfig, score_pair
from src.scoring.models import Evidence, runtime_grade
from src.scoring.models_v3 import display_score
from src.scoring.evaluation import _grade_values, evaluate
from src.scoring.skills_v3 import score_skills_v3
from src.scoring import runtime_v3
from tests.unit.test_ats_v3_contrast_set import (
    Candidate, Fixture, Requirement, build_fixture, requirement,
)

AS_OF = date(2026, 9, 30)


def fixture(candidate="Python", level="normal", minimum="normal", target="Python"):
    return build_fixture(Fixture((requirement(target, minimum),), (Candidate(candidate, level),), {}))


@pytest.mark.parametrize("minimum", ["normal", "intermediate", "advanced", "expert"])
@pytest.mark.parametrize("level", ["normal", "intermediate", "advanced", "expert"])
def test_q6_reporting_orthogonal_to_requirement_minimum(minimum, level):
    cv, jd, config = fixture(level=level, minimum=minimum)
    result = score_skills_v3(cv, jd, config)
    row = result["requirements"][0]
    assert row["candidate_depth_evidence_status"] == ("assumed_baseline" if level == "normal" else "stated_level")
    if level == "normal":
        assert row["effective_fulfillment"] == 1 and result["core_gate_fulfillment"] == 1
        assert result["bonus"] == 0


def test_q6_alias_unmatched_related_and_unsupported_remain_distinct():
    cv, jd, config = fixture("Golang", "advanced", "expert", "Go")
    assert score_skills_v3(cv, jd, config)["requirements"][0]["candidate_depth_evidence_status"] == "stated_level"
    cv, jd, config = fixture(target="Java")
    row = score_skills_v3(cv, jd, config)["requirements"][0]
    assert row["depth_status"] == "unmatched" and row["candidate_depth_evidence_status"] is None
    cv, jd, config = fixture()
    cv.annotation.skills_status = "partial"
    row = score_skills_v3(cv, jd, config)["requirements"][0]
    assert row["depth_status"] == "unsupported_annotation_depth" and row["candidate_depth_evidence_status"] is None


def test_q3_gate_is_aggregate_not_an_individual_hurdle():
    f = Fixture(tuple(requirement(f"Capability{i}") for i in range(10)),
                tuple(Candidate(f"Capability{i}") for i in range(7)), {})
    cv, jd, config = build_fixture(f)
    result = score_skills_v3(cv, jd, config)
    assert result["core_gate_fulfillment"] == .7 and result["gate_status"] == "passed"
    assert len(result["unmet_requirements"]) == 3
    assert not result["force_mismatch"]


@pytest.mark.parametrize("level", ["normal", "intermediate", "advanced", "expert"])
def test_q4_related_proficiency_never_establishes_required_depth(level):
    cv, jd, config = fixture("Django", level, "expert")
    config.relations = [SkillRelation(id="near", candidate="Django", requirement="Python",
        relation="near", review_status="approved", reason="Controlled fixture", source="test")]
    result = score_skills_v3(cv, jd, config)
    row = result["requirements"][0]
    assert row["depth_status"] == "related_skill_depth_unknown"
    assert row["depth_gap"] is None and row["depth_credit"] == 1
    assert row["candidate_depth_evidence_status"] is None
    assert row["fulfillment"] == .7 and result["core_gate_fulfillment"] == 0
    assert result["bonus"] == 0


def precision_pair(monkeypatch, target_overall):
    cv, jd, config = fixture()
    cv.document.basics.label = jd.document.title = "Software Engineer"
    for item in (cv, jd):
        item.facts.evidence["title"] = [Evidence(quote="Software Engineer")]
    # Other components yield 46.75 with title=100; solve the independent total.
    skill = (target_overall - 46.75) / .3
    original = runtime_v3.score_skills_v3

    def controlled(*args, **kwargs):
        result = original(*args, **kwargs)
        result.update(score=skill, skills_after_gate=skill)
        return result

    monkeypatch.setattr(runtime_v3, "score_skills_v3", controlled)
    return score_pair(cv, jd, config=config, as_of_date=AS_OF)


@pytest.mark.parametrize("raw,display,grade", [
    (49.96, 50.0, 0), (50.0, 50.0, 1), (50.04, 50.0, 1),
    (74.96, 75.0, 1), (75.0, 75.0, 2), (75.04, 75.0, 2),
])
def test_q7_grade_boundaries_use_unrounded_total(monkeypatch, raw, display, grade):
    result = precision_pair(monkeypatch, raw)
    assert result.unrounded_overall_score == pytest.approx(raw)
    assert result.overall_score == display and result.grade == grade


def test_q7_no_double_rounding(monkeypatch):
    result = precision_pair(monkeypatch, 49.949)
    assert result.unrounded_overall_score == pytest.approx(49.949)
    assert result.overall_score == 49.9 and result.grade == 0
    # Aggregating displayed component values would incorrectly display 50.0.
    assert display_score(sum(w * result.scores.model_dump()[k] for k, w in
        {"skills": .3, "title": .2, "experience": .15, "education": .1,
         "industry": .1, "location": .08, "salary": .07}.items())) == 50.0


@pytest.mark.parametrize("value,display", [(12.45, 12.5), (76.5, 76.5), (77.5, 77.5)])
def test_q7_half_up_and_preserved_half(value, display):
    assert display_score(value) == display


def test_q7_skills_half_is_retained_in_real_scoring():
    cv, jd, config = build_fixture(Fixture(
        (requirement("Python", "expert"), requirement("SQL", "expert")),
        (Candidate("Python", "intermediate"), Candidate("SQL", "expert")), {}))
    result = score_pair(cv, jd, config=config, as_of_date=AS_OF)
    assert result.unrounded_scores.skills == 77.5 and result.scores.skills == 77.5
    assert result.skills_breakdown["score"] == 77.5


@pytest.mark.parametrize("component", ["skills", "title"])
@pytest.mark.parametrize("raw,grade", [(69.96, 1), (70.0, 2), (70.04, 2)])
def test_q7_strong_component_gates_use_unrounded_values(component, raw, grade):
    values = {"skills": 80, "title": 80}
    values[component] = raw
    assert runtime_grade(80, values["skills"], values["title"], ScorerConfig(version="ats-en-v3")) == grade
    if raw == 69.96:
        assert display_score(raw) == 70.0


def test_q7_other_fractional_components_retain_raw_values():
    cv, jd, config = fixture()
    cv.document.basics.label = jd.document.title = "Software Engineer"
    cv.document.work = [{"position": "Software Engineer", "start_date": "2026-08", "end_date": "2026-09"}]
    cv = type(cv).model_validate(cv.model_dump(mode="json"))
    jd.document.seniority.min_years = 2
    for item in (cv, jd):
        item.facts.evidence["experience"] = [Evidence(quote="Duration source")]
    jd.document.compensation.max_monthly = 100
    jd.facts.currency = "USD"
    jd.facts.evidence["salary"] = [Evidence(quote="Salary cap source")]
    result = score_pair(cv, jd, preferences=SalaryPreference(mode="explicit", minimum=700, currency="USD"),
                        config=config, as_of_date=AS_OF)
    assert result.unrounded_scores.experience == pytest.approx(100 / 24)
    assert result.scores.experience == 4.2
    assert result.unrounded_scores.salary == pytest.approx(100 / 7)
    assert result.scores.salary == 14.3
    assert result.dimensions["salary"].unrounded_score == result.unrounded_scores.salary


def test_q7_evaluation_matches_runtime_despite_display_thresholds(monkeypatch):
    rows = []
    for i, raw in enumerate([49.96, 74.96, 75.04]):
        with monkeypatch.context() as m:
            result = precision_pair(m, raw)
        rows.append({"split": "test", "label_id": i, "pair_id": str(i), "cv_id": str(i),
                     "baseline_skill_score": [0, 60, 100][i], "result": result.model_dump(mode="json")})
    config = ScorerConfig(version="ats-en-v3")
    report = evaluate(rows, config, ScorerConfig(), 0, bootstrap=0)
    assert report["scorer"]["accuracy"] == 1
    broken = deepcopy(rows[0]["result"])
    del broken["unrounded_scores"]
    with pytest.raises(ValueError, match="unrounded"):
        _grade_values(broken, config)


def test_legacy_contract_is_integer_and_has_no_v3_fields():
    cv, jd, _ = fixture()
    for version in ("ats-en-v1", "ats-en-v2"):
        result = score_pair(cv, jd, config=ScorerConfig(version=version), as_of_date=AS_OF)
        payload = result.model_dump(mode="json")
        assert all(isinstance(v, int) for v in payload["scores"].values())
        assert "unrounded_scores" not in payload and "precision_policy_version" not in payload

