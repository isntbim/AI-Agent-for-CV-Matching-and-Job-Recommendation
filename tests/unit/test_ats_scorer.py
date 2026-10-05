"""Behavioral tests for rules, evidence and reproducibility (no live LLM)."""

from datetime import date

import pytest

from src.parser.job_schemas import JobDescriptionSchema
from src.parser.schemas import ResumeSchema
from src.scoring import (
    JobInput,
    ResumeInput,
    SalaryPreference,
    ScorerConfig,
    score_pair,
)
from src.scoring.models import Evidence, Facts, runtime_grade
from src.scoring.normalization import role

AS_OF = date(2026, 9, 30)


def inputs():
    dimensions = ["skills", "title", "experience", "education", "industry", "location", "salary"]
    evidence = {key: [Evidence(quote=key)] for key in dimensions}
    cv = ResumeInput(document=ResumeSchema.model_validate({
        "basics": {"label": "Senior Software Developer", "location": {"city": "Boston", "countryCode": "US"}},
        "skills": [{"name": "Technical", "keywords": ["Python3", "SQL"]}],
        "work": [{"position": "Software Engineer", "startDate": "2020-01", "endDate": "2023-01"},
                 {"position": "Backend Developer", "startDate": "2022-01", "endDate": "Present"}],
    }), facts=Facts(evidence=evidence, domains=["software"], completed_degrees=["bachelor"], country="US"))
    jd = JobInput(document=JobDescriptionSchema.model_validate({
        "id": "jd-test", "url": "urn:test", "scraped_at": "not-applicable",
        "title": "Software Engineer", "title_normalized": "software_engineer", "category": "software",
        "skills": {"required": [{"name": "Python"}, {"name": "SQL"}]},
        "seniority": {"min_years": 5}, "location": {"city": "Boston", "country": "US", "countryCode": "US"},
        "compensation": {"max_monthly": 5000, "currency": "USD"},
    }), facts=Facts(evidence=evidence, domains=["software"], minimum_degree="bachelor", country="US", currency="USD"))
    return cv, jd


def test_complete_match_default_salary():
    cv, jd = inputs()
    result = score_pair(cv, jd, as_of_date=AS_OF)
    assert result.overall_score == 98.2
    assert result.grade == 2
    assert result.coverage == .93
    assert result.scores.salary == 75
    assert result.matched_skills == ["python", "sql"]
    assert all(v.rule_id for v in result.dimensions.values())


def test_pure_repeatability_and_fixed_date():
    cv, jd = inputs()
    assert score_pair(cv, jd, as_of_date=AS_OF) == score_pair(cv, jd, as_of_date=AS_OF)
    with pytest.raises(ValueError, match="as_of_date"):
        score_pair(cv, jd)


def test_skill_alias_deduplication_and_monotonicity():
    cv, jd = inputs()
    cv.document.skills[0].keywords = ["Python 3", "python", "Python3"]
    partial = score_pair(cv, jd, as_of_date=AS_OF)
    assert partial.scores.skills == 50
    cv.document.skills[0].keywords.append("SQL")
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.skills == 100


def test_unknown_skills_are_retained_and_languages_distinct():
    cv, jd = inputs()
    cv.document.skills[0].keywords = ["custom-tool", "Java"]
    jd.document.skills.required[0].name = "custom-tool"
    jd.document.skills.required[1].name = "JavaScript"
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.skills == 50


def test_preferred_only_and_missing_requirements():
    cv, jd = inputs()
    jd.document.skills.preferred = jd.document.skills.required
    jd.document.skills.required = []
    assert score_pair(cv, jd, as_of_date=AS_OF).dimensions["skills"].basis == "preferred_only_coverage"
    jd.document.skills.preferred = []
    result = score_pair(cv, jd, as_of_date=AS_OF)
    assert result.scores.skills == 50 and not result.dimensions["skills"].supported


def test_missing_evidence_overrides_schema_defaults():
    cv, jd = inputs()
    cv.facts, jd.facts = Facts(), Facts()
    result = score_pair(cv, jd, as_of_date=AS_OF)
    assert result.coverage == 0
    assert result.scores.location == 50
    assert result.scores.salary == 75
    assert result.overall_score == 51.8


def test_overlapping_employment_is_not_double_counted():
    cv, jd = inputs()
    jd.document.seniority.min_years = 10
    result = score_pair(cv, jd, as_of_date=date(2025, 1, 1))
    assert result.scores.experience == 50  # 60 union months, not 72.


@pytest.mark.parametrize("start,end", [("2020", "2023"), ("2028-01", "Present"), ("2023-13", "Present"),
                                       ("2023-01", "2022-01"), ("2020-01", None)])
def test_ambiguous_or_invalid_relevant_dates_neutral(start, end):
    cv, jd = inputs()
    cv.document.work[0].start_date, cv.document.work[0].end_date = start, end
    assert not score_pair(cv, jd, as_of_date=AS_OF).dimensions["experience"].supported


def test_explicit_no_experience_required():
    cv, jd = inputs()
    jd.facts.no_experience_required = True
    cv.document.work = []
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.experience == 100


def test_title_family_and_latest_fallback():
    cv, jd = inputs()
    cv.document.basics.label = None
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.title == 70
    cv.document.basics.label = "Nurse"
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.title == 0
    assert role("Sr. Software Developer") == role("Software Engineer")
    assert role("Machine Learning Engineer")[0] != role("Data Scientist")[0]


def test_unrecognized_roles_are_unknown_rather_than_proven_mismatch():
    cv, jd = inputs()
    cv.document.basics.label = "Unlisted Research Occupation"
    result = score_pair(cv, jd, as_of_date=AS_OF)
    assert result.scores.title == 50 and not result.dimensions["title"].supported
    cv.document.work[0].position = "Unlisted Research Occupation"
    assert not score_pair(cv, jd, as_of_date=AS_OF).dimensions["experience"].supported


@pytest.mark.parametrize("completed,required,expected", [("bachelor", "bachelor", 100), ("master", "bachelor", 100),
                                                          ("bachelor", "master", 50), ("associate", "master", 0)])
def test_completed_degree_levels(completed, required, expected):
    cv, jd = inputs()
    cv.facts.completed_degrees = [completed]
    jd.facts.minimum_degree = required
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.education == expected


def test_ongoing_degree_is_not_attainment():
    cv, jd = inputs()
    cv.facts.completed_degrees = []
    assert not score_pair(cv, jd, as_of_date=AS_OF).dimensions["education"].supported


def test_domain_is_professional_role_not_company_sector():
    cv, jd = inputs()
    jd.document.company.industry = "Healthcare"
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.industry == 100


def test_remote_eligibility_requires_constraints():
    cv, jd = inputs()
    jd.facts.remote = True
    assert not score_pair(cv, jd, as_of_date=AS_OF).dimensions["location"].supported
    jd.facts.eligible_countries = ["Canada"]
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.location == 0
    jd.facts.eligible_countries = ["United States"]
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.location == 100
    jd.facts.worldwide_remote = True
    cv.facts.evidence.pop("location")
    assert score_pair(cv, jd, as_of_date=AS_OF).scores.location == 100


def test_salary_periods_unknown_currency_and_defaults():
    cv, jd = inputs()
    pref = SalaryPreference(mode="explicit", minimum=120000, currency="USD", period="annual")
    assert score_pair(cv, jd, pref, as_of_date=AS_OF).scores.salary == 50
    jd.document.compensation.max_monthly = None
    jd.document.compensation.max_annual = 60000
    assert score_pair(cv, jd, pref, as_of_date=AS_OF).scores.salary == 50
    jd.facts.currency = None
    result = score_pair(cv, jd, pref, as_of_date=AS_OF)
    assert result.scores.salary == 75 and not result.dimensions["salary"].supported
    with pytest.raises(ValueError):
        SalaryPreference(mode="explicit", minimum=100)


def test_grade_skill_title_gate_and_ordered_thresholds():
    assert runtime_grade(99, 69, 100, ScorerConfig()) == 1
    assert runtime_grade(99, 100, 69, ScorerConfig()) == 1
    assert runtime_grade(75, 70, 70, ScorerConfig()) == 2
    assert runtime_grade(49.9, 100, 100, ScorerConfig()) == 0
    with pytest.raises(ValueError):
        ScorerConfig(lower_threshold=75, upper_threshold=50)
