"""Contracts for the offline silver adapter and its limited sample selection."""

from copy import deepcopy
from datetime import date

from scripts.ats_silver_eval import select_sample
from src.scoring import score_pair
from src.scoring.silver import job_from_record, resume_from_record

AS_OF = date(2026, 9, 30)


def records():
    cv = {"basics": {"label": "Accountant"},
          "skills": [{"name": "Accounting", "keywords": ["Excel"]}],
          "work": [{"position": "Accountant", "startDate": "2020-01", "endDate": "2024-01"}]}
    jd = {"id": "jd-test", "url": "urn:test", "scraped_at": "test", "title": "Accountant",
          "title_normalized": "accountant", "category": "accountant",
          "skills": {"required": [{"name": "Accountant"}, {"name": "Excel"}]},
          "seniority": {"min_years": 2}, "description": {"requirements": []}}
    return cv, jd


def test_structured_adapter_is_independent_of_teacher_fields_and_categories():
    cv, jd = records()
    adapted_cv = resume_from_record(cv, AS_OF)
    adapted_jd, removed = job_from_record(jd)
    first = score_pair(adapted_cv, adapted_jd, as_of_date=AS_OF)
    cv.update(domain="software_engineer", grade=0, scores={"skills": 0})
    jd.update(category="software_engineer", grade=2, overall_score=100)
    changed_jd, _ = job_from_record(jd)
    second = score_pair(resume_from_record(cv, AS_OF), changed_jd, as_of_date=AS_OF)
    assert first == second
    assert removed == ["Accountant"]
    assert first.scores.skills == 100
    assert first.scores.experience == 100


def test_incomplete_or_future_cv_degrees_are_not_counted():
    cv, _ = records()
    cv["education"] = [
        {"studyType": "Bachelor of Science", "endDate": "2020"},
        {"studyType": "Ph.D. Candidate", "endDate": "2024"},
        {"studyType": "Master of Science", "endDate": "2027-01"},
        {"studyType": "Master of Science", "endDate": None},
    ]
    assert resume_from_record(cv, AS_OF).facts.completed_degrees == ["bachelor"]


def test_optional_degrees_and_office_acronyms_do_not_create_requirements():
    _, jd = records()
    jd["description"]["requirements"] = ["Proficiency in MS Office", "Master's degree preferred"]
    adapted, _ = job_from_record(jd)
    assert adapted.facts.minimum_degree is None
    assert not adapted.facts.has("education")
    jd["description"]["requirements"].append("Bachelor's or Master's degree required")
    assert job_from_record(jd)[0].facts.minimum_degree == "bachelor"


def test_missing_country_and_remote_scope_remain_unknown():
    _, jd = records()
    jd["location"] = {"remote_policy": "Remote"}
    facts = job_from_record(jd)[0].facts
    assert facts.country is None
    assert facts.remote
    assert not facts.worldwide_remote


def test_sample_is_small_reproducible_balanced_and_does_not_mutate_inputs():
    pairs = [{"cv_id": str(i), "jd_id": str(grade), "grade": grade,
              "pair_type": "heuristic_negative" if grade == 0 and i < 20 else "in_domain"}
             for grade in range(3) for i in range(40)]
    original = deepcopy(pairs)
    selected = select_sample(pairs, 10, 42)
    assert selected == select_sample(pairs, 10, 42)
    assert len(selected) == 30
    assert [sum(p["grade"] == grade for p in selected) for grade in range(3)] == [10, 10, 10]
    assert sum(p["pair_type"] == "heuristic_negative" for p in selected) == 5
    assert pairs == original
