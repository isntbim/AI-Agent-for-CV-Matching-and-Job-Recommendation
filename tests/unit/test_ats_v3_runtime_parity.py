"""Audit: copies preserve raw formulas; legacy entry points stay unchanged."""
from datetime import date
import pytest

from src.scoring import score_pair, rules, ScorerConfig, SalaryPreference
from src.scoring.models import Evidence
from src.scoring.v2_benchmark import score_stored_pair
from tests.unit.test_ats_v3_contrast_set import CASES, build_fixture
from tests.unit.test_ats_v3_design_decisions import fixture

AS_OF = date(2026, 9, 30)
CONTRAST = [(case.id+"-"+str(i), f) for case in CASES for i,f in enumerate(case.fixtures)]


def assert_raw_parity(monkeypatch, cv, jd, config, preferences=None):
    captured = {}
    original = rules._result

    def capture(score, supported, basis, rule_id, resume, job, dimension):
        captured[dimension] = score
        return original(score, supported, basis, rule_id, resume, job, dimension)

    monkeypatch.setattr(rules, "_result", capture)
    # Intentionally observe the internal legacy path before its integer conversion.
    rules.score_pair(cv, jd, preferences=preferences, config=config, as_of_date=AS_OF)
    current = score_pair(cv, jd, preferences=preferences, config=config, as_of_date=AS_OF)
    assert captured == current.unrounded_scores.model_dump()


@pytest.mark.parametrize("entry", CONTRAST, ids=lambda x:x[0])
def test_contrast_raw_formula_parity(monkeypatch, entry):
    cv, jd, config = build_fixture(entry[1])
    assert_raw_parity(monkeypatch, cv, jd, config)


@pytest.mark.parametrize("scenario", [
    "title", "experience_fraction", "explicit_no_experience", "degree_gap",
    "explicit_no_education", "industry_fraction", "worldwide_remote",
    "country_mismatch", "salary_fraction",
])
def test_design_raw_formula_parity(monkeypatch, scenario):
    cv,jd,config = fixture()
    cv.document.basics.label = jd.document.title = "Software Engineer"
    for obj in (cv,jd):
        obj.facts.evidence["title"] = [Evidence(quote="Software Engineer")]
    preferences = None
    if scenario == "experience_fraction":
        payload = cv.model_dump(mode="json")
        payload["document"]["work"] = [{"position":"Software Engineer","start_date":"2026-08","end_date":"2026-09"}]
        cv = type(cv).model_validate(payload)
        jd.document.seniority.min_years = 2
        for obj in (cv,jd):
            obj.facts.evidence["experience"] = [Evidence(quote="Duration source")]
    elif scenario == "explicit_no_experience":
        jd.facts.no_experience_required = True
        jd.facts.evidence["experience"] = [Evidence(quote="No experience required")]
    elif scenario == "degree_gap":
        cv.facts.completed_degrees = ["associate"]
        jd.facts.minimum_degree = "bachelor"
        for obj in (cv,jd):
            obj.facts.evidence["education"] = [Evidence(quote="Degree source")]
    elif scenario == "explicit_no_education":
        jd.facts.no_education_required = True
        jd.facts.evidence["education"] = [Evidence(quote="No education required")]
    elif scenario == "industry_fraction":
        cv.facts.domains = ["software"]
        jd.facts.domains = ["software","finance","data"]
        for obj in (cv,jd):
            obj.facts.evidence["industry"] = [Evidence(quote="Domain source")]
    elif scenario == "worldwide_remote":
        jd.facts.remote = jd.facts.worldwide_remote = True
        jd.facts.evidence["location"] = [Evidence(quote="Worldwide remote")]
    elif scenario == "country_mismatch":
        cv.facts.country,jd.facts.country = "US","VN"
        for obj in (cv,jd):
            obj.facts.evidence["location"] = [Evidence(quote="Country source")]
    elif scenario == "salary_fraction":
        jd.document.compensation.max_monthly = 100
        jd.facts.currency = "USD"
        jd.facts.evidence["salary"] = [Evidence(quote="Salary cap source")]
        preferences = SalaryPreference(mode="explicit",minimum=700,currency="USD")
    assert_raw_parity(monkeypatch,cv,jd,config,preferences)


def test_default_legacy_benchmark_caller_rejects_v3():
    cv,jd,config = fixture()
    documents = {"cv":{"envelope":cv.model_dump(mode="json")},
                 "jd":{"envelope":jd.model_dump(mode="json")}}
    pair = {"cv_id":"cv","jd_id":"jd","pair_type":"controlled","grade":0}
    failed = score_stored_pair(pair,"v3_full",{},documents,config,AS_OF)
    assert failed["status"] == "failed" and failed["error_type"] == "ValueError"
    assert "src.scoring.score_pair" in failed["error"]
    passed = score_stored_pair(pair,"v3_full",{},documents,config,AS_OF,scorer=score_pair)
    assert passed["status"] == "ok" and "unrounded_scores" in passed["result"]


def test_eval_cli_uses_public_dispatcher():
    from scripts import ats_eval
    assert ats_eval.score_pair is score_pair

