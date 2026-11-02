"""Unit tests for JD teacher label validation (synthetic data only)."""

from copy import deepcopy
import json

from scripts.jd_teacher_check import check_jd, jd_policy_issues

EN_TEXT = (
    "Title: Backend Developer\nLocation: Hanoi, Vietnam\n"
    "Salary: 2000 - 3000 USD/month\nRequirements: Must have Python. Docker is a plus."
)
VI_TEXT = (
    "Vị trí: Lập trình viên Backend\nĐịa điểm: Hà Nội\n"
    "Lương: 2000 - 3000 USD\nYêu cầu: Python là bắt buộc. Ưu tiên Docker."
)


def valid_label(vi: bool = False) -> dict:
    core_q = "Python là bắt buộc." if vi else "Must have Python."
    pref_q = "Ưu tiên Docker." if vi else "Docker is a plus."
    loc_q = "Hà Nội" if vi else "Hanoi, Vietnam"
    sal_q = "2000 - 3000 USD" if vi else "2000 - 3000 USD/month"
    return {
        "document": {
            "schema_version": "job-v3",
            "id": "jd_syn",
            "url": "https://example.com/jd",
            "scraped_at": "2026-10-26T00:00:00Z",
            "title": "Lập trình viên Backend" if vi else "Backend Developer",
            "title_normalized": "backend_developer",
            "category": "software_engineer",
            "location": {"city": "Hà Nội" if vi else "Hanoi", "country": "Vietnam", "countryCode": "VN"},
            "seniority": {"min_years": None, "max_years": None},
            "skills": {
                "required": [{"name": "Python", "proficiency": "required"}],
                "preferred": [{"name": "Docker", "proficiency": "preferred"}],
                "raw_text": None,
                "requirements": [
                    {
                        "id": "req_01",
                        "options": ["Python"],
                        "option_details": [{"name": "Python", "minimum_proficiency": "normal"}],
                        "importance": "core",
                        "quote": core_q,
                        "importance_quote": core_q,
                        "review_status": "pending",
                        "issues": [],
                    },
                    {
                        "id": "req_02",
                        "options": ["Docker"],
                        "option_details": [{"name": "Docker", "minimum_proficiency": "normal"}],
                        "importance": "preferred",
                        "quote": pref_q,
                        "importance_quote": pref_q,
                        "review_status": "pending",
                        "issues": [],
                    },
                ],
            },
            "description": {"raw_text": None},
            "compensation": {"currency": "USD", "min_monthly": 2000, "max_monthly": 3000, "raw_text": None},
            "benefits": {"raw_text": None},
            "metadata": {"language": "vi" if vi else "en"},
        },
        "facts": {
            "domains": ["software"],
            "completed_degrees": [],
            "minimum_degree": None,
            "country": "VN",
            "currency": "USD",
            "evidence": {
                "skills": [{"quote": "Python"}, {"quote": "Docker"}],
                "industry": [{"quote": "Lập trình viên Backend" if vi else "Backend Developer"}],
                "location": [{"quote": loc_q}],
                "salary": [{"quote": sal_q}],
            },
        },
    }


def test_valid_en_and_vi_labels_pass(tmp_path):
    for name, vi, text in (("en.json", False, EN_TEXT), ("vi.json", True, VI_TEXT)):
        path = tmp_path / name
        path.write_text(json.dumps(valid_label(vi)), encoding="utf-8")
        res = check_jd(path, text)
        assert res == {"status": "pass", "issues": [], "policy": [], "error": None}


def test_rule_2_requirement_ids():
    bad = deepcopy(valid_label())
    bad["document"]["skills"]["requirements"][0]["id"] = "req_1"
    bad["document"]["skills"]["requirements"][1]["id"] = "req_01"
    assert any("must match req_\\d{2,}" in p for p in jd_policy_issues(bad, EN_TEXT))
    dup = deepcopy(valid_label())
    dup["document"]["skills"]["requirements"][1]["id"] = "req_01"
    assert "duplicate requirement id 'req_01'" in jd_policy_issues(dup, EN_TEXT)


def test_rule_3_requirement_structure():
    bad = deepcopy(valid_label())
    req = bad["document"]["skills"]["requirements"][0]
    req["option_details"] = []
    req["review_status"] = "approved"
    req["options"] = ["Python  3"]
    issues = jd_policy_issues(bad, EN_TEXT)
    assert "req_01: option_details names must equal options in the same order" in issues
    assert "req_01: review_status='approved' (must be 'pending')" in issues
    assert "req_01: option name whitespace/empty 'Python  3'" in issues


def test_rule_4_preferred_importance_quote():
    bad = deepcopy(valid_label())
    bad["document"]["skills"]["requirements"][1]["importance_quote"] = "Docker experience"
    assert any(i.startswith("req_02: preferred requirement needs importance_quote") for i in jd_policy_issues(bad, EN_TEXT))
    for wording in ("Docker là một lợi thế", "Docker là điểm cộng", "Bonus points: Docker"):  # beyond production OPTIONAL
        ok = deepcopy(valid_label())
        ok["document"]["skills"]["requirements"][1]["importance_quote"] = wording
        assert not any("preferred requirement" in i for i in jd_policy_issues(ok, EN_TEXT + " " + wording))


def test_raw_proficiency_needs_its_literal_quote():
    bad = deepcopy(valid_label())
    bad["document"]["skills"]["requirements"][0]["option_details"][0].update(raw_proficiency="strong", proficiency_quote="strong Python")
    assert any("raw_proficiency 'strong'" in i for i in jd_policy_issues(bad, EN_TEXT))  # not in the text
    assert not any("raw_proficiency" in i for i in jd_policy_issues(bad, EN_TEXT + " strong Python"))


def test_rule_5_legacy_lists_mirror_units():
    bad = deepcopy(valid_label())
    bad["document"]["skills"]["required"] = [{"name": "Python", "proficiency": "preferred"}]
    bad["document"]["skills"]["preferred"] = []
    issues = jd_policy_issues(bad, EN_TEXT)
    assert "skills.required proficiency must be 'required'" in issues
    assert "skills.preferred must mirror preferred requirement options" in issues


def test_rule_6_title_pollution_error():
    bad = deepcopy(valid_label())
    bad["document"]["skills"]["requirements"][0]["options"] = ["Backend Developer"]
    bad["document"]["skills"]["requirements"][0]["option_details"] = [{"name": "Backend Developer"}]
    bad["document"]["skills"]["required"] = [{"name": "Backend Developer", "proficiency": "required"}]
    assert "title pollution: option 'Backend Developer' is the job title 'Backend Developer'" in jd_policy_issues(bad, EN_TEXT)
    bad["document"]["title"] = "Python Backend Developer"  # a skill named inside the title is only a WARN
    bad["document"]["skills"]["requirements"][0]["options"] = ["Python"]
    bad["document"]["skills"]["requirements"][0]["option_details"] = [{"name": "Python"}]
    bad["document"]["skills"]["required"] = [{"name": "Python", "proficiency": "required"}]
    assert not any(i.startswith("title pollution") for i in jd_policy_issues(bad, EN_TEXT))


def test_rule_7_claims_need_evidence():
    bad = deepcopy(valid_label())
    bad["facts"]["evidence"] = {}
    bad["facts"]["minimum_degree"] = "bachelor"
    bad["document"]["seniority"]["min_years"] = 3
    issues = jd_policy_issues(bad, EN_TEXT)
    assert {
        "facts.domains without evidence.industry",
        "education claim without evidence.education",
        "experience claim without evidence.experience",
        "location claim without evidence.location",
        "salary claim without evidence.salary",
        "requirements without evidence.skills",
    } <= set(issues)


def test_rule_8_salary_currency_and_ordering():
    bad = deepcopy(valid_label())
    bad["facts"]["currency"] = None
    bad["document"]["compensation"].update(min_monthly=4000, max_monthly=3000)
    issues = jd_policy_issues(bad, EN_TEXT)
    assert "salary min/max without facts.currency" in issues
    assert "compensation.min_monthly > max_monthly" in issues
    assert any(i.startswith("compensation.currency 'USD' != facts.currency None") for i in issues)


def test_rule_9_experience_bounds_and_ordering():
    bad = deepcopy(valid_label())
    bad["document"]["seniority"].update(min_years=45, max_years=10)
    issues = jd_policy_issues(bad, EN_TEXT)
    assert "WARN: seniority.min_years=45: unusually high, check the source" in issues
    assert "seniority.min_years > max_years" in issues


def test_rule_10_raw_text_copies_must_be_null():
    bad = deepcopy(valid_label())
    bad["document"]["description"]["raw_text"] = "copied text"
    assert "description.raw_text must be null" in jd_policy_issues(bad, EN_TEXT)


def test_rules_1_11_12_schema_language_domains_degrees():
    bad = deepcopy(valid_label())
    bad["document"]["schema_version"] = "job-v2"
    bad["document"]["metadata"]["language"] = "fr"
    bad["facts"]["domains"] = ["unknown_domain"]
    bad["facts"]["completed_degrees"] = ["bachelor"]
    issues = jd_policy_issues(bad, EN_TEXT)
    assert "schema_version 'job-v2'" in issues
    assert "metadata.language='fr'" in issues
    assert "facts.domains 'unknown_domain' not in ROLE_FAMILIES" in issues
    assert "facts.completed_degrees must be []" in issues


def test_check_jd_quote_not_in_text_and_policy_fail(tmp_path):
    assert check_jd(tmp_path / "missing.json", EN_TEXT)["status"] == "missing"

    extra_key = deepcopy(valid_label())
    extra_key["document"]["skills"]["requirements"][0]["extra_field"] = 1
    p0 = tmp_path / "extra.json"
    p0.write_text(json.dumps(extra_key), encoding="utf-8")
    r0 = check_jd(p0, EN_TEXT)
    assert r0["status"] == "fail" and "document.skills.requirements[].extra_field" in r0["error"]

    bad_quote = deepcopy(valid_label())
    bad_quote["document"]["skills"]["requirements"][0]["quote"] = "Not in text at all"
    bad_quote["document"]["description"]["raw_text"] = "extra copy"
    p1 = tmp_path / "bad_quote.json"
    p1.write_text(json.dumps(bad_quote), encoding="utf-8")
    r1 = check_jd(p1, EN_TEXT)
    assert r1["status"] == "fail" and "description.raw_text must be null" in r1["policy"]

    bad_policy = deepcopy(valid_label())
    bad_policy["document"]["description"]["raw_text"] = "extra copy"
    p2 = tmp_path / "bad_policy.json"
    p2.write_text(json.dumps(bad_policy), encoding="utf-8")
    r2 = check_jd(p2, EN_TEXT)
    assert r2["status"] == "policy_fail" and r2["policy"] == ["description.raw_text must be null"]


def test_role_word_option_emits_warn_not_error(tmp_path):
    text = EN_TEXT + " Experience in lead generation is a plus."
    label = deepcopy(valid_label())
    label["document"]["skills"]["requirements"][1]["options"] = ["lead generation"]
    label["document"]["skills"]["requirements"][1]["option_details"] = [{"name": "lead generation", "minimum_proficiency": "normal"}]
    label["document"]["skills"]["requirements"][1]["quote"] = "Experience in lead generation is a plus."
    label["document"]["skills"]["requirements"][1]["importance_quote"] = "Experience in lead generation is a plus."
    label["document"]["skills"]["preferred"] = [{"name": "lead generation", "proficiency": "preferred"}]
    path = tmp_path / "warn.json"
    path.write_text(json.dumps(label), encoding="utf-8")
    res = check_jd(path, text)
    assert res["status"] == "pass"
    assert res["policy"] == ["WARN: 'lead generation': is this a capability or a title/seniority word?"]
