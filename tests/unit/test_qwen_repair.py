from scripts.qwen_repair import fix_cv, fix_jd, snap, unclaim

CV_TEXT = "Skills\nPython,  SQL\nLanguages: English, Spanish\nWorked as an analyst"


def test_snap_restores_whitespace_and_case_but_not_new_words():
    assert snap("python, sql", CV_TEXT) == "Python,  SQL"
    assert snap("Python and SQL", CV_TEXT) is None
    assert snap("", CV_TEXT) is None


def test_fix_cv_relinks_drops_and_syncs_languages():
    label = {"document": {"skills": [{"name": "Tech", "level": None, "keywords": [], "details": [
        {"name": "SQL", "proficiency": "normal", "raw_proficiency": None, "skill_quote": "Python, SQL", "proficiency_quote": None},
        {"name": "Python", "proficiency": "normal", "raw_proficiency": None, "skill_quote": "Expert Python", "proficiency_quote": None},
        {"name": "Kubernetes", "proficiency": "normal", "raw_proficiency": None, "skill_quote": "Kubernetes", "proficiency_quote": None}]}],
        "languages": [{"language": "English"}, {"language": "Spanish"}]}, "facts": {}}
    log = []
    fix_cv(label, CV_TEXT, log)
    tech, langs = label["document"]["skills"]
    assert [d["name"] for d in tech["details"]] == ["SQL", "Python"]  # Kubernetes is not in the source
    assert tech["details"][1]["skill_quote"] == "Python,  SQL"  # invented quote replaced by the source line
    assert [d["name"] for d in langs["details"]] == ["English", "Spanish"]
    assert all(d["skill_quote"] in CV_TEXT for g in label["document"]["skills"] for d in g["details"])


def test_fix_jd_rebuilds_mirrors_and_downgrades_unsupported_preferred():
    text = "Must have Python.\nSQL is a plus.\nDocker experience."
    label = {"document": {"skills": {"required": [], "preferred": [{"name": "Go", "proficiency": "preferred"}], "requirements": [
        {"id": "req_07", "options": ["Python"], "importance": "core", "quote": "Must have  Python.", "importance_quote": "Must have Python."},
        {"id": "req_08", "options": ["SQL"], "importance": "preferred", "quote": "SQL is a plus.", "importance_quote": "SQL is a plus."},
        {"id": "req_09", "options": ["Docker"], "importance": "preferred", "quote": "Docker experience.", "importance_quote": "Docker experience."},
        {"id": "req_10", "options": ["Rust"], "importance": "core", "quote": "Rust required", "importance_quote": None}]}},
        "facts": {"evidence": {}}}
    fix_jd(label, text, [])
    skills = label["document"]["skills"]
    assert [r["id"] for r in skills["requirements"]] == ["req_01", "req_02", "req_03"]  # Rust dropped, ids renumbered
    assert [s["name"] for s in skills["required"]] == ["Python", "Docker"]  # Docker had no optional wording
    assert [s["name"] for s in skills["preferred"]] == ["SQL"]
    assert label["facts"]["evidence"]["skills"]


def test_unclaim_clears_claims_without_evidence():
    facts = {"evidence": {"industry": [{"quote": "x"}]}, "domains": ["finance"], "country": "US", "currency": "USD"}
    unclaim(facts, {}, "cv_extraction", [])
    assert facts == {"evidence": {"industry": [{"quote": "x"}]}, "domains": ["finance"], "country": None, "currency": None}


def test_balance_drops_only_stray_closers():
    from scripts.qwen_repair import balance
    import json
    good = '{"document":{"skills":{"requirements":[{"options":["a]"]}]}},"facts":{}}'
    assert balance(good) == good  # brackets inside strings are untouched
    slip = '{"document":{"skills":{"requirements":[{"options":["a"]}]}]},"facts":{}}'  # one extra "]"
    assert json.loads(balance(slip)) == {"document": {"skills": {"requirements": [{"options": ["a"]}]}}, "facts": {}}
    early = '{"document":{"x":1}}},"facts":{}}'  # root closed too early
    assert json.loads(balance(early)) == {"document": {"x": 1}, "facts": {}}


def test_fix_jd_clears_salary_numbers_without_a_currency():
    quote = "Salary range: $60,000 - $90,000"
    label = {"document": {"skills": {"requirements": []}, "compensation": {"currency": None, "min_monthly": 60000,
             "max_monthly": 90000, "min_annual": None, "max_annual": None}},
             "facts": {"currency": None, "evidence": {"salary": [{"quote": quote}]}}}
    log = []
    fix_jd(label, quote, log)
    comp = label["document"]["compensation"]
    assert all(comp[k] is None for k in ("min_monthly", "max_monthly", "min_annual", "max_annual"))
    assert label["facts"]["evidence"]["salary"] == [{"quote": quote}] and any("cleared salary" in line for line in log)
    label["facts"]["currency"] = "VND"  # with a currency the numbers are kept
    label["document"]["compensation"].update(min_monthly=15, currency="VND")
    fix_jd(label, quote, [])
    assert label["document"]["compensation"]["min_monthly"] == 15


def test_fix_jd_rounds_fractional_years_up():
    label = {"document": {"skills": {"requirements": []}, "seniority": {"min_years": 2.5, "max_years": 4}},
             "facts": {"evidence": {}}}
    log = []
    fix_jd(label, "", log)
    assert label["document"]["seniority"] == {"min_years": 3, "max_years": 4} and any("2.5 -> 3" in line for line in log)


def test_fix_cv_salvages_dates_and_drops_degrees_off_the_scale():
    label = {"document": {"skills": [], "education": [{"startDate": "2016-Fall", "endDate": "2017-May"},
                                                      {"startDate": "--", "endDate": "Present"}]},
             "facts": {"completed_degrees": ["master", "professional_diploma"], "evidence": {}}}
    fix_cv(label, "", [])
    assert label["document"]["education"] == [{"startDate": "2016", "endDate": "2017-05"}, {"startDate": None, "endDate": None}]
    assert label["facts"]["completed_degrees"] == ["master"]


def test_fix_jd_keeps_preferred_wording_from_a_heading():
    text = "Preferred Qualifications\nExperience with TypeScript"
    label = {"document": {"skills": {"requirements": [{"id": "r", "options": ["TypeScript"], "importance": "preferred",
             "quote": "Experience with TypeScript", "importance_quote": "Preferred Qualifications"}]}},
             "facts": {"evidence": {}}}
    fix_jd(label, text, [])
    r = label["document"]["skills"]["requirements"][0]
    assert r["importance"] == "preferred" and r["importance_quote"] == "Preferred Qualifications"


def test_mend_fixes_syntax_slips_without_inventing_numbers():
    from scripts.qwen_repair import parse_balanced
    assert parse_balanced('{"document":{"skills":[{"name[".NET","p":1}]},"facts":{}}')["document"]["skills"][0]["name"] == ".NET"
    assert parse_balanced('{"document":{"c":{"min_monthly":70*8,"max":1}},"facts":{}}')["document"]["c"]["min_monthly"] is None
    assert parse_balanced('{"document":{"a":["x","y"]),"b":1},"facts":{}}')["document"]["b"] == 1  # stray ')'
    assert parse_balanced('{"document":{"q":[{"quote":"a"},"{"quote":"b"}]},"facts":{}}')["document"]["q"][1] == {"quote": "b"}


def test_mend_closes_a_list_before_a_key():
    from scripts.qwen_repair import parse_balanced
    out = parse_balanced('{"document":{},"facts":{"evidence":{"skills":[{"quote":"a"},"title":[{"quote":"b"}]}}}')
    assert out["facts"]["evidence"] == {"skills": [{"quote": "a"}], "title": [{"quote": "b"}]}


def test_codex_review4_regressions():
    from scripts.qwen_repair import parse_balanced, salvage_date
    assert salvage_date("2017 (summary)") == "2017" and salvage_date("May 2017") == "2017-05"
    assert salvage_date("Sept 2017") == "2017-09" and salvage_date("2017/05") == "2017-05" and salvage_date("Sept") is None
    assert parse_balanced('{"document":{"seniority":{"min_years":2>5}},"facts":{}}') is None  # never 25
    kept = parse_balanced('{"document":{"summary":"rate \\"x\\":70*8, ok"},"facts":{"a":1)}}')
    assert kept is None or kept["document"]["summary"] == 'rate "x":70*8, ok'  # text inside strings untouched
    quote = "Monthly salary: 5000 USD, approximately 125000000 VND."
    label = {"document": {"skills": {"requirements": []}, "compensation": {"currency": "USD", "max_monthly": 5000}},
             "facts": {"currency": "VND", "evidence": {"salary": [{"quote": quote}]}}}
    fix_jd(label, quote, [])
    assert label["document"]["compensation"]["max_monthly"] is None  # abstain, never "5000 VND"
    bare = {"document": {"skills": {"requirements": []}, "metadata": {"language": "en"}}, "facts": {"evidence": {}}}
    fix_jd(bare, "", [])
    assert "compensation" not in bare["document"]  # an untouched JD gains nothing


