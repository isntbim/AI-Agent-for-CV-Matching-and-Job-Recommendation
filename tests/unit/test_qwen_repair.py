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
