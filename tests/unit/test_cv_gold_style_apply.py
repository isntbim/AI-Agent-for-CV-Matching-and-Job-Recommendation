from scripts.cv_gold_style_apply import apply

TEXT = "Skills: Effective time management, Python\nStrong leadership and interpersonal skills"
LABEL = {"document": {"skills": [{"name": "Skills", "level": None, "keywords": ["time management", "Python", "leadership"],
                                  "details": [{"name": "time management", "skill_quote": "Effective time management"},
                                              {"name": "Python", "skill_quote": "Python"},
                                              {"name": "leadership", "skill_quote": "Strong leadership"}]}]},
         "facts": {}}


def test_keeps_drops_and_adds():
    out, notes = apply(LABEL, {"keep": ["python"], "add": [{"name": "Effective time management",
                                                            "quote": "Effective time management"}]}, TEXT)
    assert notes == []
    names = [d["name"] for g in out["document"]["skills"] for d in g["details"]]
    assert names == ["Python", "Effective time management"]
    assert LABEL["document"]["skills"][0]["keywords"] == ["time management", "Python", "leadership"]  # input untouched


def test_keeps_languages_group_and_tidies_added_names():
    text = "Problem -\nSolving  skills\nEnglish"
    label = {"document": {"skills": [{"name": "Languages", "level": None, "keywords": ["English"],
                                      "details": [{"name": "English", "skill_quote": "English"}]}]}, "facts": {}}
    out, errors = apply(label, {"keep": [], "add": [{"name": "Problem -\nSolving  skills", "quote": "Problem -\nSolving  skills"}]}, text)
    assert errors == []
    names = [d["name"] for g in out["document"]["skills"] for d in g["details"]]
    assert names == ["English", "Problem - Solving skills"]


def test_rejects_unknown_keep_and_invented_quote():
    out, errors = apply(LABEL, {"keep": ["java"], "add": [{"name": "Excel", "quote": "Excel expert"}]}, TEXT)
    assert out is None
    assert len(errors) == 2
