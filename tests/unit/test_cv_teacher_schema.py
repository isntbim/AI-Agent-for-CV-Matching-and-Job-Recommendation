"""Teacher labels must use only v3 schema keys; ResumeSchema would silently drop the rest."""

from scripts.cv_teacher_repair import repair
from scripts.cv_teacher_validate import undefined_keys


def gemini_style_label():
    return {"document": {"schema_version": "resume-v3",
                         "work": [{"company": "Google", "position": "Intern", "current": True, "location": "CA"}],
                         "education": [{"institution": "Grinnell College", "gpa": "3.8"}]},
            "facts": {"degrees": ["bachelor"], "total_months_work": 12}}


def test_undefined_keys_are_reported_at_any_depth():
    assert undefined_keys(gemini_style_label()) == [
        "document.education[].gpa", "document.work[].company", "document.work[].current",
        "document.work[].location", "facts.degrees", "facts.total_months_work"]
    assert undefined_keys({"document": {"basics": {"location": {"countryCode": "VN", "zip": "1"}}},
                           "facts": {"evidence": {"skills": [{"quote": "Python"}], "tools": []}}, "note": 1}) == [
        "document.basics.location.zip", "facts.evidence.tools", "note"]


def test_repair_maps_onto_schema_and_keeps_values():
    label = gemini_style_label()
    renamed, dropped = repair(label)
    assert undefined_keys(label) == []
    assert label["document"]["work"][0]["name"] == "Google"
    assert label["document"]["education"][0]["score"] == "3.8"
    assert label["facts"]["completed_degrees"] == ["bachelor"]
    assert dropped == {"total_months_work": 12, "work[1].current": True, "work[1].location": "CA"}


def test_repair_never_overwrites_a_stated_employer():
    label = {"document": {"work": [{"name": "Acme", "company": "Other"}]}, "facts": {}}
    repair(label)
    assert undefined_keys(label) == ["document.work[].company"]  # left for the validator to reject


def test_repair_null_flags_become_false_but_true_is_kept():
    label = {"document": {}, "facts": {"no_skills": None, "remote": True}}
    repair(label)
    assert label["facts"] == {"no_skills": False, "remote": True}


def test_repair_drops_string_provenance():
    label = {"document": {"skills": [{"details": [{"name": "HRIS", "provenance": "tp_hr_051"}]}]}, "facts": {}}
    _, dropped = repair(label)
    assert label["document"]["skills"][0]["details"][0] == {"name": "HRIS"}
    assert dropped == {"skills[1].details[1].provenance": "tp_hr_051"}


def test_check_lint_errors_and_warnings():
    from scripts.cv_teacher_check import lint
    text = "Jane Roe\nSkills: Excel (Advanced), HTML,CSS,andJavaScript\nSpanish (intermediate)\nClerk, Acme Corp"
    detail = lambda name, quote, **kw: {"name": name, "proficiency": "normal", "skill_quote": quote, **kw}
    label = {"document": {"schema_version": "resume-v3", "basics": {"name": "Jane Roe"},
                          "work": [{"name": "Acme Inc", "position": "Clerk"}],
                          "skills": [{"name": "Tools", "keywords": ["Excel", "JavaScript", "Word"],
                                      "details": [detail("Excel", "Excel (Advanced)", proficiency="advanced"),
                                                  detail("JavaScript", "HTML,CSS,andJavaScript"),
                                                  detail("Word", "Word processing")]}],
                          "languages": [{"language": "Spanish"}]},
             "facts": {"no_skills": None, "domains": ["finance"]}}
    errors, warns = lint(label, text, "tp_x")
    assert any("no_skills" in e for e in errors)
    assert any("raw_proficiency" in e for e in errors)  # advanced without raw/quote
    assert any("not an exact substring" in e and "Word processing" in e for e in errors)
    assert not any("Spanish" in e for e in errors)  # language rules live in policy_issues (written-in-source exemption)
    assert any("evidence.industry" in e for e in errors)
    assert any("glued" in w for w in warns) and not any("JavaScript" in e for e in errors)
    assert any("Acme Inc" in w for w in warns)


def test_arbiter_merge_adds_missing_document_prefix():
    from scripts.cv_teacher_arbiter_merge import normalise
    row = {"id": "x", "kind": "teacher_flag", "subject": "education", "final": [{"institution": "U"}]}
    assert normalise(row)[0]["subject"] == "document.education"
    assert normalise({"id": "x", "kind": "skill_only_haiku", "subject": "Excel", "final": {"name": "Excel"}})[0]["subject"] == "Excel"


def test_reference_records_skip_drop_and_take_basics():
    from scripts.cv_teacher_reference import build_records
    labels = {"haiku": {"document": {"work": [{"name": "A"}, {"name": "B"}]}}, "gemini": {"document": {}}}
    final = [{"teacher_record": "haiku:1"}, {"teacher_record": "haiku:2", "action": "drop"}]
    assert [r["name"] for r in build_records("document.work", final, labels)] == ["A"]
    projects = build_records("document.projects", [{"name": "Yoda", "type": "project", "teacher_record": None}], labels)
    assert projects == [{"name": "Yoda", "type": "project"}]


def test_policy_issues_enforce_the_training_ready_standard():
    import json
    from scripts.cv_teacher_validate import policy_issues
    good = {"document": {"schema_version": "resume-v3", "work": [{"name": None, "startDate": "2019-05", "endDate": "Present"}],
                         "education": [{"institution": "MIT", "startDate": "2015", "endDate": None}],
                         "projects": [], "certificates": [{"date": "2020"}], "languages": [{"language": "English"}],
                         "skills": [{"name": "Languages", "details": [{"name": "English"}]}]},
            "facts": {"country": None, "domains": [], "completed_degrees": [], "evidence": {}}}
    assert policy_issues(good, "English") == []
    bad = json.loads(json.dumps(good))
    bad["document"]["work"][0].update(name="Company Name", startDate="May 2019")
    bad["document"]["education"][0]["endDate"] = "Present"
    bad["document"]["certificates"][0]["date"] = "2022-2024"
    bad["document"]["skills"] = [{"name": "Tools", "details": [{"name": "SQL\nServer"}]}]
    bad["facts"]["country"] = "US"
    issues = policy_issues(bad, "English")
    assert {i.split(":")[0].split("=")[0].split("[")[0] for i in issues} >= {
        "date work", "date education", "date certificates", "placeholder work.name", "language without Languages skill",
        "skill name whitespace", "facts.country without evidence.location"}
    translated = json.loads(json.dumps(good))  # a language written in another language has no literal skill quote
    translated["document"]["skills"] = []
    assert policy_issues(translated, "TIENG ANH") == []
    impossible = json.loads(json.dumps(good))  # a DD/MM date read as MM/DD gives month 13
    impossible["document"]["work"][0]["startDate"] = "2015-13-04"
    impossible["document"]["certificates"][0]["date"] = "2015-02-31"
    assert [i.split("=")[0] for i in policy_issues(impossible, "English")] == ["date work[0].startDate", "date certificates[0].date"]
    malay = json.loads(json.dumps(good))  # whole words, any-case group name
    malay["document"]["languages"] = [{"language": "Malay"}]
    malay["document"]["skills"] = [{"name": "LANGUAGES", "details": [{"name": "Malayalam"}]}]
    assert policy_issues(malay, "Malay") == ["language without Languages skill: Malay",
                                             "Languages skill not in document.languages: malayalam"]


def test_check_reports_policy_hits_on_a_depth_error_too(tmp_path):
    import json
    from scripts.cv_teacher_validate import check
    label = {"document": {"schema_version": "resume-v3", "basics": {"name": "Candidate", "label": "Engineer"}, "work": [],
                          "education": [], "projects": [], "languages": [],
                          "certificates": [{"name": "Certificate", "date": "2014-2015"}],
                          "skills": [{"name": "Core", "level": None, "keywords": ["Python"], "details": [
                              {"name": "Python", "proficiency": "advanced", "raw_proficiency": "Advanced",
                               "skill_quote": "Python", "proficiency_quote": "Advanced"}]}]},
             "facts": {"domains": ["software"], "completed_degrees": [], "country": None,
                       "evidence": {"skills": [{"quote": "Python"}], "industry": [{"quote": "Engineer"}]}}}
    (tmp_path / "l.json").write_text(json.dumps(label), encoding="utf-8")
    result = check(tmp_path / "l.json", "Engineer. Advanced degree holder. Skills: Python", {})
    assert result["status"] == "depth_error" and result["policy"][0].startswith("date certificates[0].date")
