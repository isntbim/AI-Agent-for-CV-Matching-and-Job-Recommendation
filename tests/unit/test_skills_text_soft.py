"""Unit tests for ScorerConfig.skills_policy='text_and_soft' in skills_v3 and score_pair_v3."""

from datetime import date

from src.scoring import JobInput, ResumeInput, ScorerConfig
from src.scoring.runtime_v3 import score_pair_v3
from src.scoring.skills_v3 import score_skills_v3

DATE = date(2026, 9, 30)


def make_pair(jd_skills: list[str], cv_skills: list[str] | None = None):
    cv_skills = cv_skills if cv_skills is not None else ["Python"]
    cv = ResumeInput.model_validate({
        "document": {
            "schema_version": "resume-v3",
            "basics": {"label": "Software Engineer"},
            "skills": [{
                "name": "Technical",
                "keywords": cv_skills,
                "details": [{"name": s, "proficiency": "normal", "skill_quote": s} for s in cv_skills],
            }],
        },
        "facts": {"evidence": {"skills": [{"quote": s} for s in cv_skills], "title": [{"quote": "Software Engineer"}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"},
    })
    reqs = [
        {
            "id": f"r{i}",
            "options": [opt],
            "importance": "core",
            "quote": f"{opt} is required",
            "importance_quote": f"{opt} is required",
            "review_status": "approved",
            "option_details": [{"name": opt, "minimum_proficiency": "normal"}],
        }
        for i, opt in enumerate(jd_skills, start=1)
    ]
    jd = JobInput.model_validate({
        "document": {
            "schema_version": "job-v3",
            "id": "jd",
            "url": "urn:jd",
            "scraped_at": "none",
            "title": "Software Engineer",
            "title_normalized": "software_engineer",
            "category": "software_engineer",
            "skills": {"requirements": reqs},
        },
        "facts": {"evidence": {"skills": [{"quote": f"{opt} is required"} for opt in jd_skills], "title": [{"quote": "Software Engineer"}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"},
    })
    return cv, jd


def cfg(skills_policy="extracted", **kwargs):
    return ScorerConfig(version="ats-en-v3", skills_policy=skills_policy, **kwargs)


def test_default_policy_has_no_soft_skills_and_ignores_cv_text():
    cv, jd = make_pair(["Python", "English", "C#"])
    without_text = score_skills_v3(cv, jd, cfg())
    with_text = score_skills_v3(cv, jd, cfg(), cv_text="Experienced in Python, C#, TypeScript and fluent English.")
    assert "soft_skills" not in with_text
    assert with_text["score"] == without_text["score"]
    assert with_text["matched_skills"] == ["python"]


def test_text_and_soft_moves_english_to_soft_skills():
    cv, jd = make_pair(["Python", "English"])
    result = score_skills_v3(cv, jd, cfg("text_and_soft"))
    assert "english" in result["soft_skills"]
    assert "english" not in result["missing_skills"]
    assert result["score"] == 100


def test_text_and_soft_matches_csharp_from_cv_text_and_via_score_pair_v3():
    cv, jd = make_pair(["C#"])
    cv_text = "Built backend services using ... C#, TypeScript ... in production."
    result = score_skills_v3(cv, jd, cfg("text_and_soft"), cv_text=cv_text)
    assert "c#" in result["matched_skills"]
    row = result["requirements"][0]
    assert row["match_type"] == "cv_text"
    assert "C#, TypeScript" in row["evidence"]["skill_quote"]  # quoted from the original CV text

    pair_result = score_pair_v3(cv, jd, config=cfg("text_and_soft"), as_of_date=DATE, cv_text=cv_text)
    assert "c#" in pair_result.matched_skills
    assert pair_result.scores.skills == 100.0


def test_text_and_soft_ignores_generic_lowercase_word_state():
    cv, jd = make_pair(["state"])
    result = score_skills_v3(cv, jd, cfg("text_and_soft"), cv_text="Austin, State")
    assert "state" not in result["matched_skills"]
    assert "state" in result["missing_skills"]
    assert result["requirements"][0]["match_type"] == "none"


def test_text_and_soft_only_soft_requirements_returns_neutral_50():
    cv, jd = make_pair(["English", "Communication"])
    result = score_skills_v3(cv, jd, cfg("text_and_soft"))
    assert result["basis"] == "soft_skills_only"
    assert result["score"] == 50
    assert set(result["soft_skills"]) == {"english", "communication"}


def test_text_mention_guards_from_review():
    from src.scoring.skills_v3 import _Requirement, _text_mention
    req = lambda opt: _Requirement("r", (opt.lower(),), (opt,), "standard", "", None, "approved", (), ("r",))
    found = lambda opt, text, taken=(): (_text_mention(req(opt), text, set(taken)) or (None,))[0]
    assert found("SQL", "I have no professional experience working with SQL.") is None  # negated
    assert found("Excel", "Proficient with QuickBooks, Excel, Word") == "excel"
    assert found("Excel", "I excel at teamwork. Excel in problem solving") is None  # verb, not the product
    assert found("Spring", "Spring 2024 internship") is None
    assert found("C#", "Stack: .Net, C#") == "c#"
    assert found("Python", "Python, SQL", taken={"python"}) is None  # one CV skill fulfils one requirement
    assert found("Pinia or Redux or Zustand", "State: Redux Toolkit") == "redux"  # unsplit "X or Y" option
    assert found("Object-Oriented Programming (OOP)", "Strong OOP background") == "oop"
    assert found("RESTful APIs", "Built a RESTful API in Flask") == "restful apis"  # plural JD, singular CV
    assert found("React", "She reacts fast") is None  # a singular JD name never gains an "s"
    assert found("C++ (11)", "Room 11") is None  # a part needs a letter


def test_text_and_soft_lists_vietnam_law_apart():
    cv, jd = make_pair(["Python", "Vietnam Labor Code", "SHUI"])
    result = score_skills_v3(cv, jd, cfg("text_and_soft"))
    assert set(result["soft_skills"]) == {"vietnam labor code", "shui"}
    assert result["score"] == 100
