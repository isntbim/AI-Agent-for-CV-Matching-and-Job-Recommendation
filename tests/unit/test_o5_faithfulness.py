"""O5 faithfulness check: verdict, numbers and skill names must come from the pair's facts."""
from scripts import o5_faithfulness as o5

FACTS = {"verdict": "Potential Fit", "fit": 47, "rule_score": 52, "job_title": "Data Engineer", "cv_title": "Data Analyst",
         "dimensions": {"skills": {"score": 40, "scored": True}, "experience": {"score": 50, "scored": False}},
         "matched_skills": ["python", "sql"], "missing_skills": ["airflow", "spark"], "found_in_cv_text": [],
         "soft_skills_not_scored": [], "vague_cv_titles": []}


def test_faithful_explanation_passes(monkeypatch):
    monkeypatch.setattr(o5, "vocab", lambda: ["airflow", "kubernetes", "python", "spark", "sql"])
    text = ("Potential Fit (fit 47/100). Your CV matches python and sql, so skills score 40/100. "
            "It is missing airflow and spark (2 of the 4 listed skills are matched). Experience was not scored.")
    assert o5.check(text, FACTS) == []


def test_invented_skill_number_and_verdict_are_caught(monkeypatch):
    monkeypatch.setattr(o5, "vocab", lambda: ["airflow", "kubernetes", "python", "spark", "sql"])
    problems = o5.check("Good Fit. You know Kubernetes and have 50 years in python.", FACTS)
    kinds = sorted(p.split(":")[0] for p in problems)
    assert kinds == ["number", "skill", "verdict", "verdict"]  # 50 is an unscored dimension's neutral value


def test_vietnamese_explanation_passes_and_needs_the_opening_label(monkeypatch):
    monkeypatch.setattr(o5, "vocab", lambda: ["airflow", "kubernetes", "python", "spark", "sql"])
    text = ("Có tiềm năng (độ phù hợp 47/100): bạn đang làm Data Analyst, gần với vị trí Data Engineer này. "
            "CV của bạn có python và sql, nhưng chưa thể hiện airflow và spark. Kỹ năng này chưa phù hợp hoàn toàn.")
    assert o5.check(text, FACTS) == []  # "chưa phù hợp" in prose is not a verdict claim
    wrong = o5.check("Phù hợp tốt (độ phù hợp 47/100): CV của bạn có python và sql, đây là Potential Fit.", FACTS)
    assert sorted(p.split(":")[0] for p in wrong) == ["verdict", "verdict"]


def test_wrong_stated_fit_is_caught(monkeypatch):
    monkeypatch.setattr(o5, "vocab", lambda: ["python", "sql"])
    assert [p.split(":")[0] for p in o5.check("Potential Fit (fit 40/100): your CV shows python.", FACTS)] == ["fit"]
    assert [p.split(":")[0] for p in o5.check("Có tiềm năng (độ phù hợp 52/100): CV của bạn có python và sql rồi.", FACTS)] == ["fit"]


def test_render_states_checked_results_and_hides_unscored_scores():
    from scripts.o5_export_sft import render
    facts = dict(FACTS, dimensions={"skills": {"score": 40, "scored": True}, "title": {"score": 100, "scored": True},
                                    "experience": {"score": 50, "scored": False}, "education": {"score": 0, "scored": True},
                                    "industry": {"score": 100, "scored": True}},
                 not_scored_because={"experience": "the job states no minimum years of experience"},
                 job_min_years=None, job_min_degree="bachelor", requirements=[], cv_recent_roles=[], cv_education=[])
    text = render(facts)
    checked, not_checked = text.split("NOT CHECKED")
    assert "titles match" in checked and "does not meet the bachelor degree requirement" in checked and "same field" in checked
    assert "Experience: the job states no minimum years" in not_checked and "50" not in text


def test_unscored_dimension_may_only_be_called_unchecked():
    facts = dict(FACTS, dimensions={"skills": {"score": 40, "scored": True}, "title": {"score": 50, "scored": False},
                                    "experience": {"score": 50, "scored": False}, "education": {"score": 50, "scored": False},
                                    "industry": {"score": 100, "scored": True}},
                 cv_recent_roles=[{"title": "Data Analyst", "dates": "2019-01 to Present"}], cv_education=["BS Statistics"])
    bad = ("Potential Fit (fit 47/100): your BS Statistics degree is relevant. Your Data Analyst role is close to this job. "
           "You have enough experience for it.")
    kinds = sorted(p.split(" was")[0] for p in o5.unscored(bad, facts))
    assert kinds == ["unscored: education", "unscored: experience", "unscored: job title"]
    fine = ("Potential Fit (fit 47/100): you match python and sql. Your experience could not be checked against this job. "
            "Add airflow to your CV, and use MS Office examples; I passed by the office bằng xe.")
    assert o5.unscored(fine, facts) == []


def test_vietnamese_style_rules():
    short = "Có tiềm năng (độ phù hợp 47/100): bạn hoàn toàn hoàn hảo cho vị trí này, mức lương cũng ổn."
    problems = " ".join(o5.style(short, FACTS))
    assert "tiếng" in problems and "hoàn hảo" in problems and "lương" in problems


GUARD_FACTS = dict(FACTS, matched_skills=["Word", "Excel"], missing_skills=["MS Office"], job_min_years=None,
                   job_title="Office Manager", nice_to_have_matched=[], nice_to_have_missing=["Notary"],
                   requirements=[{"requirement": "MS Office", "importance": "required", "status": "not met"},
                                 {"requirement": "Word", "importance": "required", "status": "met"}],
                   dimensions={"skills": {"score": 50, "scored": True}, "title": {"score": 100, "scored": True},
                               "experience": {"score": 50, "scored": False}, "education": {"score": 50, "scored": True},
                               "industry": {"score": 100, "scored": True}})


def test_contradictions_of_checked_facts_are_caught():
    def kinds(text):
        return sorted({p.split(":")[0] for p in o5.contradicts(text, GUARD_FACTS)})
    assert kinds("Your CV shows Word and MS Office.") == ["status"]
    assert kinds("Your CV does not show Excel.") == ["status"]
    assert kinds("It does not show MS Office or Notary, which are also required.") == ["importance"]
    assert kinds("CV chưa thể hiện MS Office và Notary, hai yêu cầu được ưu tiên.") == ["importance"]
    assert kinds("Chức danh của bạn gần với vị trí Office Manager.") == ["dimension"]
    assert kinds("Your work is in a different field, and your CV meets the degree requirement.") == ["dimension"]
    assert kinds("Kinh nghiệm của bạn không thể đối chiếu với yêu cầu 3 năm của vị trí này.") == ["years"]
    fine = ("Potential Fit (fit 47/100): your CV shows Word and Excel, but it does not show MS Office, which is required. "
            "Notary is a nice-to-have that is also not shown. Nothing else is missing. Your titles match this job title and "
            "your CV only partly meets the degree requirement. If you have used MS Office or Notary, add an example. "
            "Bạn có Word và Excel, nhưng CV chưa thể hiện MS Office; Notary là điểm cộng.")
    assert o5.contradicts(fine, GUARD_FACTS) == []


def test_fallback_template_passes_the_guard_in_both_languages(monkeypatch):
    from scripts.o5_export_sft import fallback
    monkeypatch.setattr(o5, "vocab", lambda: ["ms office", "word", "excel", "notary", "sql"])
    for vi in (False, True):
        text = fallback(GUARD_FACTS, vi)
        assert o5.violations(text, GUARD_FACTS) == [] and o5.is_vietnamese(text) == vi
        assert "MS Office" in text and "47/100" in text


def test_fallback_template_never_says_skills_are_absent_when_skills_were_not_scored():
    from scripts.o5_export_sft import fallback
    facts = dict(GUARD_FACTS, matched_skills=[], missing_skills=[], requirements=[], nice_to_have_missing=[],
                 dimensions=dict(GUARD_FACTS["dimensions"], skills={"score": 50, "scored": False}))
    text = fallback(facts)
    assert "skill requirements could not be checked" in text and "does not show" not in text
    assert text.count("could not be checked") == 1  # experience is unscored too, but only one unchecked item is named


def test_a_short_name_does_not_hide_a_longer_one():
    assert o5._named("no insurance accounting, no accounting", {"insurance accounting"}, {"accounting"}) == ["insurance accounting"]
    assert o5._named("uses sql server daily", {"sql"}, {"sql server"}) == []
