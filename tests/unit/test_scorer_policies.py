"""Opt-in V3 policies: contained title aliases and lower-bound experience."""
from datetime import date

from src.parser.schemas import WorkExperience
from src.scoring import ScorerConfig
from src.scoring.normalization import contained_role
from src.scoring.runtime_v3 import score_pair_v3
from tests.unit.test_ats_scorer import inputs

AS_OF = date(2024, 1, 1)
STRICT = ScorerConfig(version="ats-en-v3", review_policy="automatic")


def test_contained_role_finds_longest_alias_and_skips_traps():
    assert contained_role("Property Accountant") == ("accountant", "finance")
    assert contained_role("Electrical Engineer Intern") == ("electrical engineer", "engineering")
    assert contained_role("Software Engineering Manager")[1] is None  # "engineering" is not "engineer"
    assert contained_role("SQL Server DBA")[1] is None  # excluded ambiguous alias
    assert contained_role("Software Engineer") == ("software engineer", "software")


def test_default_config_serialises_without_new_policies():
    assert "title_policy" not in STRICT.model_dump() and "experience_policy" not in STRICT.model_dump()
    assert ScorerConfig(version="ats-en-v3", title_policy="contained").model_dump()["title_policy"] == "contained"


def test_contained_title_policy_supports_decorated_titles():
    cv, jd = inputs()
    jd.document.title = "Software Engineer (12-month contract, remote)"
    assert not score_pair_v3(cv, jd, config=STRICT, as_of_date=AS_OF).dimensions["title"].supported
    contained = STRICT.model_copy(update={"title_policy": "contained"})
    assert score_pair_v3(cv, jd, config=contained, as_of_date=AS_OF).dimensions["title"].supported


def test_lower_bound_only_proves_a_requirement_met():
    cv, jd = inputs()  # 2020-01..Present relevant = 48 months by 2024-01
    cv.document.work.append(WorkExperience(position="Barista"))  # unknown role vetoes strict scoring
    lower = STRICT.model_copy(update={"experience_policy": "lower_bound"})
    jd.document.seniority.min_years = 3
    assert not score_pair_v3(cv, jd, config=STRICT, as_of_date=AS_OF).dimensions["experience"].supported
    met = score_pair_v3(cv, jd, config=lower, as_of_date=AS_OF).dimensions["experience"]
    assert met.supported and met.score == 100
    jd.document.seniority.min_years = 10  # 48 known months < 120: the unknown job might cover the gap
    assert not score_pair_v3(cv, jd, config=lower, as_of_date=AS_OF).dimensions["experience"].supported


def test_contained_role_scope_words_and_compounds():
    assert contained_role("Software Engineer Manager")[1] is None  # manages engineers, not one
    assert contained_role("Data Analyst/Data Engineer") == ("data analyst/data engineer", "data")  # family only
    assert contained_role("Data Engineer/Nurse")[1] is None  # cross-family: ambiguous
    assert contained_role("Sales Manager - Retail") == ("sales manager", "sales")  # scope word inside the alias


def test_relation_memo_sees_in_place_edits():
    from src.scoring import skills_v3
    rel = [{"id": f"r{i}", "candidate": f"skill {i}", "requirement": f"need {i}", "relation": "near", "reason": "x",
            "source": "test", "review_status": "approved"} for i in range(60)]
    cfg = ScorerConfig(version="ats-en-v3", relations=rel)
    first = skills_v3._per_relations("relations", cfg, lambda: skills_v3._relations(cfg))
    assert len(first) == 60
    cfg.relations[0].review_status = "rejected"  # Codex's repro: same list, same length
    assert len(skills_v3._per_relations("relations", cfg, lambda: skills_v3._relations(cfg))) == 59
