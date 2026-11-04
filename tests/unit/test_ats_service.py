"""The service's V3.1 config and templated explanation."""
from datetime import date

from scripts.ats_service import GRADES, explain, v31_config
from src.scoring.runtime_v3 import score_pair_v3
from tests.unit.test_ats_scorer import inputs


def test_v31_config_loads_the_committed_relations_as_approved():
    cfg = v31_config()
    assert len(cfg.relations) == 2127 and {r.review_status for r in cfg.relations} == {"approved"}
    assert (cfg.title_policy, cfg.experience_policy, cfg.lower_threshold, cfg.upper_threshold) == ("contained", "lower_bound", 43, 60)


def test_explanation_reads_only_the_score_result():
    cv, jd = inputs()
    result = score_pair_v3(cv, jd, config=v31_config(), as_of_date=date(2024, 1, 1))
    lines = explain(result)
    assert lines[0].startswith(GRADES[result.grade])
    for skill in result.matched_skills[:8]:
        assert skill in lines[1]
    unscored = [k for k, v in result.dimensions.items() if not v.supported]
    assert not unscored or all(k in lines[-1] for k in unscored)
