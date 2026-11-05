"""The service's V3.1 config and templated explanation."""
from datetime import date

from scripts.ats_service import GRADES, explain, v31_config
from src.scoring.runtime_v3 import score_pair_v3
from tests.unit.test_ats_scorer import inputs


def test_v31_config_loads_the_committed_relations_as_approved():
    cfg = v31_config()
    assert len(cfg.relations) == 2127 and {r.review_status for r in cfg.relations} == {"approved"}
    assert (cfg.title_policy, cfg.experience_policy, cfg.lower_threshold, cfg.upper_threshold) == ("contained", "lower_bound", 43, 58)


def test_explanation_reads_only_the_score_result():
    cv, jd = inputs()
    result = score_pair_v3(cv, jd, config=v31_config(), as_of_date=date(2024, 1, 1))
    lines = explain(result)
    assert lines[0].startswith(GRADES[result.grade])
    for skill in result.matched_skills[:8]:
        assert skill in lines[1]
    unscored = [k for k, v in result.dimensions.items() if not v.supported]
    assert not unscored or all(k in lines[-1] for k in unscored)


def test_ranker_applies_the_saved_logistic_model():
    import numpy as np
    from scripts.ats_service import Ranker

    class Emb:  # unit vectors: CV text [1,0], JD [0.6,0.8] -> cos 0.6; titles identical -> 1.0
        def encode(self, texts, query=False):
            return np.array([[1.0, 0.0]] * len(texts))

    class Tfidf:
        def transform(self, texts):
            from scipy.sparse import csr_matrix
            return csr_matrix([[1.0, 0.0]] if texts[0] == "cv" else [[0.5, 0.0]])

    r = Ranker.__new__(Ranker)
    r.model = {"features": ["skills", "title", "experience", "education", "industry", "text_cos", "title_cos", "tfidf_cos"],
               "coefficients": [1, 0, 0, 0, 2, 3, 4, 5], "intercept": -1, "grade_cutoffs": [0.0, 5.0]}
    r.emb, r.tfidf, r.jd_texts, r.jd_emb = Emb(), Tfidf(), {"j": "jd"}, {"j": np.array([0.6, 0.8])}
    dims = {"skills": 50, "title": 0, "experience": 0, "education": 0, "industry": 100}
    (logit,) = r.scores("cv", "Dev", [("j", "Dev", dims)])
    assert abs(logit - (-1 + 0.5 + 2 + 3 * 0.6 + 4 * 1.0 + 5 * 0.5)) < 1e-9
    assert (r.grade(-0.1), r.grade(0.0), r.grade(9.8)) == (0, 1, 2)


def test_why_names_the_driver_and_good_fit_without_skills_is_flagged():
    from types import SimpleNamespace
    from scripts.ats_service import Ranker, explain
    contrib = {"skills": 0.0, "title": 0.33, "experience": 0.08, "education": -0.03, "industry": 1.38,
               "text_cos": 4.94, "title_cos": 4.15, "tfidf_cos": 3.28,
               "_values": {"text_cos": 0.60, "title_cos": 0.83, "tfidf_cos": 0.23}}
    assert Ranker.why(contrib).startswith("Fit comes mostly from how similar the CV and the job are")
    dim = lambda s, ok=True: SimpleNamespace(supported=ok, score=s, basis="", evidence=[])
    result = SimpleNamespace(grade=0, overall_score=41.8, matched_skills=[], missing_skills=["sap"],
                             dimensions={"skills": dim(0), "title": dim(50, False), "experience": dim(50, False),
                                         "education": dim(50), "industry": dim(100), "location": dim(50, False),
                                         "salary": dim(75, False)})
    lines = explain(result, grade=2, fit=94, why=Ranker.why(contrib))
    assert lines[0].startswith("Good Fit: fit 94/100") and lines[1].startswith("Fit comes") and lines[2].startswith("Caution")
    assert not any(l.startswith("Caution") for l in explain(result, grade=1, fit=60))
