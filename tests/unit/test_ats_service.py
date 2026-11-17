"""The service's V3.1 config and templated explanation."""
from datetime import date

import pytest

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
    breakdown = {"requirements": [{"match_type": "cv_text", "candidate_skill": "c#"}, {"match_type": "none"}],
                 "soft_skills": ["communication", "english"]}
    result = SimpleNamespace(grade=0, overall_score=41.8, matched_skills=[], missing_skills=["sap"],
                             skills_breakdown=breakdown, dimensions={"skills": dim(0), "title": dim(50, False), "experience": dim(50, False),
                                         "education": dim(50), "industry": dim(100), "location": dim(50, False),
                                         "salary": dim(75, False)})
    lines = explain(result, grade=2, fit=94, why=Ranker.why(contrib))
    assert lines[0].startswith("Good Fit: fit 94/100") and lines[1].startswith("Fit comes") and lines[2].startswith("Caution")
    assert not any(l.startswith("Caution") for l in explain(result, grade=1, fit=60))
    assert "Found in your CV text (not in the extracted skills list): c#." in lines
    assert "Also asked for, not scored (soft skills, languages, Vietnam-specific law): communication, english." in lines
    assert not any(l.startswith("Title and experience not scored") for l in lines)
    assert any(l.startswith("Title and experience not scored: your CV's job titles are too general to compare (IT, Leader)")
               for l in explain(result, vague=["IT", "Leader"]))
    result.dimensions["experience"] = dim(100, True)
    assert any(l.startswith("Title not scored: your CV's job titles are too general") for l in explain(result, vague=["IT"]))
    result.dimensions["skills"] = dim(54.5, True)
    assert any(l.startswith("Skills 55/100.") for l in explain(result))


def test_vague_titles_only_when_no_title_names_a_role():
    from types import SimpleNamespace
    from scripts.ats_service import vague_titles
    cv = lambda *ps: SimpleNamespace(document=SimpleNamespace(work=[SimpleNamespace(position=p) for p in ps]))
    assert vague_titles(cv("IT", "Leader", "Leader", None)) == ["IT", "Leader"]
    assert vague_titles(cv("IT", "Staff Accountant")) == []
    assert vague_titles(cv("Backend JAVA developer", "Full stack JAVA developer")) == []
    assert vague_titles(cv()) == []


def test_match_leaves_out_a_repeated_jd_text(tmp_path, monkeypatch):
    import scripts.ats_service as svc
    for name, text in (("a", "Same posting"), ("b", "Other posting"), ("c", "Same posting")):
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")

    class Cache:
        def get(self, text, kind):
            if kind == "job":
                raise ValueError(text)  # every JD lands in `skipped`, which shows which ones were tried
            return None
    monkeypatch.setattr(svc, "ROOT", tmp_path)
    monkeypatch.setattr(svc, "V4Client", lambda llm=None: None)
    monkeypatch.setattr(svc, "_llm", lambda: None)
    monkeypatch.setattr(svc, "_cache", lambda *a, **k: Cache())
    monkeypatch.setattr(svc, "_jd_paths", lambda: {j: f"{j}.txt" for j in "abc"})
    out = svc.match("cv", "2026-11-08", jd_ids=["a", "b", "c"])
    assert sorted(out["skipped"]) == ["a", "b"]  # c repeats a's text


O5_FACTS = {"verdict": "Potential Fit", "fit": 47, "matched_skills": ["Python"], "missing_skills": ["Docker"],
            "dimensions": {d: {"score": 50, "scored": d == "skills"} for d in ("skills", "title", "experience",
                                                                               "education", "industry")}}


@pytest.mark.parametrize("outputs, expected_source, calls", [
    (["good"], "o5", [1.0]),                         # passes the guard: shown as written
    (["bad", "good"], "o5", [1.0, 1.1]),             # rejected once: regenerated with repetition penalty 1.1
    (["bad", "bad"], "template", [1.0, 1.1]),        # rejected twice: the template, filled from the same facts
    ([RuntimeError("explainer down")], "template", [1.0]),  # unreachable explainer never blocks the match
])
def test_o5_explain_guard_retry_then_template(monkeypatch, outputs, expected_source, calls):
    import scripts.ats_service as svc
    import scripts.o5_faithfulness as faith
    seen = []

    def generate(facts, vi, llm, penalty=1.0):
        seen.append(penalty)
        out = outputs[len(seen) - 1]
        if isinstance(out, Exception):
            raise out
        return out
    monkeypatch.setattr(svc, "_o5_generate", generate)
    monkeypatch.setattr(faith, "violations", lambda text, facts: [] if text == "good" else ["number"])
    text, source = svc.o5_explain(O5_FACTS, False, llm=None)
    assert (source, seen) == (expected_source, calls)
    assert text == "good" if source == "o5" else text.startswith("Potential Fit (fit 47/100): your CV shows Python.")


def test_ranker_scores_pasted_jd_without_touching_jd_emb_cache(tmp_path, monkeypatch):
    import math
    import numpy as np
    import scripts.ats_service as svc
    import src.embeddings.qwen as qwen_mod

    calls = []

    class FakeEmbedder:
        def encode(self, texts, query=False):
            calls.append((list(texts), query))
            vecs = []
            for i, t in enumerate(texts, 1):
                v = np.array([float(len(t) % 11 + 1), float(i + (3 if query else 1))], dtype=np.float32)
                vecs.append(v / np.linalg.norm(v))
            return np.vstack(vecs)

    emb_file = tmp_path / "jd_emb.npz"
    monkeypatch.setattr(svc, "JD_EMB", emb_file)
    monkeypatch.setattr(qwen_mod, "QwenEmbedder", FakeEmbedder)

    corpus = {
        "jd_001": "Backend engineer experienced in Python, Django, REST APIs and PostgreSQL.",
        "jd_002": "Frontend developer skilled in TypeScript, React, Next.js and Tailwind CSS.",
    }
    ranker = svc.Ranker(corpus)
    saved_bytes = emb_file.read_bytes()
    saved_vocab = dict(ranker.tfidf.vocabulary_)
    calls.clear()

    pasted_text = "Hiring a Senior Python Backend Engineer to build FastAPI and PostgreSQL microservices."
    dims = {"skills": 80, "title": 100, "experience": 75, "education": 100, "industry": 100}
    (logit,) = ranker.scores("Senior Python engineer with FastAPI and PostgreSQL", "Python Engineer",
                             [("pasted", "Senior Python Backend Engineer", dims)], texts={"pasted": pasted_text})

    assert math.isfinite(logit)
    assert "pasted" not in ranker.jd_emb
    assert emb_file.read_bytes() == saved_bytes
    assert set(np.load(emb_file).files) == {"jd_001", "jd_002"}
    assert ranker.tfidf.vocabulary_ == saved_vocab
    assert ([pasted_text], False) in calls  # embedded on document side (query=False), like corpus JDs


def test_full_graph_pasted_jd_runs_ranker_and_o5_and_reraises_jd_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import scripts.ats_service as svc
    import src.embeddings.qwen as qwen_mod
    from src.parser.llm_extractor import ExtractionError

    cv, jd = inputs()

    class FakeEmbedder:
        def encode(self, texts, query=False):
            vecs = []
            for i, t in enumerate(texts, 1):
                v = np.array([float(len(t) % 7 + 1), float(i + (2 if query else 1))], dtype=np.float32)
                vecs.append(v / np.linalg.norm(v))
            return np.vstack(vecs)

    monkeypatch.setattr(svc, "JD_EMB", tmp_path / "jd_emb.npz")
    monkeypatch.setattr(qwen_mod, "QwenEmbedder", FakeEmbedder)
    real_ranker = svc.Ranker({
        "jd_001": "Backend engineer with Python, Django, and PostgreSQL experience.",
        "jd_002": "Frontend engineer with TypeScript, React, and Next.js experience.",
    })

    fail_jd = False

    class Cache:
        def get(self, text, kind):
            if kind == "job" and fail_jd:
                raise ExtractionError("exceeds conservative context budget")
            return cv if kind == "resume" else jd

    pasted_jd = "Senior Backend Developer role requiring Python, FastAPI, Docker, and SQL. " * 4
    steps, o5_facts = [], []
    monkeypatch.setattr(svc, "V4Client", lambda llm=None: None)
    monkeypatch.setattr(svc, "_llm", lambda: SimpleNamespace())
    monkeypatch.setattr(svc, "_cache", lambda *a, **k: Cache())
    monkeypatch.setattr(svc, "retrieve", lambda *a, **k: (_ for _ in ()).throw(AssertionError("retrieve must not run")))
    monkeypatch.setattr(svc, "o5_explain", lambda facts, vi, llm: (o5_facts.append(facts) or ("Checked O5 summary.", "o5")))

    out = svc.match("Candidate CV with Python and SQL", "2024-01-01", top=5, ranker=real_ranker,
                    on_step=steps.append, jd_text=pasted_jd)
    assert steps == ["extract", "score", "rank"]
    assert out["retrieved"] is False and out["pool"] == 1 and len(out["matches"]) == 1
    match_row = out["matches"][0]
    assert match_row["jd"] == "pasted" and match_row["job_text"] == pasted_jd
    assert match_row["summary"] == "Checked O5 summary." and match_row["summary_source"] == "o5"
    assert len(o5_facts) == 1 and o5_facts[0]["verdict"] in svc.GRADES and "dimensions" in o5_facts[0]

    # Addendum B: pasted-JD extraction failure re-raises (marked source="jd") instead of becoming an empty run
    fail_jd = True
    with pytest.raises(ExtractionError) as exc_info:
        svc.match("Candidate CV with Python and SQL", "2024-01-01", top=5, ranker=real_ranker, jd_text=pasted_jd)
    assert getattr(exc_info.value, "source", None) == "jd"
