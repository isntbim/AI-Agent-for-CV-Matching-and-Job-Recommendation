"""O5 inputs: the facts the explanation model may use, one file per (CV, JD) pair.

Pairs come from the med2425 validation set (cached V4 extractions, output/match_eval/lab_docs.pkl), balanced by the final
model's verdict (No / Potential / Good Fit), at most 4 pairs per CV, split by CV (train/val/test never share a CV).
Each pair is scored exactly as the service scores it: V3.1 (v31_config) for the dimensions, skills and evidence, and
the final ranking model (configs/hybrid_qwen_v1.json) for the verdict and the 0-100 fit.

    python -m scripts.o5_build_pairs --n 1600      # writes output/o5/inputs/<pair_id>.json + output/o5/pairs.json
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from collections import defaultdict
from datetime import date
from hashlib import md5
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.ats_service import GRADES, HYBRID, v31_config, vague_titles
from scripts.hybrid_calibrate import DIMS, load as load_dims
from scripts.scorer_lab import AS_OF, PKL
from src.scoring.rules import _latest_title
from src.scoring.runtime_v3 import score_pair_v3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/o5"
RUN = "v32_text_soft"


def split_of(cv: str) -> str:
    h = int(md5(("o5" + cv).encode()).hexdigest(), 16) % 100
    return "test" if h < 8 else "val" if h < 15 else "train"


WHY_NOT_SCORED = {  # plain reasons for a neutral (unscored) dimension, keyed by the scorer's basis code
    "missing_title_evidence": "the CV or the job gives no comparable job title",
    "missing_duration_or_requirement": "the job states no years required, or the CV gives no dates for the relevant jobs",
    "missing_degree_evidence": "the job states no degree requirement, or the CV states no completed degree",
    "missing_domain_evidence": "the CV's jobs do not show a clear field",
    "missing_location_constraints": "the job sets no location requirement",
    "negotiable_default": "no salary is stated on both sides",
    "missing_skills": "the job lists no skills",
}
IMPORTANCE = {"core": "must-have", "standard": "required", "preferred": "nice-to-have"}


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", (name or "").lower())


def wording(resume, job):
    """name -> the spelling the job or CV used ("uiux principles" -> "UI/UX principles", "gaap" -> "GAAP")."""
    names = [s.name for s in (job.document.skills.required or []) + (job.document.skills.preferred or [])]
    names += [x.name for g in resume.document.skills for x in g.details] + [k for g in resume.document.skills for k in g.keywords]
    table = {}
    for n in names:
        table.setdefault(_key(n), n)
    return lambda n: table.get(_key(n), n)


def seniority(job) -> str | None:
    sen = job.document.seniority
    return (sen.level or sen.level_raw) if sen else None


def details(resume, job, result) -> dict:
    """Facts v2: the reasons behind the scores (requirements by importance, near matches, the CV's roles and degree,
    the job's years and degree) so an explanation can say why, not only restate numbers."""
    d, bd = result.dimensions, result.skills_breakdown or {}
    say, reqs = wording(resume, job), []
    for r in bd.get("requirements", []):
        if not r.get("scoring_target", True):
            continue
        status = {"full": "met", "partial": "partly met", "unmet": "not met"}.get(r.get("status"), r.get("status"))
        item = {"requirement": say(r.get("selected_option") or (r.get("options") or [""])[0]),
                "importance": IMPORTANCE.get(r.get("importance"), r.get("importance")), "status": status}
        if r.get("candidate_skill") and r.get("match_type") in ("near", "cv_text", "alias", "normalized_exact") and status != "not met":
            item["your_skill"] = say(r["candidate_skill"])
        reqs.append(item)
    order = {"must-have": 0, "required": 1, "nice-to-have": 2}
    reqs.sort(key=lambda x: (order.get(x["importance"], 3), {"not met": 0, "partly met": 1, "met": 2}.get(x["status"], 3)))
    roles = [{"title": w.position, "dates": " to ".join(x for x in (w.start_date, w.end_date) if x) or None}
             for w in resume.document.work[:4] if w.position]
    degrees = [" ".join(x for x in (e.study_type, e.area) if x) for e in resume.document.education[:3]]
    sen = job.document.seniority
    return {
        "requirements": reqs[:14],
        "nice_to_have_matched": [say(x) for x in (bd.get("preferred_matched_skills") or [])[:6]],
        "nice_to_have_missing": [say(x) for x in (bd.get("preferred_missing_skills") or [])[:6]],
        "job_level": seniority(job),
        "cv_recent_roles": roles,
        "cv_education": [x for x in degrees if x],
        "job_min_years": sen.min_years if sen else None,
        "job_min_degree": (job.facts.minimum_degree if job.facts else None),
        "not_scored_because": {k: why_not_scored(k, v.basis, sen, job, roles) for k, v in d.items()
                               if not v.supported and k in ("title", "experience", "education", "industry", "skills")},
    }


def why_not_scored(dim: str, basis: str, sen, job, roles) -> str:
    """A reason that is true for this pair: the scorer's basis codes cover several cases, so pick the one the facts show."""
    if dim == "experience":
        if not (sen and sen.min_years):
            return "the job states no minimum years of experience"
        return "the CV's dated roles could not be matched to this kind of job" if roles else "the CV gives no dated roles"
    if dim == "education":
        if not (job.facts and job.facts.minimum_degree):
            return "the job states no degree requirement"
        return "the CV does not clearly state a completed degree"
    if dim == "title":
        return "the CV's job titles could not be compared with this job title"
    if dim == "industry":
        return "the CV's jobs do not show a clear field"
    return WHY_NOT_SCORED.get(basis, "there is not enough evidence")


def facts(resume, job, result, grade: int, fit: float, as_of=AS_OF) -> dict:
    """Everything the explanation may mention. Numbers are rounded the way the UI shows them. The service passes the
    request's as_of date (the latest CV title depends on it)."""
    d, bd, say = result.dimensions, result.skills_breakdown or {}, wording(resume, job)
    return details(resume, job, result) | {
        "verdict": GRADES[grade], "fit": round(fit), "rule_score": round(result.overall_score),
        "job_title": job.document.title, "cv_title": _latest_title(resume, as_of) or next(
            (w.position for w in resume.document.work if w.position), None),
        "dimensions": {k: {"score": round(v.score), "scored": v.supported} for k, v in d.items()},
        "matched_skills": [say(x) for x in result.matched_skills[:12]], "missing_skills": [say(x) for x in result.missing_skills[:12]],
        "found_in_cv_text": [say(r["candidate_skill"]) for r in bd.get("requirements", []) if r.get("match_type") == "cv_text"][:8],
        "soft_skills_not_scored": [say(x) for x in bd.get("soft_skills", [])[:8]],
        "experience_basis": d["experience"].basis, "education_basis": d["education"].basis,
        "vague_cv_titles": [t for t in vague_titles(resume) if len(t.split()) == 1][:4],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=1600)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    lab = pickle.loads(PKL.read_bytes())
    docs = lab["docs"]
    model = json.loads(HYBRID.read_text(encoding="utf-8"))
    w = dict(zip(model["features"], model["coefficients"]))
    lo, hi = model["grade_cutoffs"]

    rows, X = [], []
    for split in ("dev", "test"):
        Xs, _, _, _ = load_dims(RUN, split, ["text_cos", "title_cos"], "qwen3")
        rows += [json.loads(line) for line in (ROOT / f"output/match_eval/lab/{RUN}.{split}.pairs.jsonl").open()]
        X.append(Xs)
    X = np.vstack(X)
    from scripts import match_eval as m  # texts for TF-IDF, fitted on all documents as in match_ir_eval
    texts = {}
    for pr in m.load_pairs():
        texts.setdefault(m.doc_id("cv", pr["cv"]), pr["cv"])
        texts.setdefault(m.doc_id("jd", pr["jd"]), pr["jd"])
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(texts.values()))

    # the final model's logit for every pair, then balance by its verdict, at most 4 pairs per CV
    vec = {k: tfidf.transform([t]) for k, t in texts.items()}
    tf_all = np.array([float(vec[r["cv"]].multiply(vec[r["jd"]]).sum()) for r in rows])
    coef = np.array([w[f] for f in model["features"]])
    logits = model["intercept"] + np.hstack([X, tf_all[:, None]]) @ coef
    grades = np.digitize(logits, [lo, hi])
    rng, per_cv, got, picked = random.Random(a.seed), defaultdict(int), defaultdict(int), []
    order = list(range(len(rows)))
    rng.shuffle(order)
    want = {0: a.n * .35, 1: a.n * .35, 2: a.n * .30}
    for i in order:
        r, g = rows[i], int(grades[i])
        if got[g] >= want[g] or per_cv[r["cv"]] >= 4:
            continue
        got[g] += 1; per_cv[r["cv"]] += 1; picked.append(i)
        if len(picked) >= a.n:
            break

    config, out = v31_config(), []
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    for i in picked:
        r = rows[i]
        resume, job = docs[r["cv"]], docs[r["jd"]]
        result = score_pair_v3(resume, job, as_of_date=AS_OF, config=config, cv_text=texts[r["cv"]])
        tf = tf_all[i]
        feats = dict(zip(DIMS, (result.unrounded_scores.model_dump()[d] / 100 for d in DIMS)))
        feats |= {"text_cos": X[i][len(DIMS)], "title_cos": X[i][len(DIMS) + 1], "tfidf_cos": tf}
        logit = model["intercept"] + sum(w[f] * feats[f] for f in model["features"])
        grade, fit = (0 if logit < lo else 1 if logit < hi else 2), 100 / (1 + math.exp(-logit))
        pid = f"o5_{md5((r['cv'] + r['jd']).encode()).hexdigest()[:10]}"
        f = facts(resume, job, result, grade, fit)
        (OUT / "inputs" / f"{pid}.json").write_text(json.dumps({"pair_id": pid, "facts": f}, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
        out.append({"pair_id": pid, "cv": r["cv"], "jd": r["jd"], "label": r["label"], "grade": grade,
                    "split": split_of(r["cv"])})
    (OUT / "pairs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    c = defaultdict(int)
    for o in out:
        c[(o["split"], GRADES[o["grade"]])] += 1
    print(len(out), "pairs;", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
