"""Scorer lab: re-score the cached V4 extractions of the 59k validation pairs under a variant, on a CV-level split.

Tune on --split dev only; report --split test once per finished variant (CVs are hashed into halves, so no CV is in both).

  python -m scripts.scorer_lab --name baseline
  python -m scripts.scorer_lab --name rel --relations output/match_eval/lab/relations.json --lower 42
  python -m scripts.scorer_lab --name w --weights '{"skills":0.3,"title":0.2,...}'

Needs output/match_eval/lab_docs.pkl (built once with --build from the extraction cache).
"""
import argparse
import json
import pickle
from collections import defaultdict
from datetime import date
from hashlib import md5
from multiprocessing import Pool
from pathlib import Path

from src.parser import match_schemas
from src.parser.skill_v2 import SkillRelationTable
from src.scoring.models import ScorerConfig
from src.scoring.runtime_v3 import score_pair_v3

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "output/match_eval/lab_docs.pkl"
OUT = ROOT / "output/match_eval/lab"
AS_OF = date(2024, 1, 1)
DIMS = ["skills", "title", "experience", "education", "industry", "location", "salary"]


def build() -> None:
    from scripts import match_eval as m
    pairs, get, local, docs = m.load_pairs(), m.caches(m.Replay(m.OUT / "prompts.jsonl", [])), m.caches(), {}
    for p in pairs:
        for kind, key in (("resume", "cv"), ("job", "jd")):
            did = m.doc_id(key, p[key])
            if did not in docs:
                docs[did] = None  # failed extraction: its pairs are not scored, as in match_eval
                for source in (get, local):  # A100 pod replay first, then match_eval --local-fallback
                    cache = source(kind)
                    cache.client = None
                    try:
                        docs[did] = cache.get(p[key], kind)
                        break
                    except Exception:
                        pass
    slim = [{"cv": m.doc_id("cv", p["cv"]), "jd": m.doc_id("jd", p["jd"]), "label": p["label"]} for p in pairs]
    PKL.write_bytes(pickle.dumps({"docs": docs, "pairs": slim}))


def split_of(cv_id: str) -> str:
    return "dev" if int(md5(cv_id.encode()).hexdigest(), 16) % 2 == 0 else "test"


_state = {}


def _init(config_json: str, weights: dict | None) -> None:
    _state["docs"] = pickle.loads(PKL.read_bytes())["docs"]
    _state["config"] = ScorerConfig.model_validate_json(config_json)
    if weights:  # runtime_v3 reads ATS_WEIGHTS at call time
        match_schemas.ATS_WEIGHTS.clear()
        match_schemas.ATS_WEIGHTS.update(weights)


def _score(pair: dict) -> dict:
    docs, config = _state["docs"], _state["config"]
    r = score_pair_v3(docs[pair["cv"]], docs[pair["jd"]], as_of_date=AS_OF, config=config)
    dims = r.unrounded_scores.model_dump()
    return {**pair, "score": r.unrounded_overall_score, "grade": r.grade,
            "supported": {d: r.dimensions[d].supported for d in DIMS}, **{f"dim_{d}": dims[d] for d in DIMS}}


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC with ties counted half."""
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, rank_sum = 0.0, 0, 0.0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v):
        order = sorted(range(len(v)), key=v.__getitem__)
        out, i = [0.0] * len(v), 0
        while i < len(v):
            j = i
            while j < len(v) and v[order[j]] == v[order[i]]:
                j += 1
            for k in range(i, j):
                out[order[k]] = (i + j - 1) / 2
            i = j
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if not sx or not sy else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def fit_f1(rows: list[dict], lower: float) -> dict:
    tp = sum(1 for r in rows if r["label"] >= 1 and r["score"] >= lower)
    fp = sum(1 for r in rows if r["label"] == 0 and r["score"] >= lower)
    fn = sum(1 for r in rows if r["label"] >= 1 and r["score"] < lower)
    p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"cutoff": lower, "precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / max(p + rc, 1e-9), 4)}


def metrics(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    good, pot, no = by[2], by[1], by[0]
    per_cv = defaultdict(list)
    for r in rows:
        per_cv[r["cv"]].append(r)
    rhos = [s for g in per_cv.values() if len({r["label"] for r in g}) > 1
            if (s := spearman([r["score"] for r in g], [r["label"] for r in g])) is not None]
    best = max((fit_f1(rows, c) for c in range(20, 80)), key=lambda x: x["f1"])
    return {
        "pairs": len(rows),
        "auc_good_vs_no": round(auc([r["score"] for r in good], [r["score"] for r in no]), 4),
        "auc_fit_vs_no": round(auc([r["score"] for r in good + pot], [r["score"] for r in no]), 4),
        "auc_good_vs_potential": round(auc([r["score"] for r in good], [r["score"] for r in pot]), 4),
        "spearman": None if (rho := spearman([r["score"] for r in rows], [r["label"] for r in rows])) is None else round(rho, 4),
        "mean_per_cv_spearman": round(sum(rhos) / len(rhos), 4) if rhos else None,
        "per_cv_spearman_cvs": len(rhos),
        "best_f1_cutoff": best,
        "dim_auc_good_vs_no": {d: round(auc([r[f"dim_{d}"] for r in good], [r[f"dim_{d}"] for r in no]), 4) for d in DIMS},
        "dim_supported_rate": {d: round(sum(r["supported"][d] for r in rows) / len(rows), 4) for d in DIMS},
        "good_skills_zero_rate": round(sum(r["dim_skills"] == 0 for r in good) / len(good), 4),
        "mean_score": {k: round(sum(r["score"] for r in v) / len(v), 2) for k, v in (("no", no), ("potential", pot), ("good", good))},
        "grade_counts": {f"label{lab}": {g: sum(r["grade"] == g for r in by[lab]) for g in (0, 1, 2)} for lab in (0, 1, 2)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true", help="rebuild lab_docs.pkl from the extraction cache")
    p.add_argument("--name", default="run")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--relations", type=Path, help="SkillRelationTable JSON; its relations are used as approved")
    p.add_argument("--weights", help="JSON dict replacing ATS_WEIGHTS")
    p.add_argument("--lower", type=int, default=50)
    p.add_argument("--upper", type=int, default=75)
    p.add_argument("--title-policy", choices=["exact", "contained"], default="exact")
    p.add_argument("--experience-policy", choices=["strict", "lower_bound"], default="strict")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.build:
        return build()
    data = pickle.loads(PKL.read_bytes())
    docs = data["docs"]
    pairs = [x for x in data["pairs"] if docs[x["cv"]] is not None and docs[x["jd"]] is not None
             and (args.split == "all" or split_of(x["cv"]) == args.split)]
    cfg = {"version": "ats-en-v3", "review_policy": "automatic", "lower_threshold": args.lower, "upper_threshold": args.upper,
           "title_policy": args.title_policy, "experience_policy": args.experience_policy}
    if args.relations:
        table = SkillRelationTable.model_validate_json(args.relations.read_text(encoding="utf-8"))
        cfg.update(relations=[r.model_dump() | {"review_status": "approved"} for r in table.relations],
                   relation_version=table.version)
    config_json = ScorerConfig.model_validate(cfg).model_dump_json()
    weights = json.loads(args.weights) if args.weights else None
    if weights:
        assert set(weights) == set(DIMS) and abs(sum(weights.values()) - 1) < 1e-6, "weights must cover all 7 dims and sum to 1"
    with Pool(args.workers, _init, (config_json, weights)) as pool:
        rows = pool.map(_score, pairs, chunksize=200)
    report = {"name": args.name, "split": args.split, "lower": args.lower, "upper": args.upper,
              "relations": str(args.relations) if args.relations else None, "weights": weights,
              "title_policy": args.title_policy, "experience_policy": args.experience_policy, **metrics(rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / f"{args.name}.{args.split}.pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"grade_counts"}}, indent=1))


if __name__ == "__main__":
    main()
