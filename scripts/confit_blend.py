"""Blend test: final ranker + ConFit v2. blend = (1 - w) * z(final) + w * z(confit), z from dev-half stats.

w is chosen on ConFit's own validation CVs (23 dev CVs ConFit never trained on; the rest of dev is ConFit's
training data, so its scores there are inflated). Reported once on the test half; writes
output/match_eval/lab/confit_blend.json, which scripts/human_label_eval.py and scripts/vanetik_eval.py reuse.

  python -m scripts.confit_blend
"""
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import match_eval as m
from scripts.hybrid_calibrate import load as load_dims
from scripts.match_ir_eval import LAB, RUN, paired, per_cv, per_cv_metrics
from scripts.scorer_lab import split_of

ROOT = Path(__file__).resolve().parent.parent
VAL_CVS = set(json.loads(Path(r"D:\confit_v2\data_ours\recruiting_data_v2\split_summary.json").read_text())["val_cv_ids"])
OUT = LAB / "confit_blend.json"
WEIGHTS = np.round(np.arange(0, 1.01, 0.05), 2)


def blend(final, confit, z, w):
    return (1 - w) * (final - z["final"][0]) / z["final"][1] + w * (confit - z["confit"][0]) / z["confit"][1]


BLEND = "Final + ConFit blend (w from synthetic val)"


def saved_blend(final, confit):
    """The blend with the weight and z-stats fixed by main(); None before main() has run."""
    if not OUT.exists():
        return None
    d = json.loads(OUT.read_text(encoding="utf-8"))
    return blend(np.asarray(final), np.asarray(confit), d["z"], d["w"])


def main() -> None:
    pairs = m.load_pairs()
    texts = {}
    for p in pairs:
        texts.setdefault(m.doc_id("cv", p["cv"]), p["cv"])
        texts.setdefault(m.doc_id("jd", p["jd"]), p["jd"])
    rows = [json.loads(line) for s in ("dev", "test") for line in (LAB / f"{RUN}.{s}.pairs.jsonl").open()]
    key = [(r["cv"], r["jd"]) for r in rows]
    cv, label = np.array([k[0] for k in key]), np.array([r["label"] for r in rows])
    is_dev = np.array([split_of(c) == "dev" for c in cv])
    # the final ranker exactly as match_ir_eval fits it and the service loads it (configs/hybrid_qwen_v1.json)
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(texts.values()))
    vec = dict(zip(texts, tfidf.transform(list(texts.values()))))
    X = np.vstack([load_dims(RUN, s, ["text_cos", "title_cos"], "qwen3")[0] for s in ("dev", "test")])
    X = np.hstack([X, [[float(vec[c].multiply(vec[j]).sum())] for c, j in key]])
    model = json.loads((ROOT / "configs/hybrid_qwen_v1.json").read_text(encoding="utf-8"))
    final = X @ np.array(model["coefficients"]) + model["intercept"]
    cf = {(r["cv"], r["jd"]): r["cos"] for r in map(json.loads, (LAB / "confit_v2_cosine.jsonl").open(encoding="utf-8"))}
    confit = np.array([cf[k] for k in key])

    z = {n: (float(s[is_dev].mean()), float(s[is_dev].std())) for n, s in (("final", final), ("confit", confit))}
    val = np.isin(cv, list(VAL_CVS))
    curve = {float(w): float(np.mean([v[0] for v in per_cv(cv[val], label[val], blend(final[val], confit[val], z, w)).values()]))
             for w in WEIGHTS}
    w = max(curve, key=curve.get)

    test = ~is_dev
    scores = {"final": final, "confit": confit, "blend": blend(final, confit, z, w)}
    report = {n: per_cv_metrics(cv[test], label[test], s[test]) for n, s in scores.items()}
    sig = {f"blend vs {b}": paired(cv[test], label[test], scores["blend"][test], scores[b][test]) for b in ("final", "confit")}
    out = {"w": w, "z": z, "val_cvs": int(len(set(cv[val]))), "val_ndcg10_by_w": curve,
           "test": {n: {k: r[k] for k in ("NDCG@5", "NDCG@10", "MRR", "P@5")} for n, r in report.items()}, "paired": sig}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
