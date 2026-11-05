"""Calibrated hybrid layer over a scorer_lab run: logistic regression on the rule dimensions plus two bge-m3 cosines
(whole text, titles), fitted on the dev CV half; grade cut-offs chosen on dev for macro-F1; test reported once.

  python -m scripts.hybrid_calibrate --run all_relations_fn            # needs <run>.dev / <run>.test pairs.jsonl
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

LAB = Path("output/match_eval/lab")
DIMS = ["skills", "title", "experience", "education", "industry"]  # location/salary are constant here


EMBEDDINGS = {"bge_m3": ("text_cosine.jsonl", "title_emb_bge_m3.npz"),  # text_baseline.py
              "qwen3": ("qwen_embedding_cosine.jsonl", "title_emb_qwen3_embedding.npz")}  # qwen_embedding_baseline.py


def load(run: str, split: str, extra: list[str], embedding: str = "bge_m3"):
    text_file, title_file = EMBEDDINGS[embedding]
    cos = {(r["cv"], r["jd"]): r["cos"] for r in map(json.loads, (LAB / text_file).open())}
    emb = np.load(LAB / title_file)
    rows = [json.loads(line) for line in (LAB / f"{run}.{split}.pairs.jsonl").open()]
    feats = {"text_cos": lambda r: cos[(r["cv"], r["jd"])], "title_cos": lambda r: float(emb[r["cv"]] @ emb[r["jd"]])}
    X = np.array([[r[f"dim_{d}"] / 100 for d in DIMS] + [feats[k](r) for k in extra] for r in rows])
    return X, np.array([r["label"] for r in rows]), [r["cv"] for r in rows], np.array([r["score"] for r in rows])


def report(score, y, cvs, cuts=None, reps=1000):
    good_no = y != 1
    out = {"auc_good_vs_no": roc_auc_score(y[good_no] == 2, score[good_no]),
           "auc_fit_vs_no": roc_auc_score(y >= 1, score),
           "auc_good_vs_potential": roc_auc_score(y[y >= 1] == 2, score[y >= 1])}
    if cuts is not None:
        pred = np.digitize(score, cuts)
        out |= {"macro_f1": f1_score(y, pred, average="macro"), "accuracy": float((pred == y).mean()),
                "per_class_f1": f1_score(y, pred, average=None).round(4).tolist(),
                "fit_f1": f1_score(y >= 1, pred >= 1), "confusion": confusion_matrix(y, pred).tolist()}
    # 95% CI: bootstrap over CVs (pairs of one CV are dependent)
    ids = sorted(set(cvs))
    index = {c: np.flatnonzero(np.array(cvs) == c) for c in ids}
    rng, boot = np.random.default_rng(0), []
    for _ in range(reps):
        take = np.concatenate([index[c] for c in rng.choice(ids, len(ids))])
        s, t = score[take], y[take]
        boot.append(roc_auc_score(t[t != 1] == 2, s[t != 1]))
    out["auc_good_vs_no_ci"] = np.percentile(boot, [2.5, 97.5]).round(4).tolist()
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def cutoffs(score, y):
    qs = np.quantile(score, np.linspace(0.02, 0.98, 97))
    return max(((a, b) for a in qs for b in qs if b > a), key=lambda c: f1_score(y, np.digitize(score, c), average="macro"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--extra", default="text_cos,title_cos", help="comma list; empty = rule dims only")
    p.add_argument("--test", action="store_true", help="also report the held-out test half (do this once)")
    args = p.parse_args()
    extra = [e for e in args.extra.split(",") if e]
    Xa, ya, cva, rule_a = load(args.run, "dev", extra)
    model = LogisticRegression(max_iter=5000).fit(Xa, ya >= 1)
    sa = model.decision_function(Xa)
    cuts = cutoffs(sa, ya)
    rule_cuts = cutoffs(rule_a, ya)
    result = {"run": args.run, "features": DIMS + extra,
              "coefficients": dict(zip(DIMS + extra, model.coef_[0].round(3).tolist())), "intercept": round(float(model.intercept_[0]), 3),
              "grade_cutoffs_logit": [round(float(c), 4) for c in cuts], "rule_score_cutoffs": [round(float(c), 2) for c in rule_cuts],
              "dev": {"hybrid": report(sa, ya, cva, cuts), "rule_score": report(rule_a, ya, cva, rule_cuts)}}
    if args.test:
        Xb, yb, cvb, rule_b = load(args.run, "test", extra)
        result["test"] = {"hybrid": report(model.decision_function(Xb), yb, cvb, cuts),
                          "rule_score": report(rule_b, yb, cvb, rule_cuts)}
    name = f"hybrid_{args.run}_{'+'.join(extra) or 'rules'}"
    (LAB / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
