"""Matching benchmark for the paper: every variant and baseline on the same scored pairs, tuned on the dev CV half,
reported on the test half. Pair metrics (AUC, macro-F1, balanced accuracy) and per-CV ranking metrics
(P@K, R@K, HitRate@K, MRR, NDCG@K): each CV's ~100 labelled JDs are ranked by the variant's score.

  python -m scripts.match_ir_eval            # writes output/match_eval/lab/benchmark.{json,md}

Relevant = Potential or Good Fit; NDCG uses graded gain 2^label - 1. Ties are broken by a seeded random order,
averaged over 10 seeds (the rule scores tie often; sorting by id would bias them).
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from scripts import match_eval as m
from scripts.hybrid_calibrate import DIMS, cutoffs, load as load_dims
from scripts.scorer_lab import split_of
from src.retrieval.lexical import BM25
from src.retrieval.reranker import Candidate

ROOT = Path(__file__).resolve().parent.parent
LAB = Path("output/match_eval/lab")
KS = (1, 3, 5, 10)
SEEDS = 10


def per_cv_metrics(cv: np.ndarray, label: np.ndarray, score: np.ndarray) -> dict:
    out, rng = {}, np.random.default_rng(0)
    groups = {c: np.flatnonzero(cv == c) for c in np.unique(cv)}
    acc = {k: [] for k in [*(f"{n}@{k}" for n in ("P", "R", "Hit", "NDCG") for k in KS), "MRR"]}
    for _ in range(SEEDS):
        jitter = rng.random(len(score))
        for idx in groups.values():
            rel = label[idx]
            if not (rel >= 1).any():  # no relevant JD: P/R/MRR/NDCG undefined for this CV
                continue
            order = idx[np.lexsort((jitter[idx], -score[idx]))]
            ranked = label[order]
            hits = ranked >= 1
            gains = 2.0 ** ranked - 1
            ideal = np.sort(2.0 ** rel - 1)[::-1]
            disc = 1 / np.log2(np.arange(2, len(ranked) + 2))
            for k in KS:
                acc[f"P@{k}"].append(hits[:k].mean())
                acc[f"R@{k}"].append(hits[:k].sum() / hits.sum())
                acc[f"Hit@{k}"].append(float(hits[:k].any()))
                acc[f"NDCG@{k}"].append((gains[:k] * disc[:k]).sum() / (ideal[:k] * disc[:k]).sum())
            acc["MRR"].append(1 / (np.argmax(hits) + 1))
    out = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}
    out["cvs_ranked"] = len(acc["MRR"]) // SEEDS
    return out


def pair_metrics(label, score, cuts):
    pred = np.digitize(score, cuts)
    good_no = label != 1
    return {"auc_good_vs_no": roc_auc_score(label[good_no] == 2, score[good_no]),
            "auc_fit_vs_no": roc_auc_score(label >= 1, score),
            "auc_good_vs_potential": roc_auc_score(label[label >= 1] == 2, score[label >= 1]),
            "macro_f1": f1_score(label, pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(label, pred),
            "accuracy": float((pred == label).mean())}


def ndcg10_ci(cv, label, score, reps=300):
    """95% CI of NDCG@10 by bootstrapping CVs (one tie-break seed)."""
    rng = np.random.default_rng(1)
    jitter = rng.random(len(score))
    per = []
    for c in np.unique(cv):
        idx = np.flatnonzero(cv == c)
        if not (label[idx] >= 1).any():
            continue
        order = idx[np.lexsort((jitter[idx], -score[idx]))]
        disc = 1 / np.log2(np.arange(2, 12))
        g = (2.0 ** label[order][:10] - 1)
        ideal = np.sort(2.0 ** label[idx] - 1)[::-1][:10]
        per.append((g * disc[:len(g)]).sum() / (ideal * disc[:len(ideal)]).sum())
    per = np.array(per)
    return np.percentile([rng.choice(per, len(per)).mean() for _ in range(reps)], [2.5, 97.5]).round(4).tolist()


def per_cv(cv, label, score, seeds=SEEDS) -> dict:
    """{cv: (NDCG@10, MRR, P@5)} averaged over the same seeded tie-breaks as per_cv_metrics."""
    out, rng = {}, np.random.default_rng(0)
    jitters = [rng.random(len(score)) for _ in range(seeds)]
    for c in np.unique(cv):
        idx = np.flatnonzero(cv == c)
        if not (label[idx] >= 1).any():
            continue
        vals = []
        for jitter in jitters:
            ranked = label[idx[np.lexsort((jitter[idx], -score[idx]))]]
            gains, disc = 2.0 ** ranked - 1, 1 / np.log2(np.arange(2, len(ranked) + 2))
            ideal = np.sort(2.0 ** label[idx] - 1)[::-1]
            hits = ranked >= 1
            vals.append(((gains[:10] * disc[:10]).sum() / (ideal[:10] * disc[:10]).sum(), 1 / (np.argmax(hits) + 1),
                         hits[:5].mean()))
        out[c] = np.mean(vals, axis=0)
    return out


def paired(cv, label, a, b, reps=2000) -> dict:
    """a minus b on the same test CVs: mean difference, CV-bootstrap 95% CI, two-sided sign-flip permutation p."""
    pa, pb = per_cv(cv, label, a), per_cv(cv, label, b)
    diff = np.array([pa[c] - pb[c] for c in pa])  # CVs x (NDCG@10, MRR, P@5)
    rng = np.random.default_rng(7)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean(0) for _ in range(reps)])
    flips = np.array([(diff * rng.choice((-1, 1), (len(diff), 1))).mean(0) for _ in range(reps)])
    obs = diff.mean(0)
    p = ((np.abs(flips) >= np.abs(obs)).sum(0) + 1) / (reps + 1)
    ids = np.unique(cv)
    index = {c: np.flatnonzero(cv == c) for c in ids}
    auc_diff = []
    for _ in range(reps // 4):  # AUC Good vs No difference, bootstrap over CVs
        take = np.concatenate([index[c] for c in rng.choice(ids, len(ids))])
        y, keep = label[take], label[take] != 1
        auc_diff.append(roc_auc_score(y[keep] == 2, a[take][keep]) - roc_auc_score(y[keep] == 2, b[take][keep]))
    keep = label != 1
    out = {m: {"diff": round(float(obs[i]), 4), "ci95": np.percentile(boot[:, i], [2.5, 97.5]).round(4).tolist(),
               "p": round(float(p[i]), 4)} for i, m in enumerate(("NDCG@10", "MRR", "P@5"))}
    out["AUC_good_vs_no"] = {"diff": round(float(roc_auc_score(label[keep] == 2, a[keep]) - roc_auc_score(label[keep] == 2, b[keep])), 4),
                             "ci95": np.percentile(auc_diff, [2.5, 97.5]).round(4).tolist()}
    out["cvs"] = len(diff)
    return out


FINAL = "Final: Hybrid + TF-IDF (Qwen3)"
NO_RULES = "Qwen3 + TF-IDF + title (no rules)"


def main() -> None:
    pairs = m.load_pairs()
    texts = {}
    for p in pairs:
        texts.setdefault(m.doc_id("cv", p["cv"]), p["cv"])
        texts.setdefault(m.doc_id("jd", p["jd"]), p["jd"])
    rows = [json.loads(line) for split in ("dev", "test") for line in (LAB / f"all_relations_fn.{split}.pairs.jsonl").open()]
    key = [(r["cv"], r["jd"]) for r in rows]  # the 48,010 scored pairs, dev then test
    cv = np.array([k[0] for k in key])
    label = np.array([r["label"] for r in rows])
    is_dev = np.array([split_of(c) == "dev" for c in cv])

    scores = {"Random": np.random.default_rng(3).random(len(rows))}
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(texts.values()))
    vec = dict(zip(texts, tfidf.transform(list(texts.values()))))
    scores["TF-IDF + cosine"] = np.array([float(vec[c].multiply(vec[j]).sum()) for c, j in key])
    jd_ids = sorted({j for _, j in key})
    bm25 = BM25([Candidate(id=j, text=texts[j], retrieval_score=0.0) for j in jd_ids])
    by_cv = {c: bm25.scores(texts[c]) for c in sorted(set(cv))}
    scores["BM25"] = np.array([by_cv[c][j] for c, j in key])
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")  # truncates at 256 tokens
    ids = sorted({x for k in key for x in k})
    emb = dict(zip(ids, sbert.encode([texts[i] for i in ids], normalize_embeddings=True, batch_size=32)))
    scores["SBERT (MiniLM)"] = np.array([float(emb[c] @ emb[j]) for c, j in key])
    bge = {(r["cv"], r["jd"]): r["cos"] for r in map(json.loads, (LAB / "text_cosine.jsonl").open())}
    scores["BGE-M3 cosine"] = np.array([bge[k] for k in key])
    qwen_file = LAB / "qwen_embedding_cosine.jsonl"  # scripts/qwen_embedding_baseline.py
    if qwen_file.exists():
        qwen = {(r["cv"], r["jd"]): r["cos"] for r in map(json.loads, qwen_file.open(encoding="utf-8"))}
        scores["Qwen3-Embedding-4B cosine"] = np.array([qwen[k] for k in key])
    base = {(r["cv"], r["jd"]): r["score"] for s in ("dev", "test") for r in map(json.loads, (LAB / f"baseline.{s}.pairs.jsonl").open())}
    scores["V3 rule scorer"] = np.array([base[k] for k in key])
    scores["V3.1 rule scorer"] = np.array([r["score"] for r in rows])
    Xa, ya, _, _ = load_dims("all_relations_fn", "dev", ["text_cos", "title_cos"])
    Xb, _, _, _ = load_dims("all_relations_fn", "test", ["text_cos", "title_cos"])
    lr = LogisticRegression(max_iter=5000).fit(Xa, ya >= 1)
    scores["Hybrid (V3.1 + BGE-M3 text/title)"] = np.concatenate([lr.decision_function(Xa), lr.decision_function(Xb)])
    rerank_file = LAB / "reranker_scores.jsonl"  # scripts/reranker_baseline.py
    if rerank_file.exists():
        ce = {(r["cv"], r["jd"]): r["score"] for r in map(json.loads, rerank_file.open(encoding="utf-8"))}
        scores["Cross-encoder (BGE reranker)"] = np.array([ce[k] for k in key])
    extras = {"TF-IDF": scores["TF-IDF + cosine"]} | (
        {"cross-encoder": scores["Cross-encoder (BGE reranker)"]} if "Cross-encoder (BGE reranker)" in scores else {})
    for names in [["TF-IDF"], ["cross-encoder"], ["TF-IDF", "cross-encoder"]]:
        if all(n in extras for n in names):  # same protocol: fitted on dev only, test scored once
            col = np.column_stack([extras[n] for n in names])
            fit = LogisticRegression(max_iter=5000).fit(np.hstack([Xa, col[is_dev]]), ya >= 1)
            scores[f"Hybrid + {' + '.join(names)}"] = fit.decision_function(np.hstack([np.vstack([Xa, Xb]), col]))
    if "Qwen3-Embedding-4B cosine" in scores:  # the final ranking model: same hybrid with Qwen3 text/title cosines
        qa, _, _, _ = load_dims("all_relations_fn", "dev", ["text_cos", "title_cos"], "qwen3")
        qb, _, _, _ = load_dims("all_relations_fn", "test", ["text_cos", "title_cos"], "qwen3")
        Q = np.hstack([np.vstack([qa, qb]), scores["TF-IDF + cosine"][:, None]])
        for name, cols in ((NO_RULES, slice(len(DIMS), None)), (FINAL, slice(None))):  # ablation, then the full model
            fit = LogisticRegression(max_iter=5000).fit(Q[is_dev][:, cols], ya >= 1)
            scores[name] = fit.decision_function(Q[:, cols])
        model = {"features": DIMS + ["text_cos", "title_cos", "tfidf_cos"],  # the service loads it (scripts/ats_service.py)
                 "coefficients": fit.coef_[0].round(6).tolist(), "intercept": round(float(fit.intercept_[0]), 6),
                 "grade_cutoffs": [round(float(c), 6) for c in cutoffs(scores[FINAL][is_dev], label[is_dev])],
                 "fitted_on": "med2425 dev half (CV-hash split), V3.1 dims /100, Qwen3 cosines, TF-IDF cosine",
                 "text_cos": "qwen3-embedding:4b (Ollama Q4_K_M), 2048-token context, CV with retrieval instruction",
                 "title_cos": "qwen3-embedding:4b, CV latest title vs JD title, no instruction",
                 "tfidf": "sublinear_tf, english stop words, fitted on all documents of the corpus"}
        (ROOT / "configs/hybrid_qwen_v1.json").write_text(json.dumps(model, indent=1), encoding="utf-8")

    report, test = {}, ~is_dev
    for name, s in scores.items():
        cuts = cutoffs(s[is_dev], label[is_dev])
        report[name] = {**{k: round(float(v), 4) for k, v in pair_metrics(label[test], s[test], cuts).items()},
                        **per_cv_metrics(cv[test], label[test], s[test]),
                        "ndcg@10_ci": ndcg10_ci(cv[test], label[test], s[test])}
    hybrid, v31 = "Hybrid (V3.1 + BGE-M3 text/title)", "V3.1 rule scorer"
    comparisons = [(hybrid, "TF-IDF + cosine"), (hybrid, "SBERT (MiniLM)"), (hybrid, "BGE-M3 cosine"), (hybrid, v31),
                   (v31, "TF-IDF + cosine"), (v31, "V3 rule scorer")]
    for best in ("Hybrid + cross-encoder", "Hybrid + TF-IDF + cross-encoder", "Hybrid + TF-IDF"):
        if best in scores:
            comparisons += [(best, hybrid), (best, "TF-IDF + cosine")]
    if "Qwen3-Embedding-4B cosine" in scores:
        qwen = "Qwen3-Embedding-4B cosine"
        comparisons += [(qwen, "BGE-M3 cosine"), (qwen, "TF-IDF + cosine"), ("Hybrid + TF-IDF", qwen),
                        (FINAL, qwen), (FINAL, NO_RULES), (NO_RULES, qwen), (FINAL, "Hybrid + TF-IDF"),
                        (FINAL, "TF-IDF + cosine")]
    if "Cross-encoder (BGE reranker)" in scores:
        comparisons += [("Cross-encoder (BGE reranker)", "TF-IDF + cosine"), (hybrid, "Cross-encoder (BGE reranker)")]
    significance = {f"{a} vs {b}": paired(cv[test], label[test], scores[a][test], scores[b][test]) for a, b in comparisons}
    majority = float((label[test] == 0).mean())
    meta = {"test_pairs": int(test.sum()), "test_cvs": int(len(set(cv[test]))), "always_no_fit_accuracy": round(majority, 4),
            "relevant": "Potential or Good Fit", "tie_break_seeds": SEEDS}
    (LAB / "benchmark.json").write_text(json.dumps({"meta": meta, "results": report, "paired": significance}, indent=1),
                                        encoding="utf-8")
    cols = ["auc_good_vs_no", "auc_fit_vs_no", "macro_f1", "balanced_accuracy", "accuracy", "P@5", "R@10", "Hit@5", "MRR", "NDCG@5", "NDCG@10"]
    lines = [f"Test half: {meta['test_pairs']} pairs, {meta['test_cvs']} CVs; always-No-Fit accuracy {majority:.3f}.", "",
             "| Method | " + " | ".join(cols) + " | NDCG@10 95% CI |", "|---" * (len(cols) + 2) + "|"]
    lines += [f"| {n} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {r['ndcg@10_ci']} |" for n, r in report.items()]
    lines += ["", "Paired on the same test CVs (A minus B): mean difference [95% CV-bootstrap CI], sign-flip permutation p.",
              "", "| A vs B | NDCG@10 | MRR | P@5 | AUC Good vs No |", "|---|---|---|---|---|"]
    cell = lambda d: f"{d['diff']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" + (f", p={d['p']:.4f}" if "p" in d else "")
    lines += [f"| {k} | " + " | ".join(cell(v[m]) for m in ("NDCG@10", "MRR", "P@5", "AUC_good_vs_no")) + " |"
              for k, v in significance.items()]
    (LAB / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
