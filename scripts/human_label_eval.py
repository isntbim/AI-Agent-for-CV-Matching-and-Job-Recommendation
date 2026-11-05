"""Evaluate every method against the team's human labels on the CareerViet corpus (scripts/build_label_set.py).

  python -m scripts.human_label_eval output/labeling/labels_export.json

The export is {labeller id: {cv id: {jd id: 0|1|2}}} (No / Potential / Good Fit), read from the labelling page's
store. Reported: inter-annotator agreement (percent, Cohen's kappa, quadratic-weighted kappa) on pairs both labelled,
then each method ranking every labelled candidate of a CV, against the consensus (mean of the two labels): NDCG@5 with
gain = mean label, P@3 (relevant = mean >= 1), MRR, Kendall tau; the service ranking vs TF-IDF is tested by a paired
CV bootstrap and sign-flip permutation, as in scripts/match_ir_eval.py.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
SET = ROOT / "output/labeling/label_set_v1.json"
OUT = ROOT / "output/labeling"


SERVICE = "Final: Hybrid + TF-IDF (Qwen3, service)"


def agreement(export: dict) -> tuple[dict, dict]:
    """(agreement stats, consensus {(cv, jd): mean label}) over pairs every labeller labelled."""
    people = sorted(export)
    keys = sorted({(c, j) for p in people for c, js in export[p].items() for j in js})
    both = [k for k in keys if all(k[1] in export[p].get(k[0], {}) for p in people)]
    stats = {"labellers": len(people), "pairs_labelled_by_all": len(both)}
    if len(people) >= 2 and both:
        a = [export[people[0]][c][j] for c, j in both]
        b = [export[people[1]][c][j] for c, j in both]
        stats |= {"percent_agreement": round(float(np.mean(np.array(a) == np.array(b))), 4),
                  "cohen_kappa": round(float(cohen_kappa_score(a, b)), 4),
                  "weighted_kappa_quadratic": round(float(cohen_kappa_score(a, b, weights="quadratic")), 4),
                  "label_counts": {p: np.bincount([export[p][c][j] for c, j in both], minlength=3).tolist() for p in people[:2]}}
    consensus = {k: float(np.mean([export[p][k[0]][k[1]] for p in people])) for k in both}
    return stats, consensus


def ranking_metrics(score: np.ndarray, gain: np.ndarray) -> dict:
    order = np.argsort(-score, kind="stable")
    g = gain[order]
    disc = 1 / np.log2(np.arange(2, len(g) + 2))
    ideal = np.sort(gain)[::-1]
    k = min(5, len(g))
    rel = g >= 1
    return {"NDCG@5": float((g[:k] * disc[:k]).sum() / (ideal[:k] * disc[:k]).sum()) if ideal[:k].sum() else np.nan,
            "P@3": float(rel[:3].mean()), "MRR": float(1 / (np.argmax(rel) + 1)) if rel.any() else 0.0,
            "Kendall": float(kendalltau(score, gain).statistic) if len(set(gain)) > 1 else np.nan}


def method_scores(pairs: list[tuple[str, str]]) -> dict[str, dict]:
    """{method: {(cv, jd): score}} for every labelled pair, computed the same way as the benchmark and service."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    from scripts import ats_service as svc
    from scripts.build_label_set import CVS, MANIFEST
    from src.scoring.rules import _latest_title
    from src.scoring.runtime_v3 import score_pair_v3

    rows = {r["id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]["cv_extraction"]}
    corpus = [d["id"] for d in json.loads((ROOT / "output/retrieval/corpus_v2/corpus.json").read_text(encoding="utf-8"))]
    paths = svc._jd_paths()
    jd_text = {j: (ROOT / paths[j]).read_text(encoding="utf-8") for j in corpus}
    cv_text = {c: (ROOT / rows[c]["text"]).read_text(encoding="utf-8") for c in CVS}
    from sentence_transformers import SentenceTransformer

    ranker = svc.Ranker(jd_text)  # the service's final model (Qwen3)
    bge = SentenceTransformer("BAAI/bge-m3", device="cpu")  # the bge-m3 baseline and the older bge-m3 hybrid
    bge.max_seq_length = 2048
    bge_jd = dict(zip(corpus, bge.encode([jd_text[j] for j in corpus], normalize_embeddings=True)))
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(jd_text.values()) + list(cv_text.values()))
    hybrid = json.loads((ROOT / "output/match_eval/lab/hybrid_all_relations_fn_text_cos+title_cos.json").read_text(encoding="utf-8"))
    cache, config, out = svc._cache, svc.v31_config(), {m: {} for m in
                                                          ("TF-IDF + cosine", "BGE-M3 cosine", "Qwen3-Embedding cosine",
                                                           "V3.1 rule scorer", "Hybrid (V3.1 + BGE-M3)", SERVICE)}
    by_cv = {}
    for c, j in pairs:
        by_cv.setdefault(c, []).append(j)
    for c, jds in by_cv.items():
        resume = cache(rows[c]["as_of"]).get(cv_text[c], "resume")
        as_of = date.fromisoformat(rows[c]["as_of"])
        cv_emb = bge.encode([cv_text[c]], normalize_embeddings=True)[0]
        cv_qwen = ranker.emb.encode([cv_text[c]], query=True)[0]
        title = _latest_title(resume, as_of) or next((w.position for w in resume.document.work if w.position), None)
        results, jobs = {}, {}
        for j in jds:
            jobs[j] = cache().get(jd_text[j], "job")
            results[j] = score_pair_v3(resume, jobs[j], as_of_date=as_of, config=config)
        service = ranker.scores(cv_text[c], title, [(j, jobs[j].document.title, results[j].unrounded_scores.model_dump()) for j in jds])
        titles = bge.encode([title or "unknown"] + [jobs[j].document.title or "unknown" for j in jds], normalize_embeddings=True)
        w = hybrid["coefficients"]
        for k, j in enumerate(jds):
            dims = results[j].unrounded_scores.model_dump()
            text_cos, title_cos = float(cv_emb @ bge_jd[j]), float(titles[0] @ titles[k + 1])
            out["Qwen3-Embedding cosine"][(c, j)] = float(cv_qwen @ ranker.jd_emb[j])
            out["TF-IDF + cosine"][(c, j)] = float((tfidf.transform([cv_text[c]]) @ tfidf.transform([jd_text[j]]).T).toarray()[0, 0])
            out["BGE-M3 cosine"][(c, j)] = text_cos
            out["V3.1 rule scorer"][(c, j)] = results[j].unrounded_overall_score
            out["Hybrid (V3.1 + BGE-M3)"][(c, j)] = (sum(w[d] * dims[d] / 100 for d in ("skills", "title", "experience", "education", "industry"))
                                                     + w["text_cos"] * text_cos + w["title_cos"] * title_cos)
            out[SERVICE][(c, j)] = service[k]
    return out


def evaluate(consensus: dict, scores: dict, reps: int = 2000) -> tuple[dict, dict]:
    cvs = sorted({c for c, _ in consensus})
    per = {m: [] for m in scores}
    for c in cvs:
        jds = sorted(j for cc, j in consensus if cc == c)
        gain = np.array([consensus[(c, j)] for j in jds])
        if gain.max() == 0:  # no relevant job for this CV: ranking metrics undefined
            continue
        for m, s in scores.items():
            per[m].append(ranking_metrics(np.array([s[(c, j)] for j in jds]), gain))
    table = {m: {k: round(float(np.nanmean([r[k] for r in rs])), 4) for k in rs[0]} | {"cvs": len(rs)}
             for m, rs in per.items() if rs}
    a, b = SERVICE, "TF-IDF + cosine"
    diff = np.array([ra["NDCG@5"] - rb["NDCG@5"] for ra, rb in zip(per[a], per[b])])
    rng = np.random.default_rng(7)
    boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(reps)]
    flips = [(diff * rng.choice((-1, 1), len(diff))).mean() for _ in range(reps)]
    test = {"comparison": f"{a} vs {b}", "NDCG@5_diff": round(float(diff.mean()), 4),
            "ci95": np.percentile(boot, [2.5, 97.5]).round(4).tolist(),
            "p": round(float((np.sum(np.abs(flips) >= abs(diff.mean())) + 1) / (reps + 1)), 4), "cvs": len(diff)}
    return table, test


def main() -> None:
    export = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    stats, consensus = agreement(export)
    table, test = evaluate(consensus, method_scores(sorted(consensus)))
    report = {"agreement": stats, "methods": table, "paired": test}
    (OUT / "human_label_eval.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"Human labels: {stats['labellers']} labellers, {stats['pairs_labelled_by_all']} pairs labelled by all.",
             f"Agreement: {stats.get('percent_agreement')}, Cohen kappa {stats.get('cohen_kappa')}, "
             f"quadratic kappa {stats.get('weighted_kappa_quadratic')}.", "",
             "| Method | NDCG@5 | P@3 | MRR | Kendall | CVs |", "|---|---|---|---|---|---|"]
    lines += [f"| {m} | {r['NDCG@5']:.3f} | {r['P@3']:.3f} | {r['MRR']:.3f} | {r['Kendall']:.3f} | {r['cvs']} |"
              for m, r in table.items()]
    lines += ["", f"{test['comparison']}: NDCG@5 {test['NDCG@5_diff']:+.3f} {test['ci95']}, p={test['p']} ({test['cvs']} CVs)"]
    (OUT / "human_label_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
