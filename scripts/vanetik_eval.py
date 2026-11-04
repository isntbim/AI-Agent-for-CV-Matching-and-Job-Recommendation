"""Real human labels: Vanetik & Kogan (2023) vacancy-resume set. Two annotators (senior-HR instructions) rank 5 real
software-developer vacancies for each of 30 anonymised real CVs. Every method ranks the same 5 vacancies per CV; we
report agreement with each annotator and, as the ceiling, the annotators' agreement with each other.

  python -m scripts.vanetik_eval          # V4 extraction runs on local Ollama (cached); needs the GPU free of other jobs

Metrics per CV, averaged: Kendall tau-b and Spearman rho with the annotator's ranking, top-1 match, NDCG@5 with gain
(5 - human rank). Annotator 1 rows 9 and 28 repeat a vacancy id (not a ranking) and are skipped for annotator 1.
Hybrid = the logistic weights fitted on the med2425 dev half (lab/hybrid_*.json), applied unchanged: no tuning here.
"""
import ast
import csv
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/raw/external/vanetik_vacancy_resume"
LAB = ROOT / "output/match_eval/lab"
OUT = LAB / "vanetik"
AS_OF = "2023-06-01"  # the vacancies and CVs predate the 2023 paper
DIMS = ["skills", "title", "experience", "education", "industry"]


def annotations() -> dict[str, list[list[int] | None]]:
    text = (DATA / "annotations-for-the-first-30-vacancies.txt").read_text(encoding="utf-8")
    out = {}
    for name in ("ANNOTATOR_1", "ANNOTATOR_2"):
        body = re.search(name + r"_RANKINGS\s*=\s*(\[.*?\]\])", text, re.S).group(1)
        rows = ast.literal_eval(re.sub(r"#[^\n]*", "", body))
        out[name] = [r if sorted(r) == [1, 2, 3, 4, 5] else None for r in rows]  # invalid permutation -> skip
    return out


def ranks_from_order(order: list[int]) -> np.ndarray:
    """[best vacancy, ..., worst] -> rank of vacancies 1..5 (1 = best)."""
    r = np.empty(5)
    for place, vacancy in enumerate(order, 1):
        r[vacancy - 1] = place
    return r


def agreement(score: np.ndarray, human: np.ndarray) -> dict:
    """score: higher = better (5 vacancies); human: rank 1 = best."""
    gain = 5 - human
    order = np.argsort(-score, kind="stable")
    dcg = (gain[order] / np.log2(np.arange(2, 7))).sum()
    ideal = (np.sort(gain)[::-1] / np.log2(np.arange(2, 7))).sum()
    best = np.flatnonzero(score == score.max())
    return {"kendall": kendalltau(score, -human).statistic, "spearman": spearmanr(score, -human).statistic,
            "top1": float(np.argmin(human) in best) / len(best), "ndcg5": dcg / ideal}


def method_scores() -> dict[str, np.ndarray]:
    """{method: array (30 CVs x 5 vacancies)}."""
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    from scripts.ats_service import v31_config
    from scripts.match_eval import AUTO
    from src.retrieval.reranker import BGEReranker
    from src.scoring.extraction import ExtractionCache
    from src.scoring.rules import _latest_title
    from src.scoring.runtime_v3 import score_pair_v3
    from src.scoring.v4_client import V4Client

    vac = [v["job_description"] for v in csv.DictReader((DATA / "5_vacancies.csv").open(encoding="utf-8"))]
    cvs = [(DATA / f"CV/{i}.txt").read_text(encoding="utf-8") for i in range(1, 31)]
    client = V4Client()
    cache = lambda as_of=None: ExtractionCache(OUT / "extractions", client, "ollama", "qwen2.5-7b-v4-qlora",
                                               max_tokens=16384, ruleset="v3", as_of=as_of)
    jobs = [cache().get(t, "job") for t in vac]
    resumes = []
    for i, t in enumerate(cvs, 1):
        resumes.append(cache(AS_OF).get(t, "resume"))
        print(f"extracted CV {i}/30", flush=True)
    import subprocess
    subprocess.run(["ollama", "stop", "qwen-v4"], check=False, capture_output=True)  # free ~10 GB for the encoders
    s = {}
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(cvs + vac)
    s["TF-IDF + cosine"] = (tfidf.transform(cvs) @ tfidf.transform(vac).T).toarray()
    for name, model_id in (("SBERT (MiniLM)", "sentence-transformers/all-MiniLM-L6-v2"), ("BGE-M3 cosine", "BAAI/bge-m3")):
        model = SentenceTransformer(model_id, device="cuda")
        a, b = model.encode(cvs, normalize_embeddings=True), model.encode(vac, normalize_embeddings=True)
        s[name] = a @ b.T
        if name == "BGE-M3 cosine":
            titles_cv = [_latest_title(r, date.fromisoformat(AS_OF))
                         or next((w.position for w in r.document.work if w.position), None) or "unknown" for r in resumes]
            titles_jd = [j.document.title or "unknown" for j in jobs]
            title_cos = model.encode(titles_cv, normalize_embeddings=True) @ model.encode(titles_jd, normalize_embeddings=True).T
    reranker = BGEReranker(device="cuda", batch_size=4, max_length=2048)
    s["Cross-encoder (BGE reranker)"] = np.array(reranker.score_pairs([(c, v) for c in cvs for v in vac])).reshape(30, 5)
    as_of = date.fromisoformat(AS_OF)
    v3 = [[score_pair_v3(r, j, as_of_date=as_of, config=AUTO) for j in jobs] for r in resumes]
    v31 = [[score_pair_v3(r, j, as_of_date=as_of, config=v31_config()) for j in jobs] for r in resumes]
    s["V3 rule scorer"] = np.array([[x.unrounded_overall_score for x in row] for row in v3])
    s["V3.1 rule scorer"] = np.array([[x.unrounded_overall_score for x in row] for row in v31])
    hybrid = json.loads((LAB / "hybrid_all_relations_fn_text_cos+title_cos.json").read_text(encoding="utf-8"))
    w = hybrid["coefficients"]
    dims = np.array([[[x.unrounded_scores.model_dump()[d] / 100 for d in DIMS] for x in row] for row in v31])
    s["Hybrid (V3.1 + BGE-M3 text/title)"] = (dims @ np.array([w[d] for d in DIMS]) + w["text_cos"] * s["BGE-M3 cosine"]
                                              + w["title_cos"] * title_cos)
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ann = annotations()
    human = {k: [ranks_from_order(r) if r else None for r in rows] for k, rows in ann.items()}
    report = {}
    a1, a2 = human["ANNOTATOR_1"], human["ANNOTATOR_2"]
    pairs = [agreement(-a1[i], a2[i]) for i in range(30) if a1[i] is not None]
    report["Annotator 1 vs annotator 2 (ceiling)"] = {"vs_annotator_1": {k: float(np.mean([p[k] for p in pairs])) for k in pairs[0]},
                                                      "cvs": len(pairs)}
    scores = method_scores()
    rng = np.random.default_rng(0)
    scores["Random"] = rng.random((30, 5))
    for name, sc in scores.items():
        row = {}
        for k, ranks in human.items():
            per = [agreement(sc[i], ranks[i]) for i in range(30) if ranks[i] is not None]
            row[k.lower()] = {m: round(float(np.mean([p[m] for p in per])), 4) for m in per[0]} | {"cvs": len(per)}
        report[name] = row
    (OUT / "vanetik_eval.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    ceiling = report["Annotator 1 vs annotator 2 (ceiling)"]["vs_annotator_1"]
    lines = ["Vanetik & Kogan (2023): 30 real CVs x 5 real vacancies, rankings by 2 human annotators.", "",
             f"Human ceiling (annotator 1 vs 2): Kendall {ceiling['kendall']:.3f}, top-1 {ceiling['top1']:.3f}, "
             f"NDCG@5 {ceiling['ndcg5']:.3f}", "",
             "| Method | Kendall vs A1 | Kendall vs A2 | Top-1 vs A1 | Top-1 vs A2 | NDCG@5 vs A1 | NDCG@5 vs A2 |",
             "|---|---|---|---|---|---|---|"]
    for name, r in report.items():
        if "annotator_1" in r:
            a, b = r["annotator_1"], r["annotator_2"]
            lines.append(f"| {name} | {a['kendall']:.3f} | {b['kendall']:.3f} | {a['top1']:.3f} | {b['top1']:.3f} | "
                         f"{a['ndcg5']:.3f} | {b['ndcg5']:.3f} |")
    (OUT / "vanetik_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
