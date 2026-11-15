"""O3 check on the public select/reject set (AzharAli05/Resume-Screening-Dataset, 10,174 pairs).

Promised at Review 2: AUC of the match score against select/reject (main metric, no threshold), then a
threshold picked on a 20% calibration split and per-label F1 on the other 80%, plus an error look at rejected
pairs that score high (against the dataset's Reason_for_decision).

Text features (Qwen3 cosine, TF-IDF cosine) need no extraction and run on all pairs. With --extractions,
pairs whose CV and JD were both extracted also get the V3.1 rule dimensions and the final hybrid score.

    python -m scripts.select_reject_eval                 # text baselines, all pairs
    python -m scripts.select_reject_eval --sample 1000 --extract --workers 3   # V4-extract a balanced sample (4070)
    python -m scripts.select_reject_eval --sample 1000 --score                 # rules + final model on that sample
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/external/azharali_screening/dataset.csv"
OUT = ROOT / "output/select_reject"


def load(path: Path = DATA) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for i, r in enumerate(rows):
        r["uid"], r["y"] = f"sr_{i:05d}", int(r["Decision"].strip().lower() == "select")
    return rows


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def embed_cached(texts: list[str], query: bool, cache: Path) -> np.ndarray:
    """Qwen3 unit vectors keyed by text hash, saved every batch so a stopped run resumes."""
    from src.embeddings.qwen import QwenEmbedder

    store = dict(np.load(cache)) if cache.exists() else {}
    todo = list(dict.fromkeys(t for t in texts if sha(t) not in store))
    emb = QwenEmbedder()
    for i in range(0, len(todo), 64):
        chunk = todo[i:i + 64]
        store |= {sha(t): v for t, v in zip(chunk, emb.encode(chunk, query=query))}
        np.savez(cache, **store)
        print(f"  {cache.name}: {min(i + 64, len(todo))}/{len(todo)}", flush=True)
    return np.stack([store[sha(t)] for t in texts])


def auc_ci(y: np.ndarray, s: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng, n, boot = np.random.default_rng(seed), len(y), []
    for _ in range(reps):
        k = rng.integers(0, n, n)
        if 0 < y[k].sum() < n:
            boot.append(roc_auc_score(y[k], s[k]))
    return float(roc_auc_score(y, s)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def threshold_f1(y: np.ndarray, s: np.ndarray, seed: int = 0) -> dict:
    """Pick the macro-F1-best threshold on a stratified 20% split; report per-label F1 on the other 80%."""
    rng = np.random.default_rng(seed)
    calib = np.zeros(len(y), bool)
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        calib[rng.choice(idx, int(round(len(idx) * .2)), replace=False)] = True
    grid = np.quantile(s[calib], np.linspace(.02, .98, 97))
    best = max(grid, key=lambda t: f1_score(y[calib], s[calib] >= t, average="macro"))
    test = ~calib
    pred = s[test] >= best
    f1 = f1_score(y[test], pred, average=None, labels=[0, 1])
    return {"threshold": float(best), "n_test": int(test.sum()), "f1_reject": float(f1[0]), "f1_select": float(f1[1]),
            "macro_f1": float(f1.mean()), "accuracy": float((pred == y[test]).mean())}


def text_features(rows: list[dict]) -> dict[str, np.ndarray]:
    OUT.mkdir(parents=True, exist_ok=True)
    cv = embed_cached([r["Resume"] for r in rows], True, OUT / "emb_cv.npz")
    jd = embed_cached([r["Job_Description"] for r in rows], False, OUT / "emb_jd.npz")
    role = embed_cached([r["Role"] for r in rows], False, OUT / "emb_role.npz")
    jds = list(dict.fromkeys(r["Job_Description"] for r in rows))
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(jds)
    a, b = tfidf.transform([r["Resume"] for r in rows]), tfidf.transform([r["Job_Description"] for r in rows])
    return {"qwen3_text_cos": (cv * jd).sum(1), "qwen3_cv_vs_role": (cv * role).sum(1),
            "tfidf_cos": np.asarray(a.multiply(b).sum(1)).ravel()}


def report(rows: list[dict], feats: dict[str, np.ndarray]) -> dict:
    y = np.array([r["y"] for r in rows])
    out = {"n": len(rows), "select": int(y.sum()), "reject": int(len(y) - y.sum()), "methods": {}}
    for name, s in feats.items():
        auc, lo, hi = auc_ci(y, s)
        out["methods"][name] = {"auc": auc, "auc_ci95": [lo, hi]} | threshold_f1(y, s)
    return out


AS_OF = "2025-03-19"  # the dataset's publication date: "Present" jobs end here


def sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n pairs, half select and half reject, fixed by the seed."""
    import random
    rng = random.Random(seed)
    out = []
    for y in (1, 0):
        pool = [r for r in rows if r["y"] == y]
        out += rng.sample(pool, n // 2)
    return sorted(out, key=lambda r: r["uid"])


def _cache(client=None):
    from scripts.ats_service import _cache as service_cache
    c = service_cache(AS_OF, client)
    c.directory = OUT / "extractions"  # same model, prompt and rules as the service, kept apart from its cache
    return c


def extract(rows: list[dict], workers: int) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.scoring.v4_client import QwenV4, V4Client
    llm, local = QwenV4(), threading.local()
    jobs = [(r["Resume"], "resume") for r in rows] + list(dict.fromkeys((r["Job_Description"], "job") for r in rows))

    def one(job):
        if not hasattr(local, "cache"):
            local.cache = _cache(V4Client(llm))
        local.cache.get(*job)

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            try:
                f.result()
            except Exception as exc:
                failed += 1
                print("FAILED", type(exc).__name__, str(exc)[:120], flush=True)
            if i % 25 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {failed} failed", flush=True)


def score(rows: list[dict]) -> dict:
    """V3.1 rule score and the final ranking model on every sample pair whose CV and JD both extracted."""
    import math
    from datetime import date

    from scripts.ats_service import HYBRID, v31_config
    from src.embeddings.qwen import QwenEmbedder
    from src.scoring.rules import _latest_title
    from src.scoring.runtime_v3 import score_pair_v3

    cache, config, as_of = _cache(), v31_config(), date.fromisoformat(AS_OF)
    cache.client = None  # scoring reads the cache only: a missing extraction skips the pair
    model = json.loads(HYBRID.read_text(encoding="utf-8"))
    w = dict(zip(model["features"], model["coefficients"]))
    text = text_features(rows)
    kept, rule, final, titles = [], [], [], []
    for k, r in enumerate(rows):
        try:
            cv, jd = cache.get(r["Resume"], "resume"), cache.get(r["Job_Description"], "job")
        except Exception:
            continue
        res = score_pair_v3(cv, jd, as_of_date=as_of, config=config, cv_text=r["Resume"])
        cv_title = _latest_title(cv, as_of) or next((x.position for x in cv.document.work if x.position), None)
        kept.append(k); rule.append(res.overall_score)
        titles.append((cv_title or "unknown", jd.document.title or r["Role"]))
        final.append({d: res.unrounded_scores.model_dump()[d] / 100 for d in ("skills", "title", "experience", "education", "industry")})
    emb = QwenEmbedder()
    a = emb.encode([t for t, _ in titles]); b = emb.encode([t for _, t in titles])
    logits = []
    for i, k in enumerate(kept):
        f = final[i] | {"text_cos": float(text["qwen3_text_cos"][k]), "title_cos": float(a[i] @ b[i]),
                        "tfidf_cos": float(text["tfidf_cos"][k])}
        logits.append(model["intercept"] + sum(w[x] * f[x] for x in model["features"]))
    sub = [rows[k] for k in kept]
    feats = {"qwen3_text_cos": text["qwen3_text_cos"][kept], "tfidf_cos": text["tfidf_cos"][kept],
             "v31_rule_score": np.array(rule), "final_hybrid": np.array(logits)}
    out = report(sub, feats) | {"extracted": len(kept), "of": len(rows)}
    order = np.argsort(-np.array(logits))
    out["high_scored_rejects"] = [{"uid": sub[i]["uid"], "role": sub[i]["Role"], "fit": round(100 / (1 + math.exp(-logits[i])), 1),
                                   "reason": sub[i]["Reason_for_decision"]} for i in order if sub[i]["y"] == 0][:20]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit", type=int, help="first N pairs only (smoke test)")
    p.add_argument("--sample", type=int, help="balanced sample of N pairs for the extraction-based scorers")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    rows = load()[:args.limit]
    if args.sample:
        rows = sample(rows, args.sample)
        if args.extract:
            return extract(rows, args.workers)
        if args.score:
            res = score(rows)
            (OUT / f"sample{args.sample}_scored.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in res.items() if k != "high_scored_rejects"}, indent=2))
            return
    feats = text_features(rows)
    res = report(rows, feats)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "text_baselines.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
