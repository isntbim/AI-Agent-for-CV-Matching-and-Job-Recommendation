"""One CV in, ranked and explained JDs out: V4 extraction (validate-repair-retry) -> hybrid retrieval of a pool ->
the final ranking model reranks the pool -> templated explanation built only from the V3.1 scorer's result (no LLM).

Ranking = "Hybrid + TF-IDF (Qwen3)" (configs/hybrid_qwen_v1.json, written by scripts/match_ir_eval.py): logistic
weights on the V3.1 dimensions, Qwen3-Embedding-4B whole-text and title cosines and TF-IDF cosine, fitted on the
med2425 dev half (test NDCG@10 0.632 vs Qwen3 cosine 0.591, p<0.001; output/match_eval/lab/benchmark.md). The weights
come from English synthetic labels; on the Vietnamese CareerViet corpus they are unvalidated until human labels exist.

  python -m scripts.ats_service CV.txt --as-of 2026-11-02 [--top 5] [--pool 20] [--jd jd_001 ...] [--rerank]
  python -m scripts.ats_service --prepare-jds [--workers 3]   # extract every corpus JD once (offline; resumable)

Retrieval runs over the 361-JD English corpus (output/retrieval/corpus_v2 + index_v2) unless --jd names the JDs.
Extractions are cached under output/service/extractions, so a JD or CV is extracted once per model/prompt version.
Retrieval models run on the CPU; the ranking embeddings come from Ollama (qwen3-embedding:4b) next to the extractor.
On a 12 GB GPU both fit only with one extraction slot: serve with OLLAMA_NUM_PARALLEL=1 for the demo (qwen-v4 6.6 GB +
embeddings 3.1 GB); with 3 slots qwen-v4 takes 10 GB and Ollama swaps the two models (~5 s per request).
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

from src.scoring.extraction import ExtractionCache, write_json
from src.scoring.models import ScorerConfig
from src.scoring.rules import _latest_title
from src.scoring.runtime_v3 import score_pair_v3
from src.scoring.v4_client import V4Client

ROOT = Path(__file__).resolve().parent.parent
JD_MANIFEST = ROOT / "output/scorer_v3/teacher_labels/jd/manifest.json"
RELATIONS = ROOT / "configs/skill_relations_near_v1.json"
HYBRID = ROOT / "configs/hybrid_qwen_v1.json"
JD_EMB = ROOT / "output/service/jd_emb_qwen3_embedding.npz"
GRADES = ("No Fit", "Potential Fit", "Good Fit")


def v31_config() -> ScorerConfig:
    """V3.1 as evaluated (output/match_eval/lab: test-half AUC 0.910, all 59,337 pairs). The near relations are
    auto-built and treated as approved after a 60-item hand audit (~90% precise); grade cut-offs 43 / 58 (dev-fitted
    43.4 / 57.8 for macro-F1)."""
    table = json.loads(RELATIONS.read_text(encoding="utf-8"))
    return ScorerConfig(version="ats-en-v3", review_policy="automatic", title_policy="contained",
                        experience_policy="lower_bound", lower_threshold=43, upper_threshold=58,
                        relations=[r | {"review_status": "approved"} for r in table["relations"]],
                        relation_version=table["version"])


class Ranker:
    """The final ranking model over one JD corpus. Embeds with Qwen3-Embedding through Ollama, caches corpus JD
    embeddings on disk and fits TF-IDF on the corpus JD texts."""

    def __init__(self, jd_texts: dict[str, str]):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        from src.embeddings.qwen import QwenEmbedder

        self.model = json.loads(HYBRID.read_text(encoding="utf-8"))
        self.emb = QwenEmbedder()
        cached = dict(np.load(JD_EMB)) if JD_EMB.exists() else {}
        missing = [j for j in jd_texts if j not in cached]
        if missing:
            cached |= dict(zip(missing, self.emb.encode([jd_texts[j] for j in missing])))
            JD_EMB.parent.mkdir(parents=True, exist_ok=True)
            np.savez(JD_EMB, **cached)
        self.jd_emb, self.jd_texts = cached, jd_texts
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(jd_texts.values()))

    def scores(self, cv_text: str, cv_title: str | None, rows: list[tuple[str, str | None, dict]]) -> list[float]:
        """rows: (jd id, jd title, V3.1 unrounded dimension scores) -> hybrid logits (higher = better fit)."""
        cv = self.emb.encode([cv_text], query=True)[0]
        titles = self.emb.encode([cv_title or "unknown"] + [t or "unknown" for _, t, _ in rows])
        tf_cv = self.tfidf.transform([cv_text])
        w, out, self.last_features = dict(zip(self.model["features"], self.model["coefficients"])), [], []
        for k, (jd, _, dims) in enumerate(rows):
            feats = {d: dims[d] / 100 for d in ("skills", "title", "experience", "education", "industry")}
            feats |= {"text_cos": float(cv @ self.jd_emb[jd]), "title_cos": float(titles[0] @ titles[k + 1]),
                      "tfidf_cos": float((tf_cv @ self.tfidf.transform([self.jd_texts[jd]]).T).toarray()[0, 0])}
            out.append(self.model["intercept"] + sum(w[f] * feats[f] for f in self.model["features"]))
            self.last_features.append({f: w[f] * feats[f] for f in self.model["features"]} | {"_values": feats})
        return out

    @staticmethod
    def why(contrib: dict) -> str:
        """One sentence on what drives the fit: similarity (text, title, shared words) vs rule evidence."""
        sim = contrib["text_cos"] + contrib["title_cos"] + contrib["tfidf_cos"]
        rules = sum(v for k, v in contrib.items() if k not in ("text_cos", "title_cos", "tfidf_cos", "_values"))
        v = contrib["_values"]
        share = sim / (abs(sim) + abs(rules)) if sim or rules else 0
        lead = "mostly from how similar the CV and the job are" if share >= .66 else (
            "mostly from the rule evidence below" if share <= .33 else "from both similarity and the rule evidence")
        return (f"Fit comes {lead}: whole-text similarity {v['text_cos']:.2f}, job-title similarity "
                f"{v['title_cos']:.2f}, shared wording {v['tfidf_cos']:.2f}.")

    def grade(self, logit: float) -> int:
        lo, hi = self.model["grade_cutoffs"]
        return 0 if logit < lo else 1 if logit < hi else 2


def explain(result, grade: int | None = None, fit: float | None = None, why: str | None = None) -> list[str]:
    """Plain-English reasons, one line per dimension, read straight off the score result. With `grade`/`fit` the
    headline gives the ranking model's verdict and keeps the rule score beside it; `why` says what drove that fit.
    A Good Fit whose listed requirements matched no CV skill gets a caution: the verdict then rests on similarity."""
    d = result.dimensions
    head = (f"{GRADES[grade]}: fit {fit:.0f}/100 (rule score {result.overall_score:.1f}/100)." if grade is not None
            else f"{GRADES[result.grade]}: {result.overall_score:.1f}/100.")
    lines = [head] + ([why] if why else [])
    if grade == 2 and d["skills"].supported and d["skills"].score < 10:
        lines.append("Caution: almost none of the job's listed requirements matched your CV's skills exactly. "
                     "Check the Missing list before applying.")
    if d["skills"].supported:
        lines.append(f"Skills {d['skills'].score:.0f}/100. Matched: {', '.join(result.matched_skills[:8]) or 'none'}."
                     + (f" Missing: {', '.join(result.missing_skills[:8])}." if result.missing_skills else ""))
    if d["title"].supported:
        lines.append(f"Job title {d['title'].score:.0f}/100: " + ("same role as your CV title or work history." if d["title"].score >= 100
                     else "a related role in your CV title or work history." if d["title"].score > 0
                     else "no matching role in your CV title or work history."))
    if d["experience"].supported:
        s = d["experience"].score
        lines.append("Experience: no minimum required." if d["experience"].basis == "explicit_no_experience_requirement"
                     else "Experience: meets the required years." if s >= 100
                     else f"Experience: your relevant work covers about {s:.0f}% of the required years.")
    if d["education"].supported:
        s = d["education"].score
        lines.append("Education: no degree required." if d["education"].basis == "explicit_no_education_requirement"
                     else "Education: meets the required degree." if s >= 100
                     else "Education: one level below the required degree." if s >= 50 else "Education: below the required degree.")
    if d["industry"].supported:
        lines.append(f"Industry: your background covers {d['industry'].score:.0f}% of the job's domains.")
    for key in ("location", "salary"):
        if d[key].supported:
            lines.append(f"{key.capitalize()} {d[key].score:.0f}/100.")
    unknown = [k for k, v in d.items() if not v.supported]
    if unknown:
        lines.append(f"Not scored (evidence missing or inconclusive, neutral value used): {', '.join(unknown)}.")
    return lines


def retrieve(cv_document: dict, top: int, rerank: bool) -> list[str]:
    from types import SimpleNamespace

    from scripts.retrieval import load_index
    from src.embeddings.embedder import BGEEmbedder
    from src.retrieval.cache import EmbeddingCache
    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.reranker import BGEReranker, Candidate
    from src.retrieval.text import resume_text
    from src.vector_db.pgvector_store import InMemoryVectorStore

    args = SimpleNamespace(corpus=ROOT / "output/retrieval/corpus_v2", index=ROOT / "output/retrieval/index_v2")
    docs, _, vectors, _, _, im = load_index(args)
    cache = ROOT / ".cache/retrieval"
    text = resume_text(cv_document)
    embedder = BGEEmbedder(revision=im["revision"], device="cpu", batch_size=1, cache_folder=str(cache / "huggingface"))
    vector = EmbeddingCache(cache / "embeddings", im["model"], im["revision"]).embed([text], embedder)[0]
    store = InMemoryVectorStore(dimension=im["dimension"])
    store.create_table("service")
    store.upsert_batch([d["id"] for d in docs], list(vectors), docs)
    reranker = BGEReranker(device="cpu", cache_dir=str(cache / "huggingface")) if rerank else None
    result = RetrievalPipeline([Candidate(d["id"], d["text"], 0) for d in docs], store, None, reranker).query(text, vector)
    # "results" is capped at 10 for the retrieval benchmark; without the reranker the full fused list is the same order
    return [c.id for c in (result["results"] if rerank else result["hybrid"])[:top]]


def _cache(as_of: str | None = None, client: V4Client | None = None) -> ExtractionCache:
    return ExtractionCache(ROOT / "output/service/extractions", client or V4Client(), "ollama", "qwen2.5-7b-v4-qlora",
                           max_tokens=16384, ruleset="v3", as_of=as_of)


def _jd_paths() -> dict[str, str]:
    return {d["id"]: d["text_path"] for d in json.loads(JD_MANIFEST.read_text(encoding="utf-8"))["documents"]}


def corpus_ids() -> list[str]:
    return [d["id"] for d in json.loads((ROOT / "output/retrieval/corpus_v2/corpus.json").read_text(encoding="utf-8"))]


def corpus_ranker() -> "Ranker":
    """The ranking model over the whole 361-JD corpus (the service's search space)."""
    paths = _jd_paths()
    return Ranker({j: (ROOT / paths[j]).read_text(encoding="utf-8") for j in corpus_ids()})


def match(cv_text: str, as_of: str, top: int = 5, pool: int = 20, jd_ids: list[str] | None = None,
          rerank: bool = False, ranker: "Ranker | None" = None) -> dict:
    """The service call: extract the CV, retrieve `pool` JDs, score each with V3.1, return the best `top` explained.
    A JD whose extraction fails is reported in `skipped`, not silently dropped."""
    client, config = V4Client(), v31_config()
    resume = _cache(as_of, client).get(cv_text, "resume")
    retrieved = not jd_ids
    jd_ids = jd_ids or retrieve(resume.document.model_dump(mode="json"), pool, rerank)
    paths, rows, skipped, scored = _jd_paths(), [], {}, []
    for rank, jd in enumerate(jd_ids, 1):
        try:
            job = _cache(client=client).get((ROOT / paths[jd]).read_text(encoding="utf-8"), "job")
        except Exception as exc:  # one bad JD must not sink the whole answer
            skipped[jd] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        scored.append((rank, jd, job, score_pair_v3(resume, job, as_of_date=date.fromisoformat(as_of), config=config)))
    if scored:
        ranker = ranker or corpus_ranker()  # TF-IDF weights come from the whole corpus, never just the pool
        cv_title = _latest_title(resume, date.fromisoformat(as_of)) or next(
            (w.position for w in resume.document.work if w.position), None)
        logits = ranker.scores(cv_text, cv_title, [(jd, job.document.title, r.unrounded_scores.model_dump())
                                                   for _, jd, job, r in scored])
        for (rank, jd, job, result), logit, contrib in zip(scored, logits, ranker.last_features):
            grade, fit = ranker.grade(logit), 100 / (1 + math.exp(-logit))
            rows.append({"jd": jd, "title": job.document.title, "retrieval_rank": rank, "fit": round(fit, 1),
                         "_logit": logit, "grade": GRADES[grade], "rule_score": result.overall_score,
                         "scores": result.scores.model_dump(), "matched_skills": result.matched_skills,
                         "missing_skills": result.missing_skills, "explanation": explain(result, grade, fit, Ranker.why(contrib)),
                         "evidence": {k: v.evidence[:3] for k, v in result.dimensions.items() if v.supported and v.evidence}})
    rows.sort(key=lambda r: (-r.pop("_logit"), r["retrieval_rank"]))
    return {"as_of": as_of, "retrieved": retrieved, "pool": len(jd_ids), "matches": rows[:top], "skipped": skipped,
            "ranking_model": "hybrid_qwen_v1"}


def prepare_jds(workers: int = 1) -> None:
    """Extract every corpus JD into the cache so a live request only waits for its CV. workers > 1 needs
    OLLAMA_NUM_PARALLEL >= workers on the server (as in scripts/match_eval.py)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from types import SimpleNamespace

    from scripts.retrieval import load_index
    from src.scoring.v4_client import QwenV4

    docs = load_index(SimpleNamespace(corpus=ROOT / "output/retrieval/corpus_v2", index=ROOT / "output/retrieval/index_v2"))[0]
    llm, local, paths, failed = QwenV4(), threading.local(), _jd_paths(), []  # tokenizer loaded once, in this thread

    def one(jd: str) -> None:
        if not hasattr(local, "cache"):  # V4Client keeps per-call metadata on itself: one per thread
            local.cache = _cache(client=V4Client(llm))
        local.cache.get((ROOT / paths[jd]).read_text(encoding="utf-8"), "job")

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, d["id"]): d["id"] for d in docs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                status = "ok"
            except Exception as exc:
                failed.append(futures[future])
                status = f"FAILED {type(exc).__name__}"
            print(f"{i}/{len(docs)} {futures[future]} {status}", flush=True)
    print(f"done: {len(docs) - len(failed)} extracted, {len(failed)} failed: {sorted(failed)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cv", type=Path, nargs="?", help="CV as plain text")
    p.add_argument("--as-of", help="YYYY-MM-DD: the date employment durations are measured to")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--pool", type=int, default=20, help="how many retrieved JDs the scorer reranks")
    p.add_argument("--jd", nargs="*", help="score these corpus JD ids instead of retrieving")
    p.add_argument("--rerank", action="store_true", help="also apply the BGE cross-encoder inside retrieval")
    p.add_argument("--prepare-jds", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="--prepare-jds: parallel requests (OLLAMA_NUM_PARALLEL)")
    p.add_argument("--out", type=Path, default=ROOT / "output/service/last_match.json")
    args = p.parse_args()
    if args.prepare_jds:
        return prepare_jds(args.workers)
    if not (args.cv and args.as_of):
        p.error("cv and --as-of are required")
    out = match(args.cv.read_text(encoding="utf-8"), args.as_of, args.top, args.pool, args.jd, args.rerank)
    write_json(args.out, {"cv": str(args.cv), **out})
    for r in out["matches"]:
        print(f"\n{r['fit']:5.1f}  {r['jd']}  {r['title']}  (retrieval rank {r['retrieval_rank']})")
        print("\n".join(f"   - {line}" for line in r["explanation"]))
    if out["skipped"]:
        print(f"\nskipped (extraction failed): {', '.join(out['skipped'])}")
    print(f"details: {args.out}")


if __name__ == "__main__":
    main()
