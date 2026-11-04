"""One CV in, ranked and explained JDs out: V4 extraction (validate-repair-retry) -> hybrid retrieval of a pool ->
V3.1 rule scorer reranks the pool -> templated explanation built only from the scorer's own result (no LLM).

  python -m scripts.ats_service CV.txt --as-of 2026-11-02 [--top 5] [--pool 20] [--jd jd_001 ...] [--rerank]
  python -m scripts.ats_service --prepare-jds [--workers 3]   # extract every corpus JD once (offline; resumable)

Retrieval runs over the 361-JD English corpus (output/retrieval/corpus_v2 + index_v2) unless --jd names the JDs.
Extractions are cached under output/service/extractions, so a JD or CV is extracted once per model/prompt version.
Retrieval models run on the CPU: the GPU is left to the extractor.
"""

import argparse
import json
from datetime import date
from pathlib import Path

from src.scoring.extraction import ExtractionCache, write_json
from src.scoring.models import ScorerConfig
from src.scoring.runtime_v3 import score_pair_v3
from src.scoring.v4_client import V4Client

ROOT = Path(__file__).resolve().parent.parent
JD_MANIFEST = ROOT / "output/scorer_v3/teacher_labels/jd/manifest.json"
RELATIONS = ROOT / "configs/skill_relations_near_v1.json"
GRADES = ("No Fit", "Potential Fit", "Good Fit")


def v31_config() -> ScorerConfig:
    """V3.1 as evaluated (output/match_eval/lab, test half AUC 0.917). The near relations are auto-built and treated as
    approved after a 60-item hand audit (~90% precise); grade cut-offs 43 / 60 were fitted on the dev half for macro-F1."""
    table = json.loads(RELATIONS.read_text(encoding="utf-8"))
    return ScorerConfig(version="ats-en-v3", review_policy="automatic", title_policy="contained",
                        experience_policy="lower_bound", lower_threshold=43, upper_threshold=60,
                        relations=[r | {"review_status": "approved"} for r in table["relations"]],
                        relation_version=table["version"])


def explain(result) -> list[str]:
    """Plain-English reasons, one line per dimension, read straight off the score result."""
    d, lines = result.dimensions, [f"{GRADES[result.grade]}: {result.overall_score:.1f}/100."]
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
    return [c.id for c in result["results"][:top]]


def _cache(as_of: str | None = None, client: V4Client | None = None) -> ExtractionCache:
    return ExtractionCache(ROOT / "output/service/extractions", client or V4Client(), "ollama", "qwen2.5-7b-v4-qlora",
                           max_tokens=16384, ruleset="v3", as_of=as_of)


def _jd_paths() -> dict[str, str]:
    return {d["id"]: d["text_path"] for d in json.loads(JD_MANIFEST.read_text(encoding="utf-8"))["documents"]}


def match(cv_text: str, as_of: str, top: int = 5, pool: int = 20, jd_ids: list[str] | None = None,
          rerank: bool = False) -> dict:
    """The service call: extract the CV, retrieve `pool` JDs, score each with V3.1, return the best `top` explained.
    A JD whose extraction fails is reported in `skipped`, not silently dropped."""
    client, config = V4Client(), v31_config()
    resume = _cache(as_of, client).get(cv_text, "resume")
    retrieved = not jd_ids
    jd_ids = jd_ids or retrieve(resume.document.model_dump(mode="json"), pool, rerank)
    paths, rows, skipped = _jd_paths(), [], {}
    for rank, jd in enumerate(jd_ids, 1):
        try:
            job = _cache(client=client).get((ROOT / paths[jd]).read_text(encoding="utf-8"), "job")
        except Exception as exc:  # one bad JD must not sink the whole answer
            skipped[jd] = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue
        result = score_pair_v3(resume, job, as_of_date=date.fromisoformat(as_of), config=config)
        rows.append({"jd": jd, "title": job.document.title, "retrieval_rank": rank, "overall": result.overall_score,
                     "_exact": result.unrounded_overall_score,
                     "grade": GRADES[result.grade], "scores": result.scores.model_dump(),
                     "matched_skills": result.matched_skills, "missing_skills": result.missing_skills,
                     "explanation": explain(result),
                     "evidence": {k: v.evidence[:3] for k, v in result.dimensions.items() if v.supported and v.evidence}})
    rows.sort(key=lambda r: (-r.pop("_exact"), r["retrieval_rank"]))  # rank on the unrounded score
    return {"as_of": as_of, "retrieved": retrieved, "pool": len(jd_ids), "matches": rows[:top], "skipped": skipped}


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
        print(f"\n{r['overall']:5.1f}  {r['jd']}  {r['title']}  (retrieval rank {r['retrieval_rank']})")
        print("\n".join(f"   - {line}" for line in r["explanation"]))
    if out["skipped"]:
        print(f"\nskipped (extraction failed): {', '.join(out['skipped'])}")
    print(f"details: {args.out}")


if __name__ == "__main__":
    main()
