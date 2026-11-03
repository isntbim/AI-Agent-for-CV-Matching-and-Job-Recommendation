"""One CV in, ranked JDs out: V4 extraction (validate-repair-retry) -> hybrid retrieval -> v3 scorer.

  python -m scripts.ats_service CV.txt --as-of 2026-11-02 [--top 5] [--jd jd_001 jd_042 ...] [--rerank]

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
AUTO = ScorerConfig(version="ats-en-v3", review_policy="automatic")  # unreviewed V4 extractions
JD_MANIFEST = ROOT / "output/scorer_v3/teacher_labels/jd/manifest.json"


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cv", type=Path, help="CV as plain text")
    p.add_argument("--as-of", required=True, help="YYYY-MM-DD: the date employment durations are measured to")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--jd", nargs="*", help="score these corpus JD ids instead of retrieving")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "output/service/last_match.json")
    args = p.parse_args()

    client = V4Client()
    cache = lambda as_of=None: ExtractionCache(ROOT / "output/service/extractions", client, "ollama",
                                               "qwen2.5-7b-v4-qlora", max_tokens=16384, ruleset="v3", as_of=as_of)
    resume = cache(args.as_of).get(args.cv.read_text(encoding="utf-8"), "resume")
    jd_ids = args.jd or retrieve(resume.document.model_dump(mode="json"), args.top, args.rerank)
    paths = {d["id"]: d["text_path"] for d in json.loads(JD_MANIFEST.read_text(encoding="utf-8"))["documents"]}
    rows = []
    for jd in jd_ids:
        job = cache().get((ROOT / paths[jd]).read_text(encoding="utf-8"), "job")
        result = score_pair_v3(resume, job, as_of_date=date.fromisoformat(args.as_of), config=AUTO)
        rows.append({"jd": jd, "title": job.document.title, "overall": result.overall_score, "grade": result.grade,
                     "scores": result.scores.model_dump(), "matched_skills": result.matched_skills,
                     "missing_skills": result.missing_skills})
    rows.sort(key=lambda r: -r["overall"])
    write_json(args.out, {"cv": str(args.cv), "as_of": args.as_of, "retrieved": not args.jd, "matches": rows})
    for r in rows:
        print(f"{r['overall']:5.1f}  grade {r['grade']}  {r['jd']}  {r['title']}  "
              f"(missing: {', '.join(r['missing_skills'][:5]) or '-'})")
    print(f"details: {args.out}")


if __name__ == "__main__":
    main()
