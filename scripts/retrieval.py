"""Corpus nomination/preparation, local indexing, CV queries and ranking diagnostics."""

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from src.scoring.extraction import write_json
from src.retrieval.corpus import nominations, prepare_corpus, load_json, digest
from src.retrieval.text import resume_text, TEXT_VERSION
from src.retrieval.cache import EmbeddingCache, save_array, valid_vectors
from src.retrieval.reranker import Candidate, MockReranker, BGEReranker
from src.retrieval.evaluate import frozen_pools, evaluate, CachedReranker


def index(args):
    from src.embeddings.embedder import BGEEmbedder, MockEmbedder
    from huggingface_hub import HfApi
    import torch
    documents = load_json(args.corpus / "corpus.json")
    cvs = load_json(args.cvs)
    texts = [resume_text(cv) for cv in cvs]
    revision = "mock-hash-v1" if args.mock else (args.embedding_revision or HfApi().model_info("BAAI/bge-m3").sha)
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    embedder = MockEmbedder() if args.mock else BGEEmbedder(revision=revision, cache_folder=str(args.cache / "huggingface"),
                                                         batch_size=1, device=args.device)
    cache = EmbeddingCache(args.cache / "embeddings", "mock" if args.mock else "BAAI/bge-m3", revision)
    vectors = cache.embed([d["text"] for d in documents], embedder)
    queries = cache.embed(texts, embedder)
    truncations = []
    if not args.mock:
        tokenizer = embedder._model.tokenizer
        for id, text in [(d["id"], d["text"]) for d in documents] + [(cv["cv_id"], t) for cv, t in zip(cvs, texts)]:
            n = len(tokenizer(text, truncation=False, verbose=False)["input_ids"])
            if n > 8192:
                truncations.append({"id": id, "input_tokens": n, "limit": 8192, "removed_tokens": n - 8192})
    args.index.mkdir(parents=True, exist_ok=True)
    save_array(args.index / "vectors.npy", vectors)
    save_array(args.index / "query_vectors.npy", queries)
    write_json(args.index / "queries.json", [{"id": cv["cv_id"], "domain": cv["domain"], "text": t,
                                               "text_sha256": digest(t.encode())} for cv, t in zip(cvs, texts)])
    manifest = load_json(args.corpus / "manifest.json")
    report = {"corpus_manifest_hash": manifest["manifest_hash"], "document_ids": [d["id"] for d in documents],
              "query_ids": [cv["cv_id"] for cv in cvs], "query_text_sha256": [digest(t.encode()) for t in texts],
              "model": "mock" if args.mock else "BAAI/bge-m3", "revision": revision, "text_version": TEXT_VERSION,
              "dimension": embedder.dimension, "device": getattr(embedder, "_device", "mock"),
              "dtype": "float32", "max_length": 8192, "batch_size": 1, "truncations": truncations,
              "cache_hits": cache.hits, "cache_misses": cache.misses, "elapsed_seconds": time.perf_counter() - start,
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
              "vectors_sha256": digest((args.index / "vectors.npy").read_bytes()),
              "query_vectors_sha256": digest((args.index / "query_vectors.npy").read_bytes())}
    write_json(args.index / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"document_ids", "query_ids", "query_text_sha256"}}))


def load_index(args):
    manifest, index_manifest = load_json(args.corpus / "manifest.json"), load_json(args.index / "manifest.json")
    docs, queries = load_json(args.corpus / "corpus.json"), load_json(args.index / "queries.json")
    if manifest["manifest_hash"] != index_manifest["corpus_manifest_hash"] or [d["id"] for d in docs] != index_manifest["document_ids"]:
        raise ValueError("Index/corpus mismatch; rebuild embeddings")
    if [q["id"] for q in queries] != index_manifest["query_ids"] or [digest(q["text"].encode()) for q in queries] != index_manifest["query_text_sha256"]:
        raise ValueError("Stale query embeddings")
    for name, field in (("vectors.npy", "vectors_sha256"), ("query_vectors.npy", "query_vectors_sha256")):
        if digest((args.index / name).read_bytes()) != index_manifest[field]:
            raise ValueError("Embedding file checksum mismatch")
    vectors = valid_vectors(np.load(args.index / "vectors.npy", allow_pickle=False), len(docs), index_manifest["dimension"])
    query_vectors = valid_vectors(np.load(args.index / "query_vectors.npy", allow_pickle=False), len(queries), index_manifest["dimension"])
    return docs, queries, vectors, query_vectors, manifest, index_manifest


def benchmark(args):
    import torch
    docs, queries, vectors, query_vectors, manifest, index_manifest = load_index(args)
    if (index_manifest["model"] == "mock") != args.mock:
        raise ValueError("Mock/real-model index mismatch")
    cvs = load_json(args.cvs)
    if [cv["cv_id"] for cv in cvs] != [q["id"] for q in queries] or any(resume_text(cv) != q["text"] for cv, q in zip(cvs, queries)):
        raise ValueError("CV records changed after indexing")
    pairs = load_json(args.pairs)
    pools = frozen_pools(cvs, pairs, digest(args.pairs.read_bytes()))
    path = args.output / "frozen_pools.json"
    if path.exists() and load_json(path) != pools:
        raise ValueError("Frozen evaluation pool changed; use a new versioned output directory")
    write_json(path, pools)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    client = MockReranker() if args.mock else BGEReranker(revision=args.reranker_revision, device=args.device,
                                                        cache_dir=str(args.cache / "huggingface"))
    cached = CachedReranker(client, args.output / "pair_scores.json")
    report = evaluate(docs, cvs, [q["text"] for q in queries], vectors, query_vectors, pairs, manifest, pools, cached, args.output)
    report["runtime"] = {"model": "mock" if args.mock else client.model_name, "revision": client.revision,
                          "device": getattr(client, "device", "mock"), "dtype": str(getattr(client, "dtype", "mock")),
                          "batch_size": 1, "pair_limit": 8192, "cache_hits": cached.hits, "cache_misses": cached.misses,
                          "elapsed_seconds": time.perf_counter() - start,
                          "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                          "torch": torch.__version__, "embedding_manifest": index_manifest}
    write_json(args.output / "report.json", report)
    print(json.dumps({"recommendation": report["recommendation"], "failures": len(report["failures"]),
                      "truncations": report["truncation_count"],
                      "runtime": {k: v for k, v in report["runtime"].items() if k != "embedding_manifest"}}))


def query(args):
    import torch
    from src.vector_db.pgvector_store import InMemoryVectorStore
    from src.retrieval.pipeline import RetrievalPipeline
    docs, queries, vectors, query_vectors, _, im = load_index(args)
    embedding_truncations = []
    if args.cv_json:
        from src.embeddings.embedder import BGEEmbedder, MockEmbedder
        cv = load_json(args.cv_json)
        if not isinstance(cv, dict):
            raise ValueError("CV JSON must contain one structured CV or document envelope")
        text = resume_text(cv.get("document", cv))
        if not text.strip():
            raise ValueError("CV JSON has no populated retrieval fields")
        query_id = "external_cv"
        start = time.perf_counter()
        embedder = MockEmbedder() if args.mock else BGEEmbedder(revision=im["revision"], device=args.device,
                           batch_size=1, cache_folder=str(args.cache / "huggingface"))
        cache = EmbeddingCache(args.cache / "embeddings", im["model"], im["revision"])
        vector = cache.embed([text], embedder)[0]
        if not args.mock:
            length = len(embedder._model.tokenizer(text, truncation=False, verbose=False)["input_ids"])
            if length > 8192:
                embedding_truncations.append({"id": query_id, "input_tokens": length, "limit": 8192, "removed_tokens": length - 8192})
        embedding_ms = (time.perf_counter() - start) * 1000
        del embedder, cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        matches = [i for i, q in enumerate(queries) if q["id"] == args.cv_id]
        if len(matches) != 1:
            raise ValueError("Unknown CV ID")
        qi = matches[0]
        query_id, text, vector = args.cv_id, queries[qi]["text"], query_vectors[qi]
        embedding_ms = 0
    store = InMemoryVectorStore(dimension=im["dimension"])
    store.create_table("retrieval_local")
    store.upsert_batch([d["id"] for d in docs], list(vectors), docs)
    client = (MockReranker() if args.mock else BGEReranker(revision=args.reranker_revision, device=args.device,
                                                        cache_dir=str(args.cache / "huggingface"))) if args.rerank else None
    pipeline = RetrievalPipeline([Candidate(d["id"], d["text"], 0) for d in docs], store, None, client)
    result = pipeline.query(text, vector)
    report = {"cv_id": query_id, "latency_ms": {"embedding": embedding_ms, **result["latency_ms"]},
              "embedding_truncations": embedding_truncations, "truncations": result["truncations"],
              "embedding_revision": im["revision"], "reranker_revision": getattr(client, "revision", None),
              "query_text_sha256": digest(text.encode()), "results": [asdict(c) for c in result["results"]]}
    path = args.output / f"query_{query_id}.json"
    write_json(path, report)
    print(json.dumps({**report, "report_path": str(path),
                      "results": [{k: v for k, v in c.items() if k != "text"} for c in report["results"]]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["nominate", "prepare", "index", "query", "evaluate", "models", "azure-verify"])
    parser.add_argument("--jobs", type=Path, default=Path("data/processed/structured_jobs/all_jobs_en.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/jobs"))
    parser.add_argument("--cvs", type=Path, default=Path("data/processed/structured_cvs/all_cvs.json"))
    parser.add_argument("--pairs", type=Path, default=Path("data/datasets/matching_pairs/benchmark_matching_pairs.json"))
    parser.add_argument("--review", type=Path, default=Path("output/retrieval/cleaning_review.json"))
    parser.add_argument("--corpus", type=Path, default=Path("output/retrieval/corpus_v2"))
    parser.add_argument("--index", type=Path, default=Path("output/retrieval/index_v2"))
    parser.add_argument("--output", type=Path, default=Path("output/retrieval/benchmark_v2"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/retrieval"))
    parser.add_argument("--embedding-revision")
    parser.add_argument("--reranker-revision")
    parser.add_argument("--tokenizer-revision", help="Qwen tokenizer revision for the models command; weights are not downloaded")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    query_input = parser.add_mutually_exclusive_group()
    query_input.add_argument("--cv-id", default="cv_001")
    query_input.add_argument("--cv-json", type=Path, help="Structured CV JSON; embeds before loading the reranker")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="Optional stage; default recommendation comes from benchmark")
    args = parser.parse_args()
    if args.command == "azure-verify":
        import torch
        from src.retrieval.azure import verify
        docs, queries, vectors, query_vectors, manifest, index_manifest = load_index(args)
        if index_manifest["model"] == "mock" or args.mock:
            raise ValueError("Live verification requires real BGE-M3 embeddings")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        client = BGEReranker(revision=args.reranker_revision, device=args.device, cache_dir=str(args.cache / "huggingface"))
        cached = CachedReranker(client, args.output / "pair_scores.json")
        try:
            report = verify(docs, queries, vectors, query_vectors, manifest, cached, args.output)
        except Exception as error:
            write_json(args.output / "report.json", {"status": "failed", "error_type": type(error).__name__,
                       "corpus_manifest_hash": manifest["manifest_hash"], "embedding_revision": index_manifest["revision"],
                       "reranker_revision": client.revision,
                       "reason": "Live verification did not complete. Existing tables are preserved; inspect the command error and retry after resolving it."})
            raise
        report["runtime"] = {"embedding_revision": index_manifest["revision"], "reranker_model": client.model_name,
                             "reranker_revision": client.revision, "device": client.device, "dtype": str(client.dtype),
                             "batch_size": 1, "pair_limit": 8192, "cache_hits": cached.hits, "cache_misses": cached.misses,
                             "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        write_json(args.output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}))
    elif args.command == "models":
        from huggingface_hub import HfApi, snapshot_download
        revisions = {}
        requested = {"BAAI/bge-m3": args.embedding_revision,
                     "BAAI/bge-reranker-v2-m3": args.reranker_revision,
                     "Qwen/Qwen2.5-7B-Instruct": args.tokenizer_revision}
        for model in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "Qwen/Qwen2.5-7B-Instruct"):
            revision = HfApi().model_info(model, revision=requested[model]).sha
            patterns = ["*.json", "*.txt", "*.model", "tokenizer*", "*.jinja", "1_Pooling/*"]
            if model != "Qwen/Qwen2.5-7B-Instruct":
                patterns.append("model.safetensors")
            else:
                patterns = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]
            snapshot_download(model, revision=revision, cache_dir=str(args.cache / "huggingface"), allow_patterns=patterns)
            revisions[model] = revision
            print(json.dumps({"downloaded": model, "revision": revision}), flush=True)
        write_json(args.cache / "model_revisions.json", revisions)
    elif args.command == "nominate":
        rows = nominations(load_json(args.jobs), args.raw)
        write_json(args.review.with_name("cleaning_nominations.json"), rows)
        print(json.dumps({"nominations": len(rows)}))
    elif args.command == "prepare":
        report = prepare_corpus(args.jobs, args.raw, args.review, args.corpus)
        print(json.dumps({k: v for k, v in report.items() if k != "documents"}))
    elif args.command == "index":
        index(args)
    elif args.command == "evaluate":
        benchmark(args)
    else:
        query(args)


if __name__ == "__main__":
    main()
