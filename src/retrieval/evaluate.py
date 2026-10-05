"""Frozen-pool diagnostics and full-corpus known-positive retrieval evaluation."""

from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np

from src.scoring.extraction import write_json
from .corpus import digest, load_json
from .lexical import BM25, rrf, TOKENIZATION_VERSION
from .reranker import Candidate, rerank
from .metrics import reciprocal_rank, ndcg, known_positive_recall, judged_coverage
from .text import TEXT_VERSION


class CachedReranker:
    """Cache relevance by model, pair texts and token/truncation configuration."""

    def __init__(self, client, path: Path):
        self.client, self.path = client, path
        self.entries = load_json(path) if path.exists() else {}
        self.last_truncations = []
        self.last_pair_lengths = []
        self.hits = self.misses = 0

    def score_pairs(self, pairs):
        config = [self.client.revision, getattr(self.client, "max_length", None),
                  str(getattr(self.client, "dtype", "mock")), "longest_first", "pair-score-cache-v1"]
        keys = [digest([config, a, b]) for a, b in pairs]
        missing = list(dict.fromkeys(k for k in keys if k not in self.entries))
        lookup = dict(zip(keys, pairs))
        self.hits += len(keys) - len(missing)
        if missing:
            values = list(self.client.score_pairs([lookup[k] for k in missing]))
            if len(values) != len(missing) or not np.isfinite(values).all():
                raise ValueError("Invalid reranker response")
            truncations = {r["pair_index"]: r for r in getattr(self.client, "last_truncations", [])}
            lengths = getattr(self.client, "last_pair_lengths", [])
            for i, (key, score) in enumerate(zip(missing, values)):
                self.entries[key] = {"score": float(score), "tokens": lengths[i] if lengths else None,
                                     "truncation": truncations.get(i)}
            self.misses += len(missing)
            write_json(self.path, self.entries)
        self.last_pair_lengths = [self.entries[k]["tokens"] for k in keys]
        self.last_truncations = [{**self.entries[k]["truncation"], "pair_index": i}
                                 for i, k in enumerate(keys) if self.entries[k]["truncation"]]
        return [self.entries[k]["score"] for k in keys]


def frozen_pools(cvs, pairs, benchmark_hash):
    by_cv = defaultdict(list)
    identities = set()
    for pair in pairs:
        key = (pair["cv_id"], pair["jd_id"])
        if key in identities:
            raise ValueError("Duplicate labelled pair")
        identities.add(key)
        if pair["grade"] not in (0, 1, 2):
            raise ValueError("Invalid relevance grade")
        by_cv[pair["cv_id"]].append(pair)
    queries = []
    for cv in cvs:
        labelled = sorted(by_cv[cv["cv_id"]], key=lambda p: p["jd_id"])
        # The existing generator marks the six script-generated examples explicitly.
        primary = [p for p in labelled if p["pair_type"] != "heuristic_negative"]
        queries.append({"cv_id": cv["cv_id"], "domain": cv["domain"],
                        "primary": {p["jd_id"]: p["grade"] for p in primary},
                        "expanded": {p["jd_id"]: p["grade"] for p in labelled}})
    identity = {"version": "frozen-labelled-pools-v1", "benchmark_sha256": benchmark_hash, "queries": queries}
    return {**identity, "manifest_hash": digest(identity)}


def aggregate(rows, fields):
    domains = sorted({r["domain"] for r in rows})
    by_domain = {domain: {field: float(np.mean([r[field] for r in rows if r["domain"] == domain and r[field] is not None]))
                          if any(r[field] is not None for r in rows if r["domain"] == domain) else None
                          for field in fields} for domain in domains}
    macro = {field: float(np.mean([v[field] for v in by_domain.values() if v[field] is not None]))
             if any(v[field] is not None for v in by_domain.values()) else None for field in fields}
    return {"macro_by_domain": macro, "by_domain": by_domain}


def evaluate(documents, cvs, query_texts, vectors, query_vectors, pairs, manifest,
             pool_manifest, reranker, output: Path):
    candidates = [Candidate(d["id"], d["text"], 0, {"text_sha256": d["text_sha256"]}) for d in documents]
    index = {c.id: i for i, c in enumerate(candidates)}
    lexical = BM25(candidates)
    queries = {q["cv_id"]: q for q in pool_manifest["queries"]}
    rows, full_rows, failures, truncations, latencies = [], [], [], [], []
    for qi, (cv, text, vector) in enumerate(zip(cvs, query_texts, query_vectors)):
        try:
            started = time.perf_counter()
            dense_scores = vectors @ vector
            dense_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            bm25_scores = lexical.scores(text)
            bm25_ms = (time.perf_counter() - started) * 1000
            query = queries[cv["cv_id"]]
            for pool in ("primary", "expanded"):
                grades = query[pool]
                pool_candidates = [candidates[index[id]] for id in grades]
                dense = sorted([replace(c, retrieval_score=float(dense_scores[index[c.id]])) for c in pool_candidates],
                               key=lambda c: (-c.retrieval_score, c.id))
                bm25 = sorted([replace(c, retrieval_score=bm25_scores[c.id]) for c in pool_candidates],
                              key=lambda c: (-c.retrieval_score, c.id))
                started = time.perf_counter()
                hybrid = rrf([dense[:50], bm25[:50]])
                fusion_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                ranked = rerank(text, hybrid[:20], reranker)
                # Preserve any candidates outside the reranked head for meaningful @10 with small pools.
                reranked = ranked + hybrid[20:]
                rerank_ms = (time.perf_counter() - started) * 1000
                for row in reranker.last_truncations:
                    truncations.append({**row, "cv_id": cv["cv_id"], "pool": pool,
                                        "jd_id": hybrid[row["pair_index"]].id})
                for method, ranking in (("dense", dense), ("bm25", bm25), ("hybrid", hybrid), ("reranked", reranked)):
                    ids = [c.id for c in ranking]
                    rows.append({"cv_id": cv["cv_id"], "domain": cv["domain"], "pool": pool, "method": method,
                                 "no_labelled_positive": not any(g >= 1 for g in grades.values()),
                                 "mrr10": reciprocal_rank(ids, grades), "ndcg5": ndcg(ids, grades, 5),
                                 "ndcg10": ndcg(ids, grades, 10), "ranking": ids,
                                 "scores": [{"id": c.id, "retrieval": c.retrieval_score,
                                             "reranker": c.reranker_score} for c in ranking]})
                latencies.append({"cv_id": cv["cv_id"], "domain": cv["domain"], "pool": pool,
                                  "dense_ms": dense_ms, "bm25_ms": bm25_ms, "fusion_ms": fusion_ms,
                                  "rerank_ms": rerank_ms})
            started = time.perf_counter()
            full_dense = sorted([replace(c, retrieval_score=float(dense_scores[index[c.id]])) for c in candidates],
                                key=lambda c: (-c.retrieval_score, c.id))[:50]
            full_bm25 = lexical.search(text, 50)
            full_hybrid = rrf([full_dense, full_bm25])
            full_ranked = rerank(text, full_hybrid[:20], reranker)
            for row in reranker.last_truncations:
                truncations.append({**row, "cv_id": cv["cv_id"], "pool": "full_corpus",
                                    "jd_id": full_hybrid[row["pair_index"]].id})
            latencies.append({"cv_id": cv["cv_id"], "domain": cv["domain"], "pool": "full_corpus",
                              "flow_ms": dense_ms + (time.perf_counter() - started) * 1000})
            for method, ranking in (("dense", full_dense), ("bm25", full_bm25), ("hybrid", full_hybrid),
                                    ("reranked", full_ranked + full_hybrid[20:])):
                ids = [c.id for c in ranking]
                full_rows.append({"cv_id": cv["cv_id"], "domain": cv["domain"], "method": method,
                                  "known_positive_recall50": known_positive_recall(ids, query["primary"]),
                                  "judged_coverage10": judged_coverage(ids, query["primary"], 10),
                                  "judged_coverage50": judged_coverage(ids, query["primary"], 50), "ranking": ids})
            print(f"Evaluated {qi + 1}/{len(cvs)}: {cv['cv_id']}", flush=True)
        except Exception as error:
            failures.append({"cv_id": cv["cv_id"], "type": type(error).__name__, "error": str(error)})
            print(f"Evaluation failed: {cv['cv_id']} {type(error).__name__}", flush=True)
    report = {"corpus_manifest_hash": manifest["manifest_hash"], "pool_manifest_hash": pool_manifest["manifest_hash"],
              "text_version": TEXT_VERSION, "tokenization_version": TOKENIZATION_VERSION,
              "bm25": {"k1": 1.5, "b": .75}, "rrf_constant": 60, "dense_k": 50, "bm25_k": 50,
              "rerank_k": 20, "return_k": 10, "relevance": "grade >= 1", "gain": "2^grade - 1",
              "evaluation_scope": "Previously inspected silver labels; diagnostic, not independent held-out test data.",
              "unlabelled_policy": "Full corpus: known-positive recall and judged coverage only; unjudged pairs are not negatives.",
              "failures": failures, "truncation_count": len(truncations),
              "no_labelled_positive_queries": {pool: sum(not any(g >= 1 for g in q[pool].values())
                                                        for q in queries.values()) for pool in ("primary", "expanded")},
              "frozen_pool": {}, "full_corpus": {}, "latency": {}}
    for pool in ("primary", "expanded"):
        report["frozen_pool"][pool] = {}
        for method in ("dense", "bm25", "hybrid", "reranked"):
            subset = [r for r in rows if r["pool"] == pool and r["method"] == method]
            positives = [r for r in subset if not r["no_labelled_positive"]]
            report["frozen_pool"][pool][method] = {"all_queries": aggregate(subset, ["mrr10", "ndcg5", "ndcg10"]),
                                                   "queries_with_positives": aggregate(positives, ["mrr10", "ndcg5", "ndcg10"])}
    for method in ("dense", "bm25", "hybrid", "reranked"):
        report["full_corpus"][method] = aggregate([r for r in full_rows if r["method"] == method],
                                                   ["known_positive_recall50", "judged_coverage10", "judged_coverage50"])
    for pool in ("primary", "expanded", "full_corpus"):
        subset = [r for r in latencies if r["pool"] == pool]
        fields = ["flow_ms"] if pool == "full_corpus" else ["dense_ms", "bm25_ms", "fusion_ms", "rerank_ms"]
        report["latency"][pool] = {field: {"median": float(np.median([r[field] for r in subset])),
                                             "p95": float(np.percentile([r[field] for r in subset], 95))}
                                   for field in fields} if subset else {}
        report["latency"][pool]["by_domain"] = aggregate(subset, fields)["by_domain"]
    baseline = report["frozen_pool"]["primary"]["hybrid"]["all_queries"]["macro_by_domain"]
    candidate = report["frozen_pool"]["primary"]["reranked"]["all_queries"]["macro_by_domain"]
    improve = (not failures and candidate["ndcg10"] is not None and baseline["ndcg10"] is not None and
               candidate["ndcg10"] > baseline["ndcg10"] and candidate["mrr10"] >= baseline["mrr10"])
    report["recommendation"] = "eligible_for_default" if improve else "optional_stage"
    write_json(output / "rankings.json", rows)
    write_json(output / "full_corpus_rankings.json", full_rows)
    write_json(output / "latencies.json", latencies)
    write_json(output / "truncations.json", truncations)
    write_json(output / "report.json", report)
    return report
