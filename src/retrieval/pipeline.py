"""Dense/BM25 top-50 → RRF(60) → optional top-20 reranking → top-10."""

from dataclasses import replace
import time

from .lexical import BM25, rrf
from .reranker import Candidate, rerank


class RetrievalPipeline:
    def __init__(self, documents: list[Candidate], store, embedder, reranker=None):
        self.documents = {c.id: c for c in documents}
        if len(self.documents) != len(documents):
            raise ValueError("Duplicate corpus IDs")
        self.lexical, self.store, self.embedder, self.reranker = BM25(documents), store, embedder, reranker

    def query(self, text: str, query_vector=None):
        start = time.perf_counter()
        vector = query_vector if query_vector is not None else self.embedder.embed(text)
        hits = self.store.search(vector, top_k=50)
        dense = []
        for rank, hit in enumerate(hits, 1):
            if hit.id not in self.documents:
                raise ValueError(f"Dense store returned an unknown ID: {hit.id}")
            dense.append(replace(self.documents[hit.id], retrieval_score=float(hit.score), original_rank=rank))
        dense_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        lexical = self.lexical.search(text, 50)
        fused = rrf([dense, lexical])
        fusion_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        ranked = rerank(text, fused[:20], self.reranker) if self.reranker else fused
        return {"dense": dense, "bm25": lexical, "hybrid": fused, "results": ranked[:10],
                "latency_ms": {"dense": dense_ms, "bm25_fusion": fusion_ms,
                               "rerank": (time.perf_counter() - start) * 1000},
                "truncations": getattr(self.reranker, "last_truncations", [])}
