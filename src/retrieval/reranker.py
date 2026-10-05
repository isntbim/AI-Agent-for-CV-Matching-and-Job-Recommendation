"""Replaceable relevance scoring, independent of deterministic ATS scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str
    retrieval_score: float
    metadata: dict = field(default_factory=dict)
    original_rank: int = 0
    reranker_score: float | None = None
    final_rank: int = 0


@runtime_checkable
class RerankerClient(Protocol):
    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class MockReranker:
    """Deterministic technical-token overlap for integration checks."""

    revision = "mock-token-overlap-v1"

    def score_pairs(self, pairs):
        from .lexical import tokenize
        return [float(len(set(tokenize(query)) & set(tokenize(text)))) for query, text in pairs]


def rerank(query: str, candidates: Sequence[Candidate], client: RerankerClient) -> list[Candidate]:
    if not candidates:
        return []
    ids = [c.id for c in candidates]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidates require unique, nonempty stable IDs")
    if any(not math.isfinite(c.retrieval_score) for c in candidates):
        raise ValueError("Non-finite retrieval score")
    scores = list(client.score_pairs([(query, c.text) for c in candidates]))
    if len(scores) != len(candidates):
        raise ValueError("Reranker score count does not match candidates")
    if any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("Non-finite reranker score")
    values = [replace(c, original_rank=i + 1, reranker_score=float(score))
              for i, (c, score) in enumerate(zip(candidates, scores))]
    values.sort(key=lambda c: (-c.reranker_score, c.original_rank, c.id))
    return [replace(c, final_rank=i + 1) for i, c in enumerate(values)]


class BGEReranker:
    model_name = "BAAI/bge-reranker-v2-m3"

    def __init__(self, revision: str | None = None, device: str | None = None,
                 batch_size: int = 1, max_length: int = 8192, cache_dir: str | None = None):
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1 or not 1 <= max_length <= 8192:
            raise ValueError("Invalid reranker batch size or token limit")
        self.revision = revision or HfApi().model_info(self.model_name).sha
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, revision=self.revision, cache_dir=cache_dir, torch_dtype=self.dtype,
            attn_implementation="sdpa",
        ).to(self.device).eval()
        self.last_pair_lengths: list[int] = []
        self.last_truncations: list[dict] = []
        self.total_pairs = self.total_truncations = 0

    def score_pairs(self, pairs):
        import torch

        self.last_pair_lengths, self.last_truncations = [], []
        if not pairs:
            return []
        pairs = list(pairs)
        # Measure both texts with the tokenizer's actual pair special tokens.
        lengths = [len(self.tokenizer(a, b, truncation=False, verbose=False)["input_ids"])
                   for a, b in pairs]
        self.last_pair_lengths = lengths
        self.last_truncations = [{"pair_index": i, "input_tokens": n, "limit": self.max_length,
                                  "strategy": "longest_first", "removed_tokens": n - self.max_length}
                                 for i, n in enumerate(lengths) if n > self.max_length]
        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            inputs = self.tokenizer([p[0] for p in batch], [p[1] for p in batch], padding=True,
                                    truncation="longest_first", max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            with torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits
            if logits.shape != (len(batch), 1):
                raise ValueError("Expected one BGE relevance logit per pair")
            scores.extend(logits[:, 0].float().cpu().tolist())
        self.total_pairs += len(pairs)
        self.total_truncations += len(self.last_truncations)
        return scores
