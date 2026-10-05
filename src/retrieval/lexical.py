"""Versioned BM25 and equal-weight reciprocal rank fusion."""

import math
import re
from collections import Counter

from .reranker import Candidate

TOKENIZATION_VERSION = "technical-tokens-v1"
_TOKEN = re.compile(r"(?i)(?<!\w)(?:c\+\+|c#|\.net)(?!\w)|[\w]+(?:[.+#-][\w]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [m.group().casefold() for m in _TOKEN.finditer(text)]


class BM25:
    def __init__(self, documents: list[Candidate], k1: float = 1.5, b: float = 0.75):
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        if len({c.id for c in documents}) != len(documents):
            raise ValueError("Duplicate document IDs")
        self.documents, self.k1, self.b = list(documents), k1, b
        self.counts = [Counter(tokenize(c.text)) for c in documents]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / len(documents) if documents else 0
        df = Counter(t for c in self.counts for t in c)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> dict[str, float]:
        terms = set(tokenize(query))
        result = {}
        for document, counts, length in zip(self.documents, self.counts, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.average) if self.average else self.k1
            result[document.id] = sum(self.idf.get(t, 0) * counts[t] * (self.k1 + 1) /
                                      (counts[t] + norm) for t in terms if counts[t])
        return result

    def search(self, query: str, top_k: int = 50) -> list[Candidate]:
        from dataclasses import replace
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        scores = self.scores(query)
        ordered = sorted(enumerate(self.documents), key=lambda ic: (-scores[ic[1].id], ic[0], ic[1].id))
        return [replace(c, retrieval_score=scores[c.id], original_rank=i + 1)
                for i, (_, c) in enumerate(ordered[:top_k])]


def rrf(rankings: list[list[Candidate]], constant: int = 60) -> list[Candidate]:
    from dataclasses import replace
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores, first, candidates, ranks = {}, {}, {}, {}
    for stage, ranking in enumerate(rankings):
        if len({c.id for c in ranking}) != len(ranking):
            raise ValueError("Duplicate ID in RRF input ranking")
        for rank, c in enumerate(ranking, 1):
            scores[c.id] = scores.get(c.id, 0) + 1 / (constant + rank)
            first.setdefault(c.id, len(first))
            candidates.setdefault(c.id, c)
            ranks.setdefault(c.id, {})[str(stage)] = {"rank": rank, "score": c.retrieval_score}
    ids = sorted(scores, key=lambda i: (-scores[i], first[i], i))
    return [replace(candidates[id], retrieval_score=scores[id], original_rank=rank,
                    metadata={**candidates[id].metadata, "retrieval_stages": ranks[id]})
            for rank, id in enumerate(ids, 1)]
