"""Ranking diagnostics using only judged relevance."""

import math


def reciprocal_rank(ids: list[str], grades: dict[str, int], k: int = 10) -> float:
    return next((1 / rank for rank, id in enumerate(ids[:k], 1) if grades.get(id, 0) >= 1), 0.0)


def ndcg(ids: list[str], grades: dict[str, int], k: int = 10) -> float:
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ranked ID")
    # This metric is for frozen judged pools; unjudged IDs must never be implicit negatives.
    if any(id not in grades for id in ids):
        raise ValueError("NDCG requires a fully judged candidate pool")
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    dcg = sum((2 ** grades[id] - 1) / math.log2(rank + 1) for rank, id in enumerate(ids[:k], 1))
    return dcg / idcg if idcg else 0.0


def known_positive_recall(ids: list[str], grades: dict[str, int], k: int = 50) -> float | None:
    positives = {id for id, g in grades.items() if g >= 1}
    return len(set(ids[:k]) & positives) / len(positives) if positives else None


def judged_coverage(ids: list[str], grades: dict[str, int], k: int) -> float:
    selected = ids[:k]
    return sum(id in grades for id in selected) / len(selected) if selected else 0.0
