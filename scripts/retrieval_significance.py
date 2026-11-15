"""Paired significance for the retrieval benchmark (O2): reranked vs each other stage, same 88 queries.

  python -m scripts.retrieval_significance   # writes output/retrieval/benchmark_v2/significance.md

Same protocol as the matching benchmark: mean per-query difference (A minus B), 95% bootstrap CI over queries
(resampled within CV domain, then domain-macro averaged like report.md), and a sign-flip permutation p-value.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parent.parent / "output/retrieval/benchmark_v2"
METRICS = ("mrr10", "ndcg5", "ndcg10")
N = 10000


def macro(values: np.ndarray, domains: np.ndarray) -> float:
    return float(np.mean([values[domains == d].mean() for d in np.unique(domains)]))


def compare(a: np.ndarray, b: np.ndarray, domains: np.ndarray, rng) -> tuple[float, float, float, float]:
    diff = a - b
    groups = [np.flatnonzero(domains == d) for d in np.unique(domains)]
    boot = [np.mean([diff[rng.choice(g, len(g))].mean() for g in groups]) for _ in range(N)]
    observed = macro(diff, domains)
    flips = rng.choice([-1.0, 1.0], size=(N, len(diff)))
    null = np.array([macro(diff * f, domains) for f in flips])
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (N + 1)
    return observed, *np.percentile(boot, [2.5, 97.5]), p


def main() -> None:
    rows = json.loads((DIR / "rankings.json").read_text(encoding="utf-8"))
    out = ["Paired on the same 88 queries, primary pool (A minus B): domain-macro mean difference "
           "[95% bootstrap CI over queries within domain], sign-flip permutation p.", ""]
    rng = np.random.default_rng(0)
    for pool in ("primary", "expanded"):
        by = defaultdict(dict)
        for r in rows:
            if r["pool"] == pool:
                by[r["method"]][r["cv_id"]] = r
        ids = sorted(by["reranked"])
        domains = np.array([by["reranked"][i]["domain"] for i in ids])
        vals = {m: {k: np.array([by[m][i][k] for i in ids]) for k in METRICS} for m in by}
        out += [f"## {pool} pool", "", "| Stage | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        out += [f"| {m} | " + " | ".join(f"{macro(vals[m][k], domains):.4f}" for k in METRICS) + " |"
                for m in ("bm25", "dense", "hybrid", "reranked")]
        out += ["", "| A vs B | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for b in ("bm25", "dense", "hybrid"):
            cells = []
            for k in METRICS:
                d, lo, hi, p = compare(vals["reranked"][k], vals[b][k], domains, rng)
                cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.4f}")
            out.append(f"| reranked vs {b} | " + " | ".join(cells) + " |")
        out.append("")
    (DIR / "significance.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
