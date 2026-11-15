"""Retry the select/reject sample documents whose V4 extraction failed (invalid after the standard 4 attempts).

Most failures are short marketing-style JDs (~370 characters) where the model keeps inventing skills until the length
cap. The retry runs the same validate-repair-retry graph with stronger repetition penalties. Cut-off answers are
still never accepted. Documents rescued this way are listed in output/select_reject/retry_rescued.json so the report
can say which extractions used the stronger setting.

    python -m scripts.select_reject_retry --workers 3
"""
import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import src.scoring.v4_client as v4
from scripts.select_reject_eval import OUT, _cache, load, sample

STRONGER = (1.25, 1.3, 1.35, 1.4)


def failed_docs(rows: list[dict]) -> list[tuple[str, str]]:
    saved = {json.loads(f.read_text(encoding="utf-8"))["identity"]["text_hash"]
             for f in (OUT / "extractions").glob("*.json") if not f.name.endswith(".error.json")}
    docs = {}
    for r in rows:
        for text, kind in ((r["Resume"], "resume"), (r["Job_Description"], "job")):
            h = hashlib.sha256(text.encode()).hexdigest()
            if h not in saved:
                docs[h] = (text, kind)
    return list(docs.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    todo = failed_docs(sample(load(), 1000))
    print(len(todo), "documents to retry at repetition penalties", STRONGER, flush=True)
    v4.PENALTIES = STRONGER  # read by build_graph when each client builds its graph
    llm, local, rescued, still = v4.QwenV4(), threading.local(), [], []

    def one(job):
        if not hasattr(local, "cache"):
            local.cache = _cache(v4.V4Client(llm))
        local.cache.get(*job)
        return hashlib.sha256(job[0].encode()).hexdigest(), job[1]

    with ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(one, j): j for j in todo}
        for i, f in enumerate(as_completed(futures), 1):
            text, kind = futures[f]
            try:
                rescued.append(dict(zip(("text_sha256", "kind"), f.result())))
            except Exception as exc:
                still.append({"text_sha256": hashlib.sha256(text.encode()).hexdigest(), "kind": kind,
                              "error": f"{type(exc).__name__}: {str(exc)[:150]}"})
            print(f"{i}/{len(todo)} rescued {len(rescued)}, still failing {len(still)}", flush=True)
    (OUT / "retry_rescued.json").write_text(json.dumps({"penalties": STRONGER, "rescued": rescued, "still_failing": still},
                                                       indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
