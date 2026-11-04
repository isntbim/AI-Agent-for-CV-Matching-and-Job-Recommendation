"""Cross-encoder baseline: BAAI/bge-reranker-v2-m3 reads each (CV text, JD text) pair together and scores relevance.
Scores every pair the scorer scores (lab_docs.pkl), resumably, into lab/reranker_scores.jsonl. GPU strongly advised:
free the extractor's VRAM first (`ollama stop qwen-v4`).

  python -m scripts.reranker_baseline [--max-length 2048] [--batch 8]
"""
import argparse
import json
import pickle

from scripts import match_eval as m
from scripts.scorer_lab import PKL
from src.retrieval.reranker import BGEReranker

OUT = m.OUT / "lab" / "reranker_scores.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    docs = pickle.loads(PKL.read_bytes())["docs"]
    texts, todo = {}, []
    done = {(r["cv"], r["jd"]) for r in map(json.loads, OUT.open(encoding="utf-8"))} if OUT.exists() else set()
    for pair in m.load_pairs():
        cv, jd = m.doc_id("cv", pair["cv"]), m.doc_id("jd", pair["jd"])
        texts[cv], texts[jd] = pair["cv"], pair["jd"]
        if docs[cv] is not None and docs[jd] is not None and (cv, jd) not in done:
            todo.append((cv, jd))
    todo.sort(key=lambda k: len(texts[k[0]]) + len(texts[k[1]]))  # similar lengths per batch: less padding
    reranker = BGEReranker(device="cuda", batch_size=args.batch, max_length=args.max_length)
    with OUT.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), 512):
            chunk = todo[start:start + 512]
            for (cv, jd), score in zip(chunk, reranker.score_pairs([(texts[cv], texts[jd]) for cv, jd in chunk])):
                f.write(json.dumps({"cv": cv, "jd": jd, "score": score}) + "\n")
            f.flush()
            print(f"{start + len(chunk)}/{len(todo)} truncated so far: {reranker.total_truncations}", flush=True)


if __name__ == "__main__":
    main()
