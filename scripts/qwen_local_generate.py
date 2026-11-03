"""Local twin of training/scripts/generate.py: greedy V4 outputs from Ollama (src.scoring.v4_client.QwenV4), same
row format, so qwen_repair / qwen_eval score them exactly like the A100 run.

  python -m scripts.qwen_local_generate --out output/review2/qwen/v4_local_test.jsonl [--ids cv_007,jd_233]
         [--repetition-penalty 1.05 [1.1 ...]]   (rows already in --out are skipped, so a stopped run resumes)
"""

import argparse
import json
import time
from pathlib import Path

from src.scoring.v4_client import QwenV4

DATA = Path(__file__).resolve().parent.parent / "data/datasets/qwen_training_v4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--ids", default=None)
    p.add_argument("--repetition-penalty", type=float, nargs="+", default=[1.05])
    args = p.parse_args()
    rows = [dict(json.loads(line), task=task) for task in ("cv_extraction", "jd_extraction")
            for line in (DATA / task / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids.split(","))]
    done = set()
    if args.out.exists():
        done = {(r["id"], r["repetition_penalty"]) for r in map(json.loads, args.out.read_text(encoding="utf-8").splitlines())}
    llm = QwenV4()
    with args.out.open("a", encoding="utf-8") as f:
        for rp in args.repetition_penalty:
            for r in rows:
                if (r["id"], rp) in done:
                    continue
                start = time.time()
                text, finish = llm.complete(r["messages"][0]["content"], r["messages"][1]["content"], rp)
                f.write(json.dumps({"id": r["id"], "task": r["task"], "split": args.split, "prediction": text,
                                    "finish_reason": finish, "repetition_penalty": rp,
                                    "seconds": round(time.time() - start, 1)}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{r['id']} rp {rp} {finish} {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
