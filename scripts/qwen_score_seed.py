"""Score a V4-recipe seed run the way the headline V4 result was scored (validate -> repair -> retry, then qwen_eval).

The pod generates every test document at all four retry penalties in one file (run_task3.sh). This splits it into the
first answer (1.05) and the retry rows, runs scripts.qwen_repair with --retry (lowest penalty that repairs to a valid
label wins, gold never consulted), then scripts.qwen_eval with bootstrap CIs.

    python -m scripts.qwen_score_seed output/task3/preds/v4_s1_test.jsonl
"""
import json
import subprocess
import sys
from pathlib import Path

FIRST = 1.05


def main(path: Path) -> Path:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    first = [r for r in rows if abs(r["repetition_penalty"] - FIRST) < 1e-9]
    retry = [r for r in rows if abs(r["repetition_penalty"] - FIRST) >= 1e-9]
    assert first and len({r["id"] for r in first}) == len(first), "need exactly one 1.05 answer per document"
    a, b = path.with_suffix(".first.jsonl"), path.with_suffix(".retry.jsonl")
    for p, part in ((a, first), (b, retry)):
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
    final = path.with_suffix(".final.jsonl")
    py = sys.executable
    subprocess.run([py, "-m", "scripts.qwen_repair", str(a), str(final)] + (["--retry", str(b)] if retry else []), check=True)
    subprocess.run([py, "-m", "scripts.qwen_eval", str(final)], check=True)
    return final.with_suffix(".scores.json")


if __name__ == "__main__":
    print(main(Path(sys.argv[1])))
