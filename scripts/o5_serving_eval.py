"""O5 as served: the 244 test prompts (122 EN + 122 VI) through the service's own call (scripts.ats_service._o5_generate,
Ollama model qwen-o5 = O5 v3 seed 3407 merged into Qwen2.5-7B-Instruct, Q4_K_M). The adapter's reported numbers
(output/o5v3/o5v3_seeds.md) come from a 4-bit bitsandbytes base; this checks the quantised model the service runs.

  OLLAMA_URL=http://127.0.0.1:11434 python -m scripts.o5_serving_eval [--limit 10] [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.ats_service import O5_MODEL, _o5_generate
from scripts.o5_export_sft import facts_of
from scripts.o5_faithfulness import check, is_vietnamese, violations

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/datasets/o5_explanation/test_prompts.jsonl"
OUT = ROOT / "output/o5v3/preds/o5v3_ollama_q4_test.jsonl"
REFERENCE = ROOT / "output/o5v3/preds/o5v3_adapter_test.jsonl"  # seed 3407 on the 4-bit base, as evaluated
SMALL_WORDS = {"the", "to", "of", "a", "an", "your", "is", "are", "in", "for", "with"}


def small_words(text: str) -> float:
    """Function words per 100 words (English): a quantised model that drops them ("add examples CV") scores low."""
    words = text.lower().replace(",", " ").replace(".", " ").split()
    return 100 * sum(w in SMALL_WORDS for w in words) / max(len(words), 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    from src.scoring.v4_client import QwenV4

    llm, rows = QwenV4(), [json.loads(x) for x in PROMPTS.read_text(encoding="utf-8").splitlines()][:args.limit]
    run = None if args.no_wandb else __import__("wandb").init(
        entity="itsbim-fpt-university", project="capstone_project", group="review3-o5-serving", job_type="eval",
        name=f"o5v3_s3407_{O5_MODEL}_q4km_test", config={"model": O5_MODEL, "quant": "Q4_K_M", "prompts": len(rows),
                                                          "decoding": "greedy; retry repeat_penalty 1.1; then template"})
    out = []
    for i, r in enumerate(rows, 1):
        f, vi = facts_of(r["id"].removesuffix("_vi")), r["id"].endswith("_vi")
        t = time.time()
        first = _o5_generate(f, vi, llm)
        row = {"id": r["id"], "lang": "vi" if vi else "en", "first": first, "faithful": not check(first, f),
               "guard_first": not violations(first, f), "right_language": is_vietnamese(first) == vi,
               "words": len(first.split()), "small_words": small_words(first), "seconds": round(time.time() - t, 2)}
        if not row["guard_first"]:
            retry = _o5_generate(f, vi, llm, 1.1)
            row |= {"retry": retry, "guard_retry": not violations(retry, f)}
        row["source"] = "o5" if row["guard_first"] or row.get("guard_retry") else "template"
        out.append(row)
        print(f"{i}/{len(rows)} {r['id']} faithful={row['faithful']} source={row['source']} {row['seconds']}s", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    summary = {}
    for lang in ("en", "vi"):
        g = [x for x in out if x["lang"] == lang]
        if g:
            n = len(g)
            summary |= {f"{lang}/n": n, f"{lang}/faithful": sum(x["faithful"] for x in g) / n,
                        f"{lang}/guard_first": sum(x["guard_first"] for x in g) / n,
                        f"{lang}/template": sum(x["source"] == "template" for x in g) / n,
                        f"{lang}/right_language": sum(x["right_language"] for x in g) / n,
                        f"{lang}/median_words": sorted(x["words"] for x in g)[n // 2],
                        f"{lang}/median_seconds": sorted(x["seconds"] for x in g)[n // 2]}
    ref = {(r := json.loads(x))["id"]: r for x in REFERENCE.read_text(encoding="utf-8").splitlines()}
    en = [x for x in out if x["lang"] == "en" and x["id"] in ref]
    if en:  # same prompts, the evaluated 4-bit outputs
        key = next(k for k in ("output", "text", "prediction") if k in ref[en[0]["id"]])
        summary["en/small_words_per_100"] = sum(x["small_words"] for x in en) / len(en)
        summary["en/small_words_per_100_4bit_reference"] = sum(small_words(ref[x["id"]][key]) for x in en) / len(en)
    print(json.dumps(summary, indent=1))
    if run:
        run.summary.update(summary)
        run.finish()


if __name__ == "__main__":
    main()
