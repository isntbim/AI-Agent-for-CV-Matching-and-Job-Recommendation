"""Paper-style baseline for O4: schema-constrained decoding instead of validate-repair-retry.

Ollama's structured outputs (`format` = a JSON schema, compiled to a grammar by llama.cpp) is the constrained-decoding
approach that JSONSchemaBench (Geng et al., 2025) benchmarks. Same test prompts, same greedy settings and the same
token cap as scripts.qwen_local_generate, so scripts.qwen_eval scores the rows like every other run.

  python -m scripts.constrained_baseline --model qwen2.5:7b-instruct --out output/review3/baselines/constrained_base_test.jsonl
  python -m scripts.constrained_baseline --model qwen-v4 --out output/review3/baselines/constrained_v4_test.jsonl
  (rows already in --out are skipped, so a stopped run resumes)
"""
import argparse
import json
import os
import time
from pathlib import Path

import httpx

from scripts.cv_teacher_validate import LABEL_SCHEMA
from scripts.jd_teacher_check import JD_SCHEMA
from scripts.qwen_local_generate import DATA
from src.scoring.v4_client import token_cap


def schema(parts: dict) -> dict:
    """{"document": ..., "facts": ...} as one schema; pydantic's $defs move to the root so "#/$defs/X" refs resolve."""
    defs, props = {}, {}
    for key, sub in parts.items():
        sub = dict(sub)
        defs.update(sub.pop("$defs", {}))
        props[key] = sub
    return {"type": "object", "properties": props, "required": list(parts), "$defs": defs}


SCHEMAS = {"cv_extraction": schema(LABEL_SCHEMA), "jd_extraction": schema(JD_SCHEMA)}
# The grammar fixes object keys to the schema's order. JD training targets write evidence as title, industry, location,
# experience, education, salary, skills (321 of 694 rows; skills always last), so the schema order (skills first) makes
# V4 skip the evidence it writes later. --training-order gives the JD evidence the training order (CVs already match).
JD_TRAINING_ORDER = ("title", "industry", "location", "experience", "education", "salary", "skills")


def training_order(s: dict) -> dict:
    s = json.loads(json.dumps(s))
    # schema_version is optional in the pydantic schema (it has a default), so the grammar lets the model skip it, and
    # V4 then omits it on every JD; the validator needs job-v3. A fair constrained setup pins it.
    doc = s["properties"]["document"]
    doc["properties"]["schema_version"] = {"const": "job-v3", "type": "string"}
    doc["required"] = ["schema_version"] + [k for k in doc.get("required", []) if k != "schema_version"]
    ev = s["properties"]["facts"]["properties"]["evidence"]
    ev["properties"] = {k: ev["properties"][k] for k in JD_TRAINING_ORDER}
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--ids", default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--training-order", action="store_true", help="JD evidence keys in training order (see above)")
    args = p.parse_args()
    if args.training_order:
        SCHEMAS["jd_extraction"] = training_order(SCHEMAS["jd_extraction"])
        assert set(SCHEMAS["jd_extraction"]["properties"]["facts"]["properties"]["evidence"]["properties"]) == set(JD_TRAINING_ORDER)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    rows = [dict(json.loads(line), task=task) for task in ("cv_extraction", "jd_extraction")
            for line in (DATA / task / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids.split(","))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in args.out.read_text(encoding="utf-8").splitlines()} if args.out.exists() else set()
    with args.out.open("a", encoding="utf-8") as f:
        for r in rows:
            if r["id"] in done:
                continue
            system, user = r["messages"][0]["content"], r["messages"][1]["content"]
            prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True)
            cap = token_cap(len(tok(json.loads(user)["source_text"])["input_ids"]))
            start = time.time()
            res = httpx.post(f"{url}/api/generate", timeout=3600, json={
                "model": args.model, "prompt": prompt, "raw": True, "stream": False, "format": SCHEMAS[r["task"]],
                "options": {"temperature": 0, "seed": 0, "num_ctx": 32768, "num_predict": cap,
                            "repeat_penalty": args.repetition_penalty, "repeat_last_n": 32768}}).json()
            if "error" in res and "repeat" not in res["error"]:
                raise RuntimeError(f"{r['id']}: {res['error']}")
            if "error" in res:  # Ollama aborts a looping generation: no output, scored as a failed (invalid) document
                res = {"response": "", "done_reason": "length", "eval_count": None, "error": res["error"]}
            finish = "length" if res.get("done_reason") == "length" else "stop"
            f.write(json.dumps({"id": r["id"], "task": r["task"], "split": args.split, "prediction": res["response"],
                                "finish_reason": finish, "repetition_penalty": args.repetition_penalty,
                                "output_tokens": res.get("eval_count"), "seconds": round(time.time() - start, 1),
                                "model": args.model, "constrained": True}
                               | ({"error": res["error"]} if "error" in res else {}), ensure_ascii=False) + "\n")
            f.flush()
            print(f"{r['id']} {finish} {res.get('eval_count')} tok {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
