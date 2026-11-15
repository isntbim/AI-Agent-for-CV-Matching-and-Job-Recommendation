"""Greedy-decode a prompts file on a local Windows GPU (no vLLM there): Unsloth 4-bit, batched. Same output rows,
meta and W&B logging as generate.py --prompts, so scripts/o5_faithfulness.py and the reviews work unchanged.
Note: the base is 4-bit here; generate.py on the pod runs it in 16-bit.

  .venv-train/Scripts/python.exe training/scripts/generate_local.py --prompts data/datasets/o5_explanation/test_prompts.jsonl \
      --out output/o5v2/preds/o5v2_base_test.jsonl [--adapter output/o5v2/runs/qlora_r16_o5v2/adapter] --wandb-group review3-o5v2

Guard (the service's validate step for explanations): --guard-from <greedy preds> keeps every row that passes
scripts.o5_faithfulness.violations and replaces the others with the fallback template. --guard-retries 2 first
resamples them twice; measured 2026-11-12, the resampled English texts were wrong 7 times in 18 (greedy: 13%), so 0.
"""
from unsloth import FastLanguageModel  # must import before transformers

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, required=True, help="jsonl of {id, kind, system, user}")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--max-tokens", type=int, default=700)
    # 16 overflowed 12 GB on the longest prompts; Windows then spills to system RAM and a batch took 29 min (2026-11-11)
    p.add_argument("--batch", type=int, default=8)
    # a full batch of the longest prompts still overflowed at --batch 4 (2026-11-12): long prompts go in smaller batches
    p.add_argument("--batch-tokens", type=int, default=2800, help="max rows x longest prompt (tokens) in one batch")
    p.add_argument("--split", default="test", help="label written to the rows and meta (dev = validation pairs)")
    p.add_argument("--wandb-entity", default="itsbim-fpt-university")
    p.add_argument("--wandb-project", default="capstone_project")
    p.add_argument("--wandb-group", default="review3-o5v2")
    p.add_argument("--adapter-artifact", default=None)
    p.add_argument("--guard-from", type=Path, default=None, help="greedy preds of the same prompts; only failing rows are redone")
    p.add_argument("--guard-retries", type=int, default=0, help="sampled retries before the template")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    run = None
    if not args.no_wandb:
        assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY or pass --no-wandb"
        import unsloth
        import wandb

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
                         job_type="baseline" if not args.adapter else "generate", name=args.out.stem,
                         config={"script_args": {k: str(v) for k, v in vars(args).items()}, "gpu": torch.cuda.get_device_name(0),
                                 "versions": {"unsloth": unsloth.__version__, "torch": torch.__version__},
                                 "decoding": (f"guard: greedy rows kept if they pass the checker, else sampled T=0.7 top_p=0.9 x{args.guard_retries}, else template"
                                              if args.guard_from else "greedy, temperature 0, repetition_penalty 1.0") + ", 4-bit base (local)"})
        if args.adapter_artifact:
            run.use_artifact(args.adapter_artifact)

    rows = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.guard_from and not args.guard_retries:
        model = tok = prompts = lengths = None  # template-only guard: nothing is generated, so no model on the GPU
    else:
        model, tok = FastLanguageModel.from_pretrained(args.adapter or args.model, max_seq_length=4096, load_in_4bit=True, dtype=None)
        FastLanguageModel.for_inference(model)
        tok.padding_side = "left"
        prompts = [tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                           tokenize=False, add_generation_prompt=True) for r in rows]
        lengths = [len(tok(q, add_special_tokens=False).input_ids) for q in prompts]
    res = {}
    start = time.time()

    def decode(todo: list[int], sample: bool) -> None:
        order = sorted(todo, key=lambda i: lengths[i])  # similar lengths per batch = less padding
        batches = [[]]
        for i in order:  # ascending, so the new row is the longest: it sets the padded width
            if batches[-1] and (len(batches[-1]) == args.batch or (len(batches[-1]) + 1) * lengths[i] > args.batch_tokens):
                batches.append([])
            batches[-1].append(i)
        done = 0
        for idx in batches:
            # Unsloth's fast generate drops tokens when the batch holds ONE sequence (2026-11-13: "fit 21/100" came out
            # as "fit 2/100", dates lost digits). A single prompt is therefore sent twice and the first copy is kept.
            texts = [prompts[i] for i in idx] * (2 if len(idx) == 1 else 1)
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            how = {"do_sample": True, "temperature": 0.7, "top_p": 0.9} if sample else {"do_sample": False}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_tokens, repetition_penalty=1.0, pad_token_id=tok.pad_token_id, **how)
            for i, seq, mask in zip(idx, out[:, enc.input_ids.shape[1]:], enc.attention_mask):
                ids = seq.tolist()
                stop = next((k for k, t in enumerate(ids) if t in (tok.eos_token_id, tok.pad_token_id)), None)
                ids = ids[:stop] if stop is not None else ids
                res[i] = {"text": tok.decode(ids, skip_special_tokens=True), "finish_reason": "stop" if stop is not None else "length",
                          "prompt_tokens": int(mask.sum()), "output_tokens": len(ids), "attempts": res.get(i, {}).get("attempts", 0) + 1}
            done += len(idx)
            print(f"{done}/{len(order)} rows, {time.time() - start:.0f}s", flush=True)

    guard = {}
    if args.guard_from:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.o5_export_sft import facts_of, fallback
        from scripts.o5_faithfulness import violations

        first = {r["id"]: r for r in map(json.loads, args.guard_from.read_text(encoding="utf-8").splitlines())}
        facts = [facts_of(r["id"].removesuffix("_vi")) for r in rows]
        for i, r in enumerate(rows):  # KeyError = the greedy file is not for these prompts
            f = first[r["id"]]
            res[i] = {"text": f["prediction"], "finish_reason": f["finish_reason"], "prompt_tokens": f["prompt_tokens"],
                      "output_tokens": f["output_tokens"], "attempts": 1}
        failing = lambda: [i for i in range(len(rows)) if violations(res[i]["text"], facts[i])]
        todo = failing()
        guard["rejected_greedy"] = len(todo)
        for attempt in range(2, 2 + args.guard_retries):
            if todo:
                torch.manual_seed(attempt)
                decode(todo, sample=True)
                todo = failing()
            guard[f"still_failing_after_{attempt}"] = len(todo)
        for i in todo:
            res[i].update(text=fallback(facts[i], vi=rows[i]["id"].endswith("_vi")), finish_reason="stop", fallback=True)
        guard["fallback"] = len(todo)
        print("guard", guard, flush=True)
    else:
        decode(list(range(len(rows))), sample=False)
    seconds = time.time() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": r["id"], "task": r["kind"], "split": args.split, "prediction": res[i]["text"],
                                "finish_reason": res[i]["finish_reason"], "repetition_penalty": 1.0,
                                "attempts": res[i]["attempts"], "fallback": res[i].get("fallback", False),
                                "prompt_tokens": res[i]["prompt_tokens"], "output_tokens": res[i]["output_tokens"]},
                               ensure_ascii=False) + "\n")
    out_tokens = sum(x["output_tokens"] for x in res.values())
    meta = {"model": args.model, "adapter": args.adapter, "split": args.split, "rows": len(rows), "repetition_penalty": [1.0],
            "seconds": round(seconds, 1), "output_tokens": out_tokens, "tokens_per_second": round(out_tokens / max(seconds, 1e-9), 1),
            "backend": "unsloth 4-bit, local", **guard}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)
    if run is None:
        return
    import wandb

    run.summary.update({k: v for k, v in meta.items() if not isinstance(v, list)})
    table = wandb.Table(columns=["id", "task", "finish_reason", "prompt_tokens", "output_tokens", "prediction"])
    for i, r in enumerate(rows):
        table.add_data(r["id"], r["kind"], res[i]["finish_reason"], res[i]["prompt_tokens"], res[i]["output_tokens"], res[i]["text"])
    run.log({"predictions": table})
    art = wandb.Artifact(f"predictions-{args.out.stem}", type="predictions", metadata=meta)
    art.add_file(str(args.out))
    art.add_file(str(args.out.with_suffix(".meta.json")))
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
