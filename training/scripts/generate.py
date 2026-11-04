"""Greedy-decode test/validation rows with vLLM, for the raw-Qwen baseline or a trained adapter (vLLM venv).

Baseline:   python generate.py --data data --split test --out preds/base_test.jsonl
Fine-tuned: python generate.py --data data --split test --adapter runs/qlora_r16_e1/adapter --out preds/ft_test.jsonl
Retry:      python generate.py ... --ids cv_007,jd_233 --repetition-penalty 1.1 1.2 1.3   (one row per id per penalty)
Prompts:    python generate.py --prompts prompts.jsonl [--ids-file retry_ids.txt] ...   (scripts/match_eval.py --export-prompts)
Score locally afterwards with scripts/qwen_eval.py.
"""

import argparse
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

TASKS = ("cv_extraction", "jd_extraction")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path)
    p.add_argument("--prompts", type=Path, help="jsonl of {id, kind, system, user} instead of --data/--split")
    p.add_argument("--ids-file", type=Path, help="newline-separated ids, like --ids")
    p.add_argument("--split", default="test", choices=("validation", "test"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter", default=None)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--repetition-penalty", type=float, nargs="+", default=[1.0],
                   help=">1 breaks greedy repetition loops (1.05 for Review 2); several values = one output per value (retry)")
    p.add_argument("--ids", default=None, help="comma-separated row ids to generate (retry of failed outputs); default all")
    p.add_argument("--wandb-entity", default="itsbim-fpt-university")
    p.add_argument("--wandb-project", default="capstone_project")
    p.add_argument("--wandb-group", default="review2-qwen-qlora")
    p.add_argument("--adapter-artifact", default=None, help="W&B artifact the adapter came from, e.g. adapter-qlora_r16_e1:best")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    run = None
    if not args.no_wandb:
        assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY (pod: source /workspace/.wandb_env) or pass --no-wandb"
        import subprocess

        import torch
        import vllm
        import wandb

        # nvidia-smi, not torch.cuda: initialising CUDA here before vLLM starts its engine process breaks its startup.
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        # Opened first so W&B records GPU/system metrics, console output and any crash for the whole step.
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
                         job_type="baseline" if not args.adapter else "generate", name=args.out.stem,
                         config={"script_args": {k: str(v) for k, v in vars(args).items()}, "gpu": gpu,
                                 "versions": {"vllm": vllm.__version__, "torch": torch.__version__}, "decoding": f"greedy, temperature 0, repetition_penalty {args.repetition_penalty}"})
        if args.adapter_artifact:
            run.use_artifact(args.adapter_artifact)  # lineage: adapter -> these predictions

    if args.prompts:
        rows = [{"id": r["id"], "task": r["kind"], "messages": [{"role": "system", "content": r["system"]},
                                                              {"role": "user", "content": r["user"]}]}
                for r in map(json.loads, args.prompts.read_text(encoding="utf-8").splitlines())]
    else:
        rows = [dict(json.loads(line), task=task) for task in TASKS
                for line in (args.data / task / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.ids_file:
        args.ids = ",".join(args.ids_file.read_text(encoding="utf-8").split())
        assert args.ids, "empty --ids-file would regenerate every row"
    if args.ids:
        wanted = set(args.ids.split(","))
        rows = [r for r in rows if r["id"] in wanted]
        assert {r["id"] for r in rows} == wanted, f"unknown ids {wanted - {r['id'] for r in rows}}"
    rows = [dict(r, rp=rp) for rp in args.repetition_penalty for r in rows]  # one output per id per penalty
    llm = LLM(model=args.model, max_model_len=32768, enable_lora=bool(args.adapter), max_lora_rank=args.rank,
              gpu_memory_utilization=0.9, seed=0)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template(r["messages"][:2], tokenize=False, add_generation_prompt=True) for r in rows]
    lora = LoRARequest("ft", 1, args.adapter) if args.adapter else None

    start = time.time()
    outs = llm.generate(prompts, [SamplingParams(temperature=0, max_tokens=args.max_tokens, repetition_penalty=r["rp"]) for r in rows],
                        lora_request=lora)
    seconds = time.time() - start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            f.write(json.dumps({"id": r["id"], "task": r["task"], "split": args.split, "prediction": o.outputs[0].text,
                                "finish_reason": o.outputs[0].finish_reason, "repetition_penalty": r["rp"], "prompt_tokens": len(o.prompt_token_ids),
                                "output_tokens": len(o.outputs[0].token_ids)}, ensure_ascii=False) + "\n")
    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    meta = {"model": args.model, "adapter": args.adapter, "split": args.split, "rows": len(rows), "repetition_penalty": args.repetition_penalty,
            "seconds": round(seconds, 1), "output_tokens": out_tokens, "tokens_per_second": round(out_tokens / seconds, 1)}
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)
    if run is None:
        return
    run.summary.update(meta)
    table = wandb.Table(columns=["id", "task", "finish_reason", "prompt_tokens", "output_tokens", "prediction"])
    for r, o in zip(rows, outs):
        table.add_data(r["id"], r["task"], o.outputs[0].finish_reason, len(o.prompt_token_ids), len(o.outputs[0].token_ids), o.outputs[0].text)
    run.log({"predictions": table})
    art = wandb.Artifact(f"predictions-{args.out.stem}", type="predictions", metadata=meta)
    art.add_file(str(args.out))
    art.add_file(str(args.out.with_suffix(".meta.json")))
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
