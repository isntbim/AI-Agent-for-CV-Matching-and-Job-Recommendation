"""QLoRA SFT of Qwen2.5-7B-Instruct on the CV + JD extraction sets (runs on the GPU pod, Unsloth venv).

Smoke test:  python train_qlora.py --data data --out runs/smoke --max-steps 20
Full run:    python train_qlora.py --data data --out runs/qlora_r16_e1
Resume:      add --resume runs/qlora_r16_e1/checkpoint-50
"""

from unsloth import FastLanguageModel  # must import before transformers/trl
from unsloth.chat_templates import train_on_responses_only

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

TASKS = ("cv_extraction", "jd_extraction")


def rows(data: Path, split: str, exclude: tuple[str, ...] = (), upsample: dict[str, int] | None = None) -> list[dict]:
    """exclude/upsample match the row's `source` by prefix (e.g. 'external', 'gold'); applied to train only by the caller."""
    out = []
    for task in TASKS:
        for line in (data / task / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("source", "").startswith(exclude) if exclude else False:
                continue
            copies = next((n for p, n in (upsample or {}).items() if r.get("source", "").startswith(p)), 1)
            out += [{"id": r["id"], "task": task, "messages": r["messages"]}] * copies
    return out


def to_dataset(items: list[dict], tok, max_len: int) -> tuple[Dataset, list[str]]:
    kept, dropped = [], []
    for r in items:
        text = tok.apply_chat_template(r["messages"], tokenize=False)
        # Drop, never truncate: a cut-off target would teach the model to emit broken JSON.
        if len(tok(text, add_special_tokens=False).input_ids) > max_len:
            dropped.append(r["id"])
        else:
            kept.append({"text": text})
    return Dataset.from_list(kept), dropped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="folder holding cv_extraction/ and jd_extraction/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-len", type=int, default=16384)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--eval-steps", type=int, default=25)
    p.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke test)")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--resume", default=None)
    p.add_argument("--exclude-source", nargs="*", default=[], help="drop train rows whose source starts with these, e.g. external")
    p.add_argument("--upsample-source", nargs="*", default=[], help="repeat train rows, e.g. gold:4")
    p.add_argument("--wandb-entity", default="itsbim-fpt-university")
    p.add_argument("--wandb-project", default="capstone_project")
    p.add_argument("--wandb-group", default="review2-qwen-qlora")
    p.add_argument("--no-wandb", action="store_true", help="local-only run (no W&B logging or uploads)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_wandb:
        # Every checkpoint (adapter + optimizer state) is uploaded as a W&B artifact as it is saved.
        os.environ.update(WANDB_ENTITY=args.wandb_entity, WANDB_PROJECT=args.wandb_project, WANDB_LOG_MODEL="checkpoint")
        assert os.environ.get("WANDB_API_KEY"), "set WANDB_API_KEY (pod: source /workspace/.wandb_env) or pass --no-wandb"
        import peft, transformers, trl, unsloth, wandb

        # Opened before the Trainer so it reuses this run: config, versions, input dataset and code all land on it.
        wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group, job_type="train",
                   name=args.out.name, config={"script_args": {k: str(v) for k, v in vars(args).items()},
                   "gpu": torch.cuda.get_device_name(0), "versions": {m.__name__: m.__version__ for m in
                   (torch, transformers, trl, peft, unsloth, wandb)}})
        data_art = wandb.Artifact("qwen-sft-data", type="dataset", description="CV + JD extraction SFT JSONL + manifest")
        data_art.add_dir(str(args.data))
        wandb.run.use_artifact(data_art)
        wandb.run.log_code(str(Path(__file__).parent))

    model, tok = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_len, load_in_4bit=True, dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed,
    )
    upsample = {s.split(":")[0]: int(s.split(":")[1]) for s in args.upsample_source}
    train, train_dropped = to_dataset(rows(args.data, "train", tuple(args.exclude_source), upsample), tok, args.max_len)
    val, val_dropped = to_dataset(rows(args.data, "validation"), tok, args.max_len)
    print(f"train {len(train)} (dropped {len(train_dropped)} over {args.max_len} tokens), val {len(val)} (dropped {len(val_dropped)})")
    if not args.no_wandb:
        wandb.run.summary.update({"train_rows": len(train), "val_rows": len(val), "train_dropped": train_dropped, "val_dropped": val_dropped,
                                  "trainable_params": sum(x.numel() for x in model.parameters() if x.requires_grad)})

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=train, eval_dataset=val,
        args=SFTConfig(
            output_dir=str(args.out), dataset_text_field="text", max_length=args.max_len, packing=False,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
            lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01, optim="adamw_8bit",
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=1, eval_strategy="steps", eval_steps=args.eval_steps,
            save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            seed=args.seed, report_to="none" if args.no_wandb else "wandb", run_name=args.out.name,
        ),
    )
    # Loss only on the assistant JSON, not the system prompt or the CV/JD text.
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")

    start = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    model.save_pretrained(args.out / "adapter")
    tok.save_pretrained(args.out / "adapter")
    (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    (args.out / "run_config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_rows": len(train), "val_rows": len(val), "train_dropped": train_dropped, "val_dropped": val_dropped,
        "gpu": torch.cuda.get_device_name(0), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "train_seconds": round(time.time() - start), "best_checkpoint": trainer.state.best_model_checkpoint,
    }, indent=2))
    if not args.no_wandb:
        wandb.run.summary.update({"peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "train_seconds": round(time.time() - start)})
        # The final (best, since load_best_model_at_end) adapter plus the run's logs and config.
        art = wandb.Artifact(f"adapter-{args.out.name}", type="model", metadata=json.loads((args.out / "run_config.json").read_text()))
        art.add_dir(str(args.out / "adapter"), name="adapter")
        for name in ("run_config.json", "log_history.json"):
            art.add_file(str(args.out / name))
        wandb.log_artifact(art, aliases=["final", "best"])
        wandb.finish()


if __name__ == "__main__":
    main()
