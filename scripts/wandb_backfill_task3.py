"""Log a finished --no-wandb pod run (output/task3) to W&B after the fact, in the same shape the live scripts use.

Train runs: config + log_history as the step series + the data and adapter artifacts.
Generate runs: meta as summary + predictions table + predictions artifact, using the adapter artifact (lineage).
Every run is tagged "backfilled": system metrics and console logs from the pod are not available.

  .venv/Scripts/python.exe -m scripts.wandb_backfill_task3 [--dir output/task3] [--group review3-task3] [--dry-run]
"""
import argparse
import json
from pathlib import Path

ENTITY, PROJECT = "itsbim-fpt-university", "capstone_project"
# prediction file stem -> (adapter run dir or None for the base model, training data dir on this PC)
PREDS = {"o5_base_test": (None, None), "o5_adapter_test": ("qlora_r16_o5", "data/datasets/o5_explanation"),
         "v4_s1_test": ("qlora_r16_v4gs_s1", "data/datasets/qwen_training_v4"),
         "v4_s2_test": ("qlora_r16_v4gs_s2", "data/datasets/qwen_training_v4")}
DATA = {"qlora_r16_o5": ("o5-explanation-sft-data", "data/datasets/o5_explanation"),
        "qlora_r16_v4gs_s1": ("qwen-sft-data", "data/datasets/qwen_training_v4"),
        "qlora_r16_v4gs_s2": ("qwen-sft-data", "data/datasets/qwen_training_v4")}


def plan(root: Path) -> tuple[list[Path], list[Path]]:
    runs = sorted(p for p in (root / "runs").glob("*") if (p / "log_history.json").exists())
    preds = sorted(p for p in (root / "preds").glob("*.jsonl") if p.stem in PREDS)
    return runs, preds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", type=Path, default=Path("output/task3"))
    p.add_argument("--group", default="review3-task3")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    runs, preds = plan(a.dir)
    print("train runs:", [r.name for r in runs]); print("predictions:", [q.name for q in preds])
    assert runs or preds, f"nothing to log under {a.dir}"
    if a.dry_run:
        return
    import wandb

    adapters = {}
    for r in runs:
        cfg = json.loads((r / "run_config.json").read_text(encoding="utf-8"))
        hist = json.loads((r / "log_history.json").read_text(encoding="utf-8"))
        run = wandb.init(entity=ENTITY, project=PROJECT, group=a.group, job_type="train", name=r.name,
                         tags=["backfilled", "task3"], config={"script_args": cfg, "backfilled": True})
        name, data = DATA.get(r.name, ("sft-data", None))
        if data and Path(data).exists():
            d = wandb.Artifact(name, type="dataset"); d.add_dir(data); run.use_artifact(d)
        for h in hist:  # train rows carry "loss", eval rows "eval_loss"; the trainer's global step is the x-axis
            step = h.get("step")
            row = {("train/" + k if not k.startswith("eval_") else "eval/" + k[5:]): v for k, v in h.items()
                   if isinstance(v, (int, float)) and k != "step"}
            run.log(row, step=step)
        run.summary.update({k: cfg[k] for k in ("peak_mem_gb", "train_seconds", "train_rows", "val_rows", "gpu") if k in cfg})
        art = wandb.Artifact(f"adapter-{r.name}", type="model", metadata=cfg)
        if (r / "adapter").exists():
            art.add_dir(str(r / "adapter"), name="adapter")
        for f in ("run_config.json", "log_history.json"):
            art.add_file(str(r / f))
        run.log_artifact(art, aliases=["final", "best"])
        run.finish()
        adapters[r.name] = f"{ENTITY}/{PROJECT}/adapter-{r.name}:latest"

    for q in preds:
        meta_path = q.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        adapter, _ = PREDS[q.stem]
        run = wandb.init(entity=ENTITY, project=PROJECT, group=a.group, job_type="generate" if adapter else "baseline",
                         name=q.stem, tags=["backfilled", "task3"], config={"meta": meta, "backfilled": True})
        if adapter in adapters:
            run.use_artifact(adapters[adapter])
        run.summary.update({k: v for k, v in meta.items() if isinstance(v, (int, float, str))})
        rows = [json.loads(line) for line in q.read_text(encoding="utf-8").splitlines() if line.strip()]
        cols = ["id", "task", "repetition_penalty", "finish_reason", "prompt_tokens", "output_tokens", "prediction"]
        table = wandb.Table(columns=cols, data=[[x.get(c) for c in cols] for x in rows])
        run.log({"predictions": table})
        art = wandb.Artifact(f"predictions-{q.stem}", type="predictions", metadata=meta)
        art.add_file(str(q))
        if meta_path.exists():
            art.add_file(str(meta_path))
        run.log_artifact(art)
        run.finish()
    print("logged", len(runs), "train runs and", len(preds), "prediction runs to", f"{ENTITY}/{PROJECT}", "group", a.group)


if __name__ == "__main__":
    main()
