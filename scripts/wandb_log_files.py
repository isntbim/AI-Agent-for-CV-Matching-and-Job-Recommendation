"""Log result files of an evaluation that has no W&B logging of its own (one run, one artifact).

  python -m scripts.wandb_log_files --name injection_test --group review3-baselines output/review3/injection/*
"""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    import wandb

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--group", default="review3-baselines")
    args = p.parse_args()
    key_file = ROOT / "knowledge-base/wandb_key.txt"  # local-only credential, never printed
    if not os.environ.get("WANDB_API_KEY") and key_file.exists():
        os.environ["WANDB_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
    run = wandb.init(entity="itsbim-fpt-university", project="capstone_project", group=args.group, job_type="eval",
                     name=args.name, config={"files": [f.name for f in args.files]})
    art = wandb.Artifact(f"results-{args.name}", type="evaluation")
    for f in args.files:
        art.add_file(str(f))
        if f.suffix == ".md":
            run.summary[f.stem] = f.read_text(encoding="utf-8")[:4000]
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
