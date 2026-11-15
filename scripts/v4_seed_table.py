"""3-seed table for the V4 recipe (seed 3407 from Review 2, seeds 1 and 2 from the task3 pod): raw and after repair/retry.

Writes output/task3/v4_seeds.md and logs the table to W&B (group review3-task3).
  .venv/Scripts/python.exe -m scripts.v4_seed_table [--no-wandb]
"""
import json
import statistics as st
import sys
from pathlib import Path

Q, P = Path("output/review2/qwen"), Path("output/task3/preds")
SETS = {"V4 raw (penalty 1.05)": [Q / "v4_rp_test", P / "v4_s1_rp_test", P / "v4_s2_rp_test"],
        "V4 + repair/retry": [Q / "v4_final_test", P / "v4_s1_final_test", P / "v4_s2_final_test"]}
METRICS = {"cv_extraction": ["truncated", "schema_valid_rate", "skills_precision", "skills_recall", "skills_f1", "skills_relaxed_f1",
                             "domains_f1", "completed_degrees_accuracy", "quote_faithfulness"],
           "jd_extraction": ["truncated", "schema_valid_rate", "skills_precision", "skills_recall", "skills_f1", "skills_relaxed_f1",
                             "domains_f1", "minimum_degree_accuracy", "quote_faithfulness"]}


def value(x):
    return x["value"] if isinstance(x, dict) else x


def rows():
    for kind, files in SETS.items():
        scores = [json.loads(Path(f"{f}.scores.json").read_text(encoding="utf-8")) for f in files]
        for task, metrics in METRICS.items():
            for m in metrics:
                v = [value(s[task][m]) for s in scores]
                yield kind, task, m, v, st.mean(v), st.stdev(v)


def main() -> None:
    table = list(rows())
    out = ["# V4 recipe over 3 training seeds (test split: CV n=36, JD n=54)", "",
           "Seed 3407 = the Review 2 V4 run; seeds 1 and 2 = task3 pod (2026-11-10), same data and settings.",
           "Repair/retry: deterministic repair, then regenerate still-invalid documents at repetition penalty 1.1/1.15/1.2/1.3.", ""]
    for kind in SETS:
        for task in METRICS:
            out += [f"## {kind}: {task}", "", "| Metric | seed 3407 | seed 1 | seed 2 | mean ± sd |", "|---|---|---|---|---|"]
            for k, t, m, v, mean, sd in table:
                if k == kind and t == task:
                    fmt = (lambda x: f"{x:.0f}") if m == "truncated" else (lambda x: f"{x:.3f}")
                    out.append(f"| {m} | {' | '.join(fmt(x) for x in v)} | {fmt(mean)} ± {sd:.3f} |")
            out.append("")
    Path("output/task3/v4_seeds.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    if "--no-wandb" in sys.argv:
        return
    import wandb
    run = wandb.init(entity="itsbim-fpt-university", project="capstone_project", group="review3-task3", job_type="eval",
                     name="v4_seed_variance", tags=["task3", "seeds"])
    t = wandb.Table(columns=["setting", "task", "metric", "seed_3407", "seed_1", "seed_2", "mean", "sd"],
                    data=[[k, task, m, *v, mean, sd] for k, task, m, v, mean, sd in table])
    run.log({"seed_table": t})
    run.summary.update({f"{'final' if 'repair' in k else 'raw'}/{task[:2]}/{m}_mean": mean for k, task, m, v, mean, sd in table})
    art = wandb.Artifact("v4-seed-scores", type="evaluation")
    for files in SETS.values():
        for f in files:
            art.add_file(f"{f}.scores.json", name=f"{Path(f).name}.scores.json")
    art.add_file("output/task3/v4_seeds.md")
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
