"""Score Qwen extraction predictions (from training/scripts/generate.py) against the gold SFT targets.

Validity reuses the teacher validators (cv_teacher_validate.check / jd_teacher_check.check_jd).
Usage: python -m scripts.qwen_eval preds/base_test.jsonl preds/ft_test.jsonl
"""

import argparse
import json
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

from scripts.cv_teacher_validate import check
from scripts.jd_teacher_check import check_jd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/datasets/qwen_training"
FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def parse(text: str) -> dict | None:
    try:
        data = json.loads(FENCE.sub("", text))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _as(value, kind):
    """Untrained models emit the right keys with the wrong types; a wrong-typed part counts as nothing extracted."""
    return value if isinstance(value, kind) else kind()


def skills(label: dict, task: str) -> set[str]:
    doc = _as(label.get("document"), dict)
    if task == "jd_extraction":
        sk = _as(doc.get("skills"), dict)
        items = [s for key in ("required", "preferred") for s in _as(sk.get(key), list)]
    else:
        items = [d for s in _as(doc.get("skills"), list) if isinstance(s, dict) for d in _as(s.get("details"), list)]
    return {str(s.get("name", "")).casefold().strip() for s in items if isinstance(s, dict)} - {""}


def quotes(node) -> list[str]:
    if isinstance(node, dict):
        return [node["quote"]] if isinstance(node.get("quote"), str) else [q for v in node.values() for q in quotes(v)]
    return [q for v in node for q in quotes(v)] if isinstance(node, list) else []


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p, r = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def _words(name: str) -> tuple[str, ...]:
    return tuple(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9+#]+", name.casefold()))


def relaxed_matches(pred: set[str], gold: set[str]) -> int:
    """One-to-one matches where the names are equal up to case/punctuation/plural, or one name's words are a contiguous
    run inside the other's ('us gaap' ~ 'us gaap principles'). Secondary metric only: exact match stays the headline."""
    def close(a: tuple, b: tuple) -> bool:
        short, long_ = sorted((a, b), key=len)
        return bool(short) and any(long_[i:i + len(short)] == short for i in range(len(long_) - len(short) + 1))
    free, hits = [_words(p) for p in pred], 0
    for g in sorted((_words(g) for g in gold), key=len, reverse=True):
        j = next((i for i, p in enumerate(free) if p == g), None)  # exact (normalised) first, then containment
        j = j if j is not None else next((i for i, p in enumerate(free) if close(p, g)), None)
        if j is not None:
            free.pop(j)
            hits += 1
    return hits


def validate(label: dict, text: str, task: str) -> tuple[str, str | None]:
    """(status, error) from the teacher validator."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "label.json"
        path.write_text(json.dumps(label, ensure_ascii=False), encoding="utf-8")
        result = check_jd(path, text) if task == "jd_extraction" else check(path, text, {})
        return result["status"], result.get("error")


def valid(label: dict, text: str, task: str) -> str:
    return validate(label, text, task)[0]


def doc_counts(r: dict, g: dict, text: str, task: str, deg_key: str) -> Counter:
    """Per-document counts; metrics are ratios of their sums, so documents can be bootstrap-resampled."""
    pred = parse(r.get("repaired") or r["prediction"])  # a qwen_repair output file is scored on its repaired JSON
    c = Counter(n=1, truncated=r.get("finish_reason") == "length", gold=len(skills(g, task)),
                dom_gold=len(set(g["facts"].get("domains") or [])))
    if pred is None:
        return c
    facts = _as(pred.get("facts"), dict)
    ps, gs = skills(pred, task), skills(g, task)
    pd_, gd = {d for d in _as(facts.get("domains"), list) if isinstance(d, str)}, set(g["facts"].get("domains") or [])
    norm = (lambda v: sorted(v) if isinstance(v, list) else v)
    qs = quotes(pred)
    c.update(parsed=1, valid=valid(pred, text, task) in ("pass", "partial"), tp=len(ps & gs), pred=len(ps), tp_relaxed=relaxed_matches(ps, gs),
             dom_tp=len(pd_ & gd), dom_pred=len(pd_), deg_ok=norm(facts.get(deg_key)) == norm(g["facts"].get(deg_key)),
             quotes=len(qs), quotes_in_source=sum(q in text for q in qs))
    return c


def metrics(c: Counter, deg_key: str) -> dict:
    n = c["n"]
    return {"json_parse_rate": c["parsed"] / n, "schema_valid_rate": c["valid"] / n,
            "skills_precision": prf(c["tp"], c["pred"], c["gold"])["precision"],
            "skills_recall": prf(c["tp"], c["pred"], c["gold"])["recall"], "skills_f1": prf(c["tp"], c["pred"], c["gold"])["f1"],
            "skills_relaxed_f1": prf(c["tp_relaxed"], c["pred"], c["gold"])["f1"],
            "domains_f1": prf(c["dom_tp"], c["dom_pred"], c["dom_gold"])["f1"], f"{deg_key}_accuracy": c["deg_ok"] / n,
            "quote_faithfulness": c["quotes_in_source"] / c["quotes"] if c["quotes"] else 0.0}


def score(pred_rows: list[dict], boot: int = 1000) -> dict:
    gold = {}
    for task in {r["task"] for r in pred_rows}:
        for line in (DATA / task / f"{pred_rows[0]['split']}.jsonl").read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            gold[g["id"]] = (json.loads(g["messages"][2]["content"]), json.loads(g["messages"][1]["content"])["source_text"])
    report, rng = {}, random.Random(0)
    for task in sorted({r["task"] for r in pred_rows}):
        deg_key = "minimum_degree" if task == "jd_extraction" else "completed_degrees"
        docs = [doc_counts(r, *gold[r["id"]], task, deg_key) for r in pred_rows if r["task"] == task]
        point = metrics(sum(docs, Counter()), deg_key)
        samples = [metrics(sum(rng.choices(docs, k=len(docs)), Counter()), deg_key) for _ in range(boot)]
        report[task] = {"n": len(docs), "truncated": sum(d["truncated"] for d in docs)}
        for key, value in point.items():  # 95% percentile bootstrap over documents
            dist = sorted(s[key] for s in samples)
            report[task][key] = {"value": round(value, 3), "ci95": [round(dist[int(0.025 * boot)], 3), round(dist[int(0.975 * boot) - 1], 3)]}
    return report


def log_wandb(path: Path, report: dict, args) -> None:
    import wandb

    key_file = ROOT / "knowledge-base/wandb_key.txt"  # local-only credential, never printed
    if not os.environ.get("WANDB_API_KEY") and key_file.exists():
        os.environ["WANDB_API_KEY"] = key_file.read_text(encoding="utf-8").strip()
    run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group, job_type="eval",
                     name=f"eval-{path.stem}", config={"predictions": path.name, "bootstrap": args.boot, "ci": "95% percentile, resampled documents"})
    try:
        run.use_artifact(f"predictions-{path.stem}:latest")  # lineage: predictions -> these scores
    except Exception as exc:  # predictions logged with --no-wandb: scores still upload
        print(f"no W&B predictions artifact for {path.stem}: {exc}")
    table = wandb.Table(columns=["task", "metric", "value", "ci95_low", "ci95_high", "n"])
    for task, metrics_ in report.items():
        run.summary[f"{task}/n"], run.summary[f"{task}/truncated"] = metrics_["n"], metrics_["truncated"]
        for key, m in metrics_.items():
            if isinstance(m, dict):
                run.summary[f"{task}/{key}"] = m["value"]
                table.add_data(task, key, m["value"], m["ci95"][0], m["ci95"][1], metrics_["n"])
    run.log({"scores": table})
    art = wandb.Artifact(f"scores-{path.stem}", type="evaluation", metadata=report)
    art.add_file(str(path.with_suffix(".scores.json")))
    run.log_artifact(art)
    run.finish()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preds", nargs="+", type=Path)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--wandb", action="store_true", help="log scores to W&B, linked to the predictions artifact")
    p.add_argument("--wandb-entity", default="itsbim-fpt-university")
    p.add_argument("--wandb-project", default="capstone_project")
    p.add_argument("--wandb-group", default="review2-qwen-qlora")
    args = p.parse_args()
    for path in args.preds:
        report = score([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()], boot=args.boot)
        path.with_suffix(".scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(path.name, json.dumps(report, indent=2))
        if args.wandb:
            log_wandb(path, report, args)


if __name__ == "__main__":
    main()
