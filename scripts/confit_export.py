"""Export matching dev-half CV/JD pairs into ConFit v2 format under D:\\confit_v2\\data_ours.

Strictly reads only dev-half CVs (`scripts.scorer_lab.split_of(cv_id) == "dev"`) so
the held-out test half is never exposed during training or model selection.
Positive labels follow the benchmark definition: label >= 1 (Potential or Good Fit).

Usage:
    .venv\\Scripts\\python.exe scripts\\confit_export.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import match_eval as m
from scripts.scorer_lab import split_of

DEFAULT_OUT = Path(r"D:\confit_v2\data_ours\recruiting_data_v2")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_ranking_dicts(label_rows: list[dict]) -> tuple[dict, dict]:
    """Build ConFit rank_job and rank_resume dicts keeping only queries with >=1 pos and >=1 neg."""
    by_user: dict[str, list[tuple[str, int]]] = defaultdict(list)
    by_job: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in label_rows:
        uid = row["user_id"]
        jid = row["jd_no"]
        sat = int(row["satisfied"])
        by_user[uid].append((jid, sat))
        by_job[jid].append((uid, sat))

    rank_job = {}
    for uid, items in sorted(by_user.items()):
        sats = [s for _, s in items]
        if any(s == 1 for s in sats) and any(s == 0 for s in sats):
            rank_job[uid] = {
                "jd_nos": [j for j, _ in items],
                "satisfied": sats,
            }

    rank_resume = {}
    for jid, items in sorted(by_job.items()):
        sats = [s for _, s in items]
        if any(s == 1 for s in sats) and any(s == 0 for s in sats):
            rank_resume[jid] = {
                "user_ids": [u for u, _ in items],
                "satisfied": sats,
            }
    return rank_job, rank_resume


def _pick_smoke_subset(
    train_rows: list[dict],
    val_rows: list[dict],
    n_train: int = 200,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Build a small smoke subset (~n_train pairs) with balanced positives/negatives per entity."""
    rng = random.Random(seed)
    pos_train = [r for r in train_rows if r["satisfied"] == 1]
    neg_train = [r for r in train_rows if r["satisfied"] == 0]
    rng.shuffle(pos_train)
    rng.shuffle(neg_train)

    # Pick ~25% positives (matching corpus ~22% positive rate) so 1 epoch at bs=8 is ~6 steps.
    n_pos = min(len(pos_train), max(16, n_train // 4))
    chosen_pos = pos_train[:n_pos]
    n_neg = min(len(neg_train), max(0, n_train - len(chosen_pos)))
    chosen_neg = neg_train[:n_neg]
    smoke_train = chosen_pos + chosen_neg
    rng.shuffle(smoke_train)

    # For validation smoke: take up to 2 full dev-val CVs that contain both pos and neg pairs.
    val_by_cv: dict[str, list[dict]] = defaultdict(list)
    for r in val_rows:
        val_by_cv[r["user_id"]].append(r)
    eligible_val_cvs = [
        cv for cv, rs in sorted(val_by_cv.items())
        if any(x["satisfied"] == 1 for x in rs) and any(x["satisfied"] == 0 for x in rs)
    ]
    smoke_val = [r for cv in eligible_val_cvs[:2] for r in val_by_cv[cv]]
    return smoke_train, smoke_val


def export_confit_data(out_dir: Path, val_ratio: float = 0.10, seed: int = 42, smoke_pairs: int = 200) -> dict:
    """Convert dev-half pairs to ConFit v2 format under out_dir."""
    pairs = m.load_pairs()

    cv_texts: dict[str, str] = {}
    jd_texts: dict[str, str] = {}
    dev_pairs: list[dict] = []
    test_cvs: set[str] = set()

    for p in pairs:
        cv_id = m.doc_id("cv", p["cv"])
        jd_id = m.doc_id("jd", p["jd"])
        if split_of(cv_id) == "dev":
            cv_texts.setdefault(cv_id, p["cv"])
            jd_texts.setdefault(jd_id, p["jd"])
            dev_pairs.append({
                "user_id": cv_id,
                "jd_no": jd_id,
                "satisfied": 1 if int(p["label"]) >= 1 else 0,
            })
        else:
            test_cvs.add(cv_id)

    dev_cvs = sorted(cv_texts)
    assert not (set(dev_cvs) & test_cvs), "CV leakage detected between dev and test splits"

    # Stratify dev CVs slightly by whether the CV has >=1 positive pair.
    cv_pos_counts: dict[str, int] = defaultdict(int)
    for row in dev_pairs:
        cv_pos_counts[row["user_id"]] += row["satisfied"]

    rng = random.Random(seed)
    cvs_with_pos = [c for c in dev_cvs if cv_pos_counts[c] > 0]
    cvs_no_pos = [c for c in dev_cvs if cv_pos_counts[c] == 0]
    rng.shuffle(cvs_with_pos)
    rng.shuffle(cvs_no_pos)

    n_val_pos = max(2, round(len(cvs_with_pos) * val_ratio))
    n_val_nopos = round(len(cvs_no_pos) * val_ratio)
    val_cv_set = set(cvs_with_pos[:n_val_pos] + cvs_no_pos[:n_val_nopos])
    train_cv_set = set(dev_cvs) - val_cv_set

    train_rows = [r for r in dev_pairs if r["user_id"] in train_cv_set]
    val_rows = [r for r in dev_pairs if r["user_id"] in val_cv_set]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "resumes.csv",
        ["user_id", "text_resume"],
        [{"user_id": cid, "text_resume": cv_texts[cid]} for cid in dev_cvs],
    )
    _write_csv(
        out_dir / "jobs.csv",
        ["jd_no", "text_job"],
        [{"jd_no": jid, "text_job": jd_texts[jid]} for jid in sorted(jd_texts)],
    )
    _write_jsonl(out_dir / "train_labeled_data.jsonl", train_rows)
    _write_jsonl(out_dir / "valid_classification_data.jsonl", val_rows)

    rank_job, rank_resume = _build_ranking_dicts(val_rows)
    (out_dir / "valid_rank_job.json").write_text(json.dumps(rank_job, indent=2), encoding="utf-8")
    (out_dir / "valid_rank_resume.json").write_text(json.dumps(rank_resume, indent=2), encoding="utf-8")

    smoke_train, smoke_val = _pick_smoke_subset(train_rows, val_rows, n_train=smoke_pairs, seed=seed)
    _write_jsonl(out_dir / "train_labeled_data_smoke.jsonl", smoke_train)
    _write_jsonl(out_dir / "valid_classification_data_smoke.jsonl", smoke_val)
    s_rank_job, s_rank_resume = _build_ranking_dicts(smoke_val)
    (out_dir / "smoke_rank_job.json").write_text(json.dumps(s_rank_job, indent=2), encoding="utf-8")
    (out_dir / "smoke_rank_resume.json").write_text(json.dumps(s_rank_resume, indent=2), encoding="utf-8")

    summary = {
        "out_dir": str(out_dir),
        "dev_cvs_total": len(dev_cvs),
        "dev_jds_total": len(jd_texts),
        "dev_pairs_total": len(dev_pairs),
        "test_cvs_excluded": len(test_cvs),
        "train_cvs": len(train_cv_set),
        "val_cvs": len(val_cv_set),
        "train_pairs": len(train_rows),
        "train_positives": sum(r["satisfied"] for r in train_rows),
        "train_negatives": sum(1 - r["satisfied"] for r in train_rows),
        "val_pairs": len(val_rows),
        "val_positives": sum(r["satisfied"] for r in val_rows),
        "val_negatives": sum(1 - r["satisfied"] for r in val_rows),
        "valid_rank_job_queries": len(rank_job),
        "valid_rank_resume_queries": len(rank_resume),
        "smoke_train_pairs": len(smoke_train),
        "smoke_train_positives": sum(r["satisfied"] for r in smoke_train),
        "smoke_val_pairs": len(smoke_val),
        "val_cv_ids": sorted(val_cv_set),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-pairs", type=int, default=200)
    args = p.parse_args()
    summary = export_confit_data(args.out_dir, args.val_ratio, args.seed, args.smoke_pairs)
    print(json.dumps({k: v for k, v in summary.items() if k != "val_cv_ids"}, indent=2))


if __name__ == "__main__":
    main()
