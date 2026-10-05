"""Score stored silver pairs as a sample or full offline comparison; no API calls."""

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scoring import ScorerConfig, score_pair
from src.scoring.evaluation import classification_metrics
from src.scoring.silver import (
    ADAPTER_VERSION,
    job_from_record,
    resume_from_record,
)

LABELS = {0: "Mismatch", 1: "Partial", 2: "Strong"}


def select_sample(pairs: list[dict], per_grade: int, seed: int) -> list[dict]:
    if per_grade < 2:
        raise ValueError("per_grade must be at least two to include both negative sources")
    rng = random.Random(seed)
    result = []
    for grade in range(3):
        pool = sorted((p for p in pairs if p["grade"] == grade), key=lambda p: (p["cv_id"], p["jd_id"]))
        if grade == 0:
            heuristic = [p for p in pool if p["pair_type"] == "heuristic_negative"]
            teacher = [p for p in pool if p["pair_type"] != "heuristic_negative"]
            result.extend(rng.sample(heuristic, per_grade // 2))
            result.extend(rng.sample(teacher, per_grade - per_grade // 2))
        else:
            result.extend(rng.sample(pool, per_grade))
    return sorted(result, key=lambda p: (p["grade"], p["cv_id"], p["jd_id"]))


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    labels = [r["silver_grade"] for r in rows]
    predicted = [r["result"]["grade"] for r in rows]
    return {
        "agreement": classification_metrics(labels, predicted),
        "silver_distribution": dict(Counter(map(str, labels))),
        "scorer_distribution": dict(Counter(map(str, predicted))),
        "mean_coverage": statistics.mean(r["result"]["coverage"] for r in rows),
        "mean_scores_by_silver_grade": {str(k): statistics.mean(r["result"]["overall_score"] for r in rows if r["silver_grade"] == k)
                                      for k in range(3) if k in labels},
        "dimension_coverage": {dim: statistics.mean(r["result"]["dimensions"][dim]["supported"] for r in rows)
                               for dim in rows[0]["result"]["dimensions"]},
        "dimension_mean": {dim: statistics.mean(r["result"]["scores"][dim] for r in rows)
                           for dim in rows[0]["result"]["scores"]},
        "silver_overall_mae": statistics.mean(abs(r["result"]["overall_score"] - r["silver_overall_score"]) for r in rows),
        "silver_dimension_mae": {dim: statistics.mean(abs(r["result"]["scores"][dim] - r["silver_scores"][dim]) for r in rows)
                                 for dim in rows[0]["result"]["scores"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-grade", type=int, default=10)
    parser.add_argument("--full", action="store_true", help="Score every stored pair instead of sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or ROOT / ("output/ats/silver_full" if args.full else "output/ats/silver_sample")
    paths = {
        "pairs": ROOT / "data/datasets/matching_pairs/benchmark_matching_pairs.json",
        "cvs": ROOT / "data/processed/structured_cvs/all_cvs.json",
        "jobs": ROOT / "data/processed/structured_jobs/all_jobs_en.json",
    }
    data = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    cvs = {cv["cv_id"]: cv for cv in data["cvs"]}
    jobs = {jd["id"]: jd for jd in data["jobs"]}
    config = ScorerConfig()
    rows, failures = [], []
    selected = data["pairs"] if args.full else select_sample(data["pairs"], args.per_grade, args.seed)
    for pair in selected:
        try:
            cv = resume_from_record(cvs[pair["cv_id"]], args.as_of)
            jd, removed = job_from_record(jobs[pair["jd_id"]])
            result = score_pair(cv, jd, config=config, as_of_date=args.as_of)
        except (KeyError, ValueError, TypeError) as exc:
            failures.append({"cv_id": pair["cv_id"], "jd_id": pair["jd_id"], "error": str(exc)})
            continue
        rows.append({"cv_id": pair["cv_id"], "jd_id": pair["jd_id"], "pair_type": pair["pair_type"],
                     "silver_grade": pair["grade"], "silver_overall_score": pair["overall_score"],
                     "silver_scores": pair["scores"], "cv_title": cv.document.basics.label,
                     "cv_domain": pair["cv_domain"], "jd_domain": pair["jd_domain"],
                     "jd_title": jd.document.title, "removed_title_skill_tags": removed,
                     "result": result.model_dump(mode="json")})
    report = {
        "scope": ("Full stored dataset" if args.full else "Small stratified sample") + "; diagnostic agreement with silver labels, not held-out hiring-outcome validation",
        "full_dataset": args.full,
        "sample_size": len(selected), "successful": len(rows), "failures": failures,
        "population": len(data["pairs"]), "population_grades": dict(Counter(str(p["grade"]) for p in data["pairs"])),
        "seed": args.seed, "as_of_date": args.as_of.isoformat(), "adapter_version": ADAPTER_VERSION,
        "config": config.model_dump(), "api_requests": 0,
        "inputs": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in paths.items()},
        "unique_cvs": len({r["cv_id"] for r in rows}), "unique_jobs": len({r["jd_id"] for r in rows}),
        "all": summarize(rows),
        "llm_only": summarize([r for r in rows if r["pair_type"] != "heuristic_negative"]),
        "heuristic_only": summarize([r for r in rows if r["pair_type"] == "heuristic_negative"]),
        "pair_types": dict(Counter(r["pair_type"] for r in rows)),
        "evidence_provenance": "Quotes are values in the stored structured JSON, not verified against original CV PDFs/raw JDs. CV domain metadata and JD category are not supplied as scorer facts.",
        "limitations": [("Full stored population is imbalanced and shares CVs/JDs across pairs; agreement is not out-of-sample validation." if args.full else "Equal-grade sampling overrepresents the 34 Strong pairs; sample metrics do not estimate full-dataset accuracy."),
                        "Silver grades/subscores originate from Gemini and heuristic negatives, not observed hiring decisions.",
                        "Degree matching uses dated completed CV degrees and mandatory JD requirement sentences; ambiguous facts stay neutral.",
                        "The existing title/seniority skill-tag filter is applied before scoring; structured skills and extraction omissions remain limitations.",
                        "Negotiable salary defaults to 75 and is unsupported; unknown dimensions default to 50.",
                        "Current thresholds and weights are frozen; no calibration or training uses these comparison pairs.",
                        "Present employment is evaluated at the explicit as-of date; historical CVs can therefore overstate current tenure."],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["cv_id", "jd_id", "pair_type", "silver_grade", "scorer_grade", "silver_score", "scorer_score", "coverage",
                  "skills", "title", "experience", "education", "industry", "location", "salary"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            result = row["result"]
            writer.writerow({"cv_id": row["cv_id"], "jd_id": row["jd_id"], "pair_type": row["pair_type"],
                             "silver_grade": row["silver_grade"], "scorer_grade": result["grade"],
                             "silver_score": row["silver_overall_score"], "scorer_score": result["overall_score"],
                             "coverage": result["coverage"], **result["scores"]})
    lines = ["# ATS scorer: " + ("full silver dataset" if args.full else "small silver sample"), "", f"Scored **{len(rows)}/{len(selected)} pairs**, from a population of {len(data['pairs'])}. API requests: 0.",
             "", ("Every stored pair is included; no sampling." if args.full else f"{args.per_grade} per silver grade; {args.per_grade // 2} sampled mismatch labels are heuristic negatives. Seed: {args.seed}."), "",
             "| Silver grade | Pairs | Mean ATS score | Mean supported weight |", "|---|---:|---:|---:|"]
    for grade in range(3):
        group = [r for r in rows if r["silver_grade"] == grade]
        if group:
            lines.append(f"| {LABELS[grade]} | {len(group)} | {statistics.mean(r['result']['overall_score'] for r in group):.1f} | {statistics.mean(r['result']['coverage'] for r in group):.1%} |")
    if rows:
        metrics = report["all"]["agreement"]
        lines.extend(["", f"Silver-grade agreement: **{metrics['accuracy']:.1%}**. Macro F1: **{metrics['macro_f1']:.3f}**.",
                      "", "## Confusion matrix", "", "Rows: silver grade. Columns: runtime grade.", "", "| Silver / Runtime | Mismatch | Partial | Strong |", "|---|---:|---:|---:|"])
        for grade, counts in enumerate(metrics["confusion_matrix"]):
            lines.append(f"| {LABELS[grade]} | {' | '.join(map(str, counts))} |")
        lines.extend(["", "## Dimension support", "", "| Dimension | Supported pairs | Mean score |", "|---|---:|---:|"])
        for dimension, coverage in report["all"]["dimension_coverage"].items():
            lines.append(f"| {dimension} | {coverage:.1%} | {report['all']['dimension_mean'][dimension]:.1f} |")
        lines.extend(["", "## Label sources", ""])
        for key in ("llm_only", "heuristic_only"):
            group = report[key]
            if "agreement" in group:
                lines.append(f"- {key}: {group['agreement']['n']} pairs, grade agreement {group['agreement']['accuracy']:.1%}.")
        teacher = report["llm_only"]
        if "agreement" in teacher:
            lines.extend(["", f"Against Gemini-labelled pairs only, overall-score MAE is **{teacher['silver_overall_mae']:.1f}/100** (agreement with a teacher, not accuracy against hiring outcomes).",
                          "", "| Dimension | Mean absolute difference from Gemini score |", "|---|---:|"])
            for dimension, error in teacher["silver_dimension_mae"].items():
                lines.append(f"| {dimension} | {error:.1f} |")
        lines.extend(["", "## Findings", "",
                      "- The title dictionary currently requires an exact canonical title/alias after removing seniority. Longer job titles and role variants frequently stay unknown; the same dictionary also limits professional-domain and experience scoring.",
                      "- Skill scores compare canonical strings from the stored skill lists. The Gemini judge could also use CV/JD narrative evidence, so omitted skills and unrecognised synonyms can produce large differences.",
                      "- Review extraction completeness, title aliases and dimension rules before changing weights or thresholds. This sample does not establish whether Gemini or the code is correct."])
    lines.extend(["", "## Interpretation", "", report["evidence_provenance"], ""])
    lines.extend("- " + limitation for limitation in report["limitations"])
    if failures:
        lines.extend(["", "## Failures", "", json.dumps(failures, indent=2)])
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"successful": len(rows), "failed": len(failures), "report": str(args.output / "report.md"), "summary": report["all"]}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
