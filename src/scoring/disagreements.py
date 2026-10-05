"""Source-linked, training-partition review of historical scorer disagreements."""

from collections import Counter
from pathlib import Path
from .pilot import read_json, sha, REVIEW_STATUS
from .extraction import write_json


def nominate(root: Path, output: Path):
    split = read_json(output / "split_manifest.json")
    documents = {d["id"]: d for d in split["documents"]}
    results_path = root / "output/ats/silver_full/results.json"
    results = read_json(results_path)
    candidates = [r for r in results if r["pair_type"] != "heuristic_negative" and
                  documents[r["cv_id"]]["split"] == documents[r["jd_id"]]["split"] == "train" and
                  (r["silver_grade"] != r["result"]["grade"] or
                   abs(r["silver_overall_score"] - r["result"]["overall_score"]) >= 15)]
    chosen, seen = [], set()
    for domain in sorted({d["domain"] for d in documents.values() if d["kind"] == "resume"}):
        pool = [r for r in candidates if r["cv_domain"] == domain]
        categories = ["zero_skill_overlap", "unsupported_role_recognition", "grade_gate", "large_score_difference"]
        for category in categories:
            options = [r for r in pool if (r["cv_id"], r["jd_id"]) not in seen]
            if category == "zero_skill_overlap":
                options = [r for r in options if r["result"]["scores"]["skills"] == 0]
            elif category == "unsupported_role_recognition":
                options = [r for r in options if not r["result"]["dimensions"]["title"]["supported"]]
            elif category == "grade_gate":
                options = [r for r in options if r["result"]["overall_score"] >= 75 and r["result"]["grade"] < 2]
            fallback = not bool(options)
            if fallback:
                options = [r for r in pool if (r["cv_id"], r["jd_id"]) not in seen]
            if not options:
                raise ValueError(f"Insufficient training disagreements in {domain}")
            selected = min(options, key=lambda r: (-abs(r["silver_overall_score"] - r["result"]["overall_score"]),
                                                   r["cv_id"], r["jd_id"]))
            seen.add((selected["cv_id"], selected["jd_id"]))
            links = {kind: documents[selected[field]] for kind, field in (("cv", "cv_id"), ("jd", "jd_id"))}
            excerpts = {}
            for kind, document in links.items():
                source = (root / document["text_path"]).read_text(encoding="utf-8")
                excerpts[kind] = source[:650]
            chosen.append({"selection_reason": category, "category_fallback": fallback,
                           "original_comparison": selected, "sources": links, "source_excerpts": excerpts,
                           "classification": "unresolved", "review_status": "awaiting_assistant_review", "findings": []})
    payload = {"version": "source-disagreements-v1", "count": len(chosen), "as_of_date": results[0]["result"]["as_of_date"],
               "comparison_sha256": sha(results_path.read_bytes()), "split_manifest_sha256": sha((output / "split_manifest.json").read_bytes()),
               "status": "assistant_review_pending", "cases": chosen,
               "scope": "Historical silver diagnostics; neither labels nor scorer totals are changed."}
    write_json(output / "disagreement_nominations.json", payload)
    return payload


def validate_review(output):
    report = read_json(output / "disagreement_review.json")
    nominations = read_json(output / "disagreement_nominations.json")
    allowed = {"source_annotation_problem", "normalization_gap", "role_vocabulary_gap", "scoring_rule_behavior", "ambiguous_silver_label", "unresolved"}
    if len(report["cases"]) != 24 or len(nominations["cases"]) != 24:
        raise ValueError("Disagreement review requires 24 cases")
    if any(report[key] != nominations[key] for key in ("comparison_sha256", "split_manifest_sha256")):
        raise ValueError("Disagreement review references stale comparison or split manifests")
    counts = Counter()
    for current, original in zip(report["cases"], nominations["cases"]):
        if any(current[key] != original[key] for key in ("original_comparison", "sources", "selection_reason", "category_fallback", "source_excerpts")):
            raise ValueError("Review changed original labels, scores or source identity")
        if current["review_status"] != REVIEW_STATUS or current["classification"] not in allowed or not current["findings"]:
            raise ValueError("Incomplete disagreement review")
        if any(s["split"] != "train" for s in current["sources"].values()):
            raise ValueError("Disagreement review used a holdout")
        counts[current["original_comparison"]["cv_domain"]] += 1
    if set(counts.values()) != {4} or len(counts) != 6:
        raise ValueError("Expected four disagreements per CV domain")
    report["validation"] = {"count": 24, "by_domain": dict(counts), "original_labels_scores_preserved": True,
                            "training_documents_only": True}
    write_json(output / "disagreement_report.json", report)
    return report
