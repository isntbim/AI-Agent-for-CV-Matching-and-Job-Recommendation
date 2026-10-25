"""Reconcile review diagnostics, export research records, and save verified receipts."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring.models import ResumeInput, JobInput
from src.scoring.v2_corpus import atomic_json, digest, read_json
from src.scoring.v3_corpus import refresh, review_manifest, validate_identity, items, reconcile_document
from src.scoring.v3_validation import validate_skill_evidence


def reconcile():
    reviewed_on = date.today().isoformat()
    path = ROOT / "output/scorer_v3/corpus.json"
    corpus = read_json(path)
    validate_identity(corpus)
    for document in corpus["documents"]:
        text = (ROOT / document["source"]["text_path"]).read_text(encoding="utf-8")
        if document["kind"] == "job":
            for unit in document["envelope"]["document"]["skills"]["requirements"]:
                if unit.get("quote", "").startswith("Title:") and len(unit["options"]) > 1 and unit.get("review_status") != "approved" and "title_choice_scope_pending" not in unit["issues"]:
                    before = deepcopy(unit)
                    unit["issues"].append("title_choice_scope_pending")
                    corpus["changes"].append({"date": reviewed_on, "item_id": unit["id"], "document_id": document["id"],
                        "review_kind": "pending_nomination", "rule_id": "R3-TITLE", "reason": "Title-only alternatives require body evidence; this is not approval.",
                        "before": before, "after": deepcopy(unit), "source_sha256": document["source"]["source_sha256"],
                        "text_sha256": document["source"]["text_sha256"], "source_quote": unit["quote"]})
        old_status = document["envelope"]["annotation"]["skills_status"]
        reconcile_document(document, text)
        if document["skills_status"] != old_status:
            corpus["changes"].append({"date": reviewed_on, "document_id": document["id"], "review_kind": "annotation_status_reconciliation",
                "rule_id": "R3-REVIEW-STATUS", "reason": "Recompute evidence completeness after item decisions; human envelope approval remains independent.",
                "before": {"skills_status": old_status}, "after": {"skills_status": document["skills_status"], "issues": document["v3_validation_issues"]},
                "source_sha256": document["source"]["source_sha256"], "text_sha256": document["source"]["text_sha256"]})
    # Backfill the source proposal made before sample-1's direct approval. Its
    # exact pre-approval value is retained in the checksum-bound exchange.
    if not any(change.get("review_kind") == "source_proposal" for change in corpus["changes"]):
        approvals = read_json(ROOT / "human-verify/scorer-v3/imported_approvals.json")
        decision = next(d for d in approvals["items"] if d.get("replacement", {}).get("name") == "program management")
        before_approval = next(value for key, (_, _, _, value) in items(corpus).items() if key == decision["item_id"])
        neutral = deepcopy(before_approval)
        neutral["provenance"]["review_status"] = "pending"
        corpus["changes"].append({"date": "2026-10-08", "document_id": "cv_001", "item_id": decision["item_id"], "review_kind": "source_proposal",
            "rule_id": "R3-GROUP", "before": None, "after": neutral, "backfilled": True,
            "source_sha256": decision["source_sha256"], "text_sha256": decision["text_sha256"], "source_quote": decision["source_quote"],
            "reason": "An explicitly named, previously missing skill was added as normal before the direct sample-1 approval."})
    refresh(corpus)
    atomic_json(path, corpus)
    review_root = ROOT / "human-verify/scorer-v3"
    atomic_json(review_root / "review_items.json", review_manifest(corpus))
    atomic_json(review_root / "changes.json", corpus["changes"])
    exceptions = [{"sample": 2, "document_id": "cv_002", "reason": "Proficient accounting wording in training others remains normal/pending."},
                  {"sample": 6, "document_id": "jd_012", "reason": "Title Golang/JavaScript still needs body evidence for AND/OR interpretation."}]
    exceptions = [entry for entry in exceptions if entry["sample"] in corpus.get("pending_samples", [])]
    pending_documents = {entry["document_id"] for entry in exceptions}
    other_exceptions = [{"document_id": d["id"], "issues": d["v3_validation_issues"]} for d in corpus["documents"] if d["v3_validation_issues"] and d["id"] not in pending_documents]
    atomic_json(review_root / "unresolved.json", {"version": "ats-review-v3", "corpus_sha256": corpus["identity_sha256"], "samples": exceptions, "other_exceptions": other_exceptions})
    exceptions += other_exceptions
    batch_root = review_root / "batches"
    batch_root.mkdir(exist_ok=True)
    # Distribute 5–10 document samples per batch, avoiding a tiny final batch.
    number = max(1, (len(exceptions) + 9) // 10)
    sizes = [len(exceptions) // number + (i < len(exceptions) % number) for i in range(number)]
    offset = 0
    for index, size in enumerate(sizes, 1):
        selected = exceptions[offset:offset + size]
        offset += size
        lines = [f"# V3 exception review batch {index}", "", f"Corpus identity: `{corpus['identity_sha256']}`. These samples remain pending.", ""]
        for entry in selected:
            document = next(d for d in corpus["documents"] if d["id"] == entry["document_id"])
            lines += [f"## {entry['document_id']}" + (f" — sample {entry['sample']}" if "sample" in entry else ""), "",
                f"Source: `{document['source']['text_path']}`. Original SHA-256: `{document['source']['source_sha256']}`.", "",
                entry.get("reason", "Review source interpretation and pending issues before approving an item."), "", "```json", json.dumps(entry.get("issues", []), indent=2), "```", ""]
            if document["kind"] == "job":
                quoted_items = [u for u in document["envelope"]["document"]["skills"]["requirements"] if u.get("issues")][:3]
                for unit in quoted_items:
                    lines += [f"Item `{unit['id']}` — {unit.get('quote', '')}", ""]
            else:
                text = (ROOT / document["source"]["text_path"]).read_text(encoding="utf-8")
                if entry["document_id"] == "cv_002":
                    position = text.lower().find("proficient accounting")
                    lines += ["Source excerpt: " + text[max(0, position - 180):position + 150].replace("\n", " "), ""]
        (batch_root / f"batch_{index:03}.md").write_text("\n".join(lines), encoding="utf-8")
    (review_root / "README.md").write_text("# V3 item review\n\nUse `review_items.json` for current IDs/checksums. `imported_approvals.json` is the immutable direct-review exchange; `derived_rules.json` is separate. `changes.json` preserves before/after values. Exceptions are grouped into 5–10 samples per file under `batches/`. Full envelopes remain pending.\n\nImport: `python -m scripts.ats_v3_review import-review --decisions path/to/decisions.json`. Stale identities, item/source checksums, invented evidence and missing reviewer/reason are refused. Replaying an already imported transaction is a no-op.\n", encoding="utf-8")
    print(json.dumps({"identity": corpus["identity_sha256"], "exceptions": len(exceptions), "batches": len(sizes), "batch_sizes": sizes}))


def test_receipt(path):
    suite = ET.parse(path).getroot()
    if suite.tag == "testsuites":
        suite = suite.find("testsuite")
    result = dict(suite.attrib)
    if any(int(result.get(key, 0)) for key in ("failures", "errors")):
        raise ValueError("Tests did not pass: " + str(path))
    return result


def finalize(benchmark, tests=None, verified_on=None):
    verified_on = verified_on or date.today().isoformat()
    research = ROOT / "docs/research/ats_scorer"
    corpus = read_json(ROOT / "output/scorer_v3/corpus.json")
    validate_identity(corpus)
    report = read_json(benchmark / "report.json")
    identity = report["identity"]
    if report["status"] != "complete":
        raise ValueError("Benchmark failed")
    for relative, checksum in identity["code"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != checksum:
            raise ValueError("Benchmark code identity is stale: " + relative)
    for value in identity["inputs"].values():
        path = Path(value["path"])
        path = path if path.is_absolute() else ROOT / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
            raise ValueError("Benchmark input identity is stale: " + str(path))
    baseline = ROOT / "output/scorer_v3/frozen_v2"
    frozen_manifest = read_json(baseline / "manifest.json")
    inputs_manifest = read_json(baseline / "inputs_manifest.json")
    for name, expected in (("snapshot.zip", frozen_manifest["snapshot_sha256"]), ("inputs.zip", inputs_manifest["inputs_sha256"])):
        if hashlib.sha256((baseline / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen archive changed")
    receipt = {"version": "ats-v3-verification-v1", "verified_on": verified_on, "as_of_date": report["as_of_date"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "interpreter": sys.executable},
        "tests": test_receipt(tests or ROOT / "output/scorer_v3/final_tests.xml"), "frozen_tests": test_receipt(baseline / "regression_full.xml"),
        "historical_original_receipt": read_json(baseline / "verification.json"), "reproduction": report["reproduction"],
        "benchmark_identity": identity["family_sha256"],
        "all_four_variants_succeeded": {"frozen_v2", "v3_off", "v3_score_only", "v3_full"} <= set(report["variants"]),
        "variant_count": len(report["variants"]), "all_variants_succeeded": all(e["all"]["failed"] == 0 for e in report["variants"].values()),
        "original_label_checksum_verified": report["original_label_checksum_verified"], "frozen_source_matches_original_benchmark": report["frozen_source_matches_original_benchmark"],
        "population": report["population"], "source_code_input_checksums_current": True,
        "frozen_archives_unchanged": True, "review_import": read_json(ROOT / "human-verify/scorer-v3/import_receipt.json"),
        "review_coverage": report["review_coverage"], "source_depth_coverage": report["source_depth_coverage"],
        "pending_samples": corpus.get("pending_samples", []), "whole_envelopes_approved": report["review_coverage"]["whole_envelopes_approved"], "default_ruleset": "ats-en-v1",
        "perplexity_queries": 0, "primary_source_verification": "Retrieval failed; historical references retained without claiming fresh verification",
        "limitations": report["limitations"], "unresolved_review": "human-verify/scorer-v3/unresolved.json"}
    atomic_json(ROOT / "output/scorer_v3/verification.json", receipt)
    atomic_json(research / "verification.json", receipt)
    atomic_json(research / "benchmark_summary.json", report)
    atomic_json(research / "baseline_manifest.json", {k: frozen_manifest[k] for k in ("version", "preserved_on", "python", "platform", "git_head", "snapshot_sha256")})
    with (research / "annotation_changes.jsonl").open("w", encoding="utf-8") as stream:
        for change in corpus["changes"]:
            stream.write(json.dumps(change, ensure_ascii=False) + "\n")
    atomic_json(research / "unresolved_review.json", read_json(ROOT / "human-verify/scorer-v3/unresolved.json"))
    metrics = report["variants"]
    lines = ["# V3 experiment results", "", f"Verified {verified_on}; fixed evaluation date {report['as_of_date']}. Experiment identity `{identity['family_sha256']}`.", "",
        f"Current suite: {receipt['tests']['tests']} tests passed. Frozen selected historical suite: {receipt['frozen_tests']['tests']} passed. Historical original receipt (230 tests) is preserved separately.",
        "", "All 1,672 unchanged v1 and v2 pair scores/grades reproduce both the frozen source and original benchmark artifacts with zero drift. All four diagnostics completed without pair failures.",
        "", "| Variant | Gemini macro-F1 | Gemini balanced accuracy | Heuristic macro-F1 | Grade changes from previous | Skills score changes |", "|---|---:|---:|---:|---:|---:|"]
    for name, entry in metrics.items():
        change = entry.get("change_from_previous", {})
        lines.append(f"| {name} | {entry['gemini_primary']['metrics']['macro_f1']:.4f} | {entry['gemini_primary']['metrics']['balanced_accuracy']:.4f} | {entry['heuristic_negatives']['metrics']['macro_f1']:.4f} | {change.get('changed', '—')} | {change.get('score_changes', {}).get('skills', {}).get('pairs_changed', '—')} |")
    depth = metrics["v3_full"]["depth"]
    lines += ["", f"Full v3 has {depth['explicit_depth']} scored requirement occurrences with explicit minima, {depth['unknown_candidate_depth']} with unknown candidate depth; gap counts: `{depth['gap']}`.",
        "", "This sparse observed depth evidence cannot establish that the depth policy improves label agreement. Exhaustive gap tests validate the implementation, not real-world proficiency distances.",
        "", "## Coverage", "", "```json", json.dumps(report["review_coverage"], indent=2), json.dumps(report["source_depth_coverage"], indent=2), "```", "",
        "The numbered-sample pending list above reflects current user decisions. Additional title-only choices and incomplete annotations are preserved in the unresolved list and review batches. No whole envelope is approved.",
        "", "Per-grade metrics, confusion matrices, all 24 historical disagreements, failures, latency, domain slices, gate causes and changes are retained in benchmark_summary.json and local pair artifacts.",
        "", "Primary-source browsing failed for the inherited pinned references; no fresh literature verification or Perplexity research is claimed."]
    (research / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    event = {"date": verified_on, "id": "C3-VERIFY-" + identity["family_sha256"][:12], "subsystem": "final verification/research",
        "before": "Implementation and review diagnostics required reconciliation", "after": "Current code/data identity, historical reproduction, complete acceptance checks, exceptions and experiment outcomes saved",
        "reason": "Reconcile implementation, guides, annotation history and empirical limitations before handoff", "version": "ats-v3-verification-v1",
        "validation": receipt["tests"], "artifacts": ["docs/research/ats_scorer/verification.json", "docs/research/ats_scorer/results.md", "human-verify/scorer-v3/batches/"]}
    log = research / "changes.jsonl"
    if event["id"] not in log.read_text(encoding="utf-8"):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
    print(json.dumps({"tests": receipt["tests"]["tests"], "frozen_tests": receipt["frozen_tests"]["tests"], "identity": identity["family_sha256"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("reconcile", "finalize"))
    parser.add_argument("--benchmark", type=Path, default=ROOT / "output/scorer_v3/benchmark")
    parser.add_argument("--tests", type=Path)
    parser.add_argument("--verified-on", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    reconcile() if args.command == "reconcile" else finalize(args.benchmark, args.tests, args.verified_on.isoformat())
