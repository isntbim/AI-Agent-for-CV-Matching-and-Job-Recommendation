"""Checksum-isolated depth and Preferred-policy silver diagnostics."""
from __future__ import annotations
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
from .models import ScorerConfig
from .v2_benchmark import (canonical_hash, file_identity, grade_changes, load_relations,
                          read_json, require_same_identity, score_stored_pair, summarize)
from .v3_corpus import validate_identity
from .v3_aliases import ALIAS_POLICY_VERSION, APPROVED_ALIASES_SHA256
from .runtime_v3 import score_pair_v3

VERSION = "ats-v3-silver-diagnostic-v2"
VARIANTS = ("frozen_v2", "v3_off", "v3_score_only", "v3_full")


def depth_summary(rows):
    reqs = [req for row in rows if row["status"] == "ok" for req in (row["result"].get("skills_breakdown") or {}).get("requirements", [])]
    explicit = [req for req in reqs if req.get("required_level") not in {None, "normal"}]
    unknown = [req for req in explicit if req.get("depth_status") == "unknown_candidate_depth"]
    return {"requirements": len(reqs), "explicit_depth": len(explicit),
        "explicit_depth_fraction": len(explicit) / len(reqs) if reqs else None,
        "unknown_candidate_depth": len(unknown), "unknown_depth_fraction_of_explicit": len(unknown) / len(explicit) if explicit else None,
        "depth_status": dict(Counter(req.get("depth_status", "v2") for req in reqs)),
        "gap": dict(Counter(str(req["depth_gap"]) for req in explicit if req.get("depth_gap") is not None)),
        "gate_causes": dict(Counter("explicit_depth_gap" if (row["result"].get("skills_breakdown") or {}).get("identity", {}).get("depth_policy") == "full" and any(req.get("depth_status") == "explicit_gap" and req.get("importance") == "core" and req.get("fulfillment_factor", 1) < req.get("existing_v2_f", 1) for req in (row["result"].get("skills_breakdown") or {}).get("requirements", []))
                                   else "legacy_fulfillment_or_missing_evidence" for row in rows if (row.get("result", {}).get("skills_breakdown") or {}).get("gate_status") in {"failed", "unknown"}))}


def _arithmetic(row):
    result = row.get("result", {})
    return {k: result.get(k) for k in ("scores", "overall_score", "grade", "coverage", "matched_skills", "missing_skills", "preferred_skill_coverage")}


def verify_reproduction(frozen, current, saved):
    results = {}
    for variant in ("v1_original", "full"):
        old = {r["pair_id"]: r for r in frozen["variants"][variant]}
        new = {r["pair_id"]: r for r in current["variants"][variant]}
        historical = {r["pair_id"]: r for r in saved[variant]["rows"]}
        keys = set(old) | set(new) | set(historical)
        failures = [key for key in sorted(keys) if any(mapping.get(key, {}).get("status") != "ok" for mapping in (old, new, historical))]
        drift = [key for key in sorted(keys - set(failures)) if _arithmetic(old[key]) != _arithmetic(new[key]) or _arithmetic(old[key]) != _arithmetic(historical[key])
                 or any(len({mapping[key].get(field) for mapping in (old, new, historical)}) != 1 for field in ("silver_grade", "pair_type"))]
        results[variant] = {"pairs": len(keys), "failures": failures, "score_or_grade_drift": drift}
    if any(entry["failures"] or entry["score_or_grade_drift"] for entry in results.values()):
        raise ValueError("Historical reproduction drift: " + json.dumps(results))
    return results


def run_benchmark(root: Path, *, as_of: date, corpus_path: Path, output: Path,
                  policies=("off", "score_only", "full"), pairs=None, require_population=True,
                  frozen_reproduction: Path | None = None, current_reproduction: Path | None = None,
                  preferred_policies=("separate",), reference_corpus_path: Path | None = None,
                  reference_benchmark: Path | None = None):
    if len(set(policies)) != len(policies) or not set(policies) <= {"off", "score_only", "full"} or not policies:
        raise ValueError("Choose unique depth policies")
    if len(set(preferred_policies)) != len(preferred_policies) or not preferred_policies or not set(preferred_policies) <= {"separate", "legacy_share"}:
        raise ValueError("Choose unique Preferred policies")
    if reference_benchmark is not None and reference_corpus_path is None:
        raise ValueError("Reference benchmark requires its reference corpus")
    corpus = read_json(corpus_path)
    validate_identity(corpus)
    if corpus["as_of_date"] != as_of.isoformat():
        raise ValueError("V3 corpus and benchmark dates differ")
    documents = {d["id"]: d for d in corpus["documents"]}
    reference = read_json(reference_corpus_path) if reference_corpus_path is not None else None
    if reference is not None:
        validate_identity(reference)
        if reference["as_of_date"] != as_of.isoformat() or {d["id"]: d["source"] for d in reference["documents"]} != {d["id"]: d["source"] for d in documents.values()}:
            raise ValueError("Reference corpus date or source identities differ")
        if reference["relations"] != corpus["relations"]:
            raise ValueError("Reference relations differ; annotation-only comparison is invalid")
    if require_population and Counter(d["kind"] for d in documents.values()) != {"resume": 88, "job": 361}:
        raise ValueError("Expected all 449 source documents")
    paths = {"pairs": root / "data/datasets/matching_pairs/benchmark_matching_pairs.json", "corpus": corpus_path,
             "cvs": root / "data/processed/structured_cvs/all_cvs.json", "jobs": root / "data/processed/structured_jobs/all_jobs_en.json",
             "frozen": root / "output/scorer_v3/frozen_v2/reproduced.json", "current_reproduction": root / "output/scorer_v3/current_reproduction.json",
             "frozen_manifest": root / "output/scorer_v3/frozen_v2/manifest.json", "frozen_inputs": root / "output/scorer_v3/frozen_v2/inputs_manifest.json",
             "labels_manifest": root / "output/scorer_v3/frozen_v2/labels_manifest.json"}
    if reference_corpus_path is not None:
        paths["reference_corpus"] = reference_corpus_path
    if reference_benchmark is not None:
        paths["reference_report"] = reference_benchmark / "report.json"
        paths["reference_v3_full"] = reference_benchmark / "v3_full.json"
    if frozen_reproduction is not None:
        paths["frozen"] = frozen_reproduction
    if current_reproduction is not None:
        paths["current_reproduction"] = current_reproduction
    selected = read_json(paths["pairs"]) if pairs is None else pairs
    counts = {"all": len(selected), "gemini": sum(p["pair_type"] != "heuristic_negative" for p in selected), "heuristic": sum(p["pair_type"] == "heuristic_negative" for p in selected)}
    if require_population and counts != {"all": 1672, "gemini": 1144, "heuristic": 528}:
        raise ValueError("Silver population changed")
    if len({(p["cv_id"], p["jd_id"]) for p in selected}) != len(selected) or not selected:
        raise ValueError("Nonempty unique pairs required")
    for document in documents.values():
        for path_key, checksum_key in (("text_path", "text_sha256"), ("source_path", "source_sha256")):
            path = root / document["source"][path_key]
            payload = path.read_text(encoding="utf-8").encode() if path_key == "text_path" else path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != document["source"][checksum_key]:
                raise ValueError("Source checksum changed: " + document["id"])
            paths[document["id"] + "_" + path_key] = path
    frozen = read_json(paths["frozen"])
    current = read_json(paths["current_reproduction"])
    saved = {variant: read_json(root / f"output/scorer_v2/benchmark/{variant}.json") for variant in ("v1_original", "full")}
    pairs_sha = hashlib.sha256(paths["pairs"].read_bytes()).hexdigest()
    if len({pairs_sha, frozen["pairs_sha256"], current["pairs_sha256"], saved["full"]["identity"]["inputs"]["pairs"]["sha256"]}) != 1:
        raise ValueError("Original label-file checksum changed")
    frozen_manifest = read_json(paths["frozen_manifest"])
    preserved_code = {entry["path"]: entry["sha256"] for entry in frozen_manifest["files"]}
    for relative, checksum in saved["full"]["identity"]["code"].items():
        if preserved_code.get(relative.replace("\\", "/")) != checksum:
            raise ValueError("Frozen v2 source differs from original benchmark: " + relative)
    reproduction = verify_reproduction(frozen, current, saved)
    table = load_relations(None, corpus)
    # Pending/rejected proposals cannot earn credit. Keep their complete checksum
    # in the experiment family, without repeatedly adapting/hashing them per pair.
    active_relations = [relation for relation in table.relations if relation.review_status == "approved"]
    configs = {("v3_" if preferred == "separate" else "v3_legacy_share_") + policy:
        ScorerConfig(version="ats-en-v3", depth_policy=policy, preferred_policy=preferred,
                     relations=active_relations, relation_version=table.version)
        for preferred in preferred_policies for policy in policies}
    if reference is not None:
        configs["v3_reference_legacy_share_full"] = ScorerConfig(version="ats-en-v3", depth_policy="full",
            preferred_policy="legacy_share", relations=active_relations, relation_version=table.version)
    variant_corpus = {name: (reference["identity_sha256"] if name == "v3_reference_legacy_share_full" else corpus["identity_sha256"]) for name in configs}
    code_paths = sorted({*root.glob("src/scoring/*.py"), *root.glob("src/parser/*.py"), *root.glob("scripts/ats_v3*.py"), root / "scripts/ats_score.py", root / "scripts/ats_silver_eval.py"})
    frozen_config = frozen["variants"]["full"][0]["result"]["config"]
    family = {"benchmark_version": VERSION, "as_of_date": as_of.isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "code": {p.relative_to(root).as_posix(): file_identity(p, root)["sha256"] for p in code_paths},
        "inputs": {k: file_identity(p, root) for k, p in sorted(paths.items())},
        "selected_pairs_sha256": canonical_hash(sorted(selected, key=lambda p: (p["cv_id"], p["jd_id"]))),
        "configs": {"frozen_v2": frozen_config, **{name: cfg.model_dump(mode="json") for name, cfg in configs.items()}},
        "variant_corpus_sha256": variant_corpus,
        "relations_sha256": canonical_hash(table.model_dump(mode="json")),
        "alias_policy_version": ALIAS_POLICY_VERSION, "approved_aliases_sha256": APPROVED_ALIASES_SHA256}
    identity = {**family, "family_sha256": canonical_hash(family)}
    marker = output / "benchmark_identity.json"
    if marker.exists():
        require_same_identity(identity, read_json(marker))
    originals = {"cvs": {d["cv_id"]: d for d in read_json(paths["cvs"])}, "jobs": {d["id"]: d for d in read_json(paths["jobs"])}}
    ids = {f"{p['cv_id']}::{p['jd_id']}" for p in selected}
    rows_by_variant = {"frozen_v2": [r for r in frozen["variants"]["full"] if r["pair_id"] in ids]}
    # Labels are copied verbatim, and never supplied as scorer facts.
    expected_labels = {f"{p['cv_id']}::{p['jd_id']}": p["grade"] for p in selected}
    if any(expected_labels[r["pair_id"]] != r["silver_grade"] for r in rows_by_variant["frozen_v2"]):
        raise ValueError("Frozen labels differ")
    for name, config in configs.items():
        selected_documents = {d["id"]: d for d in reference["documents"]} if name == "v3_reference_legacy_share_full" else documents
        rows_by_variant[name] = [score_stored_pair(pair, name, originals, selected_documents, config, as_of,
            scorer=score_pair_v3) for pair in selected]
    reference_reproduction = None
    if reference_benchmark is not None:
        saved_reference = read_json(paths["reference_v3_full"])
        saved_report = read_json(paths["reference_report"])
        if saved_reference["identity"] != saved_report["identity"] or saved_report["as_of_date"] != as_of.isoformat():
            raise ValueError("Reference benchmark identities or evaluation date differ")
        if (saved_report["identity"]["inputs"]["corpus"]["sha256"] != file_identity(reference_corpus_path, root)["sha256"]
                or saved_report["identity"]["inputs"]["pairs"]["sha256"] != pairs_sha
                or saved_report["identity"]["relations_sha256"] != family["relations_sha256"]):
            raise ValueError("Reference corpus, labels or relations do not match the saved benchmark")
        old_rows = {row["pair_id"]: row for row in saved_reference["rows"]}
        drift = [row["pair_id"] for row in rows_by_variant["v3_reference_legacy_share_full"]
                 if row["status"] != "ok" or old_rows.get(row["pair_id"], {}).get("status") != "ok"
                 or _arithmetic(row) != _arithmetic(old_rows[row["pair_id"]])]
        reference_reproduction = {"benchmark_identity": saved_report["identity"]["family_sha256"],
                                  "pairs": len(selected), "arithmetic_drift": drift}
        if drift:
            raise ValueError("Legacy-share reference arithmetic drift: " + json.dumps(drift))
    changes = corpus["changes"]
    coverage = {kind: len({change.get("item_id") for change in changes if change.get("review_kind") == kind and change.get("item_id")}) for kind in ("direct_human", "rule_derived")}
    from .v3_corpus import items
    all_items = items(corpus)
    skill_item_count = sum(document is not None for document, _, _, _ in all_items.values())
    coverage.update(total_items=len(all_items), skill_items=skill_item_count,
        direct_item_fraction=coverage["direct_human"] / len(all_items), rule_derived_item_fraction=coverage["rule_derived"] / len(all_items))
    for kind in ("direct_human", "rule_derived"):
        reviewed_documents = {change["document_id"] for change in changes if change.get("review_kind") == kind and change.get("document_id")}
        coverage[kind + "_documents"] = len(reviewed_documents)
        coverage[kind + "_document_fraction"] = len(reviewed_documents) / len(documents)
    cv_details = [detail for d in documents.values() if d["kind"] == "resume" for group in d["envelope"]["document"]["skills"] for detail in group["details"]]
    options = [option for d in documents.values() if d["kind"] == "job" for req in d["envelope"]["document"]["skills"]["requirements"] for option in req["option_details"]]
    source_depth = {"cv_details": len(cv_details), "cv_explicit_depth": sum(d["proficiency"] != "normal" for d in cv_details),
        "cv_unknown_depth": sum(d["proficiency"] == "normal" for d in cv_details), "jd_options": len(options),
        "jd_explicit_minima": sum(o["minimum_proficiency"] != "normal" for o in options)}
    source_depth["cv_unknown_fraction"] = source_depth["cv_unknown_depth"] / len(cv_details)
    source_depth["jd_explicit_fraction"] = source_depth["jd_explicit_minima"] / len(options)
    coverage["whole_envelopes_approved"] = sum(d["full_envelope_review_status"] == "approved" for d in documents.values())
    coverage["documents"] = len(documents)
    nomination_path = root / "output/extraction_pilot/disagreement_nominations.json"
    nominations = read_json(nomination_path).get("cases", []) if nomination_path.exists() else []
    report = {"version": VERSION, "identity": identity, "as_of_date": as_of.isoformat(), "population": counts, "api_requests": 0,
        "scope": "Diagnostics on previously inspected silver data; not independent test accuracy or hiring outcomes.",
        "reproduction": reproduction, "reference_reproduction": reference_reproduction,
        "review_coverage": coverage, "pending_samples": corpus["pending_samples"],
        "original_label_checksum_verified": True, "frozen_source_matches_original_benchmark": True,
        "source_depth_coverage": source_depth,
        "relations": {"review_status": dict(Counter(r.review_status for r in table.relations)), "active": len(active_relations), "full_table_sha256": canonical_hash(table.model_dump(mode="json"))},
        "historical_disagreements": {"expected": 24, "available": len(nominations)}, "variants": {},
        "limitations": ["Quarter-step depth credits are project design choices, not empirically calibrated proficiency distances.",
            "Item approval and rule-derived changes do not establish whole-envelope approval.", "Missing/incomplete skill annotations retain neutral scores and unknown gates.",
            "Explicit JD depth may be sparse; missing minimum is not inferred from a title or years.", "Skill specialization years are recorded but unscored.",
             "Latency includes Pydantic adaptation and local scoring; frozen timings were measured in the same session but a separate process.",
             "Preferred policy effects are masked by neutral scoring in incomplete annotations; affected-source counts and numeric changes differ.",
             "Preferred-only fallback supplies base targets, but no proficiency bonus or Core gate.",
             "Core 1.0 / Standard 0.6 / Preferred 0.4 are official, untuned project choices (D3-28); no vector was selected by silver-label agreement.",
             "D3-31 aliases require human approval; embeddings only nominate proposals. Approved relations never enter the Core gate."]}
    previous = None
    for name, rows in rows_by_variant.items():
        indexed = {r["pair_id"]: r for r in rows}
        entry = {"all": summarize(rows), "gemini_primary": summarize([r for r in rows if r["pair_type"] != "heuristic_negative"]),
            "heuristic_negatives": summarize([r for r in rows if r["pair_type"] == "heuristic_negative"]), "depth": depth_summary(rows),
            "by_cv_domain": {str(d): summarize([r for r in rows if r["cv_domain"] == d]) for d in sorted({r["cv_domain"] for r in rows}, key=str)},
            "by_jd_domain": {str(d): summarize([r for r in rows if r["jd_domain"] == d]) for d in sorted({r["jd_domain"] for r in rows}, key=str)},
            "historical_disagreements": [{"pair_id": f"{case['original_comparison']['cv_id']}::{case['original_comparison']['jd_id']}",
                "selection_reason": case.get("selection_reason"), "result": indexed.get(f"{case['original_comparison']['cv_id']}::{case['original_comparison']['jd_id']}")} for case in nominations]}
        if previous:
            entry["change_from_previous"] = {"variant": previous, **grade_changes(rows_by_variant[previous], rows)}
        report["variants"][name] = entry
        previous = name
    preferred_jds = {d["id"] for d in documents.values() if d["kind"] == "job" and any(r["importance"] == "preferred" for r in d["envelope"]["document"]["skills"]["requirements"])}
    mixed_jds = {d["id"] for d in documents.values() if d["id"] in preferred_jds and any(r["importance"] != "preferred" for r in d["envelope"]["document"]["skills"]["requirements"])}
    report["preferred_policy_coverage"] = {"jds_with_preferred": len(preferred_jds), "mixed_jds": len(mixed_jds),
        "preferred_only_jds": len(preferred_jds - mixed_jds),
        "pairs_with_preferred": sum(p["jd_id"] in preferred_jds for p in selected),
        "mixed_pairs": sum(p["jd_id"] in mixed_jds for p in selected),
        "preferred_only_pairs": sum(p["jd_id"] in preferred_jds - mixed_jds for p in selected)}
    report["isolated_comparisons"] = {}
    for effect, before, after in (("importance_correction", "v3_reference_legacy_share_full", "v3_legacy_share_full"),
                                  ("preferred_policy", "v3_legacy_share_full", "v3_full"),
                                  ("combined", "v3_reference_legacy_share_full", "v3_full")):
        if before in rows_by_variant and after in rows_by_variant:
            old = {r["pair_id"]: r for r in rows_by_variant[before]}
            changes = [r for r in rows_by_variant[after] if r["status"] == "ok" and old[r["pair_id"]]["status"] == "ok"
                       and (r["result"]["overall_score"] != old[r["pair_id"]]["result"]["overall_score"] or r["result"]["scores"] != old[r["pair_id"]]["result"]["scores"])]
            report["isolated_comparisons"][effect] = {"before": before, "after": after,
                **grade_changes(rows_by_variant[before], rows_by_variant[after]),
                "changed_jds": sorted({r["jd_id"] for r in changes}),
                "changed_pair_ids": [r["pair_id"] for r in changes],
                "by_population": {label: grade_changes([r for r in rows_by_variant[before] if (r["pair_type"] == "heuristic_negative") == heuristic],
                                                       [r for r in rows_by_variant[after] if (r["pair_type"] == "heuristic_negative") == heuristic])
                                  for label, heuristic in (("gemini", False), ("heuristic_negatives", True))}}
    report["status"] = "failed" if any(e["all"]["failed"] for e in report["variants"].values()) else "complete"
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    for name, rows in rows_by_variant.items():
        (output / (name + ".json")).write_text(json.dumps({"identity": identity, "variant": name, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ATS v3 depth diagnostic", "", report["scope"], "", f"Date: {as_of}. Identity: `{identity['family_sha256']}`. Status: {report['status']}.",
        "", f"Population: {counts['gemini']} Gemini pairs and {counts['heuristic']} heuristic negatives, reported separately.", "", "| Variant | Gemini macro-F1 | Gemini balanced accuracy | Heuristic macro-F1 | Failures | Mean latency ms |", "|---|---:|---:|---:|---:|---:|"]
    for name, entry in report["variants"].items():
        gm, hm = entry["gemini_primary"].get("metrics"), entry["heuristic_negatives"].get("metrics")
        fmt = lambda metrics, key: f"{metrics[key]:.4f}" if metrics else "unavailable"
        lines.append(f"| {name} | {fmt(gm, 'macro_f1')} | {fmt(gm, 'balanced_accuracy')} | {fmt(hm, 'macro_f1')} | {entry['all']['failed']} | {entry['all']['latency_ms'].get('mean', 0):.3f} |")
    lines += ["", "Per-grade metrics, confusion matrices, domains, 24 historical cases, changes and depth/gate coverage are saved in report.json.", "", "## Review coverage", "", "```json", json.dumps(coverage, indent=2), "```", "", "## Limitations", ""]
    lines += ["- " + limitation for limitation in report["limitations"]]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
