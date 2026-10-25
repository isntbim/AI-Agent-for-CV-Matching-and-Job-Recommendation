"""Experiment-only two-layer weight sensitivity; runtime defaults stay unchanged.

This sequential, single-process harness temporarily replaces v3's parameter
dictionaries, restores the original objects even on failure, and binds the
parameters independently of the unchanged production ScorerConfig.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from itertools import combinations, product
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

from . import skills_v3
from .models import ScorerConfig
from .runtime_v3 import score_pair_v3
from .v2_benchmark import (
    canonical_hash, file_identity, load_relations, read_json,
    require_same_identity, score_stored_pair, summarize,
)
from .v3_corpus import validate_identity


VERSION = "ats-v3-weight-sensitivity-v1"
GROUPS = ("core", "standard", "preferred")
DEFAULT_WEIGHTS = {"core": 1.0, "standard": 0.6, "preferred": 0.4}
DEFAULT_SHARES = {"core": 0.7, "standard": 0.2, "preferred": 0.1}
SHARE_VECTORS = {
    "current": DEFAULT_SHARES,
    "equal": dict(zip(GROUPS, (1 / 3, 1 / 3, 1 / 3))),
    "rank_sum": dict(zip(GROUPS, (3 / 6, 2 / 6, 1 / 6))),
    "roc": dict(zip(GROUPS, (11 / 18, 5 / 18, 2 / 18))),
}
RAW_WEIGHT_VECTORS = {
    "current": DEFAULT_WEIGHTS,
    # Complete the historical two-tier note before normalizing the entire vector.
    "five_two": {"core": 5.0, "standard": 2.0, "preferred": 0.4},
    "roc": dict(zip(GROUPS, (11 / 18, 5 / 18, 2 / 18))),
}
BASELINE = "shares_current__weights_current"
_active = False


def specifications():
    """Prespecified full factorial, unrelated to silver labels or their grades."""
    result = []
    for share_name, weight_name in product(SHARE_VECTORS, RAW_WEIGHT_VECTORS):
        raw = RAW_WEIGHT_VECTORS[weight_name]
        item_weights = {group: raw[group] / raw["core"] for group in GROUPS}
        result.append({
            "name": f"shares_{share_name}__weights_{weight_name}",
            "shares_name": share_name, "weights_name": weight_name,
            "group_shares": dict(SHARE_VECTORS[share_name]),
            "raw_item_weights": dict(raw), "item_weights": item_weights,
            "normalization": "divide every item weight by the Core weight",
            "five_two_completion": "Preferred fixed at 0.4 before Core normalization",
        })
    return result


def parameter_config(config, specification):
    if (config.version, config.variant, config.depth_policy, config.preferred_policy) != (
        "ats-en-v3", "full", "full", "separate"
    ):
        raise ValueError("Sensitivity requires full v3, full depth and Preferred separate")
    return {"scorer": config.model_dump(mode="json"), "parameters": specification,
            "experiment_version": VERSION}


@contextmanager
def parameter_scope(specification):
    """Use only in this sequential CLI process; reject nesting and unsafe weights."""
    global _active
    weights, shares = specification["item_weights"], specification["group_shares"]
    for values in (weights, shares):
        if set(values) != set(GROUPS) or any(
            not math.isfinite(v) or not 0 < v <= 1 for v in values.values()
        ):
            raise ValueError("Use three finite positive, Core-normalized parameters <= 1")
    if _active:
        raise RuntimeError("Weight experiment scopes cannot overlap or nest")
    old_weights, old_shares = skills_v3.GROUP_WEIGHTS, skills_v3.GROUP_SHARES
    _active = True
    try:
        skills_v3.GROUP_WEIGHTS = dict(weights)
        skills_v3.GROUP_SHARES = dict(shares)
        yield
    finally:
        skills_v3.GROUP_WEIGHTS, skills_v3.GROUP_SHARES = old_weights, old_shares
        _active = False


def _close(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
    return a == b


def allocation(row):
    breakdown = row.get("result", {}).get("skills_breakdown") or {}
    return sorted((r["id"], r.get("candidate_skill"), r.get("selected_option"),
                   r.get("match_type"), r.get("fulfillment_factor"))
                  for r in breakdown.get("requirements", []))


def arithmetic(row):
    result = row.get("result", {})
    breakdown = result.get("skills_breakdown") or {}
    return {"status": row["status"], "scores": result.get("scores"),
            "overall_score": result.get("overall_score"), "grade": result.get("grade"),
            "coverage": result.get("coverage"), "dimensions": result.get("dimensions"),
            "allocation": allocation(row),
            "skills": {key: breakdown.get(key) for key in (
                "score", "base_skills", "bonus", "skills_before_gate", "skills_after_gate",
                "gate_status", "core_gate_fulfillment", "force_mismatch", "strong_blocked")}}


def changed_rows(reference, candidate):
    old = {row["pair_id"]: row for row in reference}
    if set(old) != {row["pair_id"] for row in candidate}:
        raise ValueError("Comparison pair identities differ")
    changes = []
    for row in candidate:
        before = old[row["pair_id"]]
        if before["status"] != "ok" or row["status"] != "ok":
            continue
        a, b = before["result"], row["result"]
        sa, sb = a.get("skills_breakdown") or {}, b.get("skills_breakdown") or {}
        flags = {
            "overall_score": not _close(a["overall_score"], b["overall_score"]),
            "skills_score": not _close(a["scores"]["skills"], b["scores"]["skills"]),
            "grade": a["grade"] != b["grade"],
            "allocation": allocation(before) != allocation(row),
            "base_skills": not _close(sa.get("base_skills"), sb.get("base_skills")),
            "bonus": not _close(sa.get("bonus"), sb.get("bonus")),
            "gate_fulfillment": not _close(sa.get("core_gate_fulfillment"), sb.get("core_gate_fulfillment")),
            "gate_status": sa.get("gate_status") != sb.get("gate_status"),
            "force_mismatch": sa.get("force_mismatch") != sb.get("force_mismatch"),
        }
        flags["rounding_only"] = flags["skills_score"] and not any(flags[key] for key in (
            "allocation", "base_skills", "bonus", "gate_fulfillment", "gate_status", "force_mismatch"))
        if any(flags.values()):
            changes.append({"pair_id": row["pair_id"], "cv_id": row["cv_id"],
                "jd_id": row["jd_id"], "pair_type": row["pair_type"], "flags": flags,
                "before": {"overall": a["overall_score"], "skills": a["scores"]["skills"],
                           "grade": a["grade"], "bonus": sa.get("bonus"),
                           "base_skills": sa.get("base_skills"), "after_gate": sa.get("skills_after_gate"),
                           "gate": sa.get("gate_status"), "allocation": allocation(before)},
                "after": {"overall": b["overall_score"], "skills": b["scores"]["skills"],
                          "grade": b["grade"], "bonus": sb.get("bonus"),
                          "base_skills": sb.get("base_skills"), "after_gate": sb.get("skills_after_gate"),
                          "gate": sb.get("gate_status"), "allocation": allocation(row)}})
    return changes


def counts(changes):
    keys = ("overall_score", "skills_score", "grade", "allocation", "base_skills", "bonus",
            "gate_fulfillment", "gate_status", "force_mismatch", "rounding_only")
    return {**{key: sum(c["flags"][key] for c in changes) for key in keys},
            "changed_pairs": len(changes), "affected_jds": len({c["jd_id"] for c in changes})}


def populations(rows):
    return {"all": rows,
            "gemini_primary": [r for r in rows if r["pair_type"] != "heuristic_negative"],
            "heuristic_negatives": [r for r in rows if r["pair_type"] == "heuristic_negative"]}


def ranking_changes(reference, candidate, score="overall"):
    """Strict reversals and tie changes between candidates for the same JD only."""
    old = {r["pair_id"]: r for r in reference if r["status"] == "ok"}
    jobs = defaultdict(list)
    for row in candidate:
        if row["status"] == "ok" and row["pair_id"] in old:
            jobs[row["jd_id"]].append(row)
    def value(row):
        result = row["result"]
        return result["overall_score"] if score == "overall" else result["scores"]["skills"]
    changes, comparisons = [], 0
    for jd_id, rows in sorted(jobs.items()):
        for a, b in combinations(sorted(rows, key=lambda r: r["pair_id"]), 2):
            comparisons += 1
            previous = value(old[a["pair_id"]]) - value(old[b["pair_id"]])
            current = value(a) - value(b)
            kind = ("strict_reversal" if previous * current < 0 else
                    "introduced_tie" if previous != 0 and current == 0 else
                    "broken_tie" if previous == 0 and current != 0 else None)
            if kind:
                changes.append({"jd_id": jd_id, "candidate_a": a["cv_id"],
                    "candidate_b": b["cv_id"], "pair_a": a["pair_id"], "pair_b": b["pair_id"],
                    "pair_type_a": a["pair_type"], "pair_type_b": b["pair_type"],
                    "mixed_population": (a["pair_type"] == "heuristic_negative") != (b["pair_type"] == "heuristic_negative"),
                    "kind": kind, "before_difference": previous, "after_difference": current})
    return {"scope": "within-JD candidate ordering; observed benchmark candidates only",
            "score": score, "eligible_jds": sum(len(r) > 1 for r in jobs.values()),
            "comparisons": comparisons,
            **{kind: sum(c["kind"] == kind for c in changes)
               for kind in ("strict_reversal", "introduced_tie", "broken_tie")}, "changes": changes}


def score_spreads(rows_by_variant):
    maps = {name: {r["pair_id"]: r for r in rows} for name, rows in rows_by_variant.items()}
    pair_sets = [set(rows) for rows in maps.values()]
    if not pair_sets or any(ids != pair_sets[0] for ids in pair_sets):
        raise ValueError("Spread pair identities differ")
    records = []
    for pair_id in sorted(pair_sets[0]):
        rows = [rows[pair_id] for rows in maps.values()]
        if any(r["status"] != "ok" for r in rows):
            continue
        totals = [r["result"]["overall_score"] for r in rows]
        skills = [r["result"]["scores"]["skills"] for r in rows]
        grades = sorted({r["result"]["grade"] for r in rows})
        records.append({"pair_id": pair_id, "jd_id": rows[0]["jd_id"],
            "pair_type": rows[0]["pair_type"], "overall_min": min(totals),
            "overall_max": max(totals), "overall_spread": round(max(totals)-min(totals), 10),
            "skills_min": min(skills), "skills_max": max(skills),
            "skills_spread": round(max(skills)-min(skills), 10), "grades": grades,
            "grade_unstable": len(grades) > 1})
    summaries = {}
    for population, items in populations(records).items():
        summaries[population] = {"pairs": len(items),
            "score_sensitive_pairs": sum(r["overall_spread"] > 0 for r in items),
            "grade_unstable_pairs": sum(r["grade_unstable"] for r in items),
            **{key: {"min": min((r[key] for r in items), default=None),
                     "max": max((r[key] for r in items), default=None),
                     "mean": statistics.mean(r[key] for r in items) if items else None}
               for key in ("overall_spread", "skills_spread")}}
    return {"summary": summaries, "pairs": records}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def run_status(variant_reports, non_skill_drift, defaults_restored):
    """The shared benchmark helper returns a failure list, not a failure count."""
    return "ok" if not non_skill_drift and defaults_restored and all(
        not report["all"]["failures"] for report in variant_reports.values()) else "failed"


def historical_rows(rows, cases):
    """Keep every original nomination, including pairs outside the fixed labels."""
    indexed = {row["pair_id"]: row for row in rows}
    result = []
    for case in cases:
        original = case["original_comparison"]
        pair_id = f"{original['cv_id']}::{original['jd_id']}"
        result.append({"pair_id": pair_id, "selection_reason": case.get("selection_reason"),
                       "in_fixed_labels": pair_id in indexed, "result": indexed.get(pair_id)})
    return result


def run_sweep(root: Path, *, as_of: date, corpus_path: Path, reference_path: Path,
              output: Path, progress=None):
    """Run the fixed 12 cells; grades are outcomes, never a parameter-selection rule."""
    if output.exists():
        raise ValueError("Sensitivity output must be a fresh directory")
    if skills_v3.GROUP_WEIGHTS != DEFAULT_WEIGHTS or skills_v3.GROUP_SHARES != DEFAULT_SHARES:
        raise ValueError("Production defaults differ from the prespecified reference")
    corpus = read_json(corpus_path)
    validate_identity(corpus)
    reference_report = read_json(reference_path / "report.json")
    reference_full = read_json(reference_path / "v3_full.json")
    reference_identity = require_same_identity(reference_report, reference_full)
    if corpus["as_of_date"] != as_of.isoformat() or reference_report["as_of_date"] != as_of.isoformat():
        raise ValueError("Corpus, reference and sweep evaluation dates differ")
    inputs = {}
    for name, identity in reference_report["identity"]["inputs"].items():
        actual = file_identity(root / identity["path"], root)
        if actual["sha256"] != identity["sha256"]:
            raise ValueError("Reference input checksum changed: " + name)
        inputs[name] = actual
    if file_identity(corpus_path, root)["sha256"] != inputs["corpus"]["sha256"]:
        raise ValueError("Sweep corpus is not the saved reference corpus")
    documents = {d["id"]: d for d in corpus["documents"]}
    if Counter(d["kind"] for d in documents.values()) != {"resume": 88, "job": 361}:
        raise ValueError("Expected the unchanged 449 source documents")
    for document in documents.values():
        source = document["source"]
        for label, key in (("source", "source_path"), ("text", "text_path")):
            path = root / source[key]
            payload = path.read_text(encoding="utf-8").encode() if label == "text" else path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != source[label + "_sha256"]:
                raise ValueError("Document checksum changed: " + str(path))
    pairs = read_json(root / inputs["pairs"]["path"])
    if len(pairs) != 1672 or Counter(p["pair_type"] == "heuristic_negative" for p in pairs) != {False: 1144, True: 528}:
        raise ValueError("Expected unchanged 1672 / 1144 / 528 silver populations")
    relations = load_relations(None, corpus)
    if canonical_hash(relations.model_dump(mode="json")) != reference_report["identity"]["relations_sha256"]:
        raise ValueError("Reference relation checksum differs")
    config = ScorerConfig(version="ats-en-v3", variant="full", depth_policy="full",
        preferred_policy="separate", relations=[r for r in relations.relations if r.review_status == "approved"],
        relation_version=relations.version)
    if config.model_dump(mode="json") != reference_report["identity"]["configs"]["v3_full"]:
        raise ValueError("Full scorer configuration differs from the reference")
    nominations_path = root / "output/extraction_pilot/disagreement_nominations.json"
    nominations = read_json(nominations_path).get("cases", [])
    if len(nominations) != 24:
        raise ValueError("Expected the unchanged 24 historical disagreement nominations")
    inputs["historical_nominations"] = file_identity(nominations_path, root)
    specs = specifications()
    code_paths = sorted({*reference_report["identity"]["code"],
        "src/scoring/v3_weight_sensitivity.py", "scripts/ats_v3_weight_sensitivity.py"})
    code = {path: file_identity(root / path, root)["sha256"] for path in code_paths}
    protected_drift = [path for path, checksum in reference_report["identity"]["code"].items() if code[path] != checksum]
    if protected_drift:
        raise ValueError("Existing scorer code changed: " + ", ".join(protected_drift))
    family = {"experiment_version": VERSION, "as_of_date": as_of.isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "code": code, "inputs": inputs, "corpus_sha256": corpus["identity_sha256"],
        "relations_sha256": canonical_hash(relations.model_dump(mode="json")),
        "reference_benchmark": reference_identity,
        "reference_artifacts": {name: file_identity(reference_path / name, root) for name in ("report.json", "v3_full.json")},
        "configurations": {s["name"]: parameter_config(config, s) for s in specs}}
    family["family_sha256"] = canonical_hash(family)
    output.mkdir(parents=True, exist_ok=False)
    (output / "variants").mkdir()
    _write(output / "prespecified_parameters.json", {"identity": family, "variants": specs,
        "fixed": {"default_changes": False, "label_tuning": False, "api_requests": 0,
                  "mode": "full", "preferred_policy": "separate"},
        "completion": "raw 5/2/0.4 -> Core-normalized 1/0.4/0.08; no defaults edited"})
    rows_by_variant, variant_reports = {}, {}
    started = datetime.now(timezone.utc).isoformat()
    for specification in specs:
        name = specification["name"]
        if progress:
            progress("Scoring " + name)
        parameter_hash = canonical_hash(family["configurations"][name])
        with parameter_scope(specification):
            rows = [score_stored_pair(pair, name, {}, documents, config, as_of,
                scorer=score_pair_v3) for pair in pairs]
        for row in rows:
            row["weight_config_sha256"] = parameter_hash
            if row["status"] == "ok":
                row["result"]["skills_breakdown"]["identity"]["weight_config_sha256"] = parameter_hash
        if name == BASELINE:
            previous = {row["pair_id"]: row for row in reference_full["rows"]}
            if set(previous) != {row["pair_id"] for row in rows}:
                raise ValueError("Reference pair identities differ")
            drift = [r["pair_id"] for r in rows if arithmetic(r) != arithmetic(previous[r["pair_id"]])]
            if drift:
                _write(output / "reference_failure.json", {"drift": drift})
                raise ValueError("Default cell does not reproduce the reference")
        rows_by_variant[name] = rows
        _write(output / "variants" / (name + ".json"), {"identity": family,
            "variant": name, "weight_config_sha256": parameter_hash,
            "parameters": specification, "rows": rows})
        variant_reports[name] = {population: summarize(items) for population, items in populations(rows).items()}
        variant_reports[name]["by_cv_domain"] = {domain: summarize([r for r in rows if r["cv_domain"] == domain])
            for domain in sorted({r["cv_domain"] for r in rows})}
        variant_reports[name]["by_jd_domain"] = {domain: summarize([r for r in rows if r["jd_domain"] == domain])
            for domain in sorted({r["jd_domain"] for r in rows})}
    comparisons = {}
    for name, rows in rows_by_variant.items():
        changes = changed_rows(rows_by_variant[BASELINE], rows)
        comparison = {population: counts(items) for population, items in populations(changes).items()}
        comparison["ranking"] = {population: {score: ranking_changes(populations(rows_by_variant[BASELINE])[population], items, score)
            for score in ("overall", "skills")} for population, items in populations(rows).items()}
        comparisons[name] = comparison
        _write(output / (name + "_changes.json"), {"identity": family, "changes": changes})
    pairwise = {}
    for a, b in combinations(rows_by_variant, 2):
        changes = changed_rows(rows_by_variant[a], rows_by_variant[b])
        pairwise[a + "::" + b] = {"changes": {p: counts(r) for p, r in populations(changes).items()},
            "ranking_overall": {p: ranking_changes(populations(rows_by_variant[a])[p], r)
                for p, r in populations(rows_by_variant[b]).items()}}
    spreads = score_spreads(rows_by_variant)
    _write(output / "score_spreads.json", {"identity": family, **spreads})
    _write(output / "historical_disagreements.json", {"identity": family,
        "expected": 24, "available": len(nominations),
        "variants": {name: historical_rows(rows, nominations)
                     for name, rows in rows_by_variant.items()}})
    non_skill_drift = []
    original = {r["pair_id"]: r for r in rows_by_variant[BASELINE]}
    for name, rows in rows_by_variant.items():
        for row in rows:
            if row["status"] != "ok":
                continue
            a, b = original[row["pair_id"]]["result"], row["result"]
            if any(a["scores"][key] != b["scores"][key] for key in a["scores"] if key != "skills") or a["coverage"] != b["coverage"]:
                non_skill_drift.append({"variant": name, "pair_id": row["pair_id"]})
    report = {"identity": family, "version": VERSION, "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "as_of_date": as_of.isoformat(),
        "population": {p: len(r) for p, r in populations(rows_by_variant[BASELINE]).items()},
        "api_requests": 0, "defaults_changed": False, "label_tuning": False,
        "reference_reproduction": {"benchmark": reference_identity, "pairs": len(pairs), "arithmetic_drift": 0},
        "variants": variant_reports, "comparisons_from_current": comparisons,
        "pairwise": pairwise, "score_spread": spreads["summary"],
        "non_skill_drift": non_skill_drift,
        "defaults_restored": skills_v3.GROUP_WEIGHTS == DEFAULT_WEIGHTS and skills_v3.GROUP_SHARES == DEFAULT_SHARES,
        "limitations": ["Previously inspected silver labels; no independent efficacy or hiring validity claim.",
            "Only observed candidates within each JD enter ranking comparisons; ties are separate from reversals.",
            "Preferred-separate excludes the third share for mixed JDs; Preferred-only fallback is retained.",
            "No vector is selected by silver-label agreement. Constants remain untuned project choices.",
            "Allocation integer rounding remains unchanged; normalization preserves its <=1 weight bound.",
            "Latency is descriptive for fixed sequential ordering, not a controlled performance study."]}
    report["status"] = run_status(variant_reports, non_skill_drift, report["defaults_restored"])
    _write(output / "report.json", report)
    return report
