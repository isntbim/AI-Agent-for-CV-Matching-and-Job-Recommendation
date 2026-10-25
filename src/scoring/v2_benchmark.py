"""Reproducible offline diagnostics against unchanged, previously seen silver labels.

This module never predicts annotation labels or calls a model. Every attempted
pair remains in the artifact, including failures; empty subsets have no metrics.
"""

import hashlib
import json
import statistics
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable

from src.parser.skill_v2 import SkillRelationTable

from .evaluation import classification_metrics
from .models import JobInput, ResumeInput, ScorerConfig
from .rules import score_pair
from .silver import job_from_record, resume_from_record

VARIANTS = ("v1_original", "v1_reviewed", "equal_groups", "group_weights", "semantic", "proficiency", "full")
BENCHMARK_VERSION = "ats-v2-silver-diagnostic-v1"
EXPECTED_POPULATION = {"all": 1672, "gemini": 1144, "heuristic": 528}


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_identity(path: Path, root: Path | None = None) -> dict:
    return {"path": str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(path: Path, as_of: date, *, require_population: bool = True) -> tuple[dict, dict]:
    corpus = read_json(path)
    if corpus.get("version") != "ats-source-corpus-v2":
        raise ValueError("Expected ats-source-corpus-v2 annotation corpus")
    if corpus.get("as_of_date") != as_of.isoformat():
        raise ValueError("Annotation corpus and benchmark must use the same fixed as-of date")
    expected_identity = canonical_hash({key: value for key, value in corpus.items() if key not in {"identity_sha256", "review_history"}})
    if corpus.get("identity_sha256") != expected_identity:
        raise ValueError("Annotation corpus identity checksum is stale")
    documents = corpus.get("documents", [])
    if not isinstance(documents, list) or not documents:
        raise ValueError("Annotation corpus requires nonempty documents")
    by_id = {document["id"]: document for document in documents}
    if len(by_id) != len(documents):
        raise ValueError("Annotation corpus has duplicate document IDs")
    kinds = Counter(document["kind"] for document in documents)
    if require_population and kinds != {"resume": 88, "job": 361}:
        raise ValueError(f"Expected all 88 CV + 361 JD annotations, found {dict(kinds)}")
    for document in documents:
        if document["kind"] not in {"resume", "job"} or not isinstance(document.get("source"), dict):
            raise ValueError(f"Missing document kind/provenance: {document['id']}")
        if not document.get("source", {}).get("source_sha256"):
            raise ValueError(f"Missing source checksum: {document['id']}")
        if not document.get("checksum_sha256") or not document.get("review_status"):
            raise ValueError(f"Missing annotation checksum/review status: {document['id']}")
        if not document.get("before") or not document.get("envelope"):
            raise ValueError(f"Missing original/reviewed envelope: {document['id']}")
        if document["checksum_sha256"] != canonical_hash(document["envelope"]):
            raise ValueError(f"Annotation envelope checksum is stale: {document['id']}")
    return corpus, by_id


def load_relations(path: Path | None, corpus: dict | None = None) -> SkillRelationTable:
    payload = read_json(path) if path else (corpus or {}).get("relations", {"version": "empty-v1", "relations": []})
    return SkillRelationTable.model_validate(payload)


def variant_config(variant: str, table: SkillRelationTable, allow_provisional: bool = False) -> ScorerConfig:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown benchmark variant: {variant}")
    if variant.startswith("v1_"):
        return ScorerConfig()
    usable = [relation for relation in table.relations if relation.review_status == "approved"
              or (allow_provisional and relation.review_status == "pending")]
    return ScorerConfig(version="ats-en-v2", variant=variant, relations=usable,
                        relation_version=table.version, allow_provisional_relations=allow_provisional)


def benchmark_identity(root: Path, paths: dict[str, Path], as_of: date,
                       table: SkillRelationTable, allow_provisional: bool,
                       selected_pairs: list[dict] | None = None) -> dict:
    """One family identity for comparable ablations, plus each exact configuration.

    Hash working-tree code, not only Git HEAD, so uncommitted scorer edits cannot
    be mistaken for the same implementation. Inputs include original silver
    labels, annotations, source manifests and relation-table review decisions.
    """
    code_paths = sorted({*root.glob("src/scoring/*.py"), *root.glob("src/parser/*.py"),
                         *root.glob("scripts/ats*.py")})
    code = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    configs = {variant: variant_config(variant, table, allow_provisional).model_dump(mode="json") for variant in VARIANTS}
    family = {"benchmark_version": BENCHMARK_VERSION, "as_of_date": as_of.isoformat(), "code": code,
              "inputs": {key: file_identity(path, root) for key, path in sorted(paths.items())},
              "relations": table.model_dump(mode="json"), "allow_provisional_relations": allow_provisional,
              "configs": configs}
    if selected_pairs is not None:
        family["selected_pairs_sha256"] = canonical_hash(sorted(selected_pairs, key=lambda pair: (pair["cv_id"], pair["jd_id"])))
    return {**family, "family_sha256": canonical_hash(family)}


def require_same_identity(*artifacts: dict) -> str:
    if not artifacts:
        raise ValueError("No artifacts to combine")
    identities = [artifact.get("identity", artifact) for artifact in artifacts]
    hashes = {identity.get("family_sha256") for identity in identities}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("Refusing to combine artifacts with different benchmark identities")
    for identity in identities:
        family = {key: value for key, value in identity.items() if key != "family_sha256"}
        if canonical_hash(family) != identity["family_sha256"]:
            raise ValueError("Benchmark identity checksum is inconsistent")
    return identities[0]["family_sha256"]


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "std": statistics.pstdev(values), "min": values[0], "max": values[-1],
            "p25": values[int((len(values) - 1) * .25)], "p75": values[int((len(values) - 1) * .75)],
            "p95": values[int((len(values) - 1) * .95)]}


def _is_heuristic(pair: dict) -> bool:
    return pair["pair_type"] == "heuristic_negative"


def score_stored_pair(pair: dict, variant: str, originals: dict, annotations: dict,
                      config: ScorerConfig, as_of: date, scorer: Callable = score_pair) -> dict:
    """Keep labels verbatim, supplying no label/domain metadata as scorer facts."""
    cv_id, jd_id = pair["cv_id"], pair["jd_id"]
    started = time.perf_counter()
    row = {"pair_id": f"{cv_id}::{jd_id}", "cv_id": cv_id, "jd_id": jd_id,
           "pair_type": pair["pair_type"], "silver_grade": pair["grade"],
           "silver_overall_score": pair.get("overall_score"), "silver_scores": pair.get("scores"),
           "cv_domain": pair.get("cv_domain"), "jd_domain": pair.get("jd_domain")}
    try:
        if config.version == "ats-en-v3" and scorer is score_pair:
            raise ValueError("V3 requires src.scoring.score_pair or runtime_v3.score_pair_v3; pass it explicitly")
        if variant == "v1_original":
            cv = resume_from_record(originals["cvs"][cv_id], as_of)
            jd, removed = job_from_record(originals["jobs"][jd_id])
        else:
            cv = ResumeInput.model_validate(annotations[cv_id]["envelope"])
            jd = JobInput.model_validate(annotations[jd_id]["envelope"])
            removed = []
        result = scorer(cv, jd, config=config, as_of_date=as_of)
        payload = result.model_dump(mode="json", exclude={"config": {"relations"}})
        # The artifact's identity/config snapshot stores the table once; each pair
        # references its exact configuration instead of duplicating thousands of rows.
        payload["config_sha256"] = canonical_hash(config.model_dump(mode="json"))
        row.update(status="ok", cv_title=cv.document.basics.label, jd_title=jd.document.title,
                   removed_title_skill_tags=removed, result=payload)
    except Exception as exc:  # Retain every runtime/data failure in the diagnostic artifact.
        row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    row["latency_ms"] = (time.perf_counter() - started) * 1000
    return row


def summarize(rows: list[dict]) -> dict:
    successes = [row for row in rows if row["status"] == "ok"]
    failures = [row for row in rows if row["status"] == "failed"]
    summary = {"expected": len(rows), "successful": len(successes), "failed": len(failures),
               "status": "empty" if not rows else "all_failed" if not successes else "partial_failure" if failures else "complete",
               "failures": [{key: row[key] for key in ("pair_id", "error_type", "error")} for row in failures],
               "latency_ms": _summary([row["latency_ms"] for row in rows])}
    if not successes:
        summary["metrics"] = None
        return summary
    labels = [row["silver_grade"] for row in successes]
    predictions = [row["result"]["grade"] for row in successes]
    breakdowns = [row["result"].get("skills_breakdown") or {} for row in successes]
    summary.update(metrics=classification_metrics(labels, predictions),
                   silver_distribution=dict(Counter(map(str, labels))), predicted_distribution=dict(Counter(map(str, predictions))),
                   overall_distribution=_summary([row["result"]["overall_score"] for row in successes]),
                   coverage=_summary([row["result"]["coverage"] for row in successes]),
                   skills_distribution=_summary([row["result"]["scores"]["skills"] for row in successes]),
                   scores_by_silver_grade={str(grade): _summary([row["result"]["overall_score"] for row in successes if row["silver_grade"] == grade]) for grade in range(3)},
                   gate_status=dict(Counter(str(item.get("gate_status", "v1")) for item in breakdowns)),
                   skills_basis=dict(Counter(str(item.get("basis", "v1")) for item in breakdowns)),
                   gate_warnings=dict(Counter(warning for item in breakdowns for warning in item.get("warnings", []))),
                   forced_mismatch=sum(bool(item.get("force_mismatch")) for item in breakdowns),
                   utilization={"core_pairs": sum(item.get("core_gate_fulfillment") is not None for item in breakdowns),
                                "bonus_pairs": sum(item.get("bonus", 0) > 0 for item in breakdowns),
                                "explicit_proficiency_pairs": sum(any(req.get("proficiency") not in {None, "normal"} for req in item.get("requirements", [])) for item in breakdowns),
                                "semantic_pairs": sum(any(req.get("match_type") in {"near", "transfer", "semantic_near", "semantic_transfer"} for req in item.get("requirements", [])) for item in breakdowns)})
    dimensions = sorted({key for row in successes for key in row["result"]["dimensions"]})
    summary["dimension_coverage"] = {key: statistics.mean(bool(row["result"]["dimensions"].get(key, {}).get("supported")) for row in successes) for key in dimensions}
    summary["dimension_mean"] = {key: statistics.mean(row["result"]["scores"][key] for row in successes) for key in dimensions}
    return summary


def grade_changes(previous: list[dict], current: list[dict]) -> dict:
    first = {row["pair_id"]: row for row in previous}
    second = {row["pair_id"]: row for row in current}
    changed, failed, score_deltas = [], [], []
    matrix = [[0] * 3 for _ in range(3)]
    for key in sorted(first.keys() | second.keys()):
        old, new = first.get(key), second.get(key)
        if not old or not new or old["status"] != "ok" or new["status"] != "ok":
            failed.append(key)
            continue
        before, after = old["result"]["grade"], new["result"]["grade"]
        matrix[before][after] += 1
        score_deltas.append({dimension: new["result"]["scores"][dimension] - value
                             for dimension, value in old["result"]["scores"].items()})
        if before != after:
            changed.append({"pair_id": key, "before": before, "after": after,
                            "score_before": old["result"]["overall_score"], "score_after": new["result"]["overall_score"],
                            "gate_status": (new["result"].get("skills_breakdown") or {}).get("gate_status")})
    return {"compared": sum(map(sum, matrix)), "changed": len(changed), "transition_matrix": matrix,
            "pairs": changed, "uncomparable_pairs": failed,
            "score_changes": {dimension: {"pairs_changed": sum(delta[dimension] != 0 for delta in score_deltas),
                               "mean_delta": statistics.mean(delta[dimension] for delta in score_deltas)}
                              for dimension in score_deltas[0]} if score_deltas else {},
            "non_skill_components_changed_pairs": sum(any(value != 0 for key, value in delta.items() if key != "skills") for delta in score_deltas)}


def annotation_summary(corpus: dict) -> dict:
    documents = corpus.get("documents", [])
    count = len(documents)
    reviewed = sum(document.get("skills_review_status", document.get("review_status")) == "approved" for document in documents)
    fully_reviewed = sum(document.get("full_envelope_review_status") == "approved" for document in documents)
    cv_details = [detail for document in documents if document["kind"] == "resume"
                  for group in document.get("envelope", {}).get("document", {}).get("skills", [])
                  for detail in group.get("details", [])]
    requirements = [requirement for document in documents if document["kind"] == "job"
                    for requirement in document.get("envelope", {}).get("document", {}).get("skills", {}).get("requirements", [])]
    return {"documents": count, "by_kind": dict(Counter(document["kind"] for document in documents)),
            "skills_review_status": dict(Counter(document.get("skills_review_status", document.get("review_status", "missing")) for document in documents)),
            "full_envelope_review_status": dict(Counter(document.get("full_envelope_review_status", "pending") for document in documents)),
            "skills_reviewed_fraction": reviewed / count if count else 0,
            "full_envelope_reviewed_fraction": fully_reviewed / count if count else 0,
            "skills_status": dict(Counter(document.get("skills_status", "missing") for document in documents)),
            "cv_skill_proficiency": dict(Counter(detail.get("proficiency", "normal") for detail in cv_details)),
            "jd_requirement_importance": dict(Counter(requirement.get("importance", "standard") for requirement in requirements)),
            "issues": dict(Counter(issue.get("code", "unknown") for document in documents for issue in document.get("issues", []))),
            "provisional": reviewed != count or fully_reviewed != count}


def run_benchmark(root: Path, *, as_of: date, corpus_path: Path, output: Path,
                  relations_path: Path | None = None, allow_provisional_relations: bool = False,
                  variants: tuple[str, ...] = VARIANTS, pairs: list[dict] | None = None,
                  require_population: bool = True, scorer: Callable = score_pair) -> dict:
    paths = {"pairs": root / "data/datasets/matching_pairs/benchmark_matching_pairs.json",
             "cvs": root / "data/processed/structured_cvs/all_cvs.json",
             "jobs": root / "data/processed/structured_jobs/all_jobs_en.json", "annotations": corpus_path}
    if relations_path:
        paths["relation_table"] = relations_path
    corpus, annotations = load_corpus(corpus_path, as_of, require_population=require_population)
    for document in annotations.values():
        source = document["source"]
        for path_key, hash_key in (("source_path", "source_sha256"), ("text_path", "text_sha256")):
            if source.get(path_key):
                path = (root / source[path_key]).resolve()
                payload = (path.read_text(encoding="utf-8").encode("utf-8") if path_key == "text_path" else path.read_bytes()) if path.is_file() else b""
                if not path.is_file() or not source.get(hash_key) or hashlib.sha256(payload).hexdigest() != source[hash_key]:
                    raise ValueError(f"Annotation source checksum is stale: {document['id']} {path_key}")
                paths[f"{document['id']}_{path_key}"] = path
    for name in ("disagreement_nominations", "disagreement_review"):
        path = root / f"output/extraction_pilot/{name}.json"
        if path.exists():
            paths[name] = path
    table = load_relations(relations_path, corpus)
    originals = {"cvs": {record["cv_id"]: record for record in read_json(paths["cvs"])},
                 "jobs": {record["id"]: record for record in read_json(paths["jobs"])}}
    selected = read_json(paths["pairs"]) if pairs is None else pairs
    counts = {"all": len(selected), "gemini": sum(not _is_heuristic(pair) for pair in selected),
              "heuristic": sum(_is_heuristic(pair) for pair in selected)}
    if require_population and counts != EXPECTED_POPULATION:
        raise ValueError(f"Expected unchanged complete silver population {EXPECTED_POPULATION}, found {counts}")
    if not selected or not counts["gemini"]:
        raise ValueError("Benchmark requires nonempty pairs and a nonempty primary Gemini subset")
    if len({(pair["cv_id"], pair["jd_id"]) for pair in selected}) != len(selected):
        raise ValueError("Benchmark contains duplicate pair IDs")
    if len(set(variants)) != len(variants) or not variants:
        raise ValueError("Choose unique, nonempty variants")
    if not set(variants) <= set(VARIANTS):
        raise ValueError("Unknown benchmark variant")
    variants = tuple(variant for variant in VARIANTS if variant in variants)
    identity = benchmark_identity(root, paths, as_of, table, allow_provisional_relations, selected)
    marker = output / "benchmark_identity.json"
    if marker.exists():
        require_same_identity(identity, read_json(marker))
    annotations_stats = annotation_summary(corpus)
    relation_counts = dict(Counter(relation.review_status for relation in table.relations))
    report = {"version": BENCHMARK_VERSION, "identity": identity, "as_of_date": as_of.isoformat(),
              "scope": "Diagnostic agreement on previously seen silver benchmark; not independent test accuracy or hiring-outcome validation.",
              "population": counts, "api_requests": 0, "annotations": annotations_stats,
              "relations": {"version": table.version, "review_status": relation_counts,
                            "allow_provisional": allow_provisional_relations,
                            "active": sum(relation.review_status == "approved" or (allow_provisional_relations and relation.review_status == "pending") for relation in table.relations)},
              "provisional": annotations_stats["provisional"] or (allow_provisional_relations and relation_counts.get("pending", 0) > 0),
              "variants": {}, "limitations": ["Gemini labels and heuristic negatives are preserved; labels are not ground truth hiring decisions.",
              "Document and pair overlap plus prior inspection prevent independent accuracy claims.",
              "Incomplete skill extraction produces neutral skills and an unknown gate rather than evidence of missing Core.",
              "Skill approval does not approve the full extraction envelope or permit Qwen gold export.",
              "Latency includes runtime validation/adaptation; no embedding model, GPU or API request is used."]}
    nomination_path = root / "output/extraction_pilot/disagreement_nominations.json"
    nominations = read_json(nomination_path).get("cases", []) if nomination_path.exists() else []
    report["historical_disagreements"] = {"expected": 24, "available": len(nominations), "source": str(nomination_path)}
    artifacts = {}
    previous = None
    for variant in variants:
        config = variant_config(variant, table, allow_provisional_relations)
        rows = [score_stored_pair(pair, variant, originals, annotations, config, as_of, scorer) for pair in selected]
        entry = {"config": config.model_dump(mode="json"), "all": summarize(rows),
                 "gemini_primary": summarize([row for row in rows if row["pair_type"] != "heuristic_negative"]),
                 "heuristic_negatives": summarize([row for row in rows if row["pair_type"] == "heuristic_negative"]),
                 "by_cv_domain": {str(domain): summarize([row for row in rows if row["cv_domain"] == domain]) for domain in sorted({row["cv_domain"] for row in rows}, key=str)},
                 "by_jd_domain": {str(domain): summarize([row for row in rows if row["jd_domain"] == domain]) for domain in sorted({row["jd_domain"] for row in rows}, key=str)}}
        indexed = {row["pair_id"]: row for row in rows}
        entry["historical_disagreements"] = [{"pair_id": f"{case['original_comparison']['cv_id']}::{case['original_comparison']['jd_id']}",
            "selection_reason": case.get("selection_reason"), "result": indexed.get(f"{case['original_comparison']['cv_id']}::{case['original_comparison']['jd_id']}")}
            for case in nominations]
        if previous is not None:
            entry["change_from_previous"] = {"variant": previous, **grade_changes(artifacts[previous]["rows"], rows)}
        if variant != "v1_original" and "v1_original" in artifacts:
            entry["change_from_original"] = grade_changes(artifacts["v1_original"]["rows"], rows)
        report["variants"][variant] = entry
        artifacts[variant] = {"identity": identity, "variant": variant, "config_sha256": canonical_hash(config.model_dump(mode="json")), "rows": rows}
        previous = variant
    report["status"] = "failed" if any(entry["all"]["failed"] for entry in report["variants"].values()) else "complete"
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(identity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for variant, artifact in artifacts.items():
        (output / f"{variant}.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report_markdown(report), encoding="utf-8")
    return report


def report_markdown(report: dict) -> str:
    lines = ["# ATS v2 silver diagnostic", "", report["scope"], "",
             f"Date: {report['as_of_date']}. Identity: `{report['identity']['family_sha256']}`. Status: **{report['status']}**. Provisional: **{report['provisional']}**.", "",
             f"Population: {report['population']['all']} pairs; primary Gemini: {report['population']['gemini']}; heuristic negatives: {report['population']['heuristic']}. API requests: 0.", "",
             f"Skills approved: {report['annotations']['skills_reviewed_fraction']:.1%}; full envelopes approved: {report['annotations']['full_envelope_reviewed_fraction']:.1%}.", "",
             "| Variant | Primary successful / attempted | Macro F1 | Balanced accuracy | Heuristic F1 | Failures | Mean latency ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for variant, entry in report["variants"].items():
        primary, heuristic = entry["gemini_primary"], entry["heuristic_negatives"]
        metrics, hm = primary.get("metrics"), heuristic.get("metrics")
        value = lambda m, key: f"{m[key]:.4f}" if m else "unavailable"
        latency = entry["all"]["latency_ms"].get("mean", 0)
        lines.append(f"| {variant} | {primary['successful']} / {primary['expected']} | {value(metrics, 'macro_f1')} | {value(metrics, 'balanced_accuracy')} | {value(hm, 'macro_f1')} | {entry['all']['failed']} | {latency:.3f} |")
    lines.extend(["", "Per-grade precision/recall, confusion matrices, score distributions, domain analyses, all failure details, historical cases and individual grade changes are recorded in report.json. Every attempted pair is retained in each variant artifact.", "", "## Annotation status", "", "```json", json.dumps(report["annotations"], indent=2), "```", "", "## Limitations", ""])
    lines.extend("- " + item for item in report["limitations"])
    return "\n".join(lines) + "\n"
