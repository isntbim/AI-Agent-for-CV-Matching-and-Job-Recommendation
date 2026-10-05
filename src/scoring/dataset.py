"""Dataset audit and a deterministic split disjoint in both CV and JD identity."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .extraction import text_hash, write_json

DATASET_ID = "cnamuangtoun/resume-job-description-fit"
DATASET_REVISION = "08978e21714984bb417547d2c0f9b477f5298163"
LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


def _partition(identity: str, kind: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{kind}:{identity}".encode()).hexdigest(), 16)
    return "development" if value / (2**256) < 0.75 else "test"


def prepare_dataset(paths: list[Path], output: Path, seed: int = 42,
                    revision: str = "local-input") -> dict:
    unique = {}
    excluded = []
    duplicate_count = 0
    input_count = 0
    resumes, jobs = {}, {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"resume_text", "job_description_text", "label"} <= set(reader.fieldnames or []):
                raise ValueError(f"Missing required CSV columns in {path}")
            for line, row in enumerate(reader, 2):
                input_count += 1
                source = {"file": path.name, "line": line}
                if row["label"] not in LABELS:
                    raise ValueError(f"Unknown label at {path.name}:{line}: {row['label']!r}")
                if not row["resume_text"].strip() or not row["job_description_text"].strip():
                    excluded.append(dict(source, reason="empty_document"))
                    continue
                cv_id, jd_id = text_hash(row["resume_text"]), text_hash(row["job_description_text"])
                resumes.setdefault(cv_id, row["resume_text"])
                jobs.setdefault(jd_id, row["job_description_text"])
                pair_id = hashlib.sha256((cv_id + ":" + jd_id).encode()).hexdigest()
                if pair_id in unique:
                    duplicate_count += 1
                    unique[pair_id]["labels"].add(row["label"])
                    unique[pair_id]["sources"].append(source)
                else:
                    unique[pair_id] = {"pair_id": pair_id, "cv_id": cv_id, "jd_id": jd_id,
                                           "resume_text": resumes[cv_id], "job_description_text": jobs[jd_id],
                                           "labels": {row["label"]}, "sources": [source]}
    pairs = []
    counts = {split: {label: 0 for label in LABELS} for split in ("development", "test")}
    for pair in sorted(unique.values(), key=lambda p: p["pair_id"]):
        labels = pair.pop("labels")
        if len(labels) > 1:
            excluded.append({"pair_id": pair["pair_id"], "sources": pair["sources"], "labels": sorted(labels), "reason": "conflicting_labels"})
            continue
        pair["label"] = next(iter(labels))
        cv_split = _partition(pair["cv_id"], "cv", seed)
        jd_split = _partition(pair["jd_id"], "jd", seed)
        if cv_split != jd_split:
            excluded.append({"pair_id": pair["pair_id"], "sources": pair["sources"], "reason": "cross_partition"})
            continue
        pair["split"] = cv_split
        counts[cv_split][pair["label"]] += 1
        pairs.append(pair)
    manifest = {"dataset": DATASET_ID, "revision": revision, "seed": seed,
                    "split_policy": "independent-cv-jd-hash-75-25-v1", "input_rows": input_count,
                    "duplicates": duplicate_count, "retained_rows": len(pairs), "counts": counts,
                    "excluded_by_reason": dict(Counter(e["reason"] for e in excluded)),
                    "unique_resumes": len({p["cv_id"] for p in pairs}),
                    "unique_jobs": len({p["jd_id"] for p in pairs}),
                    "ready_for_calibration": all(n > 0 for split in counts.values() for n in split.values()),
                    "inputs": [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    write_json(output / "manifest.json", manifest)
    write_json(output / "pairs.json", pairs)
    write_json(output / "excluded.json", excluded)
    return manifest


def load_pairs(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    pairs = json.loads((directory / "pairs.json").read_text(encoding="utf-8"))
    for kind in ("cv_id", "jd_id"):
        development = {p[kind] for p in pairs if p["split"] == "development"}
        test = {p[kind] for p in pairs if p["split"] == "test"}
        if development & test:
            raise ValueError(f"Split leaks {kind}")
    if len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Duplicate pair identity in prepared data")
    return manifest, pairs
