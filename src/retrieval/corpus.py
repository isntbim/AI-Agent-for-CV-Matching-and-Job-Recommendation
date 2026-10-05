"""Versioned, source-reviewed cleaning without modifying input records."""

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from src.scoring.silver import title_like_skill
from src.scoring.extraction import write_json
from .text import TEXT_VERSION, job_text

CORPUS_VERSION = "primary-jobs-retrieval-v2"


def role_nomination(name, title):
    # Reuse the historical detector and cover occupation spellings it omits.
    # Nomination only: reviewed tools/competencies can always be retained.
    return title_like_skill(name, title) or bool(re.search(
        r"(?i)\b(?:staff|assistant|tester|leader)\b|lập trình viên|kỹ sư|ky su", name))


def digest(value) -> str:
    data = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                            separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def nominations(records: list[dict], raw_dir: Path) -> list[dict]:
    result = []
    for record in records:
        source = raw_dir / (record["id"] + ".txt")
        if not source.is_file():
            raise ValueError(f"Missing original JD text: {source}")
        text = source.read_text(encoding="utf-8")
        for bucket in ("required", "preferred"):
            for item in record.get("skills", {}).get(bucket, []):
                name = item["name"]
                if role_nomination(name, record.get("title_normalized", "")):
                    key = digest([record["id"], bucket, name])
                    offsets = []
                    start = text.casefold().find(name.casefold())
                    if start >= 0:
                        offsets = [max(0, start - 100), min(len(text), start + len(name) + 160)]
                    result.append({"key": key, "jd_id": record["id"], "bucket": bucket, "skill": name,
                                   "title": record["title"], "source_path": str(source),
                                   "source_sha256": digest(source.read_bytes()),
                                   "context": text[slice(*offsets)] if offsets else "",
                                   "decision": "pending", "reason": "", "review_status": "awaiting_assistant_review"})
    return result


def clean_records(records: list[dict], review: list[dict]) -> tuple[list[dict], list[dict]]:
    decisions = {}
    for row in review:
        if row["key"] in decisions:
            raise ValueError("Duplicate cleaning review key")
        if row["decision"] not in {"remove_role_tag", "retain_skill"} or not row.get("reason"):
            raise ValueError(f"Unreviewed cleaning nomination: {row['key']}")
        if row.get("review_status") != "assistant_reviewed_awaiting_human_approval":
            raise ValueError("Cleaning requires explicit source review")
        decisions[row["key"]] = row
    cleaned, ledger = deepcopy(records), []
    ids = [r["id"] for r in cleaned]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate JD ID")
    for record in cleaned:
        for bucket in ("required", "preferred"):
            kept = []
            for item in record.get("skills", {}).get(bucket, []):
                key = digest([record["id"], bucket, item["name"]])
                nomination = role_nomination(item["name"], record.get("title_normalized", ""))
                if nomination and key not in decisions:
                    raise ValueError(f"Missing source review for {record['id']} {item['name']}")
                if key in decisions:
                    row = decisions[key]
                    if (row["jd_id"], row["bucket"], row["skill"]) != (record["id"], bucket, item["name"]):
                        raise ValueError("Review identity mismatch")
                    ledger.append({**row, "before": deepcopy(item),
                                   "after": None if row["decision"] == "remove_role_tag" else deepcopy(item)})
                    if row["decision"] == "remove_role_tag":
                        continue
                kept.append(item)
            record.setdefault("skills", {})[bucket] = kept
    return cleaned, ledger


def prepare_corpus(input_path: Path, raw_dir: Path, review_path: Path, output: Path) -> dict:
    original, review = load_json(input_path), load_json(review_path)
    current = {r["key"]: r for r in nominations(original, raw_dir)}
    for row in review:
        if row["key"] not in current or current[row["key"]]["source_sha256"] != row["source_sha256"]:
            raise ValueError("Stale or extraneous source cleaning review")
    cleaned, ledger = clean_records(original, review)
    again, _ = clean_records(cleaned, review)
    if cleaned != again:
        raise AssertionError("Cleaning is not idempotent")
    documents = [{"id": r["id"], "text": job_text(r), "text_sha256": digest(job_text(r).encode("utf-8")),
                  "record_sha256": digest(r), "domain": r["category"]} for r in cleaned]
    identity = {"version": CORPUS_VERSION, "text_version": TEXT_VERSION, "documents": documents}
    manifest = {**identity, "manifest_hash": digest(identity), "input_sha256": digest(input_path.read_bytes()),
                "review_sha256": digest(review_path.read_bytes()), "count": len(cleaned),
                "removed_skill_tags": sum(r["after"] is None for r in ledger),
                "retained_nominations": sum(r["after"] is not None for r in ledger), "idempotent": True,
                "review_status": "assistant_reviewed_awaiting_human_approval"}
    write_json(output / "all_jobs.json", cleaned)
    write_json(output / "corpus.json", documents)
    write_json(output / "manifest.json", manifest)
    write_json(output / "correction_ledger.json", ledger)
    return manifest
