"""Apply the checksum-bound sample-3 correction, retaining pre-fix research artifacts."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring.v2_corpus import read_json, atomic_json
from src.scoring.v3_corpus import import_review, sample3_repair_payload, validate_identity


def main():
    archive = ROOT / "output/scorer_v3/remediation_2026-10-08"
    archive.mkdir(exist_ok=True)
    path = ROOT / "output/scorer_v3/corpus.json"
    corpus = read_json(path)
    validate_identity(corpus)
    for source, destination in ((path, archive / "corpus_before.json"),
        (ROOT / "docs/research/ats_scorer/verification.json", archive / "verification_before.json"),
        (ROOT / "docs/research/ats_scorer/results.md", archive / "results_before.md")):
        if not destination.exists():
            shutil.copy2(source, destination)
    payload_path = archive / "sample3_repair.json"
    if payload_path.exists():
        payload = read_json(payload_path)
    else:
        payload = sample3_repair_payload(corpus, ROOT)
        atomic_json(payload_path, payload)
    receipt = import_review(corpus, payload, ROOT)
    repeated = import_review(corpus, payload, ROOT)
    assert repeated["idempotent"]
    atomic_json(path, corpus)
    result = {"first_import": receipt, "repeat_import": repeated, "corpus_sha256": corpus["identity_sha256"]}
    if not (archive / "repair_receipt.json").exists():
        atomic_json(archive / "repair_receipt.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
