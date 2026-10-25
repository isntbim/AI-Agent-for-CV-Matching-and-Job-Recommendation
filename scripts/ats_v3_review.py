"""Prepare or import item-level v3 reviews without modifying v2 artifacts."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring.v3_corpus import prepare, import_review, validate_identity, refresh
from src.scoring.v2_corpus import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "output/scorer_v3/corpus.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    imp = sub.add_parser("import-review")
    imp.add_argument("--decisions", type=Path, required=True)
    sub.add_parser("validate")
    sub.add_parser("refresh-metadata")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(ROOT, ROOT / "output/scorer_v2/corpus.json", args.corpus, ROOT / "human-verify/scorer-v3")
    else:
        corpus = read_json(args.corpus)
        if args.command == "refresh-metadata":
            validate_identity(corpus)
            previous = corpus["counts"]
            refresh(corpus)
            if previous != corpus["counts"]:
                corpus["changes"].append({"date": "2026-10-08", "review_kind": "metadata_reconciliation", "rule_id": "R3-COUNTS",
                    "before": previous, "after": corpus["counts"], "reason": "Recompute counts from current v3 annotations without changing review decisions."})
                refresh(corpus)
            atomic_json(args.corpus, corpus)
            result = {"identity": corpus["identity_sha256"], "counts": corpus["counts"]}
        elif args.command == "validate":
            validate_identity(corpus)
            result = {"identity": corpus["identity_sha256"], "valid": True}
        else:
            result = import_review(corpus, read_json(args.decisions), ROOT)
            if result["applied"]:
                atomic_json(args.corpus, corpus)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
