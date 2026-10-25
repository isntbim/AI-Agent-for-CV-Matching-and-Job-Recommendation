"""Prepare, validate and exchange source-backed ATS v2 annotations for review."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scoring.v2_corpus import (atomic_json, export_review, export_training,
    import_review, prepare_corpus, propose_relations, read_json, refresh_identity,
    validate_corpus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus", type=Path, default=ROOT / "output/scorer_v2/corpus.json")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Build assistant proposals; never mark human approved")
    prepare.add_argument("--as-of", type=date.fromisoformat, required=True)
    prepare.add_argument("--output", type=Path, default=ROOT / "output/scorer_v2")
    commands.add_parser("validate")
    review = commands.add_parser("export-review")
    review.add_argument("--output", type=Path, default=ROOT / "human-verify/scorer-v2")
    importer = commands.add_parser("import-review")
    importer.add_argument("--decisions", type=Path, required=True)
    training = commands.add_parser("export-training")
    training.add_argument("--output", type=Path, default=ROOT / "output/scorer_v2/training_export.json")
    proposals = commands.add_parser("propose-relations")
    proposals.add_argument("--model", default="BAAI/bge-m3")
    proposals.add_argument("--revision", required=True)
    proposals.add_argument("--device", default="cpu")
    proposals.add_argument("--top-k", type=int, default=5)
    proposals.add_argument("--output", type=Path, default=ROOT / "output/scorer_v2/relation_proposals.json")
    proposals.add_argument("--attach", action="store_true", help="Attach pending suggestions to corpus for review; none become approved")
    args = parser.parse_args()
    if args.command == "prepare":
        corpus = prepare_corpus(args.root, args.output, args.as_of)
        result = {"path": str(args.output / "corpus.json"), "identity_sha256": corpus["identity_sha256"], "counts": corpus["counts"]}
    elif args.command == "import-review":
        result = import_review(args.corpus, args.decisions, args.root)
    else:
        corpus = read_json(args.corpus)
        if args.command == "validate":
            result = validate_corpus(corpus, args.root)
            atomic_json(args.corpus.parent / "validation_report.json", result)
        elif args.command == "export-review":
            result = export_review(corpus, args.output)
        elif args.command == "export-training":
            result = export_training(corpus, args.output)
        elif args.command == "propose-relations":
            result = propose_relations(corpus, model_name=args.model, revision=args.revision, device=args.device, top_k=args.top_k)
            atomic_json(args.output, result)
            if args.attach:
                corpus["relations"] = {key: result[key] for key in ("version", "relations")}
                refresh_identity(corpus)
                atomic_json(args.corpus, corpus)
                atomic_json(args.corpus.parent / "relations.json", corpus["relations"])
            result = {"path": str(args.output), "relations": len(result["relations"]), "metadata": result["proposal_metadata"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "validate" and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
