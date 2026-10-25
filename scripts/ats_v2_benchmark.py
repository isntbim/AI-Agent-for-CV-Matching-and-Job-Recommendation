"""Run all seven ATS ablations against the fixed silver population, offline."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scoring.v2_benchmark import VARIANTS, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--annotation-corpus", type=Path, default=ROOT / "output/scorer_v2/corpus.json")
    parser.add_argument("--relations", type=Path)
    parser.add_argument("--allow-provisional-relations", action="store_true")
    parser.add_argument("--variant", choices=VARIANTS, action="append", help="Explicit subset; omit for all seven ablations")
    parser.add_argument("--output", type=Path, default=ROOT / "output/scorer_v2/benchmark")
    args = parser.parse_args()
    report = run_benchmark(ROOT, as_of=args.as_of, corpus_path=args.annotation_corpus,
                           output=args.output, relations_path=args.relations,
                           allow_provisional_relations=args.allow_provisional_relations,
                           variants=tuple(args.variant) if args.variant else VARIANTS)
    print(json.dumps({"status": report["status"], "provisional": report["provisional"],
                      "identity": report["identity"]["family_sha256"], "report": str(args.output / "report.md")}, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
