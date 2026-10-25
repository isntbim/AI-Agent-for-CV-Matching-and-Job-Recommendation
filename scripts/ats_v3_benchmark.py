"""Run v3 depth diagnostics and optional legacy Preferred-share comparisons."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring.v3_benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--annotation-corpus", type=Path, default=ROOT / "output/scorer_v3/corpus.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/scorer_v3/benchmark")
    parser.add_argument("--frozen-reproduction", type=Path)
    parser.add_argument("--current-reproduction", type=Path)
    parser.add_argument("--preferred-policies", nargs="+", choices=("separate", "legacy_share"), default=["separate"])
    parser.add_argument("--reference-corpus", type=Path, help="Checksum-bound pre-change corpus for isolated comparison")
    parser.add_argument("--reference-benchmark", type=Path, help="Saved reference benchmark to verify legacy-share reproduction")
    args = parser.parse_args()
    report = run_benchmark(ROOT, as_of=args.as_of, corpus_path=args.annotation_corpus, output=args.output,
                           frozen_reproduction=args.frozen_reproduction, current_reproduction=args.current_reproduction,
                           preferred_policies=tuple(args.preferred_policies), reference_corpus_path=args.reference_corpus,
                           reference_benchmark=args.reference_benchmark)
    print(json.dumps({"status": report["status"], "identity": report["identity"]["family_sha256"], "report": str(args.output / "report.md")}))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
