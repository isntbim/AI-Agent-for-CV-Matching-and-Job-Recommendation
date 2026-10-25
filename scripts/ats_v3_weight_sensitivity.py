"""Run the prespecified 4x3 full-v3 sensitivity experiment without editing defaults."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring.v3_weight_sensitivity import run_sweep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--annotation-corpus", type=Path, default=ROOT / "output/scorer_v3/corpus.json")
    parser.add_argument("--reference-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_sweep(ROOT, as_of=args.as_of, corpus_path=args.annotation_corpus,
        reference_path=args.reference_benchmark, output=args.output,
        progress=lambda message: print(message, flush=True))
    print(json.dumps({"status": report["status"], "identity": report["identity"]["family_sha256"],
                      "variants": len(report["variants"]), "spread": report["score_spread"]}))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
