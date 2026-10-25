"""Run the prespecified depth-step experiment; no production option is added."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.scoring.v3_depth_sensitivity import run_sweep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of",type=date.fromisoformat,required=True)
    parser.add_argument("--annotation-corpus",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=run_sweep(ROOT,as_of=args.as_of,corpus_path=args.annotation_corpus,
                     reference_path=args.reference,output=args.output)
    print(json.dumps({"status":report["status"],"identity":report["identity"]["family_sha256"]}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

