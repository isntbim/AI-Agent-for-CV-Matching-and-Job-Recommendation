"""Score unchanged v1/v2 inputs with a selected preserved or current source tree."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    data_root = (args.data_root or root).resolve()
    sys.path.insert(0, str(root))
    from src.scoring.v2_benchmark import score_stored_pair, variant_config, load_relations
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    pairs_path = data_root / "data/datasets/matching_pairs/benchmark_matching_pairs.json"
    pairs = read(pairs_path)
    originals = {"cvs": {d["cv_id"]: d for d in read(data_root / "data/processed/structured_cvs/all_cvs.json")},
                 "jobs": {d["id"]: d for d in read(data_root / "data/processed/structured_jobs/all_jobs_en.json")}}
    corpus = read(data_root / "output/scorer_v2/corpus.json")
    annotations = {d["id"]: d for d in corpus["documents"]}
    table = load_relations(data_root / "output/scorer_v2/relations.json")
    result = {"source_root": str(root), "as_of_date": "2026-09-30", "pairs_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
              "corpus_identity": corpus["identity_sha256"], "variants": {}}
    for variant in ("v1_original", "full"):
        rows = [score_stored_pair(pair, variant, originals, annotations, variant_config(variant, table), date(2026, 9, 30)) for pair in pairs]
        result["variants"][variant] = rows
        print(json.dumps({"variant": variant, "pairs": len(rows), "failures": sum(row["status"] != "ok" for row in rows)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
