"""python -m scripts.extraction_pilot <sources|split|select|validate|tokens>."""

import argparse
from pathlib import Path
import json
import sys
import hashlib

sys.stdout.reconfigure(encoding="utf-8")

from src.scoring.pilot import build_sources, freeze_splits, select_targets, validate_targets, token_report, review_proposals, apply_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sources", "split", "select", "reproduce", "validate", "tokens", "inspect-xlsx", "review-proposals", "apply-review", "disagreements", "validate-disagreements"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("output/extraction_pilot"))
    parser.add_argument("--public", type=Path, default=Path("output/ats/prepared"))
    parser.add_argument("--revision")
    args = parser.parse_args()
    output = args.root / args.output
    if args.command in {"disagreements", "validate-disagreements"}:
        from src.scoring.disagreements import nominate, validate_review
        result = nominate(args.root, output) if args.command == "disagreements" else validate_review(output)
        print(json.dumps({"count": result["count"], "validation": result.get("validation")}))
        return
    if args.command == "inspect-xlsx":
        import zipfile
        from xml.etree import ElementTree as ET
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for path in sorted((args.root / "data/raw/jobs").glob("*.xlsx")):
            with zipfile.ZipFile(path) as z:
                strings = []
                if "xl/sharedStrings.xml" in z.namelist():
                    strings = ["".join(t.text or "" for t in si.findall(".//m:t", ns)) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
                sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
                samples = []
                for row in sheet.findall(".//m:row", ns)[:2]:
                    values = []
                    for c in row.findall("m:c", ns):
                        v = c.find("m:v", ns)
                        text = "".join(t.text or "" for t in c.findall(".//m:t", ns))
                        if v is not None:
                            text = strings[int(v.text)] if c.get("t") == "s" else v.text
                        values.append((text or "")[:220])
                    samples.append(values)
                print(path.name)
                print("\n".join(samples[0]))
        return
    if args.command == "reproduce":
        from src.scoring.extraction import write_json
        def hashes(directory, names):
            return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}
        names = ["source_manifest.json", "split_manifest.json", "selection.json"]
        public = args.root / args.public
        original = hashes(output, names)
        public_original = hashes(public, ["manifest.json", "pairs.json"])
        runs = []
        for _ in range(2):
            build_sources(args.root, output)
            freeze_splits(args.root, output, output / "near_duplicate_review.json", public)
            select_targets(args.root, output)
            runs.append(hashes(output, names))
        passed = all(run == original for run in runs) and public_original == hashes(public, ["manifest.json", "pairs.json"])
        write_json(output / "reproducibility_report.json", {"passed": passed, "reference": original,
                   "repeated_runs": runs, "public_hashes": public_original, "public_splits_unchanged": public_original == hashes(public, ["manifest.json", "pairs.json"])})
        if not passed:
            raise ValueError("Frozen artifacts changed on regeneration; see reproducibility_report.json")
        print(json.dumps({"reproducibility_passed": True, "repeated_runs": 2}))
        return
    if args.command == "sources":
        result = build_sources(args.root, output)
    elif args.command == "split":
        result = freeze_splits(args.root, output, output / "near_duplicate_review.json", args.root / args.public)
    elif args.command == "select":
        result = select_targets(args.root, output)
    elif args.command == "apply-review":
        apply_review(args.root, output)
        return
    elif args.command == "review-proposals":
        review_proposals(args.root, output)
        return
    elif args.command == "validate":
        result = validate_targets(args.root, output)
    else:
        result = token_report(args.root, output, args.revision, str(args.root / ".cache/retrieval/huggingface"))
    if isinstance(result, list):
        print(json.dumps({"count": len(result), "ids": [r["id"] for r in result]}))
    else:
        print(json.dumps({k: v for k, v in result.items() if k not in {"documents", "filename_collisions", "public_manifest", "targets", "pilot"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
