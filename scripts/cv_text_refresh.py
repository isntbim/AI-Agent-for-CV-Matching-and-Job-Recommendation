"""Re-extract pool CV texts with the current parser and move label quotes onto the new text.

For every manifest CV whose re-extraction differs: the old text goes to texts_pre_<date>/ (once), texts/ and
inputs/ get the new text, and each label in --labels has its quotes (skills, levels, facts evidence) replaced by the
matching span of the new text. A quote with no match is reported, never guessed. Idempotent; run it as the last
step of every reference build (REFERENCE_BUILD.md).
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from scripts.cv_gold_to_v3 import literal
from scripts.cv_teacher_validate import ROOT, check
from src.parser.extractors import clean_text, extract_text_from_file
from src.scoring.extraction import write_json


def respace(name: str, quote: str) -> str:
    """A glued name ("ContentCreation") gets the spaces its re-spaced quote has ("content creation"), keeping its case;
    PDF line breaks inside a name become one space ("cross-\\ntraining" -> "cross-training")."""
    broken = "\n" in name
    name = re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "-", name)).strip()
    flat = re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "-", quote))
    if name.casefold() in flat.casefold() or " " in name and not broken:  # a spaced, unbroken name is already right
        return name
    letters = name.replace(" ", "")
    match = re.search(r"\s*".join(map(re.escape, letters)), flat, re.IGNORECASE)
    if not match:
        return name
    chars = iter(letters)
    return "".join(" " if c.isspace() else next(chars) for c in match.group(0))


def requote(label: dict, text: str) -> list[str]:
    """Point every quote and skill name at the new text; returns the ones that could not be placed."""
    missing = []
    for group in label["document"]["skills"]:
        for detail in group["details"]:
            for key in ("skill_quote", "proficiency_quote"):
                if detail.get(key) and detail[key] not in text:
                    span = literal(detail[key], text)
                    detail[key] = span or detail[key]
                    if span is None:
                        missing.append(f"{key}: {detail['name']}")
            detail["name"] = respace(clean_text(detail["name"]), detail.get("skill_quote") or "")
        group["keywords"] = [d["name"] for d in group["details"]]
    for dimension, quotes in label["facts"]["evidence"].items():
        for item in quotes:
            if item["quote"] not in text:
                span = literal(item["quote"], text)
                item["quote"] = span or item["quote"]
                if span is None:
                    missing.append(f"evidence.{dimension}: {item['quote'][:60]}")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    parser.add_argument("--labels", nargs="+", default=["reference_b", "reference", "reference_b_excluded"])
    parser.add_argument("--backup", default="texts_pre_20261024")
    args = parser.parse_args()
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report, status = [], Counter()
    for row in manifest["documents"]:
        if "text_path" not in row:
            continue
        old_path = ROOT / row["text_path"]
        old = old_path.read_text(encoding="utf-8")
        new = extract_text_from_file(ROOT / row["source_path"])
        if new.strip() != old.strip():
            backup = args.out / args.backup / old_path.name
            if not backup.exists():  # never overwrite the original text
                backup.parent.mkdir(exist_ok=True)
                shutil.copy2(old_path, backup)
            old_path.write_text(new, encoding="utf-8")
            input_path = ROOT / row["input_path"]
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            payload["source_text"] = new
            write_json(input_path, payload)
            row["characters"] = len(new)
        else:
            new = old
        # every label, not only changed texts: a rebuild after an earlier refresh brings back quotes of the old text
        for folder in args.labels:
            path = args.out / folder / f"{row['id']}.json"
            if not path.exists():
                continue
            label = json.loads(path.read_text(encoding="utf-8"))
            before = json.dumps(label, sort_keys=True)
            missing = requote(label, new)
            if json.dumps(label, sort_keys=True) == before and new is old:
                continue
            write_json(path, label)
            result = check(path, new, {"id": row["id"], "labels": folder})
            status[(folder, result["status"])] += 1
            report.append(dict(result, missing=missing))
            print(row["id"], folder, result["status"], len(missing), (result.get("error") or "")[:120])
    write_json(manifest_path, manifest)
    report_path = args.out / "text_refresh_report.json"  # merged: a later run only reports what it touched
    done = {(r["id"], r["labels"]) for r in report}
    earlier = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
    write_json(report_path, [r for r in earlier if (r["id"], r["labels"]) not in done] + report)
    print(dict(status))


if __name__ == "__main__":
    main()
