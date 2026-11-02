"""Mechanically map a teacher's off-schema keys onto the v3 label schema, keeping originals and a log.

Only renames where the meaning is unambiguous; values with no schema slot are dropped and logged.
Anything else stays in place, so the strict validator still rejects the label.
"""

import argparse
import json
import shutil
from pathlib import Path

from scripts.cv_teacher_validate import undefined_keys
from src.scoring.extraction import write_json

ROOT = Path(__file__).resolve().parent.parent


def repair(label: dict) -> tuple[list[str], dict]:
    renamed, dropped = [], {}
    facts = label.get("facts") or {}
    if "degrees" in facts and "completed_degrees" not in facts:
        facts["completed_degrees"] = facts.pop("degrees")
        renamed.append("degrees->completed_degrees")
    for key in ("certifications", "total_months_work"):
        if key in facts:
            dropped[key] = facts.pop(key)
    for key in ("no_experience_required", "no_education_required", "no_skills", "remote", "worldwide_remote"):
        if key in facts and facts[key] is None:  # null = not stated = the schema default false
            facts[key] = False
            renamed.append(f"{key}:null->false")
    doc = label.get("document") or {}
    for g, group in enumerate(doc.get("skills") or [], 1):
        for d, detail in enumerate(group.get("details") or [], 1):
            if isinstance(detail.get("provenance"), str):  # transport metadata the brief forbids; a bare id fills no slot
                dropped[f"skills[{g}].details[{d}].provenance"] = detail.pop("provenance")
    for i, work in enumerate(doc.get("work") or [], 1):
        if "company" in work and not (work.get("name") or "").strip():
            work["name"] = work.pop("company")
            renamed.append(f"work[{i}].company->name")
        for key in ("current", "location"):  # current: the arbiter's endDate decides; location: no schema slot
            if key in work:
                dropped[f"work[{i}].{key}"] = work.pop(key)
    for i, edu in enumerate(doc.get("education") or [], 1):
        if "gpa" in edu:
            if edu["gpa"] and not edu.get("score"):
                edu["score"] = edu.pop("gpa")
                renamed.append(f"education[{i}].gpa->score")
            elif not edu["gpa"] or edu["gpa"] == edu.get("score"):
                dropped[f"education[{i}].gpa"] = edu.pop("gpa")
    return renamed, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", help="label folder under --out, e.g. gemini_b")
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    args = parser.parse_args()
    folder, raw = args.out / args.teacher, args.out / f"{args.teacher}_raw"
    log_path = args.out / f"{args.teacher}_repairs.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    for path in sorted(folder.glob("*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        renamed, dropped = repair(label)
        if not (renamed or dropped):
            continue
        raw.mkdir(exist_ok=True)
        if not (raw / path.name).exists():  # never overwrite an earlier original
            shutil.copy2(path, raw / path.name)
        write_json(path, label)
        log.append({"id": path.stem, "renamed": "; ".join(renamed), "dropped": dropped})
        left = undefined_keys(label)
        print(path.stem, len(renamed), "renamed,", len(dropped), "dropped", f"| STILL UNDEFINED: {left}" if left else "")
    write_json(log_path, log)


if __name__ == "__main__":
    main()
