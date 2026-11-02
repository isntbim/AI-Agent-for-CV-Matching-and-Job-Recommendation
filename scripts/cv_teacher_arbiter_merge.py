"""Merge arbiter part files into one row file whose subjects the reference builder understands.

Arbiters label validator rows loosely ("haiku_b:skills.X:reason", "domains", "facts.evidence.industry").
The builder routes by subject: "facts.<field>", a record list ("document.work"), or a skill name.
"""

import argparse
import json
import re
from pathlib import Path

from src.scoring.extraction import write_json

ROOT = Path(__file__).resolve().parent.parent
FACTS = {"domains", "completed_degrees", "minimum_degree", "country", "currency", "remote", "worldwide_remote",
         "eligible_countries", "no_experience_required", "no_education_required", "no_skills"}
DOCUMENT = {"basics", "work", "education", "projects", "certificates", "languages"}


def full_record_list(row: dict, out: Path) -> list[dict]:
    """A records final may list only changed records; untouched records of the base teacher are kept as they are."""
    refs = [str(e.get("teacher_record") or "") for e in row["final"] or []]
    teachers = [r.split(":")[0] for r in refs if ":" in r]
    if not teachers:
        return [e for e in row["final"] or [] if e.get("action") != "drop"]
    base = max(set(teachers), key=teachers.count)
    label = json.loads((out / f"{base}_b" / f"{row['id']}.json").read_text(encoding="utf-8"))
    count = len(label["document"].get(row["subject"].split(".")[1]) or [])
    by_ref = {str(e.get("teacher_record")): e for e in row["final"]}
    entries = [by_ref.get(f"{base}:{i}", {"teacher_record": f"{base}:{i}", "action": "keep"}) for i in range(1, count + 1)]
    entries += [e for e in row["final"] if not str(e.get("teacher_record") or "").startswith(base + ":")]
    return [e for e in entries if e.get("action") != "drop"]


def normalise(row: dict, out: Path = ROOT / "output/scorer_v3/teacher_labels") -> list[dict]:
    final, subject = row["final"], row["subject"]
    if subject.split(".")[0] in DOCUMENT:  # arbiters sometimes drop the "document." prefix on record fields
        subject = "document." + subject
        row = dict(row, subject=subject)
    if row.get("kind") == "records":
        row = dict(row, subject=subject if subject.startswith("document.") else "document." + subject)
        return [dict(row, final=full_record_list(row, out))]
    if isinstance(final, dict) and not ({"name"} & set(final)) and set(final) & (FACTS | {"evidence"}):
        # a fact patch such as {"domains": [...], "evidence": {"industry": [...]}} becomes one row per field
        rows = [dict(row, subject=f"facts.{k}", final=v) for k, v in final.items() if k in FACTS]
        if "evidence" in final:
            rows.append(dict(row, subject="facts.evidence", final=final["evidence"]))
        return rows
    field = subject.split(".", 1)[1] if subject.startswith("facts.") else subject
    if field.split(".")[0] in FACTS:
        return [dict(row, subject="facts." + field.split(".")[0])]
    if subject.startswith("document."):
        return [row]
    # a skill row: the subject is the skill name
    details = [d for d in (final if isinstance(final, list) else [final]) if isinstance(d, dict)]
    name = details[0]["name"] if details else re.sub(r"^(haiku|gemini)(_b)?:(skills\.)?", "", subject).split(":")[0]
    return [dict(row, subject=name)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="+", help="arbiter part files under --out")
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    args = parser.parse_args()
    rows = [n for part in args.parts for r in json.loads((args.out / part).read_text(encoding="utf-8")) for n in normalise(r, args.out)]
    for r in rows:  # quotes come back CRLF from agents reading the raw file; texts are read in text mode
        for d in (r["final"] if isinstance(r["final"], list) else [r["final"]]):
            if isinstance(d, dict):
                for k in ("skill_quote", "proficiency_quote", "name"):
                    if isinstance(d.get(k), str):
                        d[k] = d[k].replace("\r\n", "\n")
    write_json(args.out / args.target, rows)
    print(len(rows), "rows ->", args.out / args.target)


if __name__ == "__main__":
    main()
