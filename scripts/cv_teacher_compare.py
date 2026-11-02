"""List where two teacher CV labels disagree, as arbiter/review items.

Skills compare on casefolded names; facts compare on the fields that change
scores (country, completed degrees, domains) plus level and record counts.
"""

import argparse
import json
from pathlib import Path

from scripts.cv_teacher_validate import selected
from src.scoring.extraction import write_json

ROOT = Path(__file__).resolve().parent.parent


def details(label: dict) -> dict:
    return {d["name"].casefold().strip(): dict(d, group=s.get("name"))
            for s in label["document"].get("skills", []) for d in (s.get("details") or [])}


def compare(a: dict, b: dict, names=("haiku", "gemini")) -> dict:
    da, db = details(a), details(b)
    items = []
    for key in sorted(da.keys() | db.keys()):
        x, y = da.get(key), db.get(key)
        if x and y:
            if x.get("proficiency", "normal") != y.get("proficiency", "normal"):
                items.append({"kind": "level", "skill": x["name"], names[0]: x, names[1]: y})
        else:
            items.append({"kind": "skill_only_" + (names[0] if x else names[1]), "skill": (x or y)["name"],
                          "detail": x or y})
    for field in ("country", "completed_degrees", "domains"):
        fa, fb = a["facts"].get(field), b["facts"].get(field)
        if sorted(fa or []) != sorted(fb or []) if isinstance(fa or fb, list) else fa != fb:
            items.append({"kind": "fact", "field": field, names[0]: fa, names[1]: fb})
    for field in ("work", "education", "languages"):
        ra, rb = a["document"].get(field) or [], b["document"].get(field) or []
        if len(ra) != len(rb):
            items.append({"kind": "records", "field": field, names[0]: ra, names[1]: rb})
    shared = da.keys() & db.keys()
    return {"agreed_skills": sorted(da[k]["name"] for k in shared),
            "jaccard": round(len(shared) / max(1, len(da.keys() | db.keys())), 3), "items": items}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    parser.add_argument("--teachers", nargs=2, default=["haiku_v21", "gemini_v21"])
    parser.add_argument("--split", default="pilot")
    args = parser.parse_args()
    report = {}
    for row in selected(args.out, args.split):
        paths = [args.out / t / (row["id"] + ".json") for t in args.teachers]
        if all(p.exists() for p in paths):
            a, b = (json.loads(p.read_text(encoding="utf-8")) for p in paths)
            report[row["id"]] = compare(a, b)
    write_json(args.out / f"disagreements_{args.split}.json", report)
    for i, r in report.items():
        kinds = {}
        for item in r["items"]:
            kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
        print(i, r["jaccard"], kinds)


if __name__ == "__main__":
    main()
