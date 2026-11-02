"""Apply gold-style skill decisions (GOLD_STYLE_FILTER.md) to teacher CV labels.

Reads TL/<decisions>/<id>.json ({"keep": [...], "add": [{"name", "quote"}]}) and the teacher label TL/<teacher>/<id>.json,
writes TL/<out>/<id>.json: the same label with only the kept skills plus the added ones. Bad decisions are reported and
that CV is skipped (never half-applied).

Usage: python -m scripts.cv_gold_style_apply <teacher> <decisions> <out> [ids...]   (default: every decision file)
"""

import argparse
import copy
import json
import sys
from pathlib import Path

from src.scoring.v3_aliases import name_in_quote

ROOT = Path(__file__).resolve().parent.parent
TL = ROOT / "output/scorer_v3/teacher_labels"


def apply(label: dict, decision: dict, text: str) -> tuple[dict | None, list[str]]:
    skills = label["document"].get("skills") or []
    have = {d["name"].casefold().strip() for g in skills for d in g.get("details") or []}
    keep = {k.casefold().strip() for k in decision.get("keep") or []}
    errors = [f"keep {k!r} is not a teacher skill" for k in sorted(keep - have)]
    adds, notes = [], []
    for a in decision.get("add") or []:
        name, quote = (a.get("name") or "").strip(), a.get("quote") or ""
        if not name or quote not in text:
            errors.append(f"add {name!r}: quote is not an exact substring of the CV")
        elif not name_in_quote(" ".join(name.split()), quote):  # production check; fails on glued PDF text ("SkillsSql")
            notes.append(f"dropped add {name!r}: name not linkable to its quote {quote!r}")
        else:
            adds.append(a)
    if errors:
        return None, errors
    out = copy.deepcopy(label)
    groups = []
    for g in out["document"].get("skills") or []:
        # the Languages group mirrors document.languages (training standard), so the filter never drops it
        details = [d for d in g.get("details") or [] if g.get("name") == "Languages" or d["name"].casefold().strip() in keep]
        if details:
            g["details"] = details
            g["keywords"] = [d["name"] for d in details]
            groups.append(g)
    added = [{"name": " ".join(a["name"].split()), "proficiency": "normal", "raw_proficiency": None,  # PDF spacing out of names
              "skill_quote": a["quote"], "proficiency_quote": None}
             for a in adds if a["name"].casefold().strip() not in keep]
    if added:
        groups.append({"name": "General", "level": None, "keywords": [d["name"] for d in added], "details": added})
    out["document"]["skills"] = groups
    return out, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("teacher")
    p.add_argument("decisions")
    p.add_argument("out")
    p.add_argument("ids", nargs="*")
    args = p.parse_args()
    ids = args.ids or sorted(f.stem for f in (TL / args.decisions).glob("*.json"))
    (TL / args.out).mkdir(exist_ok=True)
    bad = 0
    for i in ids:
        text = json.loads((TL / "inputs" / f"{i}.json").read_text(encoding="utf-8"))["source_text"]
        label = json.loads((TL / args.teacher / f"{i}.json").read_text(encoding="utf-8"))
        decision = json.loads((TL / args.decisions / f"{i}.json").read_text(encoding="utf-8"))
        result, errors = apply(label, decision, text)
        if result is None:
            bad += 1
            print(f"{i}: SKIPPED", *errors, sep="\n  ")
            continue
        for note in errors:
            print(f"{i}: {note}")
        (TL / args.out / f"{i}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"applied {len(ids) - bad}/{len(ids)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
