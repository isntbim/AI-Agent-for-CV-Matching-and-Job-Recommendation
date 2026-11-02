"""Merge the external-round teacher labels (haiku_b + gemini_b) and list what the arbiter must decide.

Base is the Haiku label, as in cv_teacher_reference. Skills become the union: a Gemini detail whose normalised name
Haiku lacks joins the Gemini group of the same name (created if missing). Facts and records are not merged here.
These go to ext_arbiter_items.json for a blinded arbiter (teachers shown as A/B in seeded random order, the key kept
apart in ext_arbiter_key.json):
- every fact or record-count disagreement (cv_teacher_compare),
- CVs whose text has the "Expected in" template label but a label written before the 17:20 rule (both teachers
  may agree on the old literal reading),
- named employers in a CV that uses the "Company Name" placeholder (may be borrowed from a duty line).
"""

import copy
import datetime
import hashlib
import json
from pathlib import Path

from scripts.cv_teacher_compare import compare
from src.scoring.extraction import write_json
from src.scoring.normalization import skill_name

ROOT = Path(__file__).resolve().parent.parent
TL = ROOT / "output/scorer_v3/teacher_labels"
EXPECTED_RULE = datetime.datetime(2026, 10, 27, 17, 20).timestamp()


def union_skills(base: dict, other: dict) -> int:
    """shortcut: names match on skill_name aliases only ("SOA" vs "Service Oriented Architecture" stay apart),
    add a variant list if the arbiter audit finds many duplicates."""
    groups = base["document"]["skills"]
    have = {skill_name(d["name"]) for g in groups for d in g.get("details") or []}
    added = 0
    for g in other["document"]["skills"]:
        if g["name"].casefold() == "languages":
            continue  # must mirror document.languages, which the arbiter decides
        for d in g.get("details") or []:
            if skill_name(d["name"]) in have:
                continue
            target = next((x for x in groups if x["name"].casefold() == g["name"].casefold()), None)
            if target is None:
                target = {"name": g["name"], "level": None, "keywords": [], "details": []}
                groups.append(target)
            target["details"].append(copy.deepcopy(d))
            target["keywords"].append(d["name"])
            have.add(skill_name(d["name"]))
            added += 1
    return added


def blind(cv_id: str, field: str, haiku, gemini):
    haiku_first = hashlib.sha256(f"ext-arbiter|{cv_id}|{field}".encode()).digest()[0] % 2 == 0
    return ({"A": haiku, "B": gemini}, "haiku") if haiku_first else ({"A": gemini, "B": haiku}, "gemini")


def build(ids: list[str]) -> tuple[list, dict, dict]:
    items, key, stats = [], {}, {"skills_added": 0}
    for cv_id in ids:
        paths = {t: TL / t / f"{cv_id}.json" for t in ("haiku_b", "gemini_b")}
        haiku, gemini = (json.loads(p.read_text(encoding="utf-8")) for p in paths.values())
        text = json.loads((TL / "inputs" / f"{cv_id}.json").read_text(encoding="utf-8"))["source_text"]
        merged = copy.deepcopy(haiku)
        stats["skills_added"] += union_skills(merged, gemini)
        write_json(TL / "ext_merged" / f"{cv_id}.json", merged)

        def add(kind, field, h, g):
            values, a_is = blind(cv_id, field, h, g)
            items.append({"item": f"{cv_id}|{field}", "id": cv_id, "kind": kind, "field": field, **values})
            key[f"{cv_id}|{field}"] = a_is

        fields = set()
        for x in compare(haiku, gemini)["items"]:
            if x["kind"] in ("fact", "records"):
                field = ("facts." if x["kind"] == "fact" else "document.") + x["field"]
                add(x["kind"], field, x["haiku"], x["gemini"])
                fields.add(field)
        old = min(p.stat().st_mtime for p in paths.values()) < EXPECTED_RULE
        if "Expected in" in text and old and "facts.completed_degrees" not in fields:
            add("expected_in", "facts.completed_degrees",
                {"completed_degrees": haiku["facts"]["completed_degrees"], "education": haiku["document"]["education"]},
                {"completed_degrees": gemini["facts"]["completed_degrees"], "education": gemini["document"]["education"]})
        named = [w["name"] for w in merged["document"]["work"] if w.get("name")]
        if "Company Name" in text and named and "document.work" not in fields:
            items.append({"item": f"{cv_id}|employer", "id": cv_id, "kind": "employer", "field": "document.work[].name",
                          "names": named})
    return items, key, stats


def main():
    ids = [d["id"] for d in json.loads((TL / "manifest_external.json").read_text(encoding="utf-8"))["documents"]]
    items, key, stats = build(ids)
    write_json(TL / "ext_arbiter_items.json", items)
    write_json(TL / "ext_arbiter_key.json", key)
    kinds = {}
    for i in items:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    print("merged", len(ids), "skills added", stats["skills_added"], "items", len(items), kinds,
          "CVs", len({i["id"] for i in items}))


if __name__ == "__main__":
    main()
