"""Apply the blinded arbiter decisions (ext_arbiter/<id>.json) to the merged labels (ext_merged/) -> ext_reference/.

A/B map back to a teacher's value through the item file (A/B hold the values themselves); "value" replaces the field.
CVs with no arbiter items are copied unchanged. The Languages skill group is rebuilt from the final document.languages,
because the training-ready standard requires the two to match.
"""

import copy
import json
import re
from pathlib import Path

from scripts.cv_teacher_validate import check
from src.scoring.extraction import write_json

ROOT = Path(__file__).resolve().parent.parent
TL = ROOT / "output/scorer_v3/teacher_labels"


def chosen(item: dict, decision: dict):
    return decision["value"] if decision["decision"] == "value" else item[decision["decision"]]


EVIDENCE = {"facts.completed_degrees": "education", "facts.domains": "industry", "facts.country": "location"}


def apply(label: dict, items: list[dict], decisions: list[dict], other: dict | None = None) -> dict:
    """`other` is the second teacher's label: evidence and Languages details come from it when the chosen value
    has none in the base label."""
    out = copy.deepcopy(label)
    other = other or {"document": {"skills": []}, "facts": {"evidence": {}}}
    by_item = {d["item"]: d for d in decisions}
    for item in items:
        d = by_item[item["item"]]
        if item["kind"] == "employer":
            drop = set(d["value"]["null"])
            for w in out["document"]["work"]:
                if w.get("name") in drop:
                    w["name"] = None
        elif item["kind"] == "expected_in":
            v = chosen(item, d)
            out["facts"]["completed_degrees"], out["document"]["education"] = v["completed_degrees"], v["education"]
        else:
            section, field = item["field"].split(".", 1)
            out[section][field] = chosen(item, d)
        dim = EVIDENCE.get(item["field"]) or ("education" if item["kind"] == "expected_in" else None)
        evidence = out["facts"].setdefault("evidence", {})
        if dim and not evidence.get(dim):
            evidence[dim] = copy.deepcopy(other["facts"].get("evidence", {}).get(dim)) or (
                [{"quote": d["quote"]}] if d.get("quote") else [])
    if any(i["field"] == "document.languages" for i in items):
        names = [l["language"] for l in out["document"]["languages"] if l.get("language")]
        groups = out["document"]["skills"]
        old = next((g for g in groups if g["name"].casefold() == "languages"), None)
        kept = {x["name"].casefold(): x for g in other["document"]["skills"] if g["name"].casefold() == "languages"
                for x in g.get("details", [])}
        kept |= {x["name"].casefold(): x for x in (old or {}).get("details", [])}
        groups[:] = [g for g in groups if g is not old]
        if names:  # keep a teacher's detail (with its quote) when it exists, else the checker will flag the missing quote
            details = [kept.get(n.casefold()) for n in names]
            if all(details):
                groups.append({"name": "Languages", "level": None, "keywords": names, "details": details})
    return out


def drop_bad_levels(label: dict, error: str) -> int:
    """Fix a depth_error: a rejected level is dropped (skill kept at "normal"); a skill whose own quote fails is removed."""
    names = set(re.findall(r"skills\.([^;:]+?)(?:\.proficiency)?:[a-z_]+", error))
    dropped = 0
    for g in label["document"]["skills"]:
        keep = []
        for d in g.get("details") or []:
            if d["name"] in names and (d.get("proficiency") != "normal" or d.get("proficiency_quote")):
                d.update(proficiency="normal", raw_proficiency=None, proficiency_quote=None)
                dropped += 1
            elif d["name"] in names:
                # no level left, so the skill quote itself fails: glued dataset text ("managementComplex problem
                # solvingSuperior") or a plural; the production matcher needs word boundaries, so the skill goes
                dropped += 1
                continue
            keep.append(d)
        g["details"], g["keywords"] = keep, [d["name"] for d in keep]
    label["document"]["skills"] = [g for g in label["document"]["skills"] if g["details"]]
    return dropped


def main():
    items = {}
    for i in json.loads((TL / "ext_arbiter_items.json").read_text(encoding="utf-8")):
        items.setdefault(i["id"], []).append(i)
    ids = [d["id"] for d in json.loads((TL / "manifest_external.json").read_text(encoding="utf-8"))["documents"]]
    changed = levels = 0
    for cv in ids:
        label = json.loads((TL / "ext_merged" / f"{cv}.json").read_text(encoding="utf-8"))
        if cv in items:
            decisions = json.loads((TL / "ext_arbiter" / f"{cv}.json").read_text(encoding="utf-8"))
            other = json.loads((TL / "gemini_b" / f"{cv}.json").read_text(encoding="utf-8"))
            label = apply(label, items[cv], decisions, other)
            changed += 1
        path = TL / "ext_reference" / f"{cv}.json"
        write_json(path, label)
        text = (TL / "texts" / f"{cv}.txt").read_text(encoding="utf-8")
        for _ in range(2):  # a dropped level can leave the skill quote itself failing on the next check
            result = check(path, text, {})
            if result["status"] != "depth_error":
                break
            levels += drop_bad_levels(label, result.get("error") or "")
            write_json(path, label)
    print("wrote", len(ids), "references,", changed, "with arbiter decisions,", levels, "unsupported levels/skills dropped")


if __name__ == "__main__":
    main()
