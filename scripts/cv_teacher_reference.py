"""Assemble approved reference CV labels from two teacher labels and arbiter decisions.

Base document is the Haiku label (it keeps more responsibilities); skills, the
three scored facts and record lists are replaced by the approved arbiter finals.
`overrides` (JSON) holds user-approved corrections that supersede the arbiter.
"""

import argparse
import copy
import json
import re
from pathlib import Path

from scripts.cv_teacher_validate import PLACEHOLDERS, undefined_keys
from src.scoring.extraction import write_json

ROOT = Path(__file__).resolve().parent.parent
RECORD_KEYS = {"document.work": {"name", "position", "url", "startDate", "endDate", "summary", "highlights"},
               "document.education": {"institution", "url", "area", "studyType", "startDate", "endDate", "score", "courses"},
               "document.languages": {"language", "fluency"},
               "document.projects": {"name", "startDate", "endDate", "description", "highlights", "keywords", "url", "type"},
               "document.certificates": {"name", "date", "issuer", "url", "type"}}
NOTE_KEYS = ("status", "context", "credential_evaluation")
# Teacher-invented stand-ins for an unstated employer/school; the source states none, so the target is null.


def template(ref, labels, field):
    match = re.match(r"(haiku|gemini):(\d+)", ref or "")
    if not match:
        return {}
    records = labels[match[1]]["document"].get(field.split(".")[1]) or []
    index = int(match[2]) - 1
    return copy.deepcopy(records[index]) if index < len(records) else {}


def build_records(field, final, labels, names=()):
    out, names = [], list(names)
    for entry in final:
        record = template(entry.get("teacher_record"), labels, field)
        key = "institution" if field == "document.education" else "name"
        if entry.get("action") == "drop":
            continue
        if not record and field != "document.languages" and key not in entry:
            # arbiter-added record without a name: take it from overrides, in order
            record = {key: names.pop(0) if names else None}
        record.update({k: v for k, v in entry.items() if k in RECORD_KEYS[field]})
        notes = "; ".join(str(entry[k]) for k in NOTE_KEYS if entry.get(k))
        if notes and field == "document.work":
            record["summary"] = ((record.get("summary") or "") + " (" + notes + ")").strip()
        out.append(record)
    return out


IDENTITY = {"work": ("name", "position"), "education": ("institution", "studyType"),
            "projects": ("name",), "certificates": ("name",)}


def apply_field(doc: dict, change: dict, cv_id: str) -> None:
    """One record field, {"field": "work[2].endDate", "record": {"name": ..., "position": ...}, "from": "May 2015", "value": "2015-05"},
    or one record split into several periods, {"split": "work[1]", "record": {...}, "into": [{"startDate": ..., "endDate": ...}, ...]}.
    "record" (the record's IDENTITY keys, or a list of them) and "from" guard the index, so a reordered rebuild fails loudly."""
    section, index, key = re.fullmatch(r"(\w+)\[(\d+)\](?:\.(\w+))?", change.get("field") or change["split"]).groups()
    records = doc["document"][section]
    record = records[int(index)]
    identity = {k: record.get(k) for k in IDENTITY[section]}
    allowed = change.get("record") if isinstance(change.get("record"), list) else [change.get("record")]
    if identity not in allowed:  # a list: the pilot anchors are built twice, with different name fixes
        raise ValueError(f"{cv_id} {section}[{index}]: expected record {change.get('record')!r}, found {identity!r}")
    if "split" in change:
        records[int(index):int(index) + 1] = [dict(record, **period) for period in change["into"]]
        return
    if record.get(key) != change["from"]:
        raise ValueError(f"{cv_id} {change['field']}: expected {change['from']!r}, found {record.get(key)!r}")
    record[key] = change["value"]


def field_changes(changes: list) -> list:
    """Record-field changes in apply order: splits last and from the bottom up, because they shift the indices after them."""
    return sorted((c for c in changes if c.get("field") or c.get("split")),
                  key=lambda c: ("split" in c, -int(re.search(r"\[(\d+)\]", c.get("split", "[0]")).group(1))))


def assemble(cv_id, rows, labels, overrides, agreed=()):
    doc = copy.deepcopy(labels["haiku"])
    dropped = set()
    # Batch arbiters only emit rows for agreed skills they drop; the rest are kept as Haiku wrote them.
    skills = {d["name"].casefold(): dict(d, group=s.get("name")) for s in doc["document"].get("skills", [])
              for d in (s.get("details") or []) if d["name"].casefold() in {a.casefold() for a in agreed}}
    for row in rows:
        subject, final = row["subject"], row["final"]
        if subject.startswith("facts.completed_degrees:") or subject == "documentation" and final is None:
            continue
        if subject == "facts.evidence":  # supporting quotes are added per dimension, never replace the others
            for dimension, quotes in (final or {}).items():
                doc["facts"].setdefault("evidence", {}).setdefault(dimension, []).extend(quotes)
        elif subject.startswith("facts."):
            doc["facts"][subject.split(".")[1]] = final
        elif subject == "document.basics.location":
            doc["document"]["basics"]["location"] = final or {k: None for k in ("address", "postalCode", "city", "countryCode", "region")}
        elif subject.startswith("document.basics."):  # a single contact field (name, email, phone, ...)
            doc["document"]["basics"][subject.split(".")[2]] = final
        elif subject in RECORD_KEYS:
            names = overrides.get("_records", {}).get(f"{cv_id}/{subject}", [])
            doc["document"][subject.split(".")[1]] = build_records(subject, final, labels, names)
        else:  # a skill row
            for detail in final if isinstance(final, list) else [final]:
                if detail is None:
                    dropped.add(subject.casefold())
                else:
                    skills[detail["name"].casefold()] = detail
    for key in dropped:
        if not any(r["decision"] != "drop" and isinstance(r["final"], dict) and r["final"].get("name", "").casefold() == key
                   for r in rows):
            skills.pop(key, None)
    for change in overrides.get(cv_id, []):
        if change.get("drop"):
            skills.pop(change["drop"].casefold(), None)
        if change.get("set"):
            skills[change["set"]["name"].casefold()] = change["set"]
    for field, key in (("work", "name"), ("education", "institution")):
        for record in doc["document"].get(field) or []:
            if (record.get(key) or "").strip().casefold() in PLACEHOLDERS:
                record[key] = None
    for change in field_changes(overrides.get(cv_id, [])):
        apply_field(doc, change, cv_id)
    groups = {}
    for detail in skills.values():
        detail = dict(detail)
        groups.setdefault(detail.pop("group", None) or "Skills", []).append(detail)
    doc["document"]["skills"] = [{"name": g, "level": None, "keywords": [d["name"] for d in ds], "details": ds}
                                 for g, ds in groups.items()]
    # Evidence lists from both teachers are literal quotes; claims changed by the arbiter need their support.
    evidence = doc["facts"].setdefault("evidence", {})
    for dimension, quotes in (labels["gemini"]["facts"].get("evidence") or {}).items():
        have = {q["quote"] for q in evidence.get(dimension, [])}
        evidence.setdefault(dimension, []).extend(q for q in quotes if q["quote"] not in have)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    parser.add_argument("--teachers", nargs=2, default=["haiku_v21", "gemini_v21"])
    parser.add_argument("--arbiter", nargs="+", default=["arbiter_pilot.json"], help="row files, concatenated in order")
    parser.add_argument("--overrides", nargs="+", default=["reference_overrides.json"], help="merged in order; later changes win")
    parser.add_argument("--target", default="reference")
    parser.add_argument("--disagreements", default=None, help="batch mode: disagreements_<chunk>.json with agreed skills")
    args = parser.parse_args()
    rows = [r for name in args.arbiter for r in json.loads((args.out / name).read_text(encoding="utf-8"))]
    overrides = {"_records": {}}
    for path in (args.out / name for name in args.overrides):
        for key, value in (json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}).items():
            if key == "_records":
                overrides["_records"].update(value)
            elif not key.startswith("_"):
                overrides.setdefault(key, []).extend(value)
    known = {d["id"] for d in json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))["documents"]}
    if unknown := sorted(k for k in overrides if k != "_records" and k not in known):  # a mistyped id would never apply
        raise SystemExit(f"Overrides for unknown CV ids, nothing written: {unknown}")
    agreed = json.loads((args.out / args.disagreements).read_text(encoding="utf-8")) if args.disagreements else {}
    ids = sorted({r["id"] for r in rows} | set(agreed))
    docs = {}
    for cv_id in ids:
        labels = {name: json.loads((args.out / t / f"{cv_id}.json").read_text(encoding="utf-8"))
                  for name, t in zip(("haiku", "gemini"), args.teachers)}
        docs[cv_id] = assemble(cv_id, [r for r in rows if r["id"] == cv_id], labels, overrides,
                               agreed.get(cv_id, {}).get("agreed_skills", ()))
    # Reference labels are training targets: refuse to write any off-schema key rather than let the loader drop it.
    bad = {cv_id: keys for cv_id, doc in docs.items() if (keys := undefined_keys(doc))}
    if bad:
        raise SystemExit("Undefined schema keys, nothing written:\n" + "\n".join(f"  {k}: {v}" for k, v in bad.items()))
    for cv_id, doc in docs.items():
        write_json(args.out / args.target / f"{cv_id}.json", doc)
    print("wrote", len(ids), "reference labels to", args.out / args.target)


if __name__ == "__main__":
    main()
