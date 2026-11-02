"""Self-check ONE teacher CV label before handing it in (used by the cv-teacher-label skill).

Prints ERROR lines (must fix, exit 1) and WARN lines (decide, then mention in the final note).
Read-only: never repairs or rewrites the label. Runs the same production checks as cv_teacher_validate.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from scripts.cv_teacher_reference import PLACEHOLDERS
from scripts.cv_teacher_validate import ROOT, check, undefined_keys
from src.scoring.v3_aliases import name_in_quote

OUT = ROOT / "output/scorer_v3/teacher_labels"
FLAGS = ("no_experience_required", "no_education_required", "no_skills", "remote", "worldwide_remote")
FIX = {
    "depth_not_explicit_skill_specific": "the level is not written right next to this skill name (e.g. 'Advanced Excel', "
    "'Proficient in Excel', 'Excel (Advanced)', 'Excel: Expert'). If it labels a group or sits elsewhere, set "
    "proficiency 'normal' and keep raw_proficiency + proficiency_quote. Keep it only if it clearly belongs to this one skill.",
    "depth_quote_not_in_source": "proficiency_quote is not an exact substring of the CV text",
    "depth_negated_or_ambiguous_source_scope": "the level sits in a negated or teaching/training clause: set 'normal', keep raw + quote",
    "unmapped_proficiency_wording": "fine: wording like 'fluent'/'professional' stays normal with raw_proficiency kept for review",
}


def squash(value: str) -> str:
    """Compare without whitespace or case: PDFs drop spaces ("andJavaScript") and letter-space headers ("A N T H O N Y")."""
    return re.sub(r"\s+", "", value).casefold()


def no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"duplicate keys {sorted(dup)}")
    return dict(pairs)


def quotes(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("quote") and isinstance(value, str):
                yield f"{path}.{key}", value
            else:
                yield from quotes(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from quotes(value, f"{path}[{i}]")


def source_of(cv_id: str) -> str:
    return json.loads((OUT / "inputs" / f"{cv_id}.json").read_text(encoding="utf-8"))["source_text"]


def lint(label: dict, text: str, cv_id: str) -> tuple[list[str], list[str]]:
    errors, warns = [], []
    if set(label) != {"document", "facts"}:
        errors.append(f"root keys must be exactly document and facts, found {sorted(label)}")
    errors += [f"key not in schema: {k}" for k in undefined_keys(label)]
    doc, facts = label.get("document") or {}, label.get("facts") or {}
    if doc.get("schema_version") != "resume-v3":
        errors.append('document.schema_version must be "resume-v3"')
    errors += [f"facts.{k} must be true or false, not {facts[k]!r}" for k in FLAGS if k in facts and not isinstance(facts[k], bool)]

    others = None
    for path, quote in quotes(label):
        if quote not in text:
            if others is None:
                others = {p.stem: p for p in (OUT / "inputs").glob("*.json") if p.stem != cv_id}
            elsewhere = next((o for o, p in others.items() if quote in source_of(o)), None) if len(quote) > 15 else None
            hint = f" -- it IS in {elsewhere}: you may have mixed up two CVs" if elsewhere else \
                " -- copy the exact characters (keep the CV's spacing, line breaks, typos)"
            errors.append(f"{path} not an exact substring of {cv_id}: {quote!r}{hint}")
        elif len(quote.split()) > 20 and not path.endswith("proficiency_quote"):  # no fixed cap (prompt), but long spans are rare
            warns.append(f"{path} has {len(quote.split())} words (use the shortest span that keeps the claim): {quote!r}")

    names_by_group, languages_in_skills = {}, []
    for g, group in enumerate(doc.get("skills") or []):
        details = group.get("details") or []
        names = [d.get("name") for d in details]
        if sorted(k or "" for k in group.get("keywords") or []) != sorted(n or "" for n in names):
            errors.append(f"skills[{g}] ({group.get('name')}): keywords must equal the details' names exactly")
        names_by_group[group.get("name")] = names
        if (group.get("name") or "").casefold() == "languages":
            languages_in_skills += [squash(n) for n in names if n]
        for d, detail in enumerate(details):
            where = f"skills[{g}].details[{d}] {detail.get('name')!r}"
            if detail.get("provenance") is not None:
                errors.append(f"{where}: provenance must be null or absent (not a teacher field)")
            name, quote = detail.get("name") or "", detail.get("skill_quote") or ""
            if not quote or not name_in_quote(name, quote):
                if quote and squash(name) in squash(quote):  # PDF dropped the spaces ("andJavaScript"): the source is at fault, not the label
                    warns.append(f"{where}: name is glued to other words in the CV text ({quote!r}); keep it, quote exactly as is")
                else:
                    errors.append(f"{where}: the name must appear inside its own skill_quote")
            if detail.get("proficiency", "normal") != "normal" and not (detail.get("raw_proficiency") and detail.get("proficiency_quote")):
                errors.append(f"{where}: a non-normal level needs raw_proficiency and proficiency_quote")
    for group, names in names_by_group.items():
        if (group or "").casefold() != "languages":
            for n in names:
                if n and squash(n) in languages_in_skills:
                    errors.append(f"language {n!r} repeated in skill group {group!r}: keep it only under 'Languages'")

    evidence = facts.get("evidence") or {}
    if facts.get("domains") and not evidence.get("industry"):
        errors.append("facts.domains is set, so facts.evidence.industry needs literal role/task quotes")
    if facts.get("completed_degrees") and not evidence.get("education"):
        errors.append("facts.completed_degrees is set, so facts.evidence.education needs a literal quote")
    if facts.get("country") and not evidence.get("location"):
        errors.append("facts.country is set, so facts.evidence.location needs a literal quote of the candidate's own address")

    for field, key in (("work", "name"), ("education", "institution")):
        for i, record in enumerate(doc.get(field) or []):
            value = (record.get(key) or "").strip()
            if field == "work" and value.casefold() not in PLACEHOLDERS and squash(value) not in squash(text):
                warns.append(f"work[{i}].name {value!r} is not written in the CV: never invent or paraphrase an employer")
    name = ((doc.get("basics") or {}).get("name") or "").strip()
    if name and name != "Candidate" and not any(squash(part) in squash(text) for part in name.split() if len(part) > 2):
        warns.append(f"basics.name {name!r} is not in this CV: wrong CV? Use 'Candidate' when none is stated")
    return errors, warns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", help="your label folder, e.g. haiku_b or gemini_b")
    parser.add_argument("ids", nargs="+")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.teacher):
        sys.exit("teacher must be a folder name like haiku_b")
    manifest = {r["id"]: r for name in ("manifest.json", "manifest_external.json", "manifest_goldcal.json") if (OUT / name).exists()
                for r in json.loads((OUT / name).read_text(encoding="utf-8"))["documents"]}
    failed = False
    for cv_id in args.ids:
        if cv_id not in manifest or not (OUT / "inputs" / f"{cv_id}.json").exists():
            print(f"{cv_id}: ERROR unknown id"); failed = True; continue
        path, text = OUT / args.teacher / f"{cv_id}.json", source_of(cv_id)
        if (ROOT / manifest[cv_id]["text_path"]).read_text(encoding="utf-8") != text:
            print(f"{cv_id}: ERROR input text differs from the manifest text; tell the leader"); failed = True; continue
        try:
            label = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
        except FileNotFoundError:
            print(f"{cv_id}: ERROR {path} not written"); failed = True; continue
        except ValueError as exc:
            print(f"{cv_id}: ERROR not valid JSON: {exc}"); failed = True; continue
        errors, warns = lint(label, text, cv_id)
        if not errors:  # the production path, only once the label is mechanically sound
            result = check(path, text, {"id": cv_id})
            for issue in result.get("issues", []):
                reason = next((r for r in FIX if r in issue), None)
                warns.append(f"review: {issue}" + (f" -> {FIX[reason]}" if reason else ""))
            if result["status"] == "depth_error":
                reasons = [r for r in FIX if r in result["error"]]
                # a negated/teaching clause is never a level; a level a few words away may still be right (human review)
                (errors if "depth_negated_or_ambiguous_source_scope" in result["error"] else warns).append(
                    f"level check: {result['error']}" + "".join(f" -> {FIX[r]}" for r in reasons))
            elif result["status"] == "fail":
                errors.append(f"production validator: {result['error']}")
            errors += [f"training-ready standard: {p}" for p in result.get("policy", [])]
        for line in errors:
            print(f"{cv_id}: ERROR {line}")
        for line in warns:
            print(f"{cv_id}: WARN  {line}")
        print(f"{cv_id}: {'FIX AND RERUN' if errors else 'OK'} ({len(errors)} errors, {len(warns)} warnings)")
        failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
