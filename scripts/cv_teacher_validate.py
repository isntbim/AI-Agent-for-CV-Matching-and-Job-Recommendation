"""Validate teacher CV labels with the same checks as the live v3 extraction path."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

from src.parser.schemas import ResumeSchema
from src.parser.skill_v3 import CVSkillDetailV3
from src.scoring.extraction import extract_document, write_json
from src.scoring.models import Facts
from src.scoring.v3_validation import validate_skill_evidence

ROOT = Path(__file__).resolve().parent.parent
LABEL_SCHEMA = {"document": ResumeSchema.model_json_schema(by_alias=True), "facts": Facts.model_json_schema()}
LABEL_SCHEMA["document"]["$defs"]["CVSkillDetail"] = CVSkillDetailV3.model_json_schema()
PLACEHOLDERS = {"organization", "company name", "educational institution", "faculty", "organization not stated", "not stated"}
DATE = re.compile(r"\d{4}(-\d{2}(-\d{2})?)?")
DATE_FIELDS = {"work": ("startDate", "endDate"), "education": ("startDate", "endDate"),
               "projects": ("startDate", "endDate"), "certificates": ("date",)}


def real_date(value: str) -> bool:
    """YYYY, YYYY-MM or YYYY-MM-DD that exists on the calendar (2015-13 or 2015-02-31 is a misread day/month order)."""
    try:
        return bool(DATE.fullmatch(value)) and bool(date.fromisoformat((value + "-01-01")[:10]))
    except ValueError:
        return False


def word_in(name: str, text: str) -> bool:
    """Whole-word, case-insensitive: "Mandarin" is in "Mandarin / Chinese", "Malay" is not in "Malayalam"."""
    return bool(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE))


def policy_issues(label: dict, text: str) -> list[str]:
    """Training-ready standard: label rules of the CV-v3 prompt that the production parser does not enforce."""
    doc, facts, found = label["document"], label["facts"], []
    if doc.get("schema_version") != "resume-v3":
        found.append(f"schema_version {doc.get('schema_version')!r}")
    for section, fields in DATE_FIELDS.items():
        for i, record in enumerate(doc.get(section) or []):
            for key in fields:
                value = record.get(key)
                ongoing = value == "Present" and key == "endDate" and section in ("work", "projects")
                if value is not None and not ongoing and not real_date(value):
                    found.append(f"date {section}[{i}].{key}={value!r}: use a real YYYY, YYYY-MM or YYYY-MM-DD date or null; "
                                 "Present only as a work/project endDate")
    for section, key in (("work", "name"), ("education", "institution")):
        for record in doc.get(section) or []:
            if (record.get(key) or "").strip().casefold() in PLACEHOLDERS:
                found.append(f"placeholder {section}.{key}={record[key]!r}")
    groups = [g for g in doc.get("skills") or [] if g["name"].casefold() == "languages"]
    if len(groups) > 1:
        found.append("more than one Languages group")
    spoken = [d["name"].casefold() for g in groups for d in g["details"]]
    for item in doc.get("languages") or []:  # a skill only when the name is written (a translated name has no literal quote)
        name = item["language"].casefold()
        if word_in(name, text) and not any(word_in(name, s) for s in spoken):
            found.append(f"language without Languages skill: {item['language']}")
    languages = [item["language"].casefold() for item in doc.get("languages") or []]
    found += [f"Languages skill not in document.languages: {s}" for s in spoken if not any(word_in(l, s) for l in languages)]
    for group in doc.get("skills") or []:
        for detail in group["details"]:
            if detail["name"] != " ".join(detail["name"].split()):
                found.append(f"skill name whitespace: {detail['name']!r}")
    evidence = facts.get("evidence") or {}
    for claim, dimension in (("country", "location"), ("domains", "industry"), ("completed_degrees", "education")):
        if facts.get(claim) and not evidence.get(dimension):
            found.append(f"facts.{claim} without evidence.{dimension}")
    return found


def undefined_keys(label: dict, schemas: dict = LABEL_SCHEMA) -> list[str]:
    """Keys outside the v3 label schema. ResumeSchema ignores them on load, so they would vanish silently."""
    found = [k for k in label if k not in schemas]

    def walk(value, schema, defs, path):
        while "$ref" in schema or "anyOf" in schema:
            schema = defs[schema["$ref"].split("/")[-1]] if "$ref" in schema else next(
                (s for s in schema["anyOf"] if s.get("type") != "null"), {})
        if isinstance(value, dict) and "properties" in schema:
            for key, item in value.items():
                if key in schema["properties"]:
                    walk(item, schema["properties"][key], defs, f"{path}.{key}")
                else:
                    found.append(f"{path}.{key}")
        elif isinstance(value, list) and "items" in schema:
            for item in value:
                walk(item, schema["items"], defs, path + "[]")

    for part, schema in schemas.items():
        walk(label.get(part) or {}, schema, {**schema.get("$defs", {})}, part)
    return sorted(set(found))


class FileClient:
    """Replays a teacher's saved JSON as the LLM response."""

    def __init__(self, response: str):
        self.response = response

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=4096):
        return self.response


def selected(out: Path, split: str) -> list[dict]:
    """A manifest split name, or a chunk name from chunks.json."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    chunks_path = out / "chunks.json"
    chunks = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else {}
    ids = set(chunks[split]) if split in chunks else None
    return [r for r in manifest["documents"] if "text_path" in r and (r["id"] in ids if ids is not None else r["split"] == split)]


def check(label: Path, text: str, result: dict) -> dict:
    """One label against its source text; the same checks the live v3 extraction path runs."""
    if not label.exists():
        return dict(result, status="missing")
    try:
        extra = undefined_keys(json.loads(label.read_text(encoding="utf-8")))
    except (ValueError, AttributeError):
        extra = []  # not a JSON object: extract_document below reports it
    if extra:
        return dict(result, status="fail", error="Undefined schema keys: " + ", ".join(extra))
    try:
        parsed = extract_document(text, "resume", FileClient(label.read_text(encoding="utf-8")), ruleset="v3")
        issues = validate_skill_evidence(parsed, text)
        skills = [d for s in parsed.document.skills for d in (s.details or [])]
        result.update(status="pass" if not issues else "partial", issues=[str(i) for i in issues],
                      skills=len(skills), leveled=sum(d.proficiency != "normal" for d in skills),
                      work=len(parsed.document.work), education=len(parsed.document.education))
        if policy := policy_issues(json.loads(label.read_text(encoding="utf-8")), text):
            result.update(status="policy_fail", policy=policy, error="Training-ready standard: " + "; ".join(policy)[:480])
    except Exception as exc:
        # Skill-depth evidence errors are fixable per detail (arbiter/review); anything else rejects the label.
        status = "depth_error" if "Invalid v3 skill evidence" in str(exc) else "fail"
        result.update(status=status, error=f"{type(exc).__name__}: {exc}"[:500])
        if status == "depth_error":  # the label parsed, so report its policy hits now rather than after the level fix
            result["policy"] = policy_issues(json.loads(label.read_text(encoding="utf-8")), text)
    return result


def validate(out: Path, teacher: str, split: str) -> list[dict]:
    return [check(out / teacher / (row["id"] + ".json"), (ROOT / row["text_path"]).read_text(encoding="utf-8"),
                  {"id": row["id"], "teacher": teacher})
            for row in selected(out, split)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    parser.add_argument("--teachers", nargs="+", default=["haiku", "gemini"])
    parser.add_argument("--split", default="pilot")
    args = parser.parse_args()
    results = [r for t in args.teachers for r in validate(args.out, t, args.split)]
    write_json(args.out / f"validation_{args.split}.json", results)
    for r in results:
        print(r["teacher"], r["id"], r["status"], r.get("skills", ""), r.get("leveled", ""),
              (r.get("error") or "; ".join(r.get("issues", [])))[:200])


if __name__ == "__main__":
    main()
