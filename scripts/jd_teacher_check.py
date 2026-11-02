"""Self-check JD teacher labels with the live v3 extraction path and training-ready policy rules."""

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

from scripts.cv_teacher_validate import ROOT, FileClient, undefined_keys
from src.parser.job_schemas import JobDescriptionSchema
from src.scoring.extraction import extract_document
from src.scoring.models import Facts
from src.scoring.normalization import ROLE_FAMILIES, normalize
from src.scoring.silver import title_like_skill
from src.scoring.v2_validation import OPTIONAL
from src.scoring.v3_aliases import name_in_quote
from src.scoring.v3_validation import validate_skill_evidence

OUT = ROOT / "output/scorer_v3/teacher_labels/jd"
JD_SCHEMA = {"document": JobDescriptionSchema.model_json_schema(by_alias=True), "facts": Facts.model_json_schema()}
JD_SCHEMA["document"]["$defs"]["RequirementUnit"] = JD_SCHEMA["document"]["$defs"]["RequirementUnitV3"]
REQ_ID = re.compile(r"req_\d{2,}")
RAW_SECTIONS = ("description", "skills", "benefits", "compensation")
# production OPTIONAL misses Vietnamese "lợi thế"/"điểm cộng" and "bonus points"; teachers use the full list
PREFERRED = re.compile(OPTIONAL.pattern + r"|\bbonus points?\b|\bprefer\w*|\bgood to have\b|\bappreciated\b|\bpriority\b"
                       r"|\badvantage\w*|\bdesire[ds]?\b|\bbeneficial\b"
                       r"|\b(?:an|a big) asset\b|lợi\s+thế|điểm\s+cộng|cộng\s+điểm", re.I)


def jd_policy_issues(label: dict, text: str) -> list[str]:
    """Training-ready standard: JD v3 rules that the production parser does not enforce."""
    doc, facts, found = label.get("document") or {}, label.get("facts") or {}, []
    skills = doc.get("skills") or {}
    reqs = skills.get("requirements") or []
    ev = facts.get("evidence") or {}
    if doc.get("schema_version") != "job-v3":
        found.append(f"schema_version {doc.get('schema_version')!r}")
    seen_ids = set()
    for i, r in enumerate(reqs):
        rid = r.get("id") or f"requirements[{i}]"
        if not REQ_ID.fullmatch(str(r.get("id") or "")):
            found.append(f"requirement id {r.get('id')!r}: must match req_\\d{{2,}}")
        elif rid in seen_ids:
            found.append(f"duplicate requirement id {rid!r}")
        seen_ids.add(rid)
        opts, details = r.get("options") or [], r.get("option_details") or []
        if not opts:
            found.append(f"{rid}: options must not be empty")
        if [d.get("name") for d in details] != opts:
            found.append(f"{rid}: option_details names must equal options in the same order")
        if r.get("importance") not in ("core", "standard", "preferred"):
            found.append(f"{rid}: importance={r.get('importance')!r}")
        if r.get("review_status") != "pending":
            found.append(f"{rid}: review_status={r.get('review_status')!r} (must be 'pending')")
        for name in opts:
            if not name or not name.strip() or name != " ".join(name.split()):
                found.append(f"{rid}: option name whitespace/empty {name!r}")
        # NFC: some sources store Vietnamese decomposed, which the composed regex misses
        if r.get("importance") == "preferred" and not PREFERRED.search(unicodedata.normalize("NFC", r.get("importance_quote") or "")):
            found.append(f"{rid}: preferred requirement needs importance_quote with optional wording (ưu tiên, a plus, ...)")
        for d in details:  # production skips raw wording at normal depth; a kept raw level still needs its source
            raw, quote = d.get("raw_proficiency"), d.get("proficiency_quote")
            if raw and not (quote and quote in text and raw.casefold() in quote.casefold() and name_in_quote(d.get("name") or "", quote)):
                found.append(f"{rid}: raw_proficiency {raw!r} needs a literal proficiency_quote naming {d.get('name')!r}")
            for e in d.get("experience") or []:
                lo, hi = e.get("min_years"), e.get("max_years")
                if (lo is not None and lo < 0) or (lo is not None and hi is not None and lo > hi):
                    found.append(f"{rid}: {d.get('name')!r} experience years {lo!r}..{hi!r}")
    req_list, pref_list = skills.get("required") or [], skills.get("preferred") or []
    req_names, pref_names = [s.get("name") for s in req_list], [s.get("name") for s in pref_list]
    if len(req_names) != len(set(req_names)):
        found.append("skills.required has duplicate names")
    if len(pref_names) != len(set(pref_names)):
        found.append("skills.preferred has duplicate names")
    if any(s.get("proficiency") != "required" for s in req_list):
        found.append("skills.required proficiency must be 'required'")
    if any(s.get("proficiency") != "preferred" for s in pref_list):
        found.append("skills.preferred proficiency must be 'preferred'")
    if set(req_names) != {o for r in reqs if r.get("importance") in ("core", "standard") for o in (r.get("options") or [])}:
        found.append("skills.required must mirror core+standard requirement options")
    if set(pref_names) != {o for r in reqs if r.get("importance") == "preferred" for o in (r.get("options") or [])}:
        found.append("skills.preferred must mirror preferred requirement options")
    title = doc.get("title") or ""
    norm_title = normalize(title).replace("_", " ")
    for name in dict.fromkeys(o for r in reqs for o in (r.get("options") or []) if o):
        if title_like_skill(name, title):
            if norm_title and norm_title in normalize(name):  # the title itself; "Python" inside "Python Developer" is a WARN
                found.append(f"title pollution: option {name!r} is the job title {title!r}")
            else:
                found.append(f"WARN: {name!r}: is this a capability or a title/seniority word?")
    sen, loc, comp = doc.get("seniority") or {}, doc.get("location") or {}, doc.get("compensation") or {}
    sal_vals = [comp.get(k) for k in ("min_monthly", "max_monthly", "min_annual", "max_annual")]
    if facts.get("domains") and not ev.get("industry"):
        found.append("facts.domains without evidence.industry")
    if (facts.get("minimum_degree") is not None or facts.get("no_education_required")) and not ev.get("education"):
        found.append("education claim without evidence.education")
    if reqs and not ev.get("skills"):
        found.append("requirements without evidence.skills")
    if (sen.get("min_years") is not None or sen.get("max_years") is not None or facts.get("no_experience_required")) and not ev.get("experience"):
        found.append("experience claim without evidence.experience")
    if (facts.get("country") or facts.get("remote") or loc.get("city")) and not ev.get("location"):
        found.append("location claim without evidence.location")
    if (facts.get("currency") or any(v is not None for v in sal_vals)) and not ev.get("salary"):
        found.append("salary claim without evidence.salary")
    if any(v is not None for v in sal_vals) and not facts.get("currency"):
        found.append("salary min/max without facts.currency")
    if any(v is not None and v < 0 for v in sal_vals):
        found.append("negative salary")
    if (any(v is not None for v in sal_vals) or facts.get("currency")) and comp.get("currency") != facts.get("currency"):
        found.append(f"compensation.currency {comp.get('currency')!r} != facts.currency {facts.get('currency')!r}")
    if comp.get("min_monthly") is not None and comp.get("max_monthly") is not None and comp["min_monthly"] > comp["max_monthly"]:
        found.append("compensation.min_monthly > max_monthly")
    if comp.get("min_annual") is not None and comp.get("max_annual") is not None and comp["min_annual"] > comp["max_annual"]:
        found.append("compensation.min_annual > max_annual")
    min_y, max_y = sen.get("min_years"), sen.get("max_years")
    if min_y is not None and min_y < 0:
        found.append(f"seniority.min_years={min_y!r} is negative")
    elif min_y is not None and min_y > 40:
        found.append(f"WARN: seniority.min_years={min_y!r}: unusually high, check the source")
    if min_y is not None and max_y is not None and min_y > max_y:
        found.append("seniority.min_years > max_years")
    for section in RAW_SECTIONS:
        if (doc.get(section) or {}).get("raw_text") is not None:
            found.append(f"{section}.raw_text must be null")
    if (doc.get("metadata") or {}).get("language") not in ("en", "vi", "mixed"):
        found.append(f"metadata.language={(doc.get('metadata') or {}).get('language')!r}")
    for dom in facts.get("domains") or []:
        if dom not in ROLE_FAMILIES:
            found.append(f"facts.domains {dom!r} not in ROLE_FAMILIES")
    if facts.get("completed_degrees"):
        found.append("facts.completed_degrees must be []")
    return found


def check_jd(label_path: Path, text: str) -> dict:
    """Validate one JD teacher label against its source text with live v3 + training-ready checks."""
    if not label_path.exists():
        return {"status": "missing", "issues": [], "policy": [], "error": None}
    raw = label_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        return {"status": "fail", "issues": [], "policy": [], "error": f"Invalid JSON: {exc}"}
    if not isinstance(data, dict) or set(data) != {"document", "facts"}:
        return {"status": "fail", "issues": [], "policy": [], "error": "Root keys must be exactly document and facts"}
    if not isinstance(data.get("document"), dict) or not isinstance(data.get("facts"), dict):
        return {"status": "fail", "issues": [], "policy": [], "error": "document and facts must be objects"}
    if extra := undefined_keys(data, JD_SCHEMA):
        return {"status": "fail", "issues": [], "policy": [], "error": "Undefined schema keys: " + ", ".join(extra)}
    policy = jd_policy_issues(data, text)
    errors = [p for p in policy if not p.startswith("WARN:")]
    try:
        parsed = extract_document(text, "job", FileClient(raw), ruleset="v3")
        issues = [str(i) for i in validate_skill_evidence(parsed, text)]
    except Exception as exc:
        return {"status": "fail", "issues": [], "policy": policy, "error": f"{type(exc).__name__}: {exc}"[:500]}
    if errors:
        return {"status": "policy_fail", "issues": issues, "policy": policy,
                "error": ("Training-ready standard: " + "; ".join(errors))[:480]}
    return {"status": "partial" if issues else "pass", "issues": issues, "policy": policy, "error": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", help="label folder under teacher_labels/jd/, e.g. gemini")
    parser.add_argument("ids", nargs="+")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese option names crash cp1252 consoles and redirects
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.teacher):
        sys.exit("teacher must be a folder name matching [A-Za-z0-9_]+")
    manifest = {r["id"]: r for r in json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["documents"]}
    failed = False
    for jd_id in args.ids:
        row = manifest.get(jd_id)
        input_path = ROOT / row["input_path"] if row and "input_path" in row else OUT / "inputs" / f"{jd_id}.json"
        if not row or not input_path.exists():
            print(f"{jd_id}: ERROR unknown id"); failed = True; continue
        text = json.loads(input_path.read_text(encoding="utf-8"))["source_text"]
        if (ROOT / row["text_path"]).read_text(encoding="utf-8") != text:
            print(f"{jd_id}: ERROR input text differs from the manifest text; tell the leader"); failed = True; continue
        result = check_jd(OUT / args.teacher / f"{jd_id}.json", text)
        errors = [p for p in result["policy"] if not p.startswith("WARN:")]
        warns = [f"review: {i}" for i in result["issues"]] + [p.removeprefix("WARN: ") for p in result["policy"] if p.startswith("WARN:")]
        if result["status"] == "missing":
            errors.insert(0, "missing label file")
        elif result["status"] == "fail" and result["error"]:
            errors.insert(0, result["error"])
        for line in errors:
            print(f"{jd_id}: ERROR {line}")
        for line in warns:
            print(f"{jd_id}: WARN {line}")
        if not errors:
            print(f"{jd_id}: OK {result['status']}")
        failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
