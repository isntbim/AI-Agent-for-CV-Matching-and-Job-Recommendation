"""Convert the 88 human-audited gold CVs (corpus.json envelopes) to the v3 teacher-label format.

Mechanical only: renames legacy snake_case keys, restores literal evidence quotes (or drops and logs them),
nulls placeholder employers, drops per-detail provenance and moves languages into a "Languages" skill group.
Judgement calls (domains, degrees, country, strict skill standard, certificate types) are logged for the
rules sweep, never decided here; their finals come back in gold_sweep_final.json, gold_strict_final.json and
gold_training_ready_final.json (record-field fixes such as dates). Writes teacher_labels/gold_v3/<id>.json and gold_v3_conversion.json.
"""

import argparse
import copy
import json
import re
from pathlib import Path

from scripts.cv_teacher_reference import PLACEHOLDERS, apply_field, field_changes
from scripts.cv_teacher_validate import ROOT, check
from src.parser.extractors import clean_text, extract_text_from_file
from src.scoring.extraction import write_json

RENAMES = {"basics.location": {"postal_code": "postalCode", "country_code": "countryCode"},
           "work": {"start_date": "startDate", "end_date": "endDate"},
           "education": {"start_date": "startDate", "end_date": "endDate", "study_type": "studyType"},
           "projects": {"start_date": "startDate", "end_date": "endDate", "project_type": "type"},
           "certificates": {"certificate_type": "type"}}
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
MAX_CHARS = 40000  # production extract_document limit
EXCLUDED = {"cv_042": "only job is the candidate's own college website (job vs student project unclear) and its "
                      "skill quotes break on the re-extracted text; removed (user decision 2026-10-24)"}


def literal(quote: str, text: str) -> str | None:
    """The exact source span for a quote: as-is, with flexible whitespace, or a normalised date's written form."""
    if quote in text:
        return quote
    quote = clean_text(quote)  # quotes taken from text before the parser's artifact cleanup
    if quote in text:
        return quote
    if quote.split():
        # \s* second: glued PDF text drops the spaces between words ("BachelorofScience")
        for gap in (r"\s+", r"\s*"):
            if match := re.search(gap.join(map(re.escape, quote.split())), text):
                return match.group(0)
    squeezed = re.sub(r"\s+", "", quote)
    if len(squeezed) >= 4:  # a quote taken from glued text, matched against re-extracted spaced text
        if match := re.search(r"\s*".join(map(re.escape, squeezed)), text):
            return match.group(0)
    if quote == "Present":
        patterns = [r"\b(?:present|current|now|to date)\b"]
    elif m := re.fullmatch(r"(\d{4})-(\d{2})", quote):
        year, month = m.group(1), int(m.group(2))
        # digit lookarounds, not \b: glued PDF text writes "11/1991to"
        patterns = [rf"(?<!\d)0?{month}\s*/\s*{year}(?!\d)", rf"(?<!\d)0?{month}-{year}(?!\d)",
                    rf"(?<!\d){year}\s*/\s*0?{month}(?!\d)", rf"\b{MONTHS[month - 1]}[a-z]*[.,\s-]*{year}(?!\d)"]
    elif re.fullmatch(r"\d{4}", quote):
        patterns = [rf"(?<!\d){quote}(?!\d)"]
    else:
        return None
    for pattern in patterns:
        if match := re.search(pattern, text, re.IGNORECASE):
            return match.group(0)
    return None


def convert(envelope: dict, text: str, sweep: dict | None = None, strict: list | None = None) -> tuple[dict, dict]:
    label = {"document": copy.deepcopy(envelope["document"]), "facts": copy.deepcopy(envelope["facts"])}
    doc, facts, log = label["document"], label["facts"], {"recovered": [], "dropped": [], "review": []}
    for section, keys in RENAMES.items():
        records = [doc["basics"]["location"]] if section == "basics.location" else doc.get(section) or []
        for record in filter(None, records):
            for old, new in keys.items():
                if old in record:
                    record[new] = record.pop(old)
    for record in doc["certificates"]:
        if record.get("type") == "certification":
            record["type"] = "certificate"
        elif record.get("type") not in ("certificate", "award"):
            log["review"].append(f"certificate type {record.get('type')!r}: {record['name']}")
    for record in doc["projects"]:
        if record.get("type") not in ("project", "publication"):
            log["review"].append(f"project type {record.get('type')!r}: {record['name']}")
            record["type"] = "project"
    for section, key in (("work", "name"), ("education", "institution")):
        for record in doc[section]:
            if (record.get(key) or "").strip().casefold() in PLACEHOLDERS:
                record[key] = None
    for dimension, quotes in facts["evidence"].items():
        kept = []
        for item in quotes:
            span = literal(item["quote"], text)
            if span is None:
                log["dropped"].append(f"{dimension}: {item['quote']}")
            else:
                if span != item["quote"]:
                    log["recovered"].append(f"{dimension}: {item['quote']} -> {span}")
                if {"quote": span} not in kept:
                    kept.append({"quote": span})
        facts["evidence"][dimension] = kept
    facts["evidence"] = {k: v for k, v in facts["evidence"].items() if v}
    if facts["country"] and "location" not in facts["evidence"]:  # country only when written (user rule)
        log["review"].append(f"facts.country {facts['country']} -> null: no literal location quote")
        facts["country"] = None
    for group in doc["skills"]:
        if group["name"] == "Source reviewed skills":  # pipeline name, not a CV heading
            group["name"] = "General"
        for detail in group["details"]:
            for key in ("skill_quote", "proficiency_quote"):  # re-extracted texts change the exact spans
                if detail.get(key) and detail[key] not in text:
                    if span := literal(detail[key], text):
                        detail[key] = span
                    else:
                        log["dropped"].append(f"{key} not in text: {detail['name']}")
            detail.pop("provenance", None)
    languages = {item["language"].casefold(): item for item in doc["languages"]}
    if languages and not any(g["name"] == "Languages" for g in doc["skills"]):
        moved = []
        for group in doc["skills"]:
            moved += [d for d in group["details"] if d["name"].casefold() in languages]
            group["details"] = [d for d in group["details"] if d["name"].casefold() not in languages]
            group["keywords"] = [d["name"] for d in group["details"]]
        have = {d["name"].casefold() for d in moved}
        for key, item in languages.items():
            match = None if key in have else re.search(rf"(?<!\w){re.escape(item['language'])}(?!\w)", text, re.IGNORECASE)
            if match:
                moved.append({"name": item["language"], "proficiency": "normal", "raw_proficiency": None,
                              "skill_quote": match.group(0), "proficiency_quote": None})
            elif key not in have:
                log["dropped"].append(f"language skill: {item['language']} (not found in text)")
        doc["skills"] = [g for g in doc["skills"] if g["details"]]
        if moved:
            doc["skills"].append({"name": "Languages", "level": None, "keywords": [d["name"] for d in moved],
                                  "details": moved})
    for change in strict or []:  # strict-standard finals (gold_strict_final.json): {"drop": name} / {"set": detail}
        if name := change.get("drop"):  # one row drops one entry; the quote picks it when a name is listed twice
            found = [(g, d) for g in doc["skills"] for d in g["details"] if d["name"].casefold() == name.casefold()]
            if not found:
                raise ValueError(f"strict drop not in label: {name}")
            if quote := change.get("skill_quote"):  # a stated quote must pick its own copy, never fall back to another
                spans = {quote} | ({span} if (span := literal(quote, text)) else set())  # never None: it matches a quote-less copy
                found = [f for f in found if f[1].get("skill_quote") in spans]
                if not found:
                    raise ValueError(f"strict drop quote not in label: {name}: {quote!r}")
            group, detail = found[0]
            group["details"].remove(detail)
        if detail := copy.deepcopy(change.get("set")):
            span = literal(detail["skill_quote"], text)
            if not span or " ".join(detail["name"].casefold().split()) not in " ".join(span.casefold().split()):
                raise ValueError(f"strict set quote not in text: {detail['name']}")
            group_name, detail["skill_quote"] = detail.pop("group"), span
            group = next((g for g in doc["skills"] if g["name"] == group_name), None)
            if group is None:
                doc["skills"].append(group := {"name": group_name, "level": None, "details": []})
            group["details"] = [d for d in group["details"] if d["name"].casefold() != detail["name"].casefold()] + [detail]
    for group in doc["skills"]:
        group["keywords"] = [d["name"] for d in group["details"]]
    doc["skills"] = [g for g in doc["skills"] if g["details"]]
    for field, decision in (sweep or {}).items():  # rules-sweep finals (gold_sweep_final.json)
        value = decision["value"]
        if field == "certificates":
            for index, kind in sorted(((int(k), v) for k, v in value.items()), reverse=True):
                if kind == "drop":
                    del doc["certificates"][index]
                else:
                    doc["certificates"][index]["type"] = kind
        elif field.startswith("facts.evidence."):
            spans = [literal(q, text) for q in value]
            if None in spans:
                raise ValueError(f"sweep evidence quote not in text: {value}")
            facts["evidence"][field.rsplit(".", 1)[1]] = [{"quote": s} for s in spans]
        else:
            facts[field.removeprefix("facts.")] = value
    for change in field_changes(strict or []):  # record-field finals and splits, after the sweep's certificate drops
        apply_field(label, change, envelope.get("id", ""))
    # a claim needs a quote: degrees from the education records, domains from the job titles
    for dimension, claim, records, keys in (("education", "completed_degrees", doc["education"], ("studyType", "area")),
                                            ("industry", "domains", doc["work"], ("position",))):
        if facts.get(claim) and dimension not in facts["evidence"]:
            spans = [literal(r[k], text) for r in records for k in keys if r.get(k)]
            if spans := list(dict.fromkeys(s for s in spans if s)):
                facts["evidence"][dimension] = [{"quote": s} for s in spans]
                log["recovered"].append(f"{dimension}: {len(spans)} quotes from {'/'.join(keys)}")
    if not facts["domains"]:
        log["review"].append("facts.domains empty")
    return label, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    args = parser.parse_args()
    corpus = json.loads((ROOT / "output/scorer_v3/corpus.json").read_text(encoding="utf-8"))
    target = args.out / "gold_v3"
    target.mkdir(exist_ok=True)
    texts = args.out / "gold_texts"
    sweep_path = args.out / "gold_sweep_final.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8")) if sweep_path.exists() else {}
    strict_path = args.out / "gold_strict_final.json"
    strict = json.loads(strict_path.read_text(encoding="utf-8")) if strict_path.exists() else {}
    ready_path = args.out / "gold_training_ready_final.json"  # training-ready pass: record-field fixes (dates)
    for cv_id, changes in (json.loads(ready_path.read_text(encoding="utf-8")) if ready_path.exists() else {}).items():
        strict.setdefault(cv_id, []).extend(changes)
    report = []
    for row in (d for d in corpus["documents"] if d["kind"] == "resume"):
        text = (ROOT / row["source"]["text_path"]).read_text(encoding="utf-8")
        entry = {"id": row["id"], "split": row["split"]}
        if row["id"] in EXCLUDED:
            report.append(dict(entry, status="excluded", error=EXCLUDED[row["id"]]))
            (target / f"{row['id']}.json").unlink(missing_ok=True)
            continue
        # corpus texts stay as hashed (scorer benchmark); training uses what the current parser produces
        fixed = texts / f"{row['id']}.txt"
        if not fixed.exists():
            texts.mkdir(exist_ok=True)
            fixed.write_text(extract_text_from_file(ROOT / row["source"]["source_path"]), encoding="utf-8")
        if fixed.read_text(encoding="utf-8").strip() != text.strip():
            text, entry["text_path"] = fixed.read_text(encoding="utf-8"), fixed.relative_to(ROOT).as_posix()
        if len(text) > MAX_CHARS:  # same rule as the long pool CVs (user decision 2026-10-19)
            report.append(dict(entry, status="excluded", error=f"{len(text)} chars > {MAX_CHARS}"))
            continue
        label, log = convert(row["envelope"], text, sweep.get(row["id"]), strict.get(row["id"]))
        path = target / f"{row['id']}.json"
        write_json(path, label)
        result = check(path, text, entry)
        report.append(dict(result, **log))
        print(row["id"], row["split"], result["status"], (result.get("error") or "")[:150])
    write_json(args.out / "gold_v3_conversion.json", report)


if __name__ == "__main__":
    main()
