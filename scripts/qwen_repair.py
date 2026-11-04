"""Deterministic repair of a model's extraction JSON before validation (the "repair" half of validate-and-retry).

It never invents content. It only restores literal source quotes, drops claims whose evidence cannot be found,
and rebuilds fields that are derived from other fields (JD required/preferred lists, the CV Languages group).
Whatever still fails after repair goes to a model retry (training/scripts/generate.py --retry-of).

Usage: python -m scripts.qwen_repair <preds.jsonl> <out.jsonl> [--retry retry.jsonl]   (adds "repaired" and "repair_log")
--retry: generate.py output for the ids still invalid after repair, one row per id per repetition penalty; the
lowest penalty that repairs to a valid label replaces the first answer (the gold labels are never consulted).
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

from scripts.cv_teacher_validate import undefined_keys, word_in
from scripts.jd_teacher_check import JD_SCHEMA, PREFERRED
from scripts.qwen_eval import DATA, parse, validate
from src.scoring.normalization import ROLE_FAMILIES
from src.scoring.v3_aliases import name_in_quote

MAX_ROUNDS = 8
ISSUE = re.compile(r"((?:skills|requirements)\.[^:;]+):(\w+)")


def snap(quote: str | None, text: str) -> str | None:
    """The source span a quote stands for: exact, else the same words with any whitespace/case; None if absent."""
    if not quote or not quote.strip():
        return None
    if quote in text:
        return quote
    match = re.search(r"\s+".join(re.escape(w) for w in quote.split()), text, re.I)
    return match.group(0) if match else None


def balance(raw: str) -> str:
    """Drop stray closing brackets: one that does not match its opener, or that would close the root object before
    the end of the output (the model's "]}]}]}]" slip). Content is never added or changed."""
    out, stack, in_str, esc = [], [], False, False
    end = len(raw.rstrip())
    for i, ch in enumerate(raw):
        if in_str:
            in_str = not (ch == '"' and not esc)
            esc = ch == "\\" and not esc
        elif ch == '"':
            in_str, esc = True, False
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack or "{[".index(stack[-1]) != "}]".index(ch) or (len(stack) == 1 and i < end - 1):
                continue
            stack.pop()
        out.append(ch)
    return "".join(out)


def parse_balanced(raw: str, schema: dict | None = None) -> dict | None:
    """Parse, else delete the one stray closing bracket that leaves the best-fitting JSON (fewest keys outside the
    schema: a wrongly placed delete nests later sections under the wrong parent). Falls back to balance()."""
    if (data := parse(raw)) is not None:
        return data
    found = [d for j, ch in enumerate(raw) if ch in "}]" and (d := parse(raw[:j] + raw[j + 1:])) is not None]
    if (d := parse(balance(raw))) is not None:
        found.append(d)
    fits = lambda d: len(undefined_keys(d, schema) if schema else undefined_keys(d)) if isinstance(d.get("document"), dict) else 10**6
    return min(found, key=fits) if found else None


def source_line(name: str, text: str) -> str | None:
    """The first source line that names the skill: a literal quote the validator can link."""
    return next((line.strip() for line in text.splitlines() if line.strip() and name_in_quote(name, line.strip())), None)


def drop_undefined(label: dict, schema: dict | None) -> list[str]:
    paths = undefined_keys(label, schema) if schema else undefined_keys(label)

    def remove(node, parts):
        if isinstance(node, list):
            for item in node:
                remove(item, parts)
        elif isinstance(node, dict) and parts:
            key = parts[0].removesuffix("[]")
            if len(parts) == 1:
                node.pop(key, None)
            elif key in node:
                remove(node[key], parts[1:])

    for path in paths:
        remove(label, path.split("."))
    return [f"dropped undefined key {p}" for p in paths]


def fix_evidence(facts: dict, text: str, log: list[str]) -> None:
    for dim, quotes in list((facts.get("evidence") or {}).items()):
        kept = []
        for q in quotes or []:
            s = snap((q or {}).get("quote"), text)
            if s:
                kept.append({**q, "quote": s})
            else:
                log.append(f"dropped {dim} evidence not in source")
        facts["evidence"][dim] = kept
    bad = [d for d in facts.get("domains") or [] if d not in ROLE_FAMILIES]
    if bad:
        facts["domains"] = [d for d in facts["domains"] if d in ROLE_FAMILIES]
        log.append(f"dropped off-vocabulary domains {bad}")


def unclaim(facts: dict, doc: dict, task: str, log: list[str]) -> None:
    """A claim whose evidence list is empty is dropped (the validator rejects claims without evidence)."""
    ev = facts.get("evidence") or {}
    claims = {"industry": ["domains"], "education": ["completed_degrees", "minimum_degree", "no_education_required"],
              "location": ["country", "remote", "eligible_countries", "worldwide_remote"], "salary": ["currency"],
              "experience": ["no_experience_required"]}
    empty = {"domains": [], "completed_degrees": [], "eligible_countries": [], "minimum_degree": None, "country": None,
             "currency": None, "no_education_required": False, "remote": False, "worldwide_remote": False,
             "no_experience_required": False}
    for dim, keys in claims.items():
        if ev.get(dim):
            continue
        for k in keys:
            if k in facts and facts[k] != empty[k] and not (task == "cv_extraction" and k == "completed_degrees" and facts[k] == []):
                facts[k] = empty[k]
                log.append(f"cleared facts.{k}: no {dim} evidence")
        if task == "jd_extraction" and dim == "location" and (doc.get("location") or {}).get("city"):
            doc["location"]["city"] = None
            log.append("cleared location.city: no location evidence")
        if task == "jd_extraction" and dim == "salary":
            comp = doc.get("compensation") or {}
            for k in ("min_monthly", "max_monthly", "min_annual", "max_annual", "currency"):
                if comp.get(k) is not None:
                    comp[k] = None
                    log.append(f"cleared compensation.{k}: no salary evidence")
        if task == "jd_extraction" and dim == "experience":
            sen = doc.get("seniority") or {}
            for k in ("min_years", "max_years"):
                if sen.get(k) is not None:
                    sen[k] = None
                    log.append(f"cleared seniority.{k}: no experience evidence")


def fix_cv(label: dict, text: str, log: list[str]) -> None:
    doc = label["document"]
    for group in doc.get("skills") or []:
        kept = []
        for d in group.get("details") or []:
            name = " ".join((d.get("name") or "").split())
            q = snap(d.get("skill_quote"), text)
            if not (q and name_in_quote(name, q)):
                q = source_line(name, text)
                log.append(f"relinked skill {name!r}" if q else f"dropped skill {name!r}: not linkable to the source")
            if not q:
                continue
            d.update(name=name, skill_quote=q)
            pq = snap(d.get("proficiency_quote"), text)
            if d.get("proficiency_quote") and not (pq and name_in_quote(name, pq)):
                d.update(proficiency="normal", raw_proficiency=None, proficiency_quote=None)
                log.append(f"reset level of {name!r}: level quote not in source")
            elif pq:
                d["proficiency_quote"] = pq
            kept.append(d)
        group["details"] = kept
        group["keywords"] = [d["name"] for d in kept]
    doc["skills"] = [g for g in doc.get("skills") or [] if g["details"]]
    # the Languages group mirrors document.languages (training standard)
    langs = [i["language"] for i in doc.get("languages") or [] if i.get("language")]
    groups = [g for g in doc["skills"] if g["name"].casefold() == "languages"]
    lang_group = groups[0] if groups else None
    if lang_group:
        for extra in groups[1:]:
            lang_group["details"] += extra["details"]
            doc["skills"].remove(extra)
        before = len(lang_group["details"])
        lang_group["details"] = [d for d in lang_group["details"] if any(word_in(lang, d["name"]) for lang in langs)]
        if len(lang_group["details"]) < before:
            log.append("dropped Languages skills missing from document.languages")
    for lang in langs:
        if word_in(lang, text) and not (lang_group and any(word_in(lang, d["name"]) for d in lang_group["details"])):
            q = source_line(lang, text)
            if not q:
                continue
            if lang_group is None:
                lang_group = {"name": "Languages", "level": None, "keywords": [], "details": []}
                doc["skills"].append(lang_group)
            lang_group["details"].append({"name": lang, "proficiency": "normal", "raw_proficiency": None,
                                          "skill_quote": q, "proficiency_quote": None})
            log.append(f"added Languages skill {lang!r} from document.languages")
    if lang_group is not None:
        lang_group["keywords"] = [d["name"] for d in lang_group["details"]]
        if not lang_group["details"]:
            doc["skills"].remove(lang_group)


def fix_jd(label: dict, text: str, log: list[str]) -> None:
    doc, facts = label["document"], label["facts"]
    skills = doc.get("skills") or {}
    reqs = []
    for r in skills.get("requirements") or []:
        q = snap(r.get("quote"), text)
        opts = [o for o in r.get("options") or [] if q and name_in_quote(o, q)]
        if not opts:
            log.append(f"dropped {r.get('id')}: quote not in source or names none of its options")
            continue
        if len(opts) < len(r.get("options") or []):
            log.append(f"{r.get('id')}: dropped options not named in its quote")
        details = {d.get("name"): d for d in r.get("option_details") or []}
        r.update(quote=q, options=opts, option_details=[details.get(o) or {"name": o, "minimum_proficiency": "normal",
                                                                          "raw_proficiency": None, "proficiency_quote": None,
                                                                          "experience": []} for o in opts])
        iq = snap(r.get("importance_quote"), text)
        if r.get("importance") == "preferred" and not (iq and PREFERRED.search(iq)):
            r["importance"] = "standard"
            log.append(f"{r['id']}: preferred without optional wording -> standard")
        r["importance_quote"] = iq if iq and all(name_in_quote(o, iq) for o in opts) else q
        for d in r["option_details"]:
            raw, pq = d.get("raw_proficiency"), snap(d.get("proficiency_quote"), text)
            if raw and not (pq and raw.casefold() in pq.casefold() and name_in_quote(d.get("name") or "", pq)):
                d.update(raw_proficiency=None, proficiency_quote=None, minimum_proficiency="normal")
                log.append(f"{r['id']}: reset level of {d.get('name')!r}: level quote not in source")
            elif pq:
                d["proficiency_quote"] = pq
        reqs.append(r)
    for i, r in enumerate(reqs, 1):
        r["id"] = f"req_{i:02d}"
    skills["requirements"] = reqs
    # derived lists mirror the requirements (training standard)
    req_names = list(dict.fromkeys(o for r in reqs if r["importance"] in ("core", "standard") for o in r["options"]))
    pref_names = list(dict.fromkeys(o for r in reqs if r["importance"] == "preferred" for o in r["options"]))
    skills["required"] = [{"name": n, "proficiency": "required"} for n in req_names]
    skills["preferred"] = [{"name": n, "proficiency": "preferred"} for n in pref_names]
    ev = facts.setdefault("evidence", {})
    if reqs and not ev.get("skills"):
        ev["skills"] = [{"quote": q} for q in dict.fromkeys(r["quote"] for r in reqs)]
        log.append("evidence.skills rebuilt from requirement quotes")


def fix_issues(label: dict, error: str, task: str, log: list[str]) -> bool:
    """Act on the validator's per-skill issue list; True if anything changed."""
    changed = False
    for path, reason in ISSUE.findall(error or ""):
        if task == "cv_extraction":
            name = path.removeprefix("skills.").removesuffix(".proficiency")
            for g in label["document"].get("skills") or []:
                for d in list(g.get("details") or []):
                    if d.get("name") != name:
                        continue
                    if reason.startswith("depth") or path.endswith(".proficiency"):
                        d.update(proficiency="normal", raw_proficiency=None, proficiency_quote=None)
                        log.append(f"reset level of {name!r}: {reason}")
                    else:
                        g["details"].remove(d)
                        log.append(f"dropped skill {name!r}: {reason}")
                    changed = True
        else:
            rid = path.split(".")[1]
            for r in list(label["document"]["skills"].get("requirements") or []):
                if r.get("id") != rid:
                    continue
                if ".depth." in path:
                    opt = path.split(".depth.", 1)[1]
                    for d in r.get("option_details") or []:
                        if d.get("name") == opt:
                            if reason.startswith("experience"):  # the years claim, not the level, lacks a linked quote
                                d["experience"] = []
                            else:
                                d.update(raw_proficiency=None, proficiency_quote=None, minimum_proficiency="normal")
                    log.append(f"{rid}: {'cleared years' if reason.startswith('experience') else 'reset level'} of {opt!r}: {reason}")
                elif path.endswith(".importance"):
                    r.update(importance="standard", importance_quote=r.get("quote"))
                    log.append(f"{rid}: importance reset: {reason}")
                else:
                    label["document"]["skills"]["requirements"].remove(r)
                    log.append(f"dropped {rid}: {reason}")
                changed = True
    return changed


def repair(label: dict, text: str, task: str) -> tuple[dict, str, list[str]]:
    """(repaired label, validator status after repair, log). The input label is not modified."""
    label, log = copy.deepcopy(label), []
    if not isinstance(label.get("document"), dict) or not isinstance(label.get("facts"), dict):
        return label, validate(label, text, task)[0], ["not a document/facts object: left for retry"]
    log += drop_undefined(label, JD_SCHEMA if task == "jd_extraction" else None)
    fix_evidence(label["facts"], text, log)
    (fix_jd if task == "jd_extraction" else fix_cv)(label, text, log)
    unclaim(label["facts"], label["document"], task, log)
    status, error = validate(label, text, task)
    for _ in range(MAX_ROUNDS):
        if status in ("pass", "partial") or not fix_issues(label, error, task, log):
            break
        if task == "jd_extraction":
            fix_jd(label, text, log)
        else:
            fix_cv(label, text, log)
        status, error = validate(label, text, task)
    if status == "fail":
        log.append(f"unresolved: {str(error)[:500]}")  # why the validator still rejects it
    return label, status, log


def sources(split: str) -> dict:
    out = {}
    for task in ("cv_extraction", "jd_extraction"):
        for line in (DATA / task / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            g = json.loads(line)
            out[g["id"]] = json.loads(g["messages"][1]["content"])["source_text"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preds", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--retry", type=Path, default=None)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.preds.read_text(encoding="utf-8").splitlines()]
    retries = [json.loads(line) for line in args.retry.read_text(encoding="utf-8").splitlines()] if args.retry else []
    text = sources(rows[0]["split"])
    before, after = {}, {}

    def attempt(r: dict) -> None:
        label = parse_balanced(r["prediction"], JD_SCHEMA if r["task"] == "jd_extraction" else None)
        if label is None:
            r.update(repaired=None, repair_status="unparsed", repair_log=["unparsed: left for retry"])
        else:
            fixed, status, log = repair(label, text[r["id"]], r["task"])
            r.update(repaired=json.dumps(fixed, ensure_ascii=False), repair_status=status, repair_log=log)

    for r in rows:
        label = parse(r["prediction"])
        before[r["id"]] = validate(label, text[r["id"]], r["task"])[0] if label is not None else "unparsed"
        attempt(r)
        if r["repair_status"] not in ("pass", "partial"):
            for cand in sorted((c for c in retries if c["id"] == r["id"]), key=lambda c: c["repetition_penalty"]):
                attempt(cand)
                if cand["repair_status"] in ("pass", "partial"):
                    r.update(first_prediction=r["prediction"], **{k: cand[k] for k in (
                        "prediction", "finish_reason", "repetition_penalty", "repaired", "repair_status", "repair_log")})
                    r["repair_log"] = [f"retried at repetition_penalty {cand['repetition_penalty']}"] + r["repair_log"]
                    break
        after[r["id"]] = r["repair_status"]
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    for task in ("cv_extraction", "jd_extraction"):
        ids = [r["id"] for r in rows if r["task"] == task]
        ok = lambda s: s in ("pass", "partial")
        print(f"{task}: valid before {sum(ok(before.get(i)) for i in ids)}/{len(ids)}, after repair "
              f"{sum(ok(after[i]) for i in ids)}/{len(ids)}; still invalid: "
              f"{sorted(i for i in ids if not ok(after[i]))}", file=sys.stderr)


if __name__ == "__main__":
    main()
