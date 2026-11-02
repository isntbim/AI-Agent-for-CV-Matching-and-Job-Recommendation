"""Group the errors of Qwen prediction files into failure types (counts + short examples), side by side.

Reuses scripts/qwen_eval.py helpers. Usage:
python -m scripts.qwen_failure_analysis output/review2/qwen/base_test.jsonl [output/review2/qwen/ft_test.jsonl] --out-dir output/review2/qwen
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from scripts.qwen_eval import DATA, ROOT, _as, parse, quotes, skills, validate

TYPES = ["truncated", "unparseable", "task_mismatch", "schema_invalid", "wrong_types", "invented_quotes", "missed_skills",
         "extra_skills", "degree_wrong", "domain_wrong"]
# Joint CV+JD training: signs the output follows the OTHER task's format. Gold never shows these (checked on all 1,908 rows):
# schema_version is always resume-v3 / job-v3, the two document schemas share only skills + schema_version, and JD gold never
# fills completed_degrees. CV gold does fill minimum_degree (19 rows), so that direction is not used.
SCHEMA = {"cv_extraction": "resume-v3", "jd_extraction": "job-v3"}
OTHER_KEYS = {"cv_extraction": {"company", "compensation", "benefits", "seniority", "title", "title_normalized", "category"},
              "jd_extraction": {"basics", "work", "education", "certificates", "languages", "projects", "summary_text"}}
SNIP = 160


def load_gold(task: str, split: str) -> dict:
    gold = {}
    for line in (DATA / task / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
        g = json.loads(line)
        gold[g["id"]] = (json.loads(g["messages"][2]["content"]), json.loads(g["messages"][1]["content"])["source_text"])
    return gold


def bucket(error: str | None) -> str:
    """The validator's reason without the exception class or the quoted example, e.g. 'skills evidence is not a literal source quote'."""
    error = re.sub(r"^\w*(Error|Exception): ", "", error or "unknown error")
    error = re.sub(r"\bskills\.[^:;]+:", "skills.<name>:", error)  # 'skills.Python:missing_source_quote' -> one bucket
    error = re.sub(r"^\d+ validation errors? for (\w+)\s.*", r"pydantic type errors in \1", error, flags=re.S)
    return re.split(r"[:;\n]", error, maxsplit=1)[0].strip()[:80]


def has_wrong_types(pred: dict, task: str) -> bool:
    doc, facts = pred.get("document"), pred.get("facts")
    if not isinstance(doc, dict) or not isinstance(facts, dict):
        return True
    if not isinstance(facts.get("domains"), list):
        return True
    sk = doc.get("skills")
    if task == "jd_extraction":
        return not isinstance(sk, dict) or any(not isinstance(sk.get(k, []), list) for k in ("required", "preferred"))
    return not isinstance(sk, list) or any(not isinstance(s, dict) or not isinstance(s.get("details", []), list) for s in sk)


def task_mismatch(pred: dict, task: str) -> str | None:
    doc, facts = _as(pred.get("document"), dict), _as(pred.get("facts"), dict)
    signs = []
    other = next(v for k, v in SCHEMA.items() if k != task)
    if doc.get("schema_version") == other:  # a missing/garbled version is a format error (schema_invalid), not task confusion
        signs.append(f"schema_version={other}")
    if keys := sorted(OTHER_KEYS[task] & doc.keys()):
        signs.append("other-task keys " + ",".join(keys))
    if task == "jd_extraction" and facts.get("completed_degrees"):
        signs.append("JD fills completed_degrees")
    return "; ".join(signs) or None


def analyse_doc(row: dict, gold: dict, text: str, task: str) -> dict:
    """Failure type -> short detail string, for one document."""
    found = {}
    if row.get("finish_reason") == "length":
        found["truncated"] = "hit max_tokens"
    pred = parse(row["prediction"])
    if pred is None:
        if "truncated" not in found:
            found["unparseable"] = row["prediction"][:SNIP].replace("\n", " ")
        return found
    if mismatch := task_mismatch(pred, task):
        found["task_mismatch"] = mismatch
    if has_wrong_types(pred, task):
        found["wrong_types"] = f"document={type(pred.get('document')).__name__}, facts={type(pred.get('facts')).__name__}"
    try:
        status, error = validate(pred, text, task)
    except Exception as exc:  # wrong-typed predictions can crash the validator
        status, error = "error", f"{type(exc).__name__}: {exc}"
    if status not in ("pass", "partial"):
        found["schema_invalid"] = bucket(error)
    qs = quotes(pred)
    bad = [q for q in qs if q not in text]
    if bad:
        found["invented_quotes"] = f"{len(bad)}/{len(qs)}: " + bad[0][:100].replace("\n", " ")
    ps, gs = skills(pred, task), skills(gold, task)
    if gs - ps:
        found["missed_skills"] = ", ".join(sorted(gs - ps)[:8])
    if ps - gs:
        found["extra_skills"] = ", ".join(sorted(ps - gs)[:8])
    deg_key = "minimum_degree" if task == "jd_extraction" else "completed_degrees"
    facts, norm = _as(pred.get("facts"), dict), (lambda v: sorted(v) if isinstance(v, list) else v)
    if norm(facts.get(deg_key)) != norm(gold["facts"].get(deg_key)):
        found["degree_wrong"] = f"pred {facts.get(deg_key)!r} vs gold {gold['facts'].get(deg_key)!r}"
    pd_ = {d for d in _as(facts.get("domains"), list) if isinstance(d, str)}
    if pd_ != set(gold["facts"].get("domains") or []):
        found["domain_wrong"] = f"pred {sorted(pd_)} vs gold {sorted(gold['facts'].get('domains') or [])}"
    return found


def analyse(rows: list[dict]) -> dict:
    report = {}
    for task in sorted({r["task"] for r in rows}):
        gold = load_gold(task, rows[0]["split"])
        out = {"n": 0, "counts": Counter(), "examples": {t: [] for t in TYPES}, "schema_buckets": Counter(),
               "quotes_total": 0, "quotes_invented": 0, "missed": Counter(), "extra": Counter(),
               "missed_total": 0, "extra_total": 0}
        for r in (r for r in rows if r["task"] == task):
            g, text = gold[r["id"]]
            out["n"] += 1
            found = analyse_doc(r, g, text, task)
            pred = parse(r["prediction"])
            for t, detail in found.items():
                out["counts"][t] += 1
                if len(out["examples"][t]) < 3:
                    out["examples"][t].append(f"{r['id']}: {detail}"[:SNIP])
            if "schema_invalid" in found:
                out["schema_buckets"][found["schema_invalid"]] += 1
            if pred is not None:
                qs = quotes(pred)
                out["quotes_total"] += len(qs)
                out["quotes_invented"] += sum(q not in text for q in qs)
                ps, gs = skills(pred, task), skills(g, task)
                out["missed"].update(gs - ps)
                out["extra"].update(ps - gs)
                out["missed_total"] += len(gs - ps)
                out["extra_total"] += len(ps - gs)
        report[task] = out
    return report


def as_json(report: dict) -> dict:
    return {task: {"n": o["n"], "counts": {t: o["counts"][t] for t in TYPES},
                   "top_schema_buckets": o["schema_buckets"].most_common(8),
                   "quotes_total": o["quotes_total"], "quotes_invented": o["quotes_invented"],
                   "invented_share": round(o["quotes_invented"] / o["quotes_total"], 4) if o["quotes_total"] else 0.0,
                   "missed_total": o["missed_total"], "extra_total": o["extra_total"],
                   "top_missed": o["missed"].most_common(10), "top_extra": o["extra"].most_common(10),
                   "examples": o["examples"]} for task, o in report.items()}


def cell(o: dict, t: str) -> str:
    return f"{o['counts'][t]} ({100 * o['counts'][t] / o['n']:.0f}%)" if o["n"] else "0"


def markdown(reports: dict[str, dict]) -> str:
    names, lines = list(reports), ["# Qwen extraction failure analysis", ""]
    for task in sorted({t for r in reports.values() for t in r}):
        os_ = {n: reports[n][task] for n in names if task in reports[n]}
        lines += [f"## {task}", "", "| failure type | " + " | ".join(f"{n} count (%)" for n in os_) + " |",
                  "|---|" + "---|" * len(os_)]
        lines += [f"| {t} | " + " | ".join(cell(o, t) for o in os_.values()) + " |" for t in TYPES]
        # Unparseable/truncated docs have no fields to compare; qwen_eval scores them as wrong, so its rates differ.
        lines += ["", "Rows from invented_quotes down are checked on parsed documents only; % is of all documents. "
                  "Unparseable and truncated documents are counted as wrong by qwen_eval.py, so its accuracies are lower."]
        for n, o in os_.items():
            share = o["quotes_invented"] / o["quotes_total"] if o["quotes_total"] else 0.0
            lines += ["", f"### {n}", "", f"- documents: {o['n']}",
                      f"- quotes: {o['quotes_total']} total, {o['quotes_invented']} invented ({100 * share:.1f}%)",
                      f"- skills: {o['missed_total']} missed, {o['extra_total']} extra (parsed documents only)",
                      "", "Top schema-error buckets:", ""]
            lines += [f"- {c} x {b}" for b, c in o["schema_buckets"].most_common(8)] or ["- none"]
            lines += ["", "Top missed skills: " + (", ".join(f"{s} ({c})" for s, c in o["missed"].most_common(10)) or "none"),
                      "", "Top extra skills: " + (", ".join(f"{s} ({c})" for s, c in o["extra"].most_common(10)) or "none"), ""]
            for t in TYPES:
                if o["examples"][t]:
                    lines += [f"Examples, {t}:", ""] + [f"- {e.replace('|', '/')}" for e in o["examples"][t]] + [""]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("preds", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, default=ROOT / "output/review2/qwen")
    args = p.parse_args()
    reports = {path.stem: analyse([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])
               for path in args.preds}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "failure_analysis.json").write_text(
        json.dumps({n: as_json(r) for n, r in reports.items()}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "failure_analysis.md").write_text(markdown(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
