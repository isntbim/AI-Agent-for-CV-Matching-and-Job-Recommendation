"""Build the teacher-labelling CV pool: exclude gold, reserve holdouts, extract text.

Holdouts are reserved before any teacher sees a CV. Selection is a seeded hash
order per domain, so reruns give the same split.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from src.parser.extractors import extract_text_from_file
from src.scoring.extraction import _prompts, write_json

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = {"ACCOUNTANT": "accountant", "HR": "hr", "Data Scientist Analyst": "data_scientist_analyst",
           "FrontendWebDeveloper": "frontend_web_developer", "Marketing Executive": "marketing_executive",
           "Software Engineer": "software_engineer"}
SEED = "cv-teacher-pool-2026-10-13"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path, n_val: int, n_test: int, n_pilot: int, extra_pilot: tuple = (), as_of: str | None = None) -> dict:
    corpus = json.loads((ROOT / "output/scorer_v3/corpus.json").read_text(encoding="utf-8"))
    gold = {d["source"]["source_sha256"] for d in corpus["documents"] if d["kind"] == "resume"}
    groups = {}  # byte hash -> first (domain, path); later copies are the same document
    for folder, domain in FOLDERS.items():
        for pdf in sorted((ROOT / "data/raw/cvs" / folder).glob("*.pdf")):
            groups.setdefault(sha(pdf), (domain, pdf.relative_to(ROOT).as_posix()))
    by_domain = defaultdict(list)
    for digest, (domain, path) in groups.items():
        if digest not in gold:
            by_domain[domain].append({"sha256": digest, "domain": domain, "source_path": path})
    old = out / "manifest.json"  # documents labelled in an earlier batch keep that batch's date
    dated = {d["sha256"]: d.get("as_of") for d in json.loads(old.read_text(encoding="utf-8"))["documents"]} if old.exists() else {}
    rows = []
    for domain, items in sorted(by_domain.items()):
        items.sort(key=lambda r: hashlib.sha256((SEED + r["sha256"]).encode()).hexdigest())
        for i, row in enumerate(items):
            row["split"] = ("validation" if i < n_val else "test" if i < n_val + n_test
                            else "pilot" if i < n_val + n_test + n_pilot else "train")
            row["id"] = f"tp_{domain}_{i:03d}"
            if row["id"] in extra_pilot and row["split"] == "train":  # hand-picked edge cases, never holdouts
                row["split"] = "pilot"
            rows.append(row)
    system, _ = _prompts("resume", "", "v3")
    (out / "system_prompt_v3.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "system_prompt_v3.txt").write_text(system, encoding="utf-8")
    for row in rows:
        if row["split"] not in {"pilot", "train"}:
            continue  # holdout text is extracted later, by the human-review step
        try:
            text = extract_text_from_file(ROOT / row["source_path"])
        except Exception as exc:  # scanned/broken PDFs stay out, recorded
            row["text_error"] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        row["characters"] = len(text)
        row["text_path"] = (out / "texts" / (row["id"] + ".txt")).relative_to(ROOT).as_posix()
        (ROOT / row["text_path"]).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / row["text_path"]).write_text(text, encoding="utf-8")
        _, user = _prompts("resume", text, "v3", as_of)
        row_as_of = dated[row["sha256"]] if row["sha256"] in dated else as_of
        if row_as_of:  # the training manifest carries it on to the export
            row["as_of"] = row_as_of
        row["input_path"] = (out / "inputs" / (row["id"] + ".json")).relative_to(ROOT).as_posix()
        (ROOT / row["input_path"]).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / row["input_path"]).write_text(user, encoding="utf-8")
    manifest = {"seed": SEED, "excluded_gold_hashes": len(gold), "groups": len(rows), "documents": rows}
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/scorer_v3/teacher_labels")
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--test", type=int, default=6)
    parser.add_argument("--pilot", type=int, default=1)
    parser.add_argument("--extra-pilot", nargs="*", default=[], help="training-pool ids to add to the pilot")
    parser.add_argument("--as-of", help="labelling date (YYYY-MM-DD) written into every input for past-dated degrees")
    args = parser.parse_args()
    if args.as_of and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.as_of):
        parser.error("--as-of must be YYYY-MM-DD")
    m = build(args.out, args.val, args.test, args.pilot, tuple(args.extra_pilot), args.as_of)
    counts = defaultdict(lambda: defaultdict(int))
    for r in m["documents"]:
        counts[r["domain"]][r["split"]] += 1
    print(json.dumps({d: dict(c) for d, c in counts.items()}, indent=1))
    print("groups", m["groups"], "text errors", sum("text_error" in r for r in m["documents"]))
    corpus = json.loads((ROOT / "output/scorer_v3/corpus.json").read_text(encoding="utf-8"))
    gold = {d["source"]["source_sha256"] for d in corpus["documents"] if d["kind"] == "resume"}
    assert not {r["sha256"] for r in m["documents"]} & gold, "gold CV leaked into the teacher pool"


if __name__ == "__main__":
    main()
