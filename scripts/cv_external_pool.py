"""Build the external CV teacher pool from med2425 and Saba06 (data/raw/external), 6 project domains only.

A CV is dropped when it sits on the scorer's held-out test side, is a near-copy of a local pool/gold CV, or is
a near-copy of another external CV already kept. Every kept CV is train-only (holdouts stay gold). Writes
texts/ and inputs/ next to the local pool and its own manifest_external.json, so cv_teacher_pool.py reruns
never touch it. Needs pyarrow (med2425 is parquet).
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scripts.cv_teacher_pool import FOLDERS, ROOT
from src.parser.extractors import extract_text_from_file
from src.scoring.extraction import _prompts, write_json

OUT = ROOT / "output/scorer_v3/teacher_labels"
EXT = ROOT / "data/raw/external"
SEED = "cv-external-pool-2026-10-27"
# first match wins: frontend/data before the generic "developer"
DOMAINS = [
    ("frontend_web_developer", r"front[\s-]?end|\bui developer|web developer|web designer"),
    ("data_scientist_analyst", r"data scien|data analy|machine learning|analytics|business intelligence|data engineer|\bbi developer"),
    ("software_engineer", r"software|programmer|full[\s-]?stack|back[\s-]?end|\.net developer|java developer|application developer|\bdeveloper\b"),
    ("accountant", r"accountant|accounting|bookkeep|accounts (?:payable|receivable)|\bcpa\b"),
    ("hr", r"human resource|\bhr\b|recruit|talent acquisition"),
    ("marketing_executive", r"marketing|\bseo\b|social media|digital market"),
]
NEAR = 0.5  # share of word 5-grams in common (of the shorter text) that marks a copy


def classify(text: str) -> str | None:
    """shortcut: keyword guess on the opening title/summary, upgrade to a teacher title check if domains matter."""
    head = " ".join(text.split())[:250].lower()
    return next((domain for domain, pattern in DOMAINS if re.search(pattern, head)), None)


def key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())[:3000]  # same key as the 2026-10-25 leakage audit


def shingles(text: str) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(words[i:i + 5])) for i in range(max(1, len(words) - 4))}


class CopyIndex:
    def __init__(self):
        self.docs, self.index = [], defaultdict(list)

    def add(self, name: str, text: str):
        grams = shingles(text)
        for g in grams:
            self.index[g].append(len(self.docs))
        self.docs.append((name, len(grams)))

    def match(self, text: str) -> str | None:
        grams = shingles(text)
        hits = Counter(i for g in grams for i in self.index.get(g, ()))
        for i, shared in hits.most_common(3):
            name, size = self.docs[i]
            if shared / max(1, min(size, len(grams))) >= NEAR:
                return name
        return None


def local_texts():
    """Every local CV the project already holds: pool (holdouts included, extracted here) and gold."""
    pool = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["documents"]
    for d in pool:
        if "text_path" in d:
            yield d["id"], (ROOT / d["text_path"]).read_text(encoding="utf-8")
        else:
            yield d["id"], extract_text_from_file(ROOT / d["source_path"])
    for path in sorted((OUT / "gold_texts").glob("*.txt")):
        yield path.stem, path.read_text(encoding="utf-8")
    # later byte-copies of a pool PDF are not in the manifest; their first copy is


def external():
    import pyarrow.parquet as pq
    for path in sorted((EXT / "med2425_resume_job_fit_merged_v1/data").glob("*.parquet")):
        for row in pq.read_table(path, columns=["resume"]).to_pylist():
            yield "med2425", row["resume"]
    with open(EXT / "saba06_resume_classification/resume_dataset_huggingface.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield "saba06", row["text"]


def build(as_of: str | None) -> dict:
    pairs = json.loads((ROOT / "output/ats/prepared/pairs.json").read_text(encoding="utf-8"))
    test = {key(p["resume_text"]) for p in pairs if p["split"] == "test"}
    copies = CopyIndex()
    for name, text in local_texts():
        copies.add(name, text)
    for p in pairs:
        if p["split"] == "test":
            copies.add("scorer_test", p["resume_text"])
    seen, rows, dropped = set(), [], Counter()
    for source, raw in external():
        text = raw.strip()
        k = key(text)
        if not text or k in seen:
            continue  # exact repeats (med2425 reuses each CV across many pairs)
        seen.add(k)
        domain = classify(text)
        if domain is None:
            dropped["other_domain"] += 1
        elif k in test:
            dropped["scorer_test"] += 1
        elif hit := copies.match(text):
            dropped["scorer_test" if hit == "scorer_test" else "external_copy" if hit.startswith("ex_") else "local_copy"] += 1
        else:
            digest = hashlib.sha256(text.encode()).hexdigest()
            rows.append({"sha256": digest, "domain": domain, "source": source, "split": "train", "text": text})
            copies.add("ex_" + digest, text)  # later near-copies of this one are dropped
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    system, _ = _prompts("resume", "", "v3")
    assert system == (OUT / "system_prompt_v3.txt").read_text(encoding="utf-8"), "v3 prompt changed since the pool"
    documents = []
    for domain, items in sorted(by_domain.items()):
        items.sort(key=lambda r: hashlib.sha256((SEED + r["sha256"]).encode()).hexdigest())
        for i, row in enumerate(items):
            text = row.pop("text")
            row["id"] = f"ex_{domain}_{i:03d}"
            row["characters"] = len(text)
            row["text_path"] = (OUT / "texts" / (row["id"] + ".txt")).relative_to(ROOT).as_posix()
            row["input_path"] = (OUT / "inputs" / (row["id"] + ".json")).relative_to(ROOT).as_posix()
            if as_of:
                row["as_of"] = as_of
            (ROOT / row["text_path"]).write_text(text, encoding="utf-8")
            (ROOT / row["input_path"]).write_text(_prompts("resume", text, "v3", as_of)[1], encoding="utf-8")
            documents.append(row)
    manifest = {"seed": SEED, "dropped": dict(dropped), "groups": len(documents), "documents": documents}
    write_json(OUT / "manifest_external.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", help="labelling date (YYYY-MM-DD) written into every input for past-dated degrees")
    args = parser.parse_args()
    if args.as_of and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.as_of):
        parser.error("--as-of must be YYYY-MM-DD")
    m = build(args.as_of)
    counts = Counter((r["domain"], r["source"]) for r in m["documents"])
    print(json.dumps({f"{d}/{s}": n for (d, s), n in sorted(counts.items())}, indent=1))
    print("kept", m["groups"], "dropped", m["dropped"])
    pool = {d["id"] for d in json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["documents"]}
    assert not pool & {r["id"] for r in m["documents"]}, "external id collides with the local pool"
    assert set(FOLDERS.values()) >= {r["domain"] for r in m["documents"]}, "domain outside the 6 project domains"


if __name__ == "__main__":
    sys.exit(main())
