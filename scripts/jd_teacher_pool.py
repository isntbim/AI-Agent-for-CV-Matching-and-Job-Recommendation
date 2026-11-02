"""Build the JD teacher pool: teacher_labels/jd/{manifest.json, inputs/<id>.json, system_prompt_v3.txt}.

Texts are the scorer corpus texts unchanged (output/scorer_v3/corpus.json source.text_path), so a label trains on
exactly the text the scorer sees at runtime. Every split is labelled; each row keeps its corpus split so validation/test
labels serve as evaluation references and never enter training. The Vietnamese-only CareerViet secondary corpus
(data/raw/jobs_vi, never in the scorer corpus) joins as train JDs, rebuilt from the original workbook cells exactly as
the corpus texts are (src.scoring.pilot.job_source); a text identical to an earlier JD after whitespace folding is skipped.
Batches are fixed once written: a rerun keeps every existing batch and fills the next one.
"""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from src.scoring.extraction import _prompts, write_json
from src.scoring.pilot import job_source, original_job_rows

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/scorer_v3/teacher_labels/jd"
SEED = "jd-teacher-pool-2026-10-26"
VIETNAMESE = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)


def language(text: str) -> str:
    marks = len(VIETNAMESE.findall(text))
    return "vi" if marks > 50 else "mixed" if marks > 5 else "en"


def vietnamese_secondary() -> list[dict]:
    """jobs_vi records as corpus-shaped train docs; texts written to teacher_labels/jd/sources/."""
    records = json.loads((ROOT / "data/processed/structured_jobs_vi/jobs_vi_all.json").read_text(encoding="utf-8"))
    originals = original_job_rows(ROOT)
    docs = []
    for record in records:
        _, text, _ = job_source(record, ROOT, originals, "data/raw/jobs_vi")
        path = OUT / "sources" / (record["id"] + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        docs.append({"id": record["id"], "kind": "job", "domain": record["category"], "split": "train",
                     "source": {"text_path": path.relative_to(ROOT).as_posix()},
                     "envelope": {"document": {"source": record["source"]}}})
    return docs


def build(size: int, quota: dict[str, int]) -> dict:
    corpus = json.loads((ROOT / "output/scorer_v3/corpus.json").read_text(encoding="utf-8"))["documents"]
    corpus += vietnamese_secondary()
    old = OUT / "manifest.json"
    manifest = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {"seed": SEED, "documents": []}
    taken = {d["id"] for d in manifest["documents"]}
    seen = set()  # folded texts, corpus first, so a secondary copy of a corpus JD is skipped
    batch = 1 + max((d["batch"] for d in manifest["documents"]), default=0)
    groups = defaultdict(list)  # (language, domain) -> JDs in seeded order
    for doc in corpus:
        if doc["kind"] != "job":
            continue
        text = (ROOT / doc["source"]["text_path"]).read_text(encoding="utf-8")
        folded = " ".join(text.split()).casefold()
        if folded in seen or doc["id"] in taken:
            seen.add(folded)
            continue
        seen.add(folded)
        groups[(language(text), doc["domain"])].append({"id": doc["id"], "domain": doc["domain"], "language": language(text), "split": doc["split"],
                                                        "source": doc["envelope"]["document"]["source"],
                                                        "text_path": doc["source"]["text_path"], "characters": len(text)})
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.sha256((SEED + r["id"]).encode()).hexdigest())
    picked = []
    for lang, want in quota.items():  # round-robin over domains so each language spreads across roles
        queues = [rows for (l, _), rows in sorted(groups.items()) if l == lang]
        while want and any(queues):
            for rows in queues:
                if rows and want:
                    picked.append(rows.pop(0)); want -= 1
    assert len(picked) == size, f"only {len(picked)} JDs left for quota {quota}"
    system, _ = _prompts("job", "", "v3")
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT / "system_prompt_v3.txt").write_text(system, encoding="utf-8")
    for row in picked:
        _, user = _prompts("job", (ROOT / row["text_path"]).read_text(encoding="utf-8"), "v3")
        row.update(batch=batch, input_path=(OUT / "inputs" / f"{row['id']}.json").relative_to(ROOT).as_posix())
        (ROOT / row["input_path"]).write_text(user, encoding="utf-8")
    manifest["documents"] += picked
    write_json(old, manifest)
    return {"batch": batch, "ids": [r["id"] for r in picked]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en", type=int, default=20)
    parser.add_argument("--mixed", type=int, default=20)
    parser.add_argument("--vi", type=int, default=40)
    args = parser.parse_args()
    quota = {"en": args.en, "mixed": args.mixed, "vi": args.vi}
    result = build(sum(quota.values()), quota)
    print("batch", result["batch"], len(result["ids"]), "JDs")


if __name__ == "__main__":
    main()
