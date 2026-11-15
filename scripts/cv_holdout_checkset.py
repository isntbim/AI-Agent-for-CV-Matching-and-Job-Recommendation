"""Build the data file for the Gold CV Check page from the Gemini drafts of the 46 usable holdout CVs, and read the
reviewers' results back into gold labels.

Assignment: per domain, the first CV (fixed order) is checked by both reviewers (agreement); the rest alternate A/B
across the whole list.
Reviewers fix only what the extraction metrics score (scripts/qwen_eval.py): skill names, domains, completed degrees.

    python -m scripts.cv_holdout_checkset build   # -> output/scorer_v3/teacher_labels/holdout/holdout_check_v1.json
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TL = ROOT / "output/scorer_v3/teacher_labels"
H = TL / "holdout"


def draft(cv_id: str) -> dict:
    path = TL / "holdout_draft" / f"{cv_id}.json"
    if not path.exists():
        return {"skills": [], "domains": [], "degrees": [], "missing": True}
    d = json.loads(path.read_text(encoding="utf-8"))
    names = list(dict.fromkeys(x["name"] for g in d["document"]["skills"] for x in g.get("details", [])))
    return {"skills": names, "domains": d["facts"].get("domains", []), "degrees": d["facts"].get("completed_degrees", [])}


def build() -> None:
    rows = [r for r in json.loads((H / "manifest.json").read_text(encoding="utf-8"))["documents"] if "excluded" not in r]
    rows.sort(key=lambda r: (r["domain"], hashlib.sha256(("check" + r["id"]).encode()).hexdigest()))
    seen, cvs, alt = defaultdict(int), [], 0
    for n, r in enumerate(rows, 1):
        k = seen[r["domain"]]; seen[r["domain"]] += 1
        if k == 0:
            group = "both"
        else:  # alternate across the whole list, so A and B get equal counts and every domain in both
            group, alt = ("A" if alt % 2 == 0 else "B"), alt + 1
        cvs.append({"id": r["id"], "n": n, "domain": r["domain"], "group": group,
                    "text": (ROOT / r["text_path"]).read_text(encoding="utf-8"), "draft": draft(r["id"])})
    out = {"version": 1, "as_of": "2026-10-25", "cvs": cvs}
    (H / "holdout_check_v1.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    g = defaultdict(int)
    for c in cvs:
        g[c["group"]] += 1
    print(len(cvs), "CVs", dict(g), "| drafts missing:", [c["id"] for c in cvs if c["draft"].get("missing")])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cmd", choices=["build"])
    if p.parse_args().cmd == "build":
        build()
