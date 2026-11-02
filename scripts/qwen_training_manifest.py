"""List every Qwen training/evaluation item by task and split: data/datasets/qwen_training/manifest.json.

Nothing is moved: each item points at its source text and its label where the build pipelines keep them.
CV extraction: pool reference labels (reference_b + the 11 non-anchor pilot labels) and gold v3 train for training,
gold v3 validation/test for evaluation, plus the external round's arbitrated references (ext_reference, train
only). JD extraction: the JD teacher reference labels (teacher_labels/jd/reference;
corpus JDs keep their corpus split, the Vietnamese-only secondary JDs are train). The training JSONL export reads
this file.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.cv_teacher_validate import ROOT, check
from scripts.jd_teacher_check import check_jd
from src.scoring.extraction import write_json

TL = ROOT / "output/scorer_v3/teacher_labels"
# pool CV that is a copy of a gold CV (same text); the gold label is kept
DUPLICATES = {"tp_hr_035": "cv_054", "tp_data_scientist_analyst_028": "cv_021"}  # 2nd: same person, other CV version
BATCH_AS_OF = "2026-10-25"  # the first batch (pool before --as-of existed, and gold): its degree calls hold on this date
# Qwen-only (2026-10-28): 4 more gold train CVs per domain move to test (12 -> 36, 6/domain), picked by
# sha256("qwen-gold-test-2026-10-28:<id>") order. corpus.json keeps its split, so scorer experiments are unchanged.
GOLD_TO_TEST = {
    "cv_003", "cv_005", "cv_007", "cv_013", "cv_017", "cv_021", "cv_022", "cv_028", "cv_030", "cv_034", "cv_040", "cv_041",
    "cv_048", "cv_055", "cv_056", "cv_058", "cv_059", "cv_063", "cv_067", "cv_073", "cv_081", "cv_082", "cv_085", "cv_087",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def cv_items(suffix: str = "") -> list[dict]:
    """suffix "_gs": the GOLD_STYLE_FILTER.md-filtered copies of the pool and external labels (V4)."""
    pool = {d["id"]: d for d in json.loads((TL / "manifest.json").read_text(encoding="utf-8"))["documents"]}
    items = []
    for folder in ("reference_b", "reference"):
        for path in sorted((TL / (folder + suffix)).glob("*.json")):
            if path.stem in DUPLICATES or any(i["id"] == path.stem for i in items):  # anchors sit in both folders
                continue
            items.append({"id": path.stem, "split": "train", "source": f"pool/{folder}", "text": pool[path.stem]["text_path"],
                          "label": rel(path), "as_of": pool[path.stem].get("as_of", BATCH_AS_OF)})
    # external round (med2425 + Saba06, scripts/cv_external_pool.py): train only, arbitrated references
    for d in json.loads((TL / "manifest_external.json").read_text(encoding="utf-8"))["documents"]:
        items.append({"id": d["id"], "split": "train", "source": f"external/{d['source']}", "text": d["text_path"],
                      "label": rel(TL / ("ext_reference" + suffix) / f"{d['id']}.json"), "as_of": d.get("as_of", BATCH_AS_OF)})
    conversion = {r["id"]: r for r in json.loads((TL / "gold_v3_conversion.json").read_text(encoding="utf-8"))}
    corpus = json.loads((ROOT / "output/scorer_v3/corpus.json").read_text(encoding="utf-8"))["documents"]
    for doc in (d for d in corpus if d["kind"] == "resume"):
        label = TL / "gold_v3" / f"{doc['id']}.json"
        if label.exists():
            text = conversion[doc["id"]].get("text_path") or doc["source"]["text_path"]
            assert doc["id"] not in GOLD_TO_TEST or doc["split"] == "train", doc["id"]
            items.append({"id": doc["id"], "split": "test" if doc["id"] in GOLD_TO_TEST else doc["split"], "source": "gold",
                          "text": text, "label": rel(label), "as_of": BATCH_AS_OF})
    for item in items:
        item["status"] = check(ROOT / item["label"], (ROOT / item["text"]).read_text(encoding="utf-8"),
                               {"id": item["id"], "labels": item["source"]})["status"]
    return items


def jd_items() -> list[dict]:
    items = []
    for d in json.loads((TL / "jd/manifest.json").read_text(encoding="utf-8"))["documents"]:
        label = TL / "jd/reference" / f"{d['id']}.json"
        items.append({"id": d["id"], "split": d["split"], "source": f"jd_pool/batch{d['batch']}", "text": d["text_path"],
                      "label": rel(label), "status": check_jd(label, (ROOT / d["text_path"]).read_text(encoding="utf-8"))["status"]})
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/datasets/qwen_training/manifest.json")
    parser.add_argument("--gold-style", action="store_true", help="use the *_gs filtered CV labels (V4); pass a new --out")
    args = parser.parse_args()
    tasks = {"cv_extraction": cv_items("_gs" if args.gold_style else ""), "jd_extraction": jd_items(), "score_explanation": []}
    if args.gold_style:  # a missing filtered file would silently shrink the training set
        missing = {i["id"] for i in cv_items()} - {i["id"] for i in tasks["cv_extraction"]}
        assert not missing, f"{len(missing)} CVs have no *_gs label yet, e.g. {sorted(missing)[:5]}"
    for name, items in tasks.items():
        splits = {}
        for item in items:  # an id may sit in one split only
            assert splits.setdefault(item["id"], item["split"]) == item["split"], (name, item["id"])
        print(name, dict(Counter((i["split"], i["status"]) for i in items)))
    write_json(args.out, {"tasks": tasks, "excluded_duplicates": DUPLICATES})


if __name__ == "__main__":
    main()
