"""Apply the judge's change rows to JD teacher labels: jd/<TEACHER>/<id>.json + rows -> jd/<TARGET>/<id>.json.

A row is {"id", "path", "from", "to", ...}. path is a dotted path with [i] list indices; "...requirements[+]" appends
`to` as a new unit; `to: "DROP"` deletes the list item at that path. "from" must equal the current value (for [+],
null), so a stale row fails loudly. Teacher labels are never modified.
"""

import argparse
import copy
import json
import re
from pathlib import Path

from src.scoring.extraction import write_json

ROOT = Path(__file__).resolve().parent.parent
JD = ROOT / "output/scorer_v3/teacher_labels/jd"
STEP = re.compile(r"([A-Za-z_]+)(?:\[(\d+|\+)\])?")


def walk(label: dict, path: str):
    """Return (container, key) of the last step; key is a dict key, a list index, or "+"."""
    steps = [STEP.fullmatch(s).groups() for s in path.split(".")]
    node = label
    for i, (name, index) in enumerate(steps):
        last = i == len(steps) - 1
        if index is None:
            if last:
                return node, name
            node = node[name]
        else:
            if last:
                return node[name], index if index == "+" else int(index)
            node = node[name][int(index)]
    raise ValueError(path)


def apply(label: dict, rows: list, jd_id: str) -> dict:
    label = copy.deepcopy(label)
    # deletions last and bottom up, so earlier indices stay valid; appends after the edits they may rely on
    order = sorted(rows, key=lambda r: (r["to"] == "DROP", r["path"].endswith("[+]"),
                                        -int(m.group(1)) if (m := re.search(r"\[(\d+)\]$", r["path"])) else 0))
    for row in order:
        container, key = walk(label, row["path"])
        if key == "+":
            if row["from"] is not None:
                raise ValueError(f"{jd_id} {row['path']}: an append has no 'from'")
            container.append(row["to"])
            continue
        current = container[key]
        if current != row["from"]:
            raise ValueError(f"{jd_id} {row['path']}: expected {row['from']!r:.80}, found {current!r:.80}")
        if row["to"] == "DROP":
            del container[key]
        else:
            container[key] = row["to"]
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="gemini")
    parser.add_argument("--rows", nargs="+", default=["judge_batch1.json"], help="row files under jd/, in order")
    parser.add_argument("--target", default="reference")
    args = parser.parse_args()
    # each file is one stage: its paths index the label as left by the files before it
    stages = []
    for name in args.rows:
        by_id = {}
        for row in json.loads((JD / name).read_text(encoding="utf-8")):
            by_id.setdefault(row["id"], []).append(row)
        stages.append(by_id)
    ids = sorted(p.stem for p in (JD / args.teacher).glob("jd_*.json"))
    if unknown := sorted({i for by_id in stages for i in by_id} - set(ids)):
        raise SystemExit(f"rows for ids without a teacher label, nothing written: {unknown}")
    labels = {}
    for i in ids:
        label = json.loads((JD / args.teacher / f"{i}.json").read_text(encoding="utf-8"))
        for by_id in stages:
            label = apply(label, by_id.get(i, []), i)
        labels[i] = label
    for jd_id, label in labels.items():
        write_json(JD / args.target / f"{jd_id}.json", label)
    print("wrote", len(labels), "labels to", JD / args.target, "| rows applied:",
          sum(len(by_id.get(i, [])) for by_id in stages for i in ids))


if __name__ == "__main__":
    main()
