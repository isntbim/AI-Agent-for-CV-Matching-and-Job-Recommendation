"""Label-free cleanup of the automatic skill-relation table (bge-m3 confuses short acronyms: sas/saas, css/cms).

Drops cosine relations where either side is a short token (<= 5 letters), containment relations whose shorter side
is under 4 letters ('c', 'add'), and any side longer than 8 words (unsplit extraction dumps)."""
import argparse
import json
from pathlib import Path

from src.parser.skill_v2 import SkillRelationTable


def keep(rel: dict) -> bool:
    a, b = rel["candidate"], rel["requirement"]
    if max(len(a.split()), len(b.split())) > 8:
        return False
    compact = min(len(a.replace(" ", "")), len(b.replace(" ", "")))
    if rel["reason"].startswith("cosine"):
        return compact > 5
    return compact >= 4


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    table = json.loads(args.src.read_text(encoding="utf-8"))
    before = len(table["relations"])
    table["relations"] = [r for r in table["relations"] if keep(r)]
    table["version"] += "-filtered"
    SkillRelationTable.model_validate(table)
    args.out.write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{before} -> {len(table['relations'])} relations")


if __name__ == "__main__":
    main()
