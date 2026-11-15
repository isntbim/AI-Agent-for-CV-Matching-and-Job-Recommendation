"""Prepare the 48 held-out pool CVs (manifest splits test + validation, never seen by a teacher) for gold labelling:
PDF text and the teacher input (same v3 prompt and as_of as the gold test set) under output/scorer_v3/teacher_labels/holdout/.

Also flags any holdout CV whose text is a near-copy of a gold CV (word 5-gram overlap), since a copy of a gold test CV
would add no new evidence.

    python -m scripts.cv_holdout_prepare
"""
import json
import re
from pathlib import Path

from src.parser.extractors import extract_text_from_file
from src.scoring.extraction import _prompts, write_json

ROOT = Path(__file__).resolve().parents[1]
TL = ROOT / "output/scorer_v3/teacher_labels"
OUT = TL / "holdout"
AS_OF = "2026-10-25"  # the gold test labels' as_of_date


def grams(text: str, n: int = 5) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main() -> None:
    m = json.loads((TL / "manifest.json").read_text(encoding="utf-8"))
    held = [d for d in m["documents"] if d["split"] in ("test", "validation")]
    gold = {p.stem: grams(p.read_text(encoding="utf-8")) for p in (TL / "gold_texts").glob("*.txt")}
    rows = []
    for d in held:
        row = {k: d[k] for k in ("id", "domain", "split", "source_path", "sha256")}
        try:
            text = extract_text_from_file(ROOT / d["source_path"])
        except Exception as exc:
            row["text_error"] = f"{type(exc).__name__}: {exc}"[:300]
            rows.append(row)
            continue
        g = grams(text)
        best = max(((len(g & v) / max(1, min(len(g), len(v))), k) for k, v in gold.items()), default=(0, None))
        row |= {"characters": len(text), "max_gold_overlap": round(best[0], 3), "closest_gold": best[1],
                "text_path": f"output/scorer_v3/teacher_labels/holdout/texts/{d['id']}.txt",
                "input_path": f"output/scorer_v3/teacher_labels/holdout/inputs/{d['id']}.json", "as_of": AS_OF}
        (ROOT / row["text_path"]).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / row["text_path"]).write_text(text, encoding="utf-8")
        _, user = _prompts("resume", text, "v3", AS_OF)
        (ROOT / row["input_path"]).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / row["input_path"]).write_text(user, encoding="utf-8")
        rows.append(row)
    write_json(OUT / "manifest.json", {"as_of": AS_OF, "documents": rows})
    near = [(r["id"], r["closest_gold"], r["max_gold_overlap"]) for r in rows if r.get("max_gold_overlap", 0) >= 0.5]
    print(f"{sum('text_path' in r for r in rows)}/{len(rows)} extracted; text errors: "
          f"{[r['id'] for r in rows if 'text_error' in r]}; near-copies of gold (>= 50% 5-gram overlap): {near}")


if __name__ == "__main__":
    main()
