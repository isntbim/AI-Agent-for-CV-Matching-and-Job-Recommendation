"""Paper baseline for O1: SkillSpan's JobBERT taggers (Zhang et al., NAACL 2022; github.com/kris927b/SkillSpan, MIT;
models huggingface.co/jjzha) run zero-shot on our extraction test split.

  python -m scripts.jobbert_baseline --out output/review3/baselines/jobbert_test.jsonl
  python -m scripts.qwen_eval output/review3/baselines/jobbert_test.jsonl     # skill F1 / relaxed F1 only

SkillSpan tags two span types, "skill" (applied/soft) and "knowledge" (tools, languages, domains); our skill lists hold
both, so the prediction is their union. Spans go into a minimal label in our schema so qwen_eval matches names with the
same exact and relaxed rules as every other run. Schema-valid / quote metrics do not apply to a tagger: ignore them.
JobBERT was trained on English job postings only; our JD test split is mostly Vietnamese.
"""
import argparse
import json
from pathlib import Path

from scripts.qwen_local_generate import DATA

MODELS = ("jjzha/jobbert_skill_extraction", "jjzha/jobbert_knowledge_extraction")


def chunks(text: str, limit: int = 450) -> list[str]:
    """Line-based pieces under 512 BERT tokens (the taggers fail on longer input; Vietnamese needs ~1 token/char)."""
    out, cur = [], ""
    for line in text.splitlines():
        if cur and len(cur) + len(line) > limit:
            out.append(cur)
            cur = ""
        cur += line[:limit] + "\n"
    return out + [cur] if cur.strip() else out


def bio_spans(text: str, ents: list[dict]) -> list[str]:
    """The pipeline groups B and I tags separately; a span is a B group plus the I groups that directly follow it."""
    spans, cur = [], None
    for e in sorted(ents, key=lambda e: e["start"]):
        if e["entity_group"] == "I" and cur and not text[cur[1]:e["start"]].strip():
            cur[1] = e["end"]
            continue
        if cur:
            spans.append(cur)
        cur = [e["start"], e["end"]] if e["entity_group"] in ("B", "I") else None
    spans += [cur] if cur else []
    return [s for a, b in spans if (s := " ".join(text[a:b].split()))]


def _check() -> None:
    t = "Develop and execute sales strategy, Python"
    ents = [{"entity_group": "B", "start": 0, "end": 7}, {"entity_group": "I", "start": 8, "end": 34},
            {"entity_group": "B", "start": 36, "end": 42}]
    assert bio_spans(t, ents) == ["Develop and execute sales strategy", "Python"]


def main() -> None:
    _check()
    from transformers import pipeline

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    taggers = [pipeline("token-classification", model=m, aggregation_strategy="first", device=args.device) for m in MODELS]
    rows = [dict(json.loads(line), task=task) for task in ("cv_extraction", "jd_extraction")
            for line in (DATA / task / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            text = json.loads(r["messages"][1]["content"])["source_text"]
            spans = []
            for piece in chunks(text):
                for tag in taggers:
                    spans += bio_spans(piece, tag(piece))
            names = list(dict.fromkeys(spans))  # keep order, drop repeats
            items = [{"name": n} for n in names]
            doc = ({"skills": {"required": items, "preferred": []}} if r["task"] == "jd_extraction"
                   else {"skills": [{"name": "all", "details": items}]})
            f.write(json.dumps({"id": r["id"], "task": r["task"], "split": args.split, "finish_reason": "stop",
                                "prediction": json.dumps({"document": doc, "facts": {}}, ensure_ascii=False),
                                "model": "+".join(MODELS)}, ensure_ascii=False) + "\n")
            print(r["id"], len(names), names[:6], flush=True)


if __name__ == "__main__":
    main()
