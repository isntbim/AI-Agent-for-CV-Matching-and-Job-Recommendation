"""O5 text metrics (mentor M3.3): BLEU, ROUGE-L and BERTScore of each system against the team-written references.

References: 30 test pairs x EN/VI = 60 texts written by two team members (writers A/B) on the O5 reference board,
exported to output/o5/references/refs_final.json. Each text has one reference, so scores are single-reference.

Systems (all on the same test prompts, data/datasets/o5_explanation/test_prompts.jsonl):
  base      Qwen2.5-7B-Instruct, no adapter               output/o5v2/preds/o5v2_base_test.jsonl
  o5        O5 v3 adapter (seed 3407), as evaluated        output/o5v3/preds/o5v3_adapter_test.jsonl
  o5_guard  O5 v3 + fact guard (template when rejected)   output/o5v3/preds/o5v3_guard_test.jsonl
  template  fixed sentences from the facts only            scripts.o5_export_sft.fallback

BLEU = sacrebleu corpus BLEU (13a). ROUGE-L = mean F1 with a Unicode word tokenizer (rouge_score's default one drops
Vietnamese letters). BERTScore = mean F1 with xlm-roberta-large for both languages, no baseline rescaling.
95% CIs: 1,000 bootstrap resamples of the pairs.

  python -m scripts.o5_text_metrics [--no-wandb]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "output/o5/references/refs_final.json"
OUT = ROOT / "output/o5/references/text_metrics.json"
PREDS = {"base": "output/o5v2/preds/o5v2_base_test.jsonl", "o5": "output/o5v3/preds/o5v3_adapter_test.jsonl",
         "o5_guard": "output/o5v3/preds/o5v3_guard_test.jsonl"}
BERT_MODEL = "xlm-roberta-large"


class _Words:
    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())  # \w is Unicode-aware, so "kỹ năng" stays two words


def predictions(refs: list[dict]) -> dict[str, list[str]]:
    from scripts.o5_export_sft import facts_of, fallback

    out = {}
    for name, path in PREDS.items():
        rows = {r["id"]: r["prediction"] for r in map(json.loads, (ROOT / path).read_text(encoding="utf-8").splitlines())}
        out[name] = [rows[r["pair"] + ("_vi" if r["lang"] == "vi" else "")] for r in refs]
    out["template"] = [fallback(facts_of(r["pair"]), vi=r["lang"] == "vi") for r in refs]
    return out


def scores(hyps: list[str], refs: list[str], bert) -> dict:
    """Per-item ROUGE-L and BERTScore, plus what corpus BLEU needs to be recomputed on any resample."""
    from rouge_score.rouge_scorer import RougeScorer

    rouge = RougeScorer(["rougeL"], tokenizer=_Words())
    return {"hyps": hyps, "refs": refs, "rougeL": [rouge.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)],
            "bertscore": bert(hyps, refs)}


def summarise(s: dict, idx: list[int]) -> dict:
    from sacrebleu.metrics import BLEU

    pick = lambda xs: [xs[i] for i in idx]
    return {"bleu": BLEU().corpus_score(pick(s["hyps"]), [pick(s["refs"])]).score,
            "rougeL": 100 * sum(pick(s["rougeL"])) / len(idx), "bertscore": 100 * sum(pick(s["bertscore"])) / len(idx)}


def bootstrap(s: dict, n: int, rounds: int = 1000, seed: int = 20261118) -> dict:
    rng, draws = random.Random(seed), []
    for _ in range(rounds):
        draws.append(summarise(s, [rng.randrange(n) for _ in range(n)]))
    ci = lambda k: sorted(d[k] for d in draws)
    return {k: [ci(k)[int(0.025 * rounds)], ci(k)[int(0.975 * rounds) - 1]] for k in ("bleu", "rougeL", "bertscore")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    import bert_score

    refs = json.loads(REFS.read_text(encoding="utf-8"))
    assert len(refs) == 60 and len({(r["pair"], r["lang"]) for r in refs}) == 60, "expected 30 pairs x EN/VI"
    hyps = predictions(refs)
    scorer = bert_score.BERTScorer(model_type=BERT_MODEL, lang=None, rescale_with_baseline=False)
    bert = lambda h, r: scorer.score(h, r, batch_size=16)[2].tolist()

    result = {"references": len(refs), "bert_model": BERT_MODEL, "systems": {}}
    for name, hs in hyps.items():
        result["systems"][name] = {}
        for lang in ("en", "vi", "all"):
            idx = [i for i, r in enumerate(refs) if lang in (r["lang"], "all")]
            s = scores([hs[i] for i in idx], [refs[i]["text"] for i in idx], bert)
            result["systems"][name][lang] = {**summarise(s, list(range(len(idx)))), "ci95": bootstrap(s, len(idx)), "n": len(idx)}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    rows = [[n, lang, round(v["bleu"], 1), round(v["rougeL"], 1), round(v["bertscore"], 1)]
            for n, by in result["systems"].items() for lang, v in by.items()]
    print(f"{'system':10} lang  BLEU  ROUGE-L  BERTScore")
    for r in rows:
        print(f"{r[0]:10} {r[1]:4} {r[2]:5} {r[3]:8} {r[4]:9}")
    if not args.no_wandb:
        import wandb

        run = wandb.init(entity="itsbim-fpt-university", project="capstone_project", group="review3-o5-text-metrics",
                         job_type="eval", config={"references": len(refs), "bert_model": BERT_MODEL, "preds": PREDS})
        run.log({"o5_text_metrics": wandb.Table(columns=["system", "lang", "bleu", "rougeL", "bertscore"], data=rows)})
        run.summary.update({f"{n}/{lang}/{k}": v[k] for n, by in result["systems"].items() for lang, v in by.items()
                            for k in ("bleu", "rougeL", "bertscore")})
        run.finish()


if __name__ == "__main__":
    main()
