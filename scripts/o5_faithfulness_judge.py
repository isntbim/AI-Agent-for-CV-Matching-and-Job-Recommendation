"""O5 faithfulness with an LLM judge, using the RAGAS faithfulness definition (mentor M3.4).

RAGAS (Es et al., 2024, "RAGAS: Automated Evaluation of Retrieval Augmented Generation", EACL demo): break the answer into
atomic claims, ask a judge whether each claim can be inferred from the context, and score
    faithfulness = supported claims / all claims.
Here the context is the facts block the O5 model was given (the test prompt's user turn) and the answer is the explanation.
Judges (--judge), one batch of texts per call, same instructions for all:
  gemini  Gemini 4 Hydrogen (XHigh) through the Antigravity CLI (agy) in print mode   cache output/o5/judge/cache/
  gemini_high  Gemini 4 Hydrogen (High) through agy (faster; used for the paper-exact rerun)  cache output/o5/judge/gemini_high[_ragas]/cache/
  sonnet  Claude Sonnet 5.5 (High) through agy                                        cache output/o5/judge/sonnet/cache/
  terra   GPT-6.1-Terra (xhigh) via `headroom wrap codex --no-proxy --no-mcp -- exec` (existing proxy :8787), no user config/MCP    cache output/o5/judge/terra/cache/

Systems (same 244 test prompts, 122 pairs x EN/VI):
  base      Qwen2.5-7B-Instruct, no adapter               output/o5v2/preds/o5v2_base_test.jsonl
  o5        O5 v3 adapter (seed 3407)                      output/o5v3/preds/o5v3_adapter_test.jsonl
  o5_guard  O5 v3 + fact guard (template when rejected)   output/o5v3/preds/o5v3_guard_test.jsonl

Judgements are cached per (pair, text), so identical texts (o5 and o5_guard share 195) are judged once and a stopped run resumes.

  python -m scripts.o5_faithfulness_judge [--judge gemini|sonnet|terra] [--limit N] [--workers 4] [--no-wandb]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.o5_export_sft import facts_of
from scripts.o5_faithfulness import violations

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data/datasets/o5_explanation/test_prompts.jsonl"
PREDS = {"base": "output/o5v2/preds/o5v2_base_test.jsonl", "o5": "output/o5v3/preds/o5v3_adapter_test.jsonl",
         "o5_guard": "output/o5v3/preds/o5v3_guard_test.jsonl"}
OUT = ROOT / "output/o5/judge"
JUDGES = {"gemini": "gemini-4-hydrogen-xhigh", "gemini_high": "gemini-4-hydrogen-high", "sonnet": "claude-sonnet-5-5-high", "terra": "gpt-6.1-terra"}
JUDGE = "gemini"  # set from --judge in main()
VARIANT = "ours"  # "ours" = one-call adapted prompt; "ragas" = paper-exact two-step prompts
BATCH = 6

INSTRUCTIONS = """You are a strict fact-checking judge. Use the RAGAS faithfulness procedure for EACH item below.
Do not use any tools, do not read or write files: everything you need is in this message.

Each item has FACTS (the only source of truth) and an EXPLANATION written for a job seeker, in English or Vietnamese.
Step 1. Split the explanation into atomic factual claims about the CV, the job, the scores, the verdict or the match.
        Skip pure advice ("add an example to your CV") unless it asserts a fact (e.g. "X is a main gap" asserts X is a
        missing requirement). Keep each claim short, in the explanation's own language.
Step 2. For each claim, supported = 1 if it can be directly inferred from the FACTS, else 0.
Rules for reading the FACTS:
- Anything not stated in or directly inferable from the FACTS is unsupported, even if plausible.
- NOT CHECKED parts have no result: "could not be checked/compared" is supported; saying such a part matches, meets,
  is relevant, is close, is missing or is not met is unsupported.
- "Partly met" described as fully met ("meets", "has") is unsupported.
- A reason given for a score ("the skill score is low because ...") is supported only if it matches the FACTS: the
  skills score comes from the listed job requirements; nice-to-have items do not change it.
- Vietnamese verdict labels: "Phù hợp tốt" = Good Fit, "Có tiềm năng" = Potential Fit, "Chưa phù hợp" = No Fit.
- Numbers must match the FACTS exactly.

Reply with JSON only, no code fence, in this exact shape:
{"items": [{"key": "<item key>", "claims": [{"claim": "...", "supported": 1, "reason": ""}]}]}
Give a short reason only when supported is 0. Return every item key exactly once.
"""


def key_of(pid: str, text: str) -> str:
    return hashlib.sha1(f"{pid}\n{text}".encode()).hexdigest()[:16]


def cache_dir() -> Path:
    if VARIANT == "ragas":
        return OUT / f"{JUDGE}_ragas" / "cache"
    return OUT / "cache" if JUDGE == "gemini" else OUT / JUDGE / "cache"  # gemini keeps its original path


def call(prompt: str, cwd: Path, tag: str) -> str:
    """One judge call; returns the model's final text."""
    if JUDGE == "terra":
        last = cwd / f"last_{tag}.txt"
        cmd = ["headroom", "wrap", "codex", "--no-proxy", "--no-mcp", "--", "exec", "-m", JUDGES["terra"], "-c", 'model_reasoning_effort="xhigh"', "--ignore-user-config",
               "--ephemeral", "--skip-git-repo-check", "-s", "read-only", "-C", str(cwd), "-o", str(last), "-"]
        run = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=1500)
        if run.returncode != 0 or not last.exists():
            raise RuntimeError(f"codex exit {run.returncode}: {run.stderr[-200:]}")
        text, usage, seconds = last.read_text(encoding="utf-8"), {}, None
        last.unlink()
    else:
        cmd = ["agy", "-p", prompt, "--model", JUDGES[JUDGE], "--output-format", "json",
               "--disable-slash-commands", "--print-timeout", "900s"]
        run = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=960)
        envelope = json.loads(run.stdout[run.stdout.index("{"):])
        if envelope.get("status") != "SUCCESS":
            raise RuntimeError(f"agy status {envelope.get('status')}")
        text, usage, seconds = envelope["response"], envelope.get("usage", {}), envelope.get("duration_seconds")
    return text


def _json(text: str) -> dict:
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", text.strip()))


def ask(items: list[dict], cwd: Path) -> dict[str, list]:
    if VARIANT == "ragas":
        return ask_ragas(items, cwd)
    body = "\n\n".join(f"=== ITEM {it['key']} ===\nFACTS:\n{it['facts']}\n\nEXPLANATION:\n{it['text']}" for it in items)
    got = {x["key"]: x["claims"] for x in _json(call(INSTRUCTIONS + "\n\n" + body, cwd, items[0]["key"]))["items"]}
    return {"claims": got}


# Paper-exact variant (Es et al., Section 3): step 1 sees only question + answer (never the context), step 2 sees the
# context + statements and gives a brief explanation BEFORE each Yes/No verdict; both prompts carry one demonstration
# (footnote 3). The paper's prompt sentences are kept verbatim; only the batching and the JSON wrapper are ours.
QUESTION = "How well does my CV match this job, and what should I do next?"
DEMO_ANSWER = ("Potential Fit (fit 55/100): your CV shows SQL and Excel, which the job lists. It does not show Power BI. "
               "If you have used Power BI, add an example to your CV.")
DEMO_STATEMENTS = ["The verdict is Potential Fit.", "The fit score is 55/100.", "The CV shows SQL.", "The CV shows Excel.",
                   "The job lists SQL and Excel.", "The CV does not show Power BI.",
                   "The candidate should add an example of Power BI to the CV if they have used it."]
DEMO_FACTS = "Verdict: Potential Fit. Fit: 55/100.\nSKILLS: Matched on the CV: SQL. Missing from the CV: Power BI; Excel."
DEMO_VERDICTS = [
    {"statement": DEMO_STATEMENTS[0], "explanation": "The context states Verdict: Potential Fit.", "verdict": "Yes"},
    {"statement": DEMO_STATEMENTS[1], "explanation": "The context states Fit: 55/100.", "verdict": "Yes"},
    {"statement": DEMO_STATEMENTS[2], "explanation": "SQL is matched on the CV.", "verdict": "Yes"},
    {"statement": DEMO_STATEMENTS[3], "explanation": "Excel is listed as missing from the CV.", "verdict": "No"},
    {"statement": DEMO_STATEMENTS[4], "explanation": "Both appear among the job's skills.", "verdict": "Yes"},
    {"statement": DEMO_STATEMENTS[5], "explanation": "Power BI is missing from the CV.", "verdict": "Yes"},
    {"statement": DEMO_STATEMENTS[6], "explanation": "This is advice; the context says nothing about it.", "verdict": "No"}]
STEP1 = ("Given a question and answer, create one or more statements from each sentence in the given answer.\n"
         'Do not use any tools. Reply with JSON only: {"items": [{"key": "...", "statements": ["..."]}]}, '
         "with every item key exactly once and the statements in the answer's own language.\n\n"
         "Demonstration:\nquestion: " + QUESTION + "\nanswer: " + DEMO_ANSWER + "\nstatements: " + json.dumps(DEMO_STATEMENTS))
STEP2 = ("Consider the given context and following statements, then determine whether they are supported by the information "
         "present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). "
         "Provide a final verdict for each statement in order at the end in the given format. "
         "Do not deviate from the specified format.\n"
         'Do not use any tools. Reply with JSON only: {"items": [{"key": "...", "verdicts": [{"statement": "...", '
         '"explanation": "...", "verdict": "Yes"}]}]}, with every item key exactly once and the statements in the given order.\n\n'
         "Demonstration:\ncontext: " + DEMO_FACTS + "\nstatements: " + json.dumps(DEMO_STATEMENTS)
         + "\nverdicts: " + json.dumps(DEMO_VERDICTS))


def ask_ragas(items: list[dict], cwd: Path) -> dict[str, list]:
    body1 = "\n\n".join(f"=== ITEM {it['key']} ===\nquestion: {QUESTION}\nanswer: {it['text']}" for it in items)
    stmts = {x["key"]: x["statements"] for x in _json(call(STEP1 + "\n\n" + body1, cwd, items[0]["key"] + "_1"))["items"]}
    ready = [it for it in items if isinstance(stmts.get(it["key"]), list) and stmts[it["key"]]]
    body2 = "\n\n".join(f"=== ITEM {it['key']} ===\ncontext:\n{it['facts']}\nstatements: "
                        + json.dumps(stmts[it["key"]], ensure_ascii=False) for it in ready)
    got = {}
    for x in _json(call(STEP2 + "\n\n" + body2, cwd, items[0]["key"] + "_2"))["items"]:
        v = x.get("verdicts")
        if isinstance(v, list) and v and all(str(y.get("verdict", "")).strip().lower() in ("yes", "no") for y in v):
            got[x["key"]] = [{"claim": y.get("statement", ""), "supported": int(str(y["verdict"]).strip().lower() == "yes"),
                              "reason": y.get("explanation", "")} for y in v]
    return {"claims": got}


def judge_batch(items: list[dict], cwd: Path) -> int:
    """Judge one batch, retrying once for any item the reply dropped or mangled; returns how many items were cached."""
    todo, saved = items, 0
    for _ in range(2):
        try:
            res = ask(todo, cwd)
        except Exception as e:  # one failed call should not stop the run; the item stays uncached and is retried next run
            print(f"batch failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        for it in todo:
            claims = res["claims"].get(it["key"])
            if isinstance(claims, list) and all(c.get("supported") in (0, 1) for c in claims):
                (cache_dir() / f"{it['key']}.json").write_text(json.dumps(
                    {"pair": it["pid"], "text": it["text"], "claims": claims, "model": JUDGES[JUDGE]}, ensure_ascii=False), encoding="utf-8")
                saved += 1
        todo = [it for it in todo if not (cache_dir() / f"{it['key']}.json").exists()]
        if not todo:
            break
    return saved


def score(claims: list[dict]) -> float | None:
    return sum(c["supported"] for c in claims) / len(claims) if claims else None  # RAGAS leaves 0-claim answers undefined


def summarise(rows: list[dict]) -> dict:
    vals = [r["score"] for r in rows if r["score"] is not None]
    rng, n = random.Random(20261118), len(vals)
    if not n:
        return {"n": len(rows), "scored": 0, "faithfulness": float("nan"), "ci95": [float("nan")] * 2,
                "fully_faithful_pct": float("nan"), "claims": 0, "unsupported": 0}
    boots = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(1000))
    return {"n": len(rows), "scored": n, "faithfulness": 100 * sum(vals) / n, "ci95": [100 * boots[25], 100 * boots[974]],
            "fully_faithful_pct": 100 * sum(v == 1 for v in vals) / n,
            "claims": sum(len(r["claims"]) for r in rows), "unsupported": sum(len(r["claims"]) - sum(c["supported"] for c in r["claims"]) for r in rows)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="judge only the first N prompts (pilot)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--judge", choices=list(JUDGES), default="gemini")
    p.add_argument("--variant", choices=["ours", "ragas"], default="ours", help="ragas = paper-exact two-step prompts")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    global JUDGE, VARIANT
    JUDGE, VARIANT = args.judge, args.variant
    cache_dir().mkdir(parents=True, exist_ok=True)
    cwd = OUT / f"{JUDGE}_cwd"  # empty folder, so agy loads no project context
    cwd.mkdir(exist_ok=True)

    prompts = [json.loads(x) for x in PROMPTS.read_text(encoding="utf-8").splitlines()][:args.limit]
    preds = {name: {r["id"]: r["prediction"] for r in map(json.loads, (ROOT / path).read_text(encoding="utf-8").splitlines())}
             for name, path in PREDS.items()}
    jobs = {}
    for pr in prompts:
        for name in PREDS:
            text = preds[name][pr["id"]]
            k = key_of(pr["id"], text)
            jobs.setdefault(k, {"key": k, "pid": pr["id"], "facts": pr["user"], "text": text})
    todo = [j for k, j in jobs.items() if not (cache_dir() / f"{k}.json").exists()]
    print(f"{len(jobs)} distinct texts, {len(todo)} to judge in {(len(todo) + BATCH - 1) // BATCH} calls", flush=True)
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for saved in pool.map(lambda b: judge_batch(b, cwd), batches):
            done += saved
            print(f"judged {done}/{len(todo)}", flush=True)

    result, missing = {"judge": JUDGES[JUDGE], "variant": VARIANT, "definition": "RAGAS faithfulness = supported claims / claims", "systems": {}}, 0
    for name in PREDS:
        rows = []
        for pr in prompts:
            text = preds[name][pr["id"]]
            f = cache_dir() / f"{key_of(pr['id'], text)}.json"
            if not f.exists():
                missing += 1
                continue
            claims = json.loads(f.read_text(encoding="utf-8"))["claims"]
            rows.append({"id": pr["id"], "lang": "vi" if pr["id"].endswith("_vi") else "en", "claims": claims, "score": score(claims),
                         "rule_flagged": bool(violations(text, facts_of(pr["id"].removesuffix("_vi"))))})
        result["systems"][name] = {lang: summarise([r for r in rows if lang in (r["lang"], "all")]) for lang in ("en", "vi", "all")}
        flagged = [r for r in rows if r["rule_flagged"]]
        result["systems"][name]["rule_checker_agreement"] = {  # how the cheap rule checker lines up with the judge
            "rule_flagged": len(flagged), "judge_also_unfaithful": sum(r["score"] is not None and r["score"] < 1 for r in flagged),
            "rule_clean_but_judge_unfaithful": sum(r["score"] is not None and r["score"] < 1 for r in rows if not r["rule_flagged"])}
    result["missing_judgements"] = missing
    tag = ("" if JUDGE == "gemini" else f"_{JUDGE}") + ("_ragas" if VARIANT == "ragas" else "")
    name_out = f"faithfulness_judge{tag}.json" if args.limit is None else f"faithfulness_judge{tag}_pilot{args.limit}.json"
    (OUT / name_out).write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n{'system':9} lang  faithfulness [95% CI]   fully-faithful  claims  unsupported")
    for name, by in result["systems"].items():
        for lang in ("en", "vi", "all"):
            v = by[lang]
            print(f"{name:9} {lang:4} {v['faithfulness']:5.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]   {v['fully_faithful_pct']:5.1f}%"
                  f"   {v['claims']:6} {v['unsupported']:6}")
        print(f"          rule checker vs judge: {by['rule_checker_agreement']}")
    print(f"missing judgements: {missing}")
    if not args.no_wandb and args.limit is None:
        import wandb

        run = wandb.init(entity="itsbim-fpt-university", project="capstone_project", group="review3-o5-faithfulness-judge",
                         job_type="eval", name=f"o5-judge-{JUDGE}" + ("-ragas" if VARIANT == "ragas" else ""), config={"judge": JUDGES[JUDGE], "variant": VARIANT, "batch": BATCH, "preds": PREDS, "definition": result["definition"]})
        run.log({"o5_faithfulness_judge": wandb.Table(
            columns=["system", "lang", "faithfulness", "ci_low", "ci_high", "fully_faithful_pct", "claims", "unsupported"],
            data=[[n, lang, v["faithfulness"], *v["ci95"], v["fully_faithful_pct"], v["claims"], v["unsupported"]]
                  for n, by in result["systems"].items() for lang, v in by.items() if lang != "rule_checker_agreement"])})
        run.summary.update({f"{n}/{lang}/faithfulness": v["faithfulness"] for n, by in result["systems"].items()
                            for lang, v in by.items() if lang != "rule_checker_agreement"})
        run.finish()


if __name__ == "__main__":
    main()
