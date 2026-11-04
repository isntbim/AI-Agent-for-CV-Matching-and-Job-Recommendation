"""Matching evaluation on the clean validation pairs: V4 extraction (Ollama, validate-repair-retry) -> score_pair_v3,
compared with the dataset's No / Potential / Good Fit labels.

  python -m scripts.match_eval [--workers 3] [--limit 20] [--score-only]
  python -m scripts.match_eval --export-prompts output/match_eval/prompts.jsonl      (for the A100 pod)
  python -m scripts.match_eval --replay preds/match_rp105.jsonl [preds/match_retry.jsonl]  (pod outputs, no GPU)

Pairs: med2425/resume-job-fit-merged-v1 train split minus every row whose resume is one of the 157 med2425 CVs used
in extraction training -> 59,337 pairs, 485 CVs x 280 JDs. Labels are synthetic (source generated_smart).
Extractions are cached (output/match_eval/extractions), so a stopped run resumes; --score-only scores what is cached.
--workers > 1 needs OLLAMA_NUM_PARALLEL >= workers on the server. Writes pairs.csv and metrics.json next to the cache.
--replay runs the same validate-repair-retry graph on generations made by training/scripts/generate.py --prompts
(vLLM, bf16 + adapter on the A100); documents whose next repetition penalty was not generated are listed in
retry_ids.txt for the pod's retry pass.
"""

import argparse
import csv
import glob
import json
import random
import statistics
import threading
import traceback
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from hashlib import sha256
from pathlib import Path

from src.parser.llm_extractor import ExtractionError
from src.scoring.extraction import ExtractionCache, extract_document, write_json
from src.scoring.models import ScorerConfig
from src.scoring.runtime_v3 import score_pair_v3
from src.scoring.v4_client import QwenV4, V4Client

ROOT = Path(__file__).resolve().parent.parent
AUTO = ScorerConfig(version="ats-en-v3", review_policy="automatic")  # unreviewed V4 extractions
OUT = ROOT / "output/match_eval"
LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
AS_OF = "2024-01-01"  # latest year named in any of these CVs is 2023; "present" jobs end here


def load_pairs() -> list[dict]:
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(str(ROOT / "data/raw/external/med2425_resume_job_fit_merged_v1/data/train-*.parquet")))
    rows = [r for p in paths for r in pq.read_table(p, columns=["resume", "jd", "label"]).to_pylist()]
    manifest = json.loads((ROOT / "output/scorer_v3/teacher_labels/manifest_external.json").read_text(encoding="utf-8"))
    trained = {(ROOT / d["text_path"]).read_text(encoding="utf-8") for d in manifest["documents"] if d["source"] == "med2425"}
    pairs = [{"cv": r["resume"].strip(), "jd": r["jd"].strip(), "label": LABELS[r["label"]]}
             for r in rows if r["resume"].strip() not in trained]
    assert len(trained) == 157 and len(pairs) == 59337, (len(trained), len(pairs))
    return pairs


def doc_id(kind: str, text: str) -> str:
    return f"{kind}_{sha256(text.encode()).hexdigest()[:12]}"


class MissingGeneration(ExtractionError):
    """The replay file has no output for this document at this repetition penalty: the pod must generate it."""


class Replay:
    """Plays back pod generations (one row per document id and repetition penalty) through QwenV4's interface."""

    def __init__(self, prompts: Path, paths: list[Path]):
        self.ids = {sha256(r["user"].encode()).hexdigest(): r["id"] for r in map(json.loads, prompts.open(encoding="utf-8"))}
        self.rows = {(r["id"], r["repetition_penalty"]): r for p in paths for r in map(json.loads, p.open(encoding="utf-8"))}

    def complete(self, system: str, user: str, repetition_penalty: float) -> tuple[str, str]:
        key = (self.ids[sha256(user.encode()).hexdigest()], repetition_penalty)
        if key not in self.rows:
            raise MissingGeneration(f"no generation for {key}")
        row = self.rows[key]
        return row["prediction"], "length" if row["finish_reason"] == "length" else "stop"


class _Recorder:
    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=4096):
        self.prompt = (system_prompt, user_prompt)
        raise ExtractionError("recorded")


def export_prompts(docs: dict, path: Path) -> None:
    """The exact system/user prompts extract_document sends, so the pod generates what the local graph would ask for."""
    with path.open("w", encoding="utf-8") as f:
        for (kind, text), name in docs.items():
            recorder = _Recorder()
            try:
                extract_document(text, kind, recorder, 40000, 16384, "v3", AS_OF if kind == "resume" else None)
            except ExtractionError:
                pass
            system, user = recorder.prompt
            f.write(json.dumps({"id": name, "kind": kind, "system": system, "user": user}, ensure_ascii=False) + "\n")
    print(f"{len(docs)} prompts -> {path}")


def caches(replay: Replay | None = None):
    """One client per thread (V4Client keeps per-call metadata on itself), one shared tokenizer: loaded here in the
    main thread, because transformers' lazy imports race when several threads import it at once."""
    llm, local = replay or QwenV4(), threading.local()
    backend = "vllm-a100-bf16" if replay else "ollama"  # separate cache entries: different precision

    def get(kind: str) -> ExtractionCache:
        if not hasattr(local, "client"):
            local.client = V4Client(llm)
        return ExtractionCache(OUT / "extractions", local.client, backend, "qwen2.5-7b-v4-qlora", max_tokens=16384,
                               ruleset="v3", as_of=AS_OF if kind == "resume" else None)
    return get


def extract_all(docs: dict, workers: int, replay: Replay | None = None) -> dict:
    get, done, failed = caches(replay), {}, {}
    start = time.time()

    def one(key):
        kind, text = key
        return get(kind).get(text, kind)

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, key): key for key in docs}
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                done[key] = future.result()
                status = "ok"
            except Exception as exc:  # a document V4 cannot extract drops its pairs; counted in metrics.json
                failed[docs[key]] = f"{type(exc).__name__}: {str(exc)[:200]}"
                status = "NEEDS_RETRY" if isinstance(exc, MissingGeneration) else "FAILED"
                if status == "FAILED":
                    traceback.print_exc(limit=-4)
            print(f"{i}/{len(docs)} {docs[key]} {status} {time.time() - start:.0f}s", flush=True)
    return done, failed


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC with ties counted as half; None when a class is absent (small --limit runs)."""
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2  # 1-based average rank of the tie block
        rank_sum += avg * sum(flag for _, flag in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def rho(rows: list[dict]) -> float | None:
    """Spearman of score vs label; None when either is constant."""
    x, y = [r["score"] for r in rows], [r["label"] for r in rows]
    return statistics.correlation(x, y, method="ranked") if len(set(x)) > 1 and len(set(y)) > 1 else None


def metrics(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r["score"])
    per_cv = defaultdict(list)
    for r in rows:
        per_cv[r["cv_id"]].append(r)
    cv_rho = [v for g in per_cv.values() if (v := rho(g)) is not None]
    return {"spearman": rho(rows),
            "auc_good_vs_no": auc(by[2], by[0]),
            "auc_fit_vs_no": auc(by[2] + by[1], by[0]),
            "mean_per_cv_spearman": statistics.mean(cv_rho) if cv_rho else None}


def bootstrap(rows: list[dict], reps: int = 1000) -> dict:
    """95% CI resampling CVs (each CV has ~120 pairs, so pairs are not independent)."""
    per_cv = defaultdict(list)
    for r in rows:
        per_cv[r["cv_id"]].append(r)
    ids, rng, samples = list(per_cv), random.Random(0), defaultdict(list)
    for _ in range(reps):
        # a CV drawn twice counts as two CVs in the per-CV mean
        sample = [dict(r, cv_id=(i, cv)) for i, cv in enumerate(rng.choices(ids, k=len(ids))) for r in per_cv[cv]]
        for k, v in metrics(sample).items():
            if v is not None:
                samples[k].append(v)
    return {k: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]] for k, v in samples.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, help="extract only this many CVs and JDs (smoke run)")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--export-prompts", type=Path)
    p.add_argument("--replay", type=Path, nargs="+", help="pod generation files (generate.py --prompts output)")
    p.add_argument("--prompts", type=Path, default=OUT / "prompts.jsonl", help="the --export-prompts file the pod used")
    p.add_argument("--local-fallback", action="store_true", help="with --replay: extract still-failing documents locally")
    args = p.parse_args()
    pairs = load_pairs()
    docs = {}
    for pair in pairs:
        docs.setdefault(("resume", pair["cv"]), doc_id("cv", pair["cv"]))
        docs.setdefault(("job", pair["jd"]), doc_id("jd", pair["jd"]))
    if args.limit:
        cvs = [k for k in docs if k[0] == "resume"][:args.limit]
        jds = [k for k in docs if k[0] == "job"][:args.limit]
        docs = {k: docs[k] for k in cvs + jds}
    if args.export_prompts:
        return export_prompts(docs, args.export_prompts)
    replay = Replay(args.prompts, args.replay) if args.replay else None
    if args.score_only:
        get, extracted, failed = caches(replay), {}, {}
        for key, name in docs.items():
            cache = get(key[0])
            cache.client = None  # cache hits only
            try:
                extracted[key] = cache.get(key[1], key[0])
            except Exception as exc:
                failed[name] = f"{type(exc).__name__}: {str(exc)[:200]}"
    else:
        extracted, failed = extract_all(docs, args.workers, replay)
    local = {}
    if replay and args.local_fallback:  # pod outputs that still fail -> local V4 (Ollama, loop surgery), named in metrics
        redo = {k: n for k, n in docs.items() if n in failed and not failed[n].startswith("MissingGeneration")}
        if args.score_only:
            for key, name in redo.items():
                cache = caches()(key[0])
                cache.client = None
                try:
                    local[key] = cache.get(key[1], key[0])
                except Exception:
                    pass
        else:
            local = extract_all(redo, args.workers)[0]
        for key in local:
            failed.pop(docs[key])
        extracted.update(local)
    if replay:
        retry = sorted(name for name, error in failed.items() if error.startswith("MissingGeneration"))
        (OUT / "retry_ids.txt").write_text("\n".join(retry), encoding="utf-8")
        print(f"retry_ids.txt: {len(retry)} documents need the next repetition penalty")
        if retry:
            return

    as_of, rows = date.fromisoformat(AS_OF), []
    for pair in pairs:
        cv, jd = extracted.get(("resume", pair["cv"])), extracted.get(("job", pair["jd"]))
        if cv and jd:
            result = score_pair_v3(cv, jd, as_of_date=as_of, config=AUTO)
            rows.append({"cv_id": docs[("resume", pair["cv"])], "jd_id": docs[("job", pair["jd"])],
                         "label": pair["label"], "score": result.unrounded_overall_score, "grade": result.grade,
                         **{f"dim_{k}": v for k, v in result.unrounded_scores.model_dump().items()}})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [r["label"] for r in rows]
    report = {"as_of": AS_OF, "pairs_total": len(pairs), "pairs_scored": len(rows),
              "label_counts": {name: labels.count(v) for name, v in LABELS.items()},
              "documents": len(docs), "failed_documents": failed,
              "local_fallback_documents": sorted(docs[k] for k in local),
              "mean_score_by_label": {name: statistics.mean(r["score"] for r in rows if r["label"] == v)
                                      for name, v in LABELS.items() if v in labels},
              **metrics(rows), "ci95": bootstrap(rows) if len({r["cv_id"] for r in rows}) > 1 else None}
    write_json(OUT / "metrics.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "failed_documents"}, indent=2))
    print(f"failed documents: {len(failed)}")


if __name__ == "__main__":
    main()
