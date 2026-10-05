"""Reproducible ATS dataset preparation, extraction, scoring and evaluation.

Run from the repository root: python -m scripts.ats_eval --help
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import httpx

from src.parser.llm_extractor import ExtractionError, VLLMClient
from src.scoring.dataset import (
    DATASET_ID,
    DATASET_REVISION,
    LABELS,
    load_pairs,
    prepare_dataset,
)
from src.scoring.evaluation import calibrate, evaluate
from src.scoring.extraction import (
    ExtractionCache,
    ScoringOllamaClient,
    text_hash,
    write_json,
)
from src.scoring.gemini import GeminiClient
from src.scoring.models import ScorerConfig
from src.scoring.rules import score_pair

_BATCH_ERRORS = (ExtractionError, ValueError, OSError, httpx.HTTPError)


def code_digest() -> str:
    files = sorted((Path(__file__).resolve().parents[1] / "src" / "scoring").glob("*.py"))
    hasher = hashlib.sha256()
    for file in files:
        hasher.update(file.name.encode())
        hasher.update(file.read_bytes())
    return hasher.hexdigest()


def manifest_digest(directory: Path) -> str:
    hasher = hashlib.sha256()
    for name in ("manifest.json", "pairs.json"):
        hasher.update((directory / name).read_bytes())
    return hasher.hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cache(args, online=False):
    client = None
    if online:
        if args.backend == "gemini":
            client = GeminiClient(base_url=args.endpoint, model=args.model)
        elif args.backend == "ollama":
            client = ScoringOllamaClient(base_url=args.endpoint, model=args.model, context_window=args.context_window)
        else:
            client = VLLMClient(base_url=args.endpoint, model=args.model)
    return ExtractionCache(args.cache, client, args.backend + ":" + args.endpoint.rstrip("/"), args.model,
                           args.max_chars, args.max_tokens, args.context_window)


def _identity(args):
    return {"dataset_digest": manifest_digest(args.prepared), "code_digest": code_digest(),
                "as_of_date": args.as_of, "extraction": _cache(args).identity}


def download(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name in ("train.csv", "test.csv"):
            path = args.output / name
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{name}"
            with client.stream("GET", url) as response:
                response.raise_for_status()
                temporary = path.with_suffix(".csv.download")
                try:
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_bytes():
                            stream.write(chunk)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            sources.append({"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    metadata = {"dataset": DATASET_ID, "revision": DATASET_REVISION, "files": sources}
    write_json(args.output / "source.json", metadata)
    return metadata


def extract(args):
    _, pairs = load_pairs(args.prepared)
    pairs = [p for p in pairs if p["split"] == args.split]
    documents = {}
    for pair in pairs:
        for kind, field in [("resume", "resume_text"), ("job", "job_description_text")]:
            documents.setdefault((kind, text_hash(pair[field])), pair[field])
    selected = sorted(documents.items())
    if args.limit is not None:
        selected = selected[:args.limit]
    cache, failures, successful = _cache(args, online=True), [], 0
    for (kind, identity), text in selected:
        try:
            cache.get(text, kind)
            successful += 1
        except _BATCH_ERRORS as exc:
            failures.append({"kind": kind, "document_id": identity, "error_type": type(exc).__name__, "error": str(exc)[:500]})
        print(f"{kind} {identity[:12]} {successful + len(failures)}/{len(selected)}", file=sys.stderr)
    report = {"split": args.split, "selected": len(selected), "total_documents": len(documents),
                  "successful": successful, "failures": failures, "extraction": cache.identity}
    write_json(args.output / (args.split + "_extraction.json"), report)
    return report


def _calibration(args):
    payload = _read(args.calibration)
    if payload["identity"] != _identity(args):
        raise ValueError("Calibration identity differs: dataset, rule code, date or extraction configuration changed")
    return payload


def smoke(args):
    """Small, explicitly non-benchmark development run with a real model."""
    if args.split != "development" or args.pairs_per_label < 1:
        raise ValueError("Smoke testing requires development and positive pairs-per-label")
    _, pairs = load_pairs(args.prepared)
    selected = []
    for label in LABELS:
        candidates = [p for p in pairs if p["split"] == "development" and p["label"] == label]
        candidates.sort(key=lambda p: (len(p["resume_text"]) + len(p["job_description_text"]), p["pair_id"]))
        selected.extend(candidates[:args.pairs_per_label])
    cache, rows, failures = _cache(args, online=True), [], []
    metadata = {}
    if args.backend == "ollama":
        response = httpx.get(args.endpoint.rstrip("/") + "/api/tags", timeout=10)
        response.raise_for_status()
        metadata = next((model for model in response.json().get("models", []) if model["name"] == args.model), {})
    elif args.backend == "gemini":
        metadata = cache.client.describe()
    for index, pair in enumerate(selected, 1):
        print(f"Smoke pair {index}/{len(selected)} {pair['pair_id'][:12]}", file=sys.stderr, flush=True)
        try:
            resume = cache.get(pair["resume_text"], "resume")
            job = cache.get(pair["job_description_text"], "job")
            result = score_pair(resume, job, as_of_date=date.fromisoformat(args.as_of))
            rows.append({"pair_id": pair["pair_id"], "label": pair["label"], "cv_id": pair["cv_id"],
                         "jd_id": pair["jd_id"], "result": result.model_dump(mode="json")})
            print(f"Score {result.overall_score}, grade {result.grade}, coverage {result.coverage}", file=sys.stderr, flush=True)
        except _BATCH_ERRORS as exc:
            failures.append({"pair_id": pair["pair_id"], "label": pair["label"],
                             "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            print(f"Failed: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr, flush=True)
        write_json(args.output / "smoke.json", {"identity": _identity(args), "model_metadata": metadata,
                   "kind": "development_smoke_not_accuracy_benchmark", "selection": "shortest_pairs_per_label",
                   "attempted": index, "selected": len(selected), "successful": len(rows), "failures": failures, "rows": rows})
    return {"selected": len(selected), "successful": len(rows), "failures": failures,
            "output": str(args.output / "smoke.json"), "interpretation": "Smoke test only; no accuracy claim"}


def score(args):
    _, pairs = load_pairs(args.prepared)
    calibration = _calibration(args) if args.split == "test" else None
    config = ScorerConfig.model_validate(calibration["config"]) if calibration else ScorerConfig()
    cache, rows, failures = _cache(args), [], []
    memo = {}
    selected = [p for p in pairs if p["split"] == args.split]
    for pair in selected:
        try:
            objects = []
            for kind, key, field in [("resume", "cv_id", "resume_text"), ("job", "jd_id", "job_description_text")]:
                identity = (kind, pair[key])
                if identity not in memo:
                    try:
                        memo[identity] = cache.get(pair[field], kind)
                    except _BATCH_ERRORS as exc:
                        memo[identity] = exc
                if isinstance(memo[identity], Exception):
                    raise memo[identity]
                objects.append(memo[identity])
            result = score_pair(*objects, config=config, as_of_date=date.fromisoformat(args.as_of))
            # Baseline: canonical required/preferred skill coverage, no other dimensions.
            baseline = result.scores.skills if result.dimensions["skills"].supported else 0
            rows.append({"pair_id": pair["pair_id"], "cv_id": pair["cv_id"], "jd_id": pair["jd_id"], "split": args.split,
                             "label": pair["label"], "label_id": LABELS[pair["label"]], "baseline_skill_score": baseline,
                             "result": result.model_dump(mode="json")})
        except _BATCH_ERRORS as exc:
            failures.append({"pair_id": pair["pair_id"], "label": pair["label"], "error_type": type(exc).__name__, "error": str(exc)[:500]})
    payload = {"identity": _identity(args), "split": args.split, "attempted": len(selected), "successful": len(rows),
                   "failure_rate": len(failures) / len(selected) if selected else 0,
                   "failures": failures, "rows": rows}
    write_json(args.output / (args.split + "_scores.json"), payload)
    return {k: v for k, v in payload.items() if k != "rows"}


def tune(args):
    manifest, _ = load_pairs(args.prepared)
    if not manifest["ready_for_calibration"]:
        raise ValueError("Prepared development and test partitions must contain all three labels")
    scored = _read(args.scores)
    if scored["split"] != "development" or scored["identity"]["dataset_digest"] != manifest_digest(args.prepared):
        raise ValueError("Calibration requires scores from this development manifest")
    if scored["identity"]["code_digest"] != code_digest():
        raise ValueError("Scoring code changed; regenerate development scores")
    rows = scored["rows"]
    config, baseline = calibrate(rows), calibrate(rows, baseline=True)
    majority = min(range(3), key=lambda k: (-Counter(r["label_id"] for r in rows)[k], k))
    payload = {"identity": scored["identity"], "config": config.model_dump(), "baseline_config": baseline.model_dump(),
                   "majority_label": majority, "development_n": len(rows), "failures": scored["failures"],
                   "objective": "macro_f1", "tie_break": "closest_to_50_75_then_lower_cutoffs"}
    write_json(args.output, payload)
    return payload


def report(args):
    calibration, scored = _read(args.calibration), _read(args.scores)
    if calibration["identity"] != scored["identity"] or scored["identity"]["code_digest"] != code_digest():
        raise ValueError("Test and calibration identities must match current scoring code")
    result = evaluate(scored["rows"], ScorerConfig.model_validate(calibration["config"]),
                      ScorerConfig.model_validate(calibration["baseline_config"]), calibration["majority_label"],
                      args.bootstrap)
    result.update(identity=scored["identity"], attempted=scored["attempted"], failures=scored["failures"],
                  failure_rate=scored["failure_rate"],
                  interpretation="Agreement with dataset labels; no hiring-outcome or individual-subscore validation")
    write_json(args.output / "report.json", result)
    metrics = result["scorer"]
    markdown = (f"# Held-out ATS evaluation\n\nSuccessful pairs: {metrics['n']} / {scored['attempted']}\n\n"
                f"Macro-F1: {metrics['macro_f1']:.4f}; balanced accuracy: {metrics['balanced_accuracy']:.4f}.\n\n"
                f"Skill baseline macro-F1: {result['skill_baseline']['macro_f1']:.4f}; "
                f"majority baseline: {result['majority_baseline']['macro_f1']:.4f}.\n\n"
                "Results measure agreement with dataset labels. Label means are descriptive; "
                "they are not numerical ground truth. Full metrics, overlap, coverage, "
                "CV bootstrap intervals and representative errors are in report.json.\n")
    (args.output / "report.md").write_text(markdown, encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dl = commands.add_parser("download", help="Download the pinned public CSV revision")
    dl.add_argument("--output", type=Path, default=Path("output/ats/raw"))
    prep = commands.add_parser("prepare", help="Audit CSVs and freeze disjoint partitions")
    prep.add_argument("--csv", type=Path, nargs="+", required=True)
    prep.add_argument("--output", type=Path, default=Path("output/ats/prepared"))
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--revision", default="local-input")
    for name in ("extract", "score", "smoke"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, default=Path("output/ats/prepared"))
        command.add_argument("--output", type=Path, default=Path("output/ats/results"))
        command.add_argument("--cache", type=Path, default=Path("output/ats/cache"))
        command.add_argument("--split", choices=["development", "test"], required=True)
        command.add_argument("--backend", choices=["ollama", "vllm", "gemini"], required=True)
        command.add_argument("--endpoint", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--max-chars", type=int, default=40000)
        command.add_argument("--max-tokens", type=int, default=8192)
        command.add_argument("--context-window", type=int, default=32768, help="Ollama num_ctx; serving model must support it")
        if name == "extract":
            command.add_argument("--limit", type=int)
        elif name == "score":
            command.add_argument("--as-of", required=True, help="Fixed ISO date for ongoing jobs")
            command.add_argument("--calibration", type=Path)
        else:
            command.add_argument("--as-of", required=True)
            command.add_argument("--pairs-per-label", type=int, default=1)
    tune_cmd = commands.add_parser("calibrate")
    tune_cmd.add_argument("--prepared", type=Path, default=Path("output/ats/prepared"))
    tune_cmd.add_argument("--scores", type=Path, default=Path("output/ats/results/development_scores.json"))
    tune_cmd.add_argument("--output", type=Path, default=Path("output/ats/calibration.json"))
    rep = commands.add_parser("report")
    rep.add_argument("--scores", type=Path, default=Path("output/ats/results/test_scores.json"))
    rep.add_argument("--calibration", type=Path, default=Path("output/ats/calibration.json"))
    rep.add_argument("--output", type=Path, default=Path("output/ats/results"))
    rep.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    if args.command == "score" and args.split == "test" and args.calibration is None:
        parser.error("Test scoring requires --calibration")
    if args.command == "prepare":
        result = prepare_dataset(args.csv, args.output, args.seed, args.revision)
    else:
        result = {"download": download, "extract": extract, "score": score, "smoke": smoke, "calibrate": tune, "report": report}[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
