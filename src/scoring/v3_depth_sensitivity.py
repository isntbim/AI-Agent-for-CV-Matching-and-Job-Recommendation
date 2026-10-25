"""Prespecified depth-step experiment; runtime defaults remain unchanged."""
from contextlib import contextmanager
from datetime import date
import hashlib
import json
from pathlib import Path
import platform

from . import skills_v3
from .models import ScorerConfig
from .runtime_v3 import score_pair_v3
from .v2_benchmark import canonical_hash, file_identity, load_relations, read_json, score_stored_pair
from .v3_corpus import validate_identity

VERSION = "depth-step-sensitivity-v3-v1"
STEPS = (0.15, 0.25, 0.35)
_active = False


def depth_credit(gap, step):
    return max(0.0, 1.0 - step * gap)


@contextmanager
def parameter_scope(step):
    """Sequential harness only; restore the original function even on failure."""
    global _active
    if _active or step not in STEPS:
        raise ValueError("Use a prespecified step in a non-nested experiment")
    original = skills_v3._depth

    def depth(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["depth_gap"] is not None:
            result["depth_credit"] = depth_credit(result["depth_gap"], step)
        return result

    _active = True
    skills_v3._depth = depth
    try:
        yield
    finally:
        skills_v3._depth = original
        _active = False


def compare(before, after):
    a, b = {r["pair_id"]: r for r in before}, {r["pair_id"]: r for r in after}
    if a.keys() != b.keys() or any(r["status"] != "ok" for r in [*a.values(), *b.values()]):
        raise ValueError("Sensitivity cannot omit changed or failed pairs")
    flips = [i for i in a if a[i]["result"]["grade"] != b[i]["result"]["grade"]]
    return {"pairs": len(a), "grade_flips": len(flips), "grade_flip_ids": flips,
        "skills_changes": sum(a[i]["result"]["unrounded_scores"]["skills"] != b[i]["result"]["unrounded_scores"]["skills"] for i in a),
        "overall_changes": sum(a[i]["result"]["unrounded_overall_score"] != b[i]["result"]["unrounded_overall_score"] for i in a)}


def spreads(variants):
    maps = [{r["pair_id"]: r for r in rows} for rows in variants.values()]
    records = []
    if any(set(m) != set(maps[0]) for m in maps):
        raise ValueError("Spread pair identities differ")
    for i in sorted(maps[0]):
        values = [m[i]["result"] for m in maps]
        skills = [v["unrounded_scores"]["skills"] for v in values]
        totals = [v["unrounded_overall_score"] for v in values]
        records.append({"pair_id": i, "skills_spread": max(skills)-min(skills),
            "overall_spread": max(totals)-min(totals),
            "grade_unstable": len({v["grade"] for v in values}) > 1})
    return {"pairs": records, "max_skills_spread": max(r["skills_spread"] for r in records),
        "max_overall_spread": max(r["overall_spread"] for r in records),
        "grade_unstable_pairs": sum(r["grade_unstable"] for r in records)}


def controlled_rows():
    from itertools import product
    from tests.unit.test_ats_v3_contrast_set import Candidate, Fixture, build_fixture, requirement
    rows = []
    for candidate, minimum in product(("normal", "intermediate", "advanced", "expert"),
                                      ("intermediate", "advanced", "expert")):
        cv, jd, config = build_fixture(Fixture((requirement(minimum=minimum),),
                                              (Candidate("Python", candidate),), {}))
        result = score_pair_v3(cv, jd, config=config, as_of_date=date(2026, 9, 30))
        rows.append({"pair_id": candidate+"::"+minimum, "status": "ok", "result": result.model_dump(mode="json")})
    return rows


def run_sweep(root: Path, *, as_of: date, corpus_path: Path, reference_path: Path, output: Path):
    if output.exists():
        raise ValueError("Sensitivity requires a fresh output directory")
    corpus = read_json(corpus_path)
    validate_identity(corpus)
    reference = read_json(reference_path/"v3_full.json")
    if corpus["as_of_date"] != as_of.isoformat() or reference["identity"]["as_of_date"] != as_of.isoformat():
        raise ValueError("Evaluation dates differ")
    pairs_path = root/"data/datasets/matching_pairs/benchmark_matching_pairs.json"
    pairs = read_json(pairs_path)
    if len(pairs) != 1672 or sum(p["pair_type"] != "heuristic_negative" for p in pairs) != 1144:
        raise ValueError("Fixed silver population changed")
    for key, path in (("corpus", corpus_path), ("pairs", pairs_path)):
        if file_identity(path, root)["sha256"] != reference["identity"]["inputs"][key]["sha256"]:
            raise ValueError("Reference input changed: "+key)
    documents = {d["id"]: d for d in corpus["documents"]}
    table = load_relations(None, corpus)
    config = ScorerConfig(version="ats-en-v3", depth_policy="full", preferred_policy="separate",
        relations=[r for r in table.relations if r.review_status == "approved"], relation_version=table.version)
    if config.model_dump(mode="json") != reference["identity"]["configs"]["v3_full"]:
        raise ValueError("Reference scoring configuration differs")
    family = {"version": VERSION, "steps": STEPS, "floor": 0, "as_of_date": as_of.isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "reference_identity": reference["identity"]["family_sha256"],
        "corpus": file_identity(corpus_path, root), "labels": file_identity(pairs_path, root),
        "relations_sha256": canonical_hash(table.model_dump(mode="json")),
        "config": config.model_dump(mode="json"), "code": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted({*root.glob("src/scoring/*.py"), root/"scripts/ats_v3_depth_sensitivity.py"})}}
    family["family_sha256"] = canonical_hash(family)
    output.mkdir(parents=True)
    write = lambda p,v: p.write_text(json.dumps(v,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    write(output/"experiment_identity.json",family)
    original = skills_v3._depth
    variants, controlled = {}, {}
    for step in STEPS:
        name = str(step)
        with parameter_scope(step):
            rows = [score_stored_pair(p,name,{},documents,config,as_of,scorer=score_pair_v3) for p in pairs]
            probes = controlled_rows()
        for row in rows+probes:
            if row["status"] == "ok":
                row["result"]["skills_breakdown"]["identity"]["depth_step_experiment"] = {
                    "version": VERSION, "step": step, "floor": 0, "family_sha256": family["family_sha256"]}
        variants[name],controlled[name] = rows,probes
        write(output/("step_"+name+".json"),{"identity":family,"rows":rows})
    if skills_v3._depth is not original:
        raise AssertionError("Depth function was not restored")
    old = {r["pair_id"]:r for r in reference["rows"]}
    for row in variants["0.25"]:
        if row["status"] != "ok":
            raise ValueError("Default sensitivity cell failed")
        result = json.loads(json.dumps(row["result"]))
        result["skills_breakdown"]["identity"].pop("depth_step_experiment")
        if result != old[row["pair_id"]]["result"]:
            raise ValueError("Default sensitivity cell drift: "+row["pair_id"])
    report = {"identity":family,"status":"passed","defaults_restored":True,
        "population":{"all":1672,"gemini":1144,"heuristic":528},"variants":{},
        "silver_spread":spreads(variants),"controlled_spread":spreads(controlled),
        "controlled_scope":"12 constructed explicit-depth probes; not predictive-effectiveness data",
        "limitations":["The three-level ladder has at most two gaps; the zero floor does not bind at these steps.",
                       "No step is selected by agreement with previously inspected silver labels."]}
    for name,rows in variants.items():
        report["variants"][name] = {"all":compare(variants["0.25"],rows),
            "gemini":compare([r for r in variants["0.25"] if r["pair_type"]!="heuristic_negative"],
                              [r for r in rows if r["pair_type"]!="heuristic_negative"]),
            "heuristic":compare([r for r in variants["0.25"] if r["pair_type"]=="heuristic_negative"],
                                 [r for r in rows if r["pair_type"]=="heuristic_negative"]),
            "controlled":compare(controlled["0.25"],controlled[name])}
    write(output/"controlled_rows.json",controlled)
    write(output/"report.json",report)
    lines=["# Prespecified depth-step sensitivity","","Default 0.25; alternatives 0.15/0.35, zero floor. Other parameters and labels are fixed.",
        "No selection by silver-label agreement.","","| Step | Silver flips | Constructed flips |","|---|---:|---:|"]
    for name,v in report["variants"].items():
        lines.append(f"| {name} | {v['all']['grade_flips']} | {v['controlled']['grade_flips']} |")
    lines += ["",f"Silver maximum raw skills/overall spread: {report['silver_spread']['max_skills_spread']} / {report['silver_spread']['max_overall_spread']}.",
        f"Constructed maximum raw skills/overall spread: {report['controlled_spread']['max_skills_spread']} / {report['controlled_spread']['max_overall_spread']}.",
        "Constructed probes verify policy sensitivity, not effectiveness. Defaults restored."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report

