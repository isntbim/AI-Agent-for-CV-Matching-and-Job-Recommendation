"""Human-checked holdout CVs (Gold CV Check page): collect the reviewers' corrections into gold labels, build model
prompts for the 46 CVs, and score extraction against the corrected labels.

Two reviewers (A, B) each corrected 20 CVs alone and 6 shared ones (one per domain) from a Gemini draft: skill names,
professional domains and completed degrees, the three things scripts/qwen_eval.py scores. The shared CVs measure
reviewer agreement and are scored against each reviewer (52 (CV, reviewer) labels; bootstrap resamples CVs).

    python -m scripts.cv_holdout_eval collect   # checks_db/ (artifact DB export) -> gold_checked_v1.json + agreement
    python -m scripts.cv_holdout_eval rows      # -> holdout_rows.jsonl (same system prompt + user format as the test set)
    python -m scripts.cv_holdout_eval score <repaired_preds.jsonl> [...]   # -> <preds>.holdout_scores.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from scripts.qwen_eval import _as, parse, prf, relaxed_matches, skills, valid

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "output/scorer_v3/teacher_labels/holdout"
DB = H / "checks_db/checks"
GOLD = H / "gold_checked_v1.json"
ROWS = H / "holdout_rows.jsonl"
TEST = ROOT / "data/datasets/qwen_training_v4/cv_extraction/test.jsonl"
REVIEWERS = {"u_KQsTaPcDqxN3JVUtaaldaQ": "A", "u_rnr53qNDRrlz7cZmewQaoQ": "B"}  # opaque artifact ids -> group


def norm_skills(names) -> set[str]:
    return {str(n).casefold().strip() for n in names} - {""}


def label_of(review: dict) -> dict:
    return {"skills": norm_skills(review["skills"]), "domains": set(review["domains"]), "degrees": sorted(review["degrees"])}


def collect() -> None:
    check = json.loads((H / "holdout_check_v1.json").read_text(encoding="utf-8"))
    groups = {c["id"]: c["group"] for c in check["cvs"]}
    gold = {cid: {"group": g, "reviews": []} for cid, g in groups.items()}
    for uid, who in REVIEWERS.items():
        for f in sorted((DB / uid / "cvs").glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            d = d.get("data", d)
            if f.stem in gold and d.get("done"):
                gold[f.stem]["reviews"].append({"reviewer": who, "skills": d["skills"], "domains": d["domains"],
                                                "degrees": d["degrees"], "note": d.get("note", "")})
    missing = [cid for cid, g in gold.items() if not g["reviews"]]
    assert not missing, f"CVs without a finished review: {missing}"
    drafts = {c["id"]: c["draft"] for c in check["cvs"]}
    GOLD.write_text(json.dumps({"version": 1, "cvs": gold, "drafts": drafts}, ensure_ascii=False, indent=1), encoding="utf-8")

    # Agreement on the shared CVs, and how much each reviewer changed the Gemini draft.
    shared = [cid for cid, g in gold.items() if len(g["reviews"]) == 2]
    tp = fa = fb = dom_same = deg_same = 0
    for cid in shared:
        a, b = (label_of(r) for r in gold[cid]["reviews"])
        tp += len(a["skills"] & b["skills"]); fa += len(a["skills"]); fb += len(b["skills"])
        dom_same += a["domains"] == b["domains"]; deg_same += a["degrees"] == b["degrees"]
    agreement = {"shared_cvs": len(shared), "skill_f1_A_vs_B": prf(tp, fa, fb)["f1"],
                 "skills_A": fa, "skills_B": fb, "skills_both": tp,
                 "domains_identical": f"{dom_same}/{len(shared)}", "degrees_identical": f"{deg_same}/{len(shared)}"}
    edits = Counter()
    for cid, g in gold.items():
        draft = norm_skills(drafts[cid].get("skills", []))
        for r in g["reviews"]:
            lab = label_of(r)
            edits.update(draft_skills=len(draft), kept=len(draft & lab["skills"]), removed=len(draft - lab["skills"]),
                         added=len(lab["skills"] - draft), reviews=1)
    agreement["draft_edits"] = dict(edits)
    (H / "gold_checked_v1.agreement.json").write_text(json.dumps(agreement, indent=1), encoding="utf-8")
    print(f"{len(gold)} CVs, {sum(len(g['reviews']) for g in gold.values())} reviews ->", GOLD)
    print(json.dumps(agreement, indent=1))


def rows() -> None:
    system = json.loads(TEST.read_text(encoding="utf-8").splitlines()[0])["messages"][0]["content"]
    manifest = [d for d in json.loads((H / "manifest.json").read_text(encoding="utf-8"))["documents"] if "excluded" not in d]
    with ROWS.open("w", encoding="utf-8") as f:
        for d in manifest:
            user = json.dumps(json.loads((ROOT / d["input_path"]).read_text(encoding="utf-8")), ensure_ascii=False)
            f.write(json.dumps({"id": d["id"], "task": "cv_extraction",
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                               ensure_ascii=False) + "\n")
    print(len(manifest), "rows ->", ROWS)


def counts(pred: dict | None, lab: dict, text: str) -> Counter:
    c = Counter(n=1, gold=len(lab["skills"]), dom_gold=len(lab["domains"]))
    if pred is None:
        return c
    facts = _as(pred.get("facts"), dict)
    ps = skills(pred, "cv_extraction")
    pd_ = {d for d in _as(facts.get("domains"), list) if isinstance(d, str)}
    deg = facts.get("completed_degrees")
    c.update(parsed=1, valid=valid(pred, text, "cv_extraction") in ("pass", "partial"), tp=len(ps & lab["skills"]),
             pred=len(ps), tp_relaxed=relaxed_matches(ps, lab["skills"]), dom_tp=len(pd_ & lab["domains"]),
             dom_pred=len(pd_), deg_ok=(sorted(deg) if isinstance(deg, list) else deg) == lab["degrees"])
    return c


def metrics(c: Counter) -> dict:
    n = c["n"]
    return {"schema_valid_rate": c["valid"] / n, "skills_precision": prf(c["tp"], c["pred"], c["gold"])["precision"],
            "skills_recall": prf(c["tp"], c["pred"], c["gold"])["recall"], "skills_f1": prf(c["tp"], c["pred"], c["gold"])["f1"],
            "skills_relaxed_f1": prf(c["tp_relaxed"], c["pred"], c["gold"])["f1"],
            "domains_f1": prf(c["dom_tp"], c["dom_pred"], c["dom_gold"])["f1"], "completed_degrees_accuracy": c["deg_ok"] / n}


def score_labels(per_cv: dict[str, list[Counter]], boot: int = 1000) -> dict:
    """per_cv: CV id -> one Counter per reviewer label. Bootstrap resamples CVs (clusters), not labels."""
    ids, rng = sorted(per_cv), random.Random(0)
    point = metrics(sum((c for i in ids for c in per_cv[i]), Counter()))
    samples = [metrics(sum((c for i in rng.choices(ids, k=len(ids)) for c in per_cv[i]), Counter())) for _ in range(boot)]
    out = {"cvs": len(ids), "labels": sum(len(v) for v in per_cv.values())}
    for k, v in point.items():
        dist = sorted(s[k] for s in samples)
        out[k] = {"value": round(v, 3), "ci95": [round(dist[int(0.025 * boot)], 3), round(dist[int(0.975 * boot) - 1], 3)]}
    return out


def score(paths: list[Path]) -> None:
    g = json.loads(GOLD.read_text(encoding="utf-8"))
    texts = {json.loads(l)["id"]: json.loads(json.loads(l)["messages"][1]["content"])["source_text"]
             for l in ROWS.read_text(encoding="utf-8").splitlines()}
    # the Gemini draft as a "prediction": how far the teacher draft was from the corrected labels
    draft_cv = {cid: [counts({"document": {"skills": [{"details": [{"name": n} for n in d.get("skills", [])]}]},
                              "facts": {"domains": d.get("domains", []), "completed_degrees": d.get("degrees", [])}},
                             label_of(r), "") for r in g["cvs"][cid]["reviews"]] for cid, d in g["drafts"].items()}
    draft = score_labels(draft_cv)
    draft.pop("schema_valid_rate")
    for path in paths:
        preds = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
        per_cv = {cid: [counts(parse(preds[cid].get("repaired") or preds[cid]["prediction"]), label_of(r), texts[cid])
                        for r in g["cvs"][cid]["reviews"]] for cid in g["cvs"]}
        report = {"model": score_labels(per_cv), "gemini_draft": draft,
                  "seconds_total": round(sum(r.get("seconds", 0) for r in preds.values()), 1)}
        out = path.with_suffix(".holdout_scores.json")
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(path.name, json.dumps({k: report["model"][k]["value"] for k in ("skills_f1", "skills_relaxed_f1", "domains_f1",
                                                                             "completed_degrees_accuracy", "schema_valid_rate")}))
        print("  gemini draft vs reviewers:", {k: draft[k]["value"] for k in ("skills_f1", "domains_f1", "completed_degrees_accuracy")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["collect", "rows", "score"])
    p.add_argument("preds", nargs="*", type=Path)
    a = p.parse_args()
    {"collect": collect, "rows": rows, "score": lambda: score(a.preds)}[a.cmd]()
