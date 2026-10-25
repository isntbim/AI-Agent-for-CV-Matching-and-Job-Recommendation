"""Apply the user's sample 2 normal and sample 6 OR decisions with source binding."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring.v2_corpus import atomic_json, digest, read_json
from src.scoring.v3_corpus import REVIEW_VERSION, _literal, items, import_review, refresh, validate_identity

DECISIONS = {
    2: ("cv_002:skill:0:1", "proficient accounting support and software", "Keep sample 2 as normal/unspecified, retain wording, and award no proficiency bonus."),
    6: ("jd_012:req:1060285ea8fa", "Title: Senior Software Engineer (Fullstack, Golang/JavaScript)", "Approve Golang OR JavaScript as one Standard requirement for sample 6; do not infer minimum proficiency or generalize title slashes."),
}


def prepare_decisions(corpus, root):
    validate_identity(corpus)
    lookup = items(corpus)
    selected = []
    for number, (key, excerpt, reason) in DECISIONS.items():
        document, _, _, before = lookup[key]
        text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
        quote = _literal(text, excerpt)
        after = deepcopy(before)
        if number == 2:
            assert after["name"] == "accounting"
            after.update(proficiency="normal", proficiency_quote=None)
        else:
            assert set(after["options"]) == {"golang", "javascript"}
            after.update(importance="standard", importance_quote=None,
                         issues=[i for i in after["issues"] if i != "title_choice_scope_pending"])
        selected.append({"item_id": key, "item_sha256": digest(before), "decision": "edit",
            "replacement": after, "source_quote": quote, "source_sha256": document["source"]["source_sha256"],
            "text_sha256": document["source"]["text_sha256"], "reviewer": f"user:approved-sample-{number}-followup",
            "reason": reason, "rule_id": f"R3-SAMPLE-{number}-RESOLVED", "review_kind": "direct_human"})
    return {"version": REVIEW_VERSION, "corpus_sha256": corpus["identity_sha256"], "items": selected}


def apply_decisions(corpus, payload, root):
    result = import_review(corpus, payload, root)
    pending = corpus.get("pending_samples", [])
    updated = [number for number in pending if number not in DECISIONS]
    if pending != updated:
        corpus["pending_samples"] = updated
        corpus["changes"].append({"date": "2026-10-08", "review_kind": "sample_status_reconciliation",
            "rule_id": "R3-SAMPLES-2-6", "before": {"pending_samples": pending}, "after": {"pending_samples": updated},
            "reason": "User explicitly resolved sample 2 as normal and sample 6 as OR; document and envelope approval remain separate.",
            "approval_source": "user followup", "transaction_sha256": result["transaction_sha256"]})
        refresh(corpus)
    return result


def main():
    directory = ROOT / "output/scorer_v3/sample_decisions_2026-10-08"
    directory.mkdir(exist_ok=True)
    path = ROOT / "output/scorer_v3/corpus.json"
    corpus = read_json(path)
    for source, target in ((path, "corpus_before.json"),
                           (ROOT / "docs/research/ats_scorer/verification.json", "verification_before.json"),
                           (ROOT / "human-verify/scorer-v3/unresolved.json", "unresolved_before.json")):
        if not (directory / target).exists():
            shutil.copy2(source, directory / target)
    payload_path = directory / "decisions.json"
    payload = read_json(payload_path) if payload_path.exists() else prepare_decisions(corpus, ROOT)
    atomic_json(payload_path, payload)
    first = apply_decisions(corpus, payload, ROOT)
    after = deepcopy(corpus)
    replay = apply_decisions(corpus, payload, ROOT)
    assert replay["idempotent"] and corpus == after
    atomic_json(path, corpus)
    atomic_json(ROOT / "human-verify/scorer-v3/samples_2_6_approved.json", payload)
    receipt = {"first_import": first, "replay": replay, "pending_samples": corpus["pending_samples"], "corpus_sha256": corpus["identity_sha256"]}
    if not (directory / "import_receipt.json").exists():
        atomic_json(directory / "import_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
