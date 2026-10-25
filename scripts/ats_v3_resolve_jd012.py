"""Import the explicit human JD 012 list decision and retire its duplicate unit."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.scoring.skills_v3 import GROUP_WEIGHTS
from src.scoring.v2_corpus import atomic_json, digest, read_json
from src.scoring.v3_corpus import (REVIEW_VERSION, _literal, import_review, items,
                                   reconcile_document, refresh, validate_identity)

REVIEWED_ON = "2026-10-11"
TARGET_ID = "jd_012:req:538345687a95"
RETIRED_ID = "jd_012:req:4ca097c8360e"
SOURCE_SENTENCE = (
    "Experience with SaaS-specific problems (fair usage, noisy neighbor), large-scale "
    "databases (sharding, partitioning, no-downtime migration), Martech, or Fintech domains."
)
OPTIONS = ["saas-specific problems", "large-scale databases", "martech", "fintech"]
APPROVAL_SOURCE = "user 'agree' decision relayed by Claude Code in this session"
REASON = (
    "The user approves this specific trailing-'or' list as one Standard requirement: "
    "any one of SaaS-specific problems, large-scale databases, Martech, or Fintech. "
    "Retire the duplicate Martech requirement; retain its item history. "
    "This is direct human review, not a corpus-wide punctuation rule."
)


def prepare_decision(corpus, root):
    validate_identity(corpus)
    lookup = items(corpus)
    document, _, _, target = lookup[TARGET_ID]
    retired_document, _, _, retired = lookup[RETIRED_ID]
    if document is not retired_document or document["id"] != "jd_012":
        raise ValueError("JD 012 merge items must belong to the same document")
    text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
    quote = _literal(text, SOURCE_SENTENCE)
    if any(" ".join(unit["quote"].split()) != " ".join(quote.split()) for unit in (target, retired)):
        raise ValueError("JD 012 items no longer describe the approved sentence")
    if target["options"] != ["databases"] or retired["options"] != ["martech"]:
        raise ValueError("JD 012 pre-approval options changed; review the new items")
    merged = deepcopy(target)
    merged.update(options=OPTIONS.copy(), importance="standard", importance_quote=None,
                  quote=quote, issues=[], option_details=[
                      {"name": name, "minimum_proficiency": "normal", "raw_proficiency": None,
                       "proficiency_quote": None, "experience": []} for name in OPTIONS])
    retired_after = deepcopy(retired)
    retired_after.update(quote=quote, issues=[])
    selected = []
    for key, before, after in ((TARGET_ID, target, merged), (RETIRED_ID, retired, retired_after)):
        selected.append({"item_id": key, "item_sha256": digest(before), "decision": "edit",
            "replacement": after, "source_quote": quote,
            "source_sha256": document["source"]["source_sha256"],
            "text_sha256": document["source"]["text_sha256"], "reviewer": APPROVAL_SOURCE,
            "reason": REASON, "rule_id": "H3-JD012-OR", "review_kind": "direct_human"})
    return {"version": REVIEW_VERSION, "reviewed_on": REVIEWED_ON,
        "corpus_sha256": corpus["identity_sha256"], "items": selected,
        "merge": {"target_item_id": TARGET_ID, "retired_item_id": RETIRED_ID},
        "importance_weight_note": {"requested_in_relay": 2, "implemented_standard_weight": GROUP_WEIGHTS["standard"],
                                   "policy": "Preserve existing scoring arithmetic; record the discrepancy."}}


def apply_decision(corpus, payload, root):
    """Publish only after both source-bound approvals and structural merge succeed."""
    if payload.get("merge") != {"target_item_id": TARGET_ID, "retired_item_id": RETIRED_ID}:
        raise ValueError("Expected the specifically approved JD 012 merge")
    if {decision["item_id"] for decision in payload["items"]} != {TARGET_ID, RETIRED_ID}:
        raise ValueError("Both original JD 012 items must be checksum-bound")
    trial = deepcopy(corpus)
    result = import_review(trial, payload, root)
    if result["idempotent"]:
        lookup = items(trial)
        if RETIRED_ID in lookup or lookup[TARGET_ID][3]["options"] != OPTIONS:
            raise ValueError("Recorded JD 012 import is inconsistent with its merge")
        return {**result, "merged": 0}
    lookup = items(trial)
    document, _, _, target = lookup[TARGET_ID]
    retired_document, container, index, retired = lookup[RETIRED_ID]
    if document is not retired_document or target["options"] != OPTIONS or target["review_status"] != "approved":
        raise ValueError("JD 012 replacement does not implement the approved alternatives")
    if retired["review_status"] != "approved" or retired["quote"] != target["quote"]:
        raise ValueError("JD 012 duplicate must have a source-bound approval before retirement")
    before_retirement = deepcopy(retired)
    container.pop(index)
    source = document["source"]
    trial["changes"].append({"date": payload["reviewed_on"], "item_id": RETIRED_ID,
        "document_id": document["id"], "review_kind": "direct_human_merge", "rule_id": "H3-JD012-OR",
        "reason": REASON, "approval_source": APPROVAL_SOURCE, "merged_into": TARGET_ID,
        "source_sha256": source["source_sha256"], "text_sha256": source["text_sha256"],
        "source_quote": target["quote"], "before": before_retirement, "after": None,
        "before_sha256": digest(before_retirement), "after_sha256": digest(None),
        "transaction_sha256": result["transaction_sha256"]})
    before_status = document["skills_status"]
    before_issues = deepcopy(document.get("issues", []))
    document["issues"] = [issue for issue in before_issues if not (
        issue.get("code") == "complex_choice_scope" and
        " ".join(issue.get("quote", "").split()) == " ".join(target["quote"].split()))]
    text = (root / source["text_path"]).read_text(encoding="utf-8")
    reconcile_document(document, text)
    trial["changes"].append({"date": payload["reviewed_on"], "document_id": document["id"],
        "review_kind": "requirement_scope_reconciliation", "rule_id": "H3-JD012-OR",
        "reason": "Recompute completeness after the approved merge; retain pending whole-envelope review.",
        "approval_source": APPROVAL_SOURCE, "transaction_sha256": result["transaction_sha256"],
        "before": {"skills_status": before_status, "issues": before_issues},
        "after": {"skills_status": document["skills_status"], "issues": deepcopy(document["issues"]),
                  "v3_validation_issues": deepcopy(document["v3_validation_issues"])}})
    refresh(trial)
    validate_identity(trial)
    corpus.clear()
    corpus.update(trial)
    return {**result, "merged": 1}


def main():
    directory = ROOT / "output/scorer_v3/jd012_decision_2026-10-11"
    directory.mkdir(parents=True, exist_ok=True)
    path = ROOT / "output/scorer_v3/corpus.json"
    corpus = read_json(path)
    for source, target in ((path, "corpus_before.json"),
                          (ROOT / "docs/research/ats_scorer/verification.json", "verification_before.json"),
                          (ROOT / "human-verify/scorer-v3/unresolved.json", "unresolved_before.json")):
        if not (directory / target).exists():
            shutil.copy2(source, directory / target)
    payload_path = directory / "decisions.json"
    payload = read_json(payload_path) if payload_path.exists() else prepare_decision(corpus, ROOT)
    atomic_json(payload_path, payload)
    first = apply_decision(corpus, payload, ROOT)
    after = deepcopy(corpus)
    replay = apply_decision(corpus, payload, ROOT)
    assert replay["idempotent"] and corpus == after
    atomic_json(path, corpus)
    atomic_json(ROOT / "human-verify/scorer-v3/jd012_or_approved.json", payload)
    doc = next(d for d in corpus["documents"] if d["id"] == "jd_012")
    receipt = {"reviewed_on": REVIEWED_ON, "approval_source": APPROVAL_SOURCE, "first_import": first,
        "replay": replay, "corpus_sha256": corpus["identity_sha256"], "jd_012_skills_status": doc["skills_status"],
        "jd_012_validation_issues": doc["v3_validation_issues"], "whole_envelope_review_status": doc["full_envelope_review_status"]}
    if not (directory / "import_receipt.json").exists():
        atomic_json(directory / "import_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
