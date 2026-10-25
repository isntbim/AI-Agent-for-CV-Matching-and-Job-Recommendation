"""Import the follow-up human Preferred importance decision for JD 012."""
from copy import deepcopy
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ats_v3_resolve_jd012 import OPTIONS, RETIRED_ID, SOURCE_SENTENCE, TARGET_ID
from src.scoring.v2_corpus import atomic_json, digest, read_json
from src.scoring.v3_corpus import (REVIEW_VERSION, _literal, import_review, items,
                                   reconcile_document, refresh, validate_identity)

REVIEWED_ON = "2026-10-11"
APPROVAL_SOURCE = "direct user Preferred correction relayed by Claude Code: 'yes send it to codex'"
REASON = (
    "The prior Standard relay was an error caught by checking the Nice to Have heading. "
    "The user now approves Preferred importance for the existing four-way OR requirement; "
    "any one alternative satisfies it. Supersedes H3-JD012-OR importance only, retains "
    "that decision and the retired duplicate as history. Direct human review, not a derived rule."
)


def prepare_decision(corpus, root):
    validate_identity(corpus)
    document, _, _, before = items(corpus)[TARGET_ID]
    if document["id"] != "jd_012" or before["options"] != OPTIONS or before["importance"] != "standard":
        raise ValueError("Expected the approved Standard JD 012 OR merge before correction")
    if RETIRED_ID in items(corpus):
        raise ValueError("The original structural OR merge must be completed first")
    text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
    quote = _literal(text, SOURCE_SENTENCE)
    end = text.index(quote) + len(quote)
    start = text.rfind("Nice to Have", 0, text.index(quote))
    if start < 0 or " ".join(before["quote"].split()) != " ".join(quote.split()):
        raise ValueError("The approved sentence must be under a literal Nice to Have heading")
    evidence = text[start:end]
    replacement = deepcopy(before)
    replacement.update(importance="preferred", importance_quote=evidence, issues=[])
    decision = {"item_id": TARGET_ID, "item_sha256": digest(before), "decision": "edit",
        "replacement": replacement, "source_quote": evidence,
        "source_sha256": document["source"]["source_sha256"],
        "text_sha256": document["source"]["text_sha256"], "reviewer": APPROVAL_SOURCE,
        "reason": REASON, "rule_id": "H3-JD012-PREFERRED", "review_kind": "direct_human"}
    return {"version": REVIEW_VERSION, "reviewed_on": REVIEWED_ON,
        "corpus_sha256": corpus["identity_sha256"], "items": [decision],
        "supersedes": {"rule_id": "H3-JD012-OR", "field": "importance", "before": "standard", "after": "preferred"}}


def apply_decision(corpus, payload, root):
    if len(payload.get("items", [])) != 1 or payload["items"][0]["item_id"] != TARGET_ID:
        raise ValueError("Expected specifically the JD 012 Preferred correction")
    decision = payload["items"][0]
    if decision.get("replacement", {}).get("importance") != "preferred":
        raise ValueError("Expected Preferred importance")
    if (decision.get("rule_id") != "H3-JD012-PREFERRED"
            or not decision.get("source_quote", "").startswith("Nice to Have")
            or decision.get("replacement", {}).get("importance_quote") != decision.get("source_quote")):
        raise ValueError("Expected the approved Nice to Have heading evidence")
    trial = deepcopy(corpus)
    result = import_review(trial, payload, root)
    document, _, _, after = items(trial)[TARGET_ID]
    if RETIRED_ID in items(trial) or after["options"] != OPTIONS or after["importance"] != "preferred":
        raise ValueError("Recorded correction does not match the approved four-way Preferred requirement")
    if result["idempotent"]:
        return result
    text = (root / document["source"]["text_path"]).read_text(encoding="utf-8")
    reconcile_document(document, text)
    refresh(trial)
    validate_identity(trial)
    corpus.clear()
    corpus.update(trial)
    return result


def main():
    directory = ROOT / "output/scorer_v3/preferred_policy_2026-10-11"
    directory.mkdir(parents=True, exist_ok=True)
    path = ROOT / "output/scorer_v3/corpus.json"
    corpus = read_json(path)
    payload_path = directory / "jd012_preferred_decision.json"
    payload = read_json(payload_path) if payload_path.exists() else prepare_decision(corpus, ROOT)
    atomic_json(payload_path, payload)
    first = apply_decision(corpus, payload, ROOT)
    after = deepcopy(corpus)
    replay = apply_decision(corpus, payload, ROOT)
    assert replay["idempotent"] and corpus == after
    atomic_json(path, corpus)
    atomic_json(ROOT / "human-verify/scorer-v3/jd012_preferred_approved.json", payload)
    document = next(d for d in corpus["documents"] if d["id"] == "jd_012")
    receipt = {"reviewed_on": REVIEWED_ON, "approval_source": APPROVAL_SOURCE,
        "first_import": first, "replay": replay, "corpus_sha256": corpus["identity_sha256"],
        "jd_012_skills_status": document["skills_status"], "jd_012_issues": document["issues"],
        "v3_validation_issues": document["v3_validation_issues"],
        "full_envelope_review_status": document["full_envelope_review_status"]}
    if not (directory / "jd012_preferred_import_receipt.json").exists():
        atomic_json(directory / "jd012_preferred_import_receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
