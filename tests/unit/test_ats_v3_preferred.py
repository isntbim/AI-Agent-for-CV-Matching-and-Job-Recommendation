"""Preferred reporting, scoring fallback, historical arithmetic and human review."""
from copy import deepcopy
from datetime import date
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.ats_v3_correct_jd012_preferred import apply_decision, prepare_decision
from scripts.ats_v3_resolve_jd012 import OPTIONS, RETIRED_ID, SOURCE_SENTENCE, TARGET_ID
from src.scoring import JobInput, ResumeInput, ScorerConfig, score_pair
from src.scoring.models import runtime_grade
from src.scoring.skills_v3 import score_skills_v3
from src.scoring.v2_corpus import digest
from src.scoring.v3_corpus import VERSION, items, reconcile_document, refresh


def pair(requirements, candidates):
    cv = ResumeInput.model_validate({"document": {"schema_version": "resume-v3", "skills": [
        {"name": "Skills", "keywords": list(candidates), "details": [
            {"name": name, "skill_quote": name, "proficiency": level,
             "raw_proficiency": level if level != "normal" else None,
             "proficiency_quote": level + " " + name if level != "normal" else None}
            for name, level in candidates.items()]}]},
        "facts": {"evidence": {"skills": [{"quote": ", ".join(candidates)}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    units = []
    for index, (options, importance) in enumerate(requirements):
        quote = " or ".join(options) + (" is required" if importance == "core" else " experience")
        units.append({"id": str(index), "options": options, "importance": importance,
            "quote": quote, "importance_quote": quote if importance == "core" else None,
            "review_status": "approved"})
    jd = JobInput.model_validate({"document": {"schema_version": "job-v3", "id": "jd",
        "url": "urn:jd", "scraped_at": "none", "title": "Engineer", "title_normalized": "engineer",
        "category": "engineer", "skills": {"requirements": units}},
        "facts": {"evidence": {"skills": [{"quote": "requirements"}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    return cv, jd


def config(policy="full", preferred="separate", **kwargs):
    return ScorerConfig(version="ats-en-v3", depth_policy=policy, preferred_policy=preferred, **kwargs)


@pytest.mark.parametrize("depth", ["off", "score_only", "full"])
@pytest.mark.parametrize("importance", ["core", "standard"])
def test_preferred_cannot_supply_score_bonus_or_core_gate(depth, importance):
    cv, jd = pair([(["Python"], importance), (["Go"], "preferred")], {"Go": "expert"})
    result = score_skills_v3(cv, jd, config(depth))
    assert result["supported"] and result["score"] == 0 and result["base_skills"] == 0
    assert result["bonus"] == 0 and result["preferred_skill_coverage"] == 100
    assert result["group_shares"]["preferred"] == 0
    assert result["matched_skills"] == [] and result["preferred_matched_skills"] == ["go"]
    pref = next(row for row in result["requirements"] if row["importance"] == "preferred")
    assert not pref["bonus_eligible"] and not pref["scoring_target"]
    assert result["core_gate_fulfillment"] == (0 if importance == "core" else None)


@pytest.mark.parametrize("depth", ["off", "score_only", "full"])
@pytest.mark.parametrize("level", ["normal", "advanced", "expert"])
def test_preferred_cannot_dilute_required_score(depth, level):
    cv, jd = pair([(["Python"], "standard"), (["Go"], "preferred")], {"Python": level})
    baseline = deepcopy(jd)
    baseline.document.skills.requirements.pop()
    a = score_skills_v3(cv, baseline, config(depth))
    b = score_skills_v3(cv, jd, config(depth))
    for key in ("score", "base_skills", "bonus", "gate_status", "core_gate_fulfillment", "strong_blocked", "force_mismatch"):
        assert a[key] == b[key]
    assert b["preferred_skill_coverage"] == 0 and b["missing_skills"] == []
    assert b["preferred_missing_skills"] == ["go"]


@pytest.mark.parametrize("depth", ["off", "score_only", "full"])
def test_preferred_only_fallback_uses_base_targets_without_bonus_or_gate(depth):
    cv, jd = pair([(["Python"], "preferred"), (["Go"], "preferred")], {"Python": "expert"})
    result = score_skills_v3(cv, jd, config(depth))
    assert result["preferred_only_fallback"] and result["score"] == 50
    assert result["base_skills"] == 50 and result["bonus"] == 0
    assert result["preferred_skill_coverage"] == 50 and result["group_shares"]["preferred"] == 1
    assert result["core_gate_fulfillment"] is None and result["gate_status"] == "not_applicable"
    assert not result["strong_blocked"] and not result["force_mismatch"]
    assert all(row["scoring_target"] and not row["bonus_eligible"] for row in result["requirements"])


def test_optional_tie_break_cannot_change_required_bonus_and_each_cv_used_once():
    from src.parser.skill_v2 import SkillRelation
    cv, jd = pair([(["A", "B"], "standard"), (["C"], "preferred")], {"A": "advanced", "B": "normal"})
    relation = SkillRelation(id="a-to-c", candidate="A", requirement="C", relation="near",
        reason="test", source="test", review_status="approved")
    cfg = config(relations=[relation])
    only_required = deepcopy(jd)
    only_required.document.skills.requirements.pop()
    baseline = score_skills_v3(cv, only_required, cfg)
    result = score_skills_v3(cv, jd, cfg)
    assert result["requirements"][0]["candidate_skill"] == baseline["requirements"][0]["candidate_skill"]
    assert result["bonus"] == baseline["bonus"] == 2.5
    assert result["preferred_skill_coverage"] == 0
    legacy = score_skills_v3(cv, jd, config(preferred="legacy_share", relations=[relation]))
    assert legacy["bonus"] == 0 and legacy["preferred_skill_coverage"] == pytest.approx(70)
    assigned = [r["candidate_skill"] for r in result["requirements"] if r["candidate_skill"]]
    assert len(assigned) == len(set(assigned))
    assert score_skills_v3(cv, jd, cfg) == result


@pytest.mark.parametrize("depth", ["off", "score_only", "full"])
def test_legacy_share_retains_historical_70_20_10_numbers(depth):
    cv, jd = pair([(["Python"], "core"), (["Java"], "standard"), (["Go"], "preferred")],
                  {"Python": "normal", "Java": "normal"})
    legacy = score_skills_v3(cv, jd, config(depth, "legacy_share"))
    separate = score_skills_v3(cv, jd, config(depth))
    assert legacy["score"] == 90 and legacy["group_shares"] == pytest.approx({"core": .7, "standard": .2, "preferred": .1})
    assert separate["score"] == 100 and separate["group_shares"]["preferred"] == 0


@pytest.mark.parametrize("version", ["ats-en-v1", "ats-en-v2"])
def test_v1_v2_config_bytes_match_preserved_pre_change_code(version):
    reference = Path(__file__).resolve().parents[2] / "output/scorer_v3/preferred_policy_2026-10-11/reference_code"
    script = "from src.scoring.models import ScorerConfig; print(ScorerConfig(version='" + version + "').model_dump_json())"
    old = subprocess.run([sys.executable, "-X", "utf8", "-c", script], cwd=reference,
                         check=True, capture_output=True, text=True).stdout.strip()
    assert ScorerConfig(version=version).model_dump_json() == old
    with pytest.raises(ValueError, match="Preferred policy requires ats-en-v3"):
        ScorerConfig(version=version, preferred_policy="legacy_share")


def test_v3_policy_defaults_identities_and_runtime_grade_agree():
    from src.scoring.evaluation import classification_metrics, evaluate
    assert ScorerConfig().version == "ats-en-v1"
    assert ScorerConfig(version="ats-en-v3").preferred_policy == "separate"
    cv, jd = pair([(["Python"], "core"), (["Go"], "preferred")], {"Go": "expert"})
    results = [score_pair(cv, jd, config=config(preferred=p), as_of_date=date(2026, 9, 30))
               for p in ("separate", "legacy_share")]
    for result in results:
        assert result.grade == runtime_grade(result.overall_score, result.scores.skills,
            result.scores.title, result.config, result.skills_breakdown)
        rows = [{"pair_id": str(label), "cv_id": str(label), "split": "test", "label_id": label,
                 "baseline_skill_score": 50, "result": result.model_dump(mode="json")} for label in range(3)]
        report = evaluate(rows, result.config, ScorerConfig(), 0, bootstrap=0)
        assert report["scorer"] == classification_metrics([0, 1, 2], [result.grade] * 3)
    assert results[0].skills_breakdown["identity"]["config_sha256"] != results[1].skills_breakdown["identity"]["config_sha256"]


def review_fixture(tmp_path):
    text = "Nice to Have\n\n" + SOURCE_SENTENCE + "\nPython is required."
    path = tmp_path / "source.txt"
    path.write_text(text, encoding="utf-8")
    source = {"source_path": "source.txt", "text_path": "source.txt",
              "source_sha256": digest(path.read_bytes()), "text_sha256": digest(text.encode())}
    cv, jd = pair([(OPTIONS, "standard"), (["Python"], "core")], {"Python": "normal"})
    jd.document.id = "jd_012"
    unit = jd.document.skills.requirements[0]
    unit.id, unit.quote = TARGET_ID, SOURCE_SENTENCE
    jd.annotation.source_sha256, jd.annotation.text_sha256 = source["source_sha256"], source["text_sha256"]
    jd.annotation.source_path = "source.txt"
    doc = {"id": "jd_012", "kind": "job", "source": source, "envelope": jd.model_dump(mode="json"),
           "skills_status": "complete", "full_envelope_review_status": "pending", "issues": []}
    reconcile_document(doc, text)
    historical = {"rule_id": "H3-JD012-OR", "item_id": TARGET_ID, "after": {"importance": "standard"}}
    corpus = {"version": VERSION, "documents": [doc], "relations": {"version": "empty", "relations": []},
              "changes": [historical], "review_history": []}
    refresh(corpus)
    return corpus


def test_jd012_preferred_correction_is_heading_bound_and_history_preserving(tmp_path):
    corpus = review_fixture(tmp_path)
    history = deepcopy(corpus["changes"])
    payload = prepare_decision(corpus, tmp_path)
    assert payload["items"][0]["source_quote"].startswith("Nice to Have")
    result = apply_decision(corpus, payload, tmp_path)
    assert result["applied"] == 1
    unit = items(corpus)[TARGET_ID][3]
    assert unit["importance"] == "preferred" and unit["options"] == OPTIONS
    assert unit["importance_quote"].startswith("Nice to Have") and unit["review_status"] == "approved"
    assert RETIRED_ID not in items(corpus) and corpus["changes"][:len(history)] == history
    assert corpus["changes"][-1]["review_kind"] == "direct_human"
    assert corpus["changes"][-1]["date"] == "2026-10-11"
    assert corpus["documents"][0]["skills_status"] == "complete"
    assert corpus["documents"][0]["full_envelope_review_status"] == "pending"
    after = deepcopy(corpus)
    assert apply_decision(corpus, payload, tmp_path)["idempotent"] and corpus == after


@pytest.mark.parametrize("field", ["item_sha256", "source_sha256", "text_sha256"])
def test_jd012_preferred_stale_binding_fails_atomically(tmp_path, field):
    corpus = review_fixture(tmp_path)
    payload = prepare_decision(corpus, tmp_path)
    payload["items"][0][field] = "stale"
    before = deepcopy(corpus)
    with pytest.raises(ValueError, match="Stale|stale"):
        apply_decision(corpus, payload, tmp_path)
    assert corpus == before


def test_jd012_preferred_changed_source_fails_atomically(tmp_path):
    corpus = review_fixture(tmp_path)
    payload = prepare_decision(corpus, tmp_path)
    before = deepcopy(corpus)
    (tmp_path / "source.txt").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="Source text changed"):
        apply_decision(corpus, payload, tmp_path)
    assert corpus == before


def test_jd012_preferred_cannot_drop_heading_binding(tmp_path):
    corpus = review_fixture(tmp_path)
    payload = prepare_decision(corpus, tmp_path)
    payload["items"][0]["replacement"]["importance_quote"] = None
    before = deepcopy(corpus)
    with pytest.raises(ValueError, match="Nice to Have"):
        apply_decision(corpus, payload, tmp_path)
    assert corpus == before
