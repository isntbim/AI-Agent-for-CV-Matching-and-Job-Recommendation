"""Behavioral checks for the isolated two-layer sensitivity harness."""
from copy import deepcopy
from datetime import date

import pytest

from src.parser.skill_v2 import SkillRelation
from src.scoring import JobInput, ResumeInput, ScorerConfig
from src.scoring import skills_v3
from src.scoring.skills_v3 import score_skills_v3
from src.scoring.v2_benchmark import canonical_hash, require_same_identity
from src.scoring.v3_weight_sensitivity import (
    BASELINE, DEFAULT_SHARES, DEFAULT_WEIGHTS, changed_rows, historical_rows, parameter_config,
    parameter_scope, ranking_changes, run_status, run_sweep, score_spreads, specifications,
)


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
        quote = " or ".join(options) + (" required" if importance == "core" else " experience")
        units.append({"id": str(index), "options": options, "importance": importance,
            "quote": quote, "importance_quote": quote if importance == "core" else None,
            "review_status": "approved"})
    jd = JobInput.model_validate({"document": {"schema_version": "job-v3", "id": "jd",
        "url": "urn:jd", "scraped_at": "none", "title": "Engineer", "title_normalized": "engineer",
        "category": "engineer", "skills": {"requirements": units}},
        "facts": {"evidence": {"skills": [{"quote": "requirements"}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    return cv, jd


def config(**kwargs):
    return ScorerConfig(version="ats-en-v3", variant="full", depth_policy="full",
                        preferred_policy="separate", **kwargs)


def spec(shares="current", weights="current"):
    return next(s for s in specifications() if s["name"] == f"shares_{shares}__weights_{weights}")


def test_prespecified_factorial_and_five_two_completion():
    items = specifications()
    assert len(items) == len({s["name"] for s in items}) == 12
    assert items[0]["name"] == BASELINE
    assert spec(weights="five_two")["raw_item_weights"] == {"core": 5, "standard": 2, "preferred": .4}
    assert spec(weights="five_two")["item_weights"] == {"core": 1, "standard": .4, "preferred": .08}
    assert spec(weights="roc")["item_weights"]["standard"] == pytest.approx(5 / 11)
    assert spec(weights="roc")["item_weights"]["preferred"] == pytest.approx(2 / 11)


@pytest.mark.parametrize("specification", specifications(), ids=lambda s: s["name"])
def test_parameter_scope_restores_original_objects_and_default_configuration(specification):
    weights, shares = skills_v3.GROUP_WEIGHTS, skills_v3.GROUP_SHARES
    original_config = ScorerConfig().model_dump_json()
    with parameter_scope(specification):
        assert skills_v3.GROUP_WEIGHTS == specification["item_weights"]
        assert skills_v3.GROUP_SHARES == specification["group_shares"]
    assert skills_v3.GROUP_WEIGHTS is weights
    assert skills_v3.GROUP_SHARES is shares
    assert weights == DEFAULT_WEIGHTS and shares == DEFAULT_SHARES
    assert ScorerConfig().model_dump_json() == original_config
    assert ScorerConfig().version == "ats-en-v1"
    assert config().preferred_policy == "separate"


def test_exception_and_nesting_cannot_leak_parameters():
    weights, shares = skills_v3.GROUP_WEIGHTS, skills_v3.GROUP_SHARES
    with pytest.raises(RuntimeError, match="deliberate"):
        with parameter_scope(spec("equal", "five_two")):
            with pytest.raises(RuntimeError, match="overlap"):
                with parameter_scope(spec()):
                    pass
            raise RuntimeError("deliberate scoring failure")
    assert skills_v3.GROUP_WEIGHTS is weights and skills_v3.GROUP_SHARES is shares


@pytest.mark.parametrize("value", [0, -1, 5, float("nan"), float("inf")])
def test_unbounded_or_nonfinite_parameters_are_rejected(value):
    changed = spec()
    changed["item_weights"]["core"] = value
    with pytest.raises(ValueError, match="Core-normalized"):
        with parameter_scope(changed):
            pass
    assert skills_v3.GROUP_WEIGHTS == DEFAULT_WEIGHTS


@pytest.mark.parametrize("share,expected", [("current", 7 / 9), ("equal", .5), ("rank_sum", .6), ("roc", 11 / 16)])
def test_share_sweep_retains_full_mode_and_core_gate(share, expected):
    cv, jd = pair([(["Python"], "core"), (["SQL"], "standard"), (["Go"], "preferred")],
                  {"Python": "normal", "Go": "expert"})
    with parameter_scope(spec(share)):
        result = score_skills_v3(cv, jd, config())
    assert result["base_skills"] == pytest.approx(100 * expected)
    assert result["group_shares"]["core"] == pytest.approx(expected)
    assert result["group_shares"]["preferred"] == 0
    assert result["core_gate_fulfillment"] == 1
    assert result["gate_status"] == "passed" and result["bonus"] == 0


@pytest.mark.parametrize("weight", ["current", "five_two", "roc"])
def test_item_weights_change_required_bonus_without_optional_bonus(weight):
    cv, jd = pair([(["Python"], "core"), (["SQL"], "standard"), (["Go"], "preferred")],
                  {"Python": "expert", "Go": "expert"})
    settings = spec(weights=weight)
    with parameter_scope(settings):
        result = score_skills_v3(cv, jd, config())
    assert result["base_skills"] == pytest.approx(100 * 7 / 9)
    assert result["bonus"] == pytest.approx(5 / (1 + settings["item_weights"]["standard"]))
    assert result["core_gate_fulfillment"] == 1


def test_item_weights_also_change_allocation_before_group_cancellation():
    cv, jd = pair([(["B"], "core"), (["C"], "standard")], {"X": "normal", "Y": "normal"})
    relations = [SkillRelation(id=str(i), candidate=a, requirement=b, relation=r,
        reason="constructed allocation contrast", source="test", review_status="approved")
        for i, (a, b, r) in enumerate((("X", "B", "transfer"), ("Y", "B", "near"), ("Y", "C", "near")))]
    results = []
    for weight in ("current", "five_two"):
        with parameter_scope(spec(weights=weight)):
            results.append(score_skills_v3(cv, jd, config(relations=relations)))
    assert [r["candidate_skill"] for r in results[0]["requirements"]] == ["x", "y"]
    assert [r["candidate_skill"] for r in results[1]["requirements"]] == ["y", None]
    assert results[0]["base_skills"] != results[1]["base_skills"]
    assert all(r["core_gate_fulfillment"] == 0 for r in results)


def test_parameter_hash_binds_both_layers_and_artifact_tampering_fails():
    hashes = {canonical_hash(parameter_config(config(), s)) for s in specifications()}
    assert len(hashes) == 12
    family = {"configurations": {"baseline": parameter_config(config(), spec())}}
    family["family_sha256"] = canonical_hash(family)
    assert require_same_identity({"identity": family}) == family["family_sha256"]
    changed = deepcopy(family)
    changed["configurations"]["baseline"]["parameters"]["group_shares"]["core"] = .5
    with pytest.raises(ValueError, match="inconsistent"):
        require_same_identity({"identity": changed})


@pytest.mark.parametrize("kwargs", [{"variant": "equal_groups"}, {"depth_policy": "off"}, {"preferred_policy": "legacy_share"}])
def test_confounded_scoring_modes_are_rejected(kwargs):
    values = config().model_dump(mode="json")
    values.update(kwargs)
    with pytest.raises(ValueError, match="full v3"):
        parameter_config(ScorerConfig(**values), spec())


def row(candidate, score, grade=1, job="j", population="gemini"):
    return {"pair_id": candidate + "::" + job, "cv_id": candidate, "jd_id": job,
            "pair_type": population, "status": "ok", "result": {
                "overall_score": score, "scores": {"skills": score}, "grade": grade,
                "skills_breakdown": {"requirements": [], "bonus": 0, "gate_status": "not_applicable"}}}


def test_ranking_reversals_are_within_jd_and_ties_are_separate():
    previous = [row("a", 40), row("b", 60), row("c", 100, job="other")]
    current = [row("a", 70), row("b", 60), row("c", 0, job="other")]
    assert ranking_changes(previous, current)["strict_reversal"] == 1
    assert ranking_changes(previous, current)["comparisons"] == 1
    tie = ranking_changes(previous, [row("a", 60), row("b", 60)])
    assert tie["strict_reversal"] == 0 and tie["introduced_tie"] == 1
    assert ranking_changes([row("a", 60), row("b", 60)], current)["broken_tie"] == 1


def test_spread_and_change_reports_split_populations_and_do_not_select_weights():
    before = [row("a", 49, 0), row("b", 20, 0, population="heuristic_negative")]
    after = [row("a", 54, 1), row("b", 20, 0, population="heuristic_negative")]
    spread = score_spreads({"current": before, "comparison": after})
    assert spread["summary"]["gemini_primary"]["grade_unstable_pairs"] == 1
    assert spread["summary"]["all"]["overall_spread"]["max"] == 5
    assert spread["summary"]["heuristic_negatives"]["score_sensitive_pairs"] == 0
    assert changed_rows(before, after)[0]["flags"]["grade"]
    with pytest.raises(ValueError, match="identities"):
        score_spreads({"a": before, "b": after[:1]})


def test_fresh_output_is_required(tmp_path):
    with pytest.raises(ValueError, match="fresh directory"):
        run_sweep(tmp_path, as_of=date(2026, 9, 30), corpus_path=tmp_path / "absent",
                  reference_path=tmp_path / "absent", output=tmp_path)


def test_run_status_accepts_empty_failure_lists_and_rejects_actual_failures():
    assert run_status({"a": {"all": {"failures": []}}}, [], True) == "ok"
    assert run_status({"a": {"all": {"failures": ["pair"]}}}, [], True) == "failed"
    assert run_status({"a": {"all": {"failures": []}}}, ["changed"], True) == "failed"
    assert run_status({"a": {"all": {"failures": []}}}, [], False) == "failed"


def test_rounding_only_effects_are_retained_in_score_deltas():
    before, after = row("a", 12), row("a", 13)
    before["result"]["skills_breakdown"].update(base_skills=12.5, skills_after_gate=12.5)
    after["result"]["skills_breakdown"].update(base_skills=12.500000000000004, skills_after_gate=12.500000000000004)
    change = changed_rows([before], [after])[0]
    assert change["flags"]["rounding_only"] and change["flags"]["overall_score"]
    assert change["before"]["base_skills"] != change["after"]["base_skills"]


def test_historical_nomination_nested_ids_and_missing_pairs_are_retained():
    cases = [{"original_comparison": {"cv_id": "a", "jd_id": "j"}, "selection_reason": "review"},
             {"original_comparison": {"cv_id": "missing", "jd_id": "j"}}]
    results = historical_rows([row("a", 40)], cases)
    assert [r["pair_id"] for r in results] == ["a::j", "missing::j"]
    assert results[0]["result"]["result"]["overall_score"] == 40
    assert results[1]["in_fixed_labels"] is False and results[1]["result"] is None
