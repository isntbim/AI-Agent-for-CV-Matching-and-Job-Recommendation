"""Behavioral tests for the opt-in evidence-backed skills policy."""

from copy import deepcopy
import itertools
import json
import random

import pytest

from src.scoring.skills_v2 import score_skills_v2


def cv(skills=(), *, complete=True, group_level=None):
    details = []
    for value in skills:
        name, proficiency = (value, "normal") if isinstance(value, str) else value
        details.append({"name": name, "proficiency": proficiency, "raw_proficiency": proficiency if proficiency != "normal" else None, "skill_quote": f"Skills: {name}.", "proficiency_quote": f"{proficiency.title()} in {name}." if proficiency != "normal" else None})
    return {
        "document": {"skills": [{"keywords": [item["name"] for item in details], "details": details, "level": group_level}]},
        "annotation": {"skills_status": "complete" if complete else "partial", "review_status": "approved", "source_sha256": "cv-source", "text_sha256": "cv-text"},
        "facts": {"no_skills": not bool(skills)},
    }


def jd(requirements=(), *, complete=True):
    items = []
    for index, value in enumerate(requirements):
        options, importance = value
        options = [options] if isinstance(options, str) else options
        items.append({"id": f"r-{index:03d}", "options": options, "importance": importance, "quote": "Requirements: " + " or ".join(options), "importance_quote": "Must have " + " or ".join(options) if importance == "core" else None, "review_status": "approved", "issues": []})
    return {
        "document": {"skills": {"requirements": items}},
        "annotation": {"skills_status": "complete" if complete else "partial", "review_status": "approved", "source_sha256": "jd-source", "text_sha256": "jd-text"},
    }


def config(**kwargs):
    return {"version": "ats-en-v2", "variant": "full", "relations": [], "relation_version": "test-v1", "allow_provisional_relations": False, **kwargs}


def test_missing_jd_classification_keeps_core_gate_unknown():
    legacy_job = {"document": {"skills": {"required": [{"name": "Python"}]}}}
    result = score_skills_v2(cv(["Python"]), legacy_job, config())
    assert result["score"] == 50 and result["basis"] == "missing_evidence"
    assert result["gate_status"] == "unknown" and not result["force_mismatch"]
    partial_job = jd([("Python", "standard")], complete=False)
    assert score_skills_v2(cv(["Python"]), partial_job, config())["gate_status"] == "unknown"
    assert score_skills_v2(cv(["Python"]), jd(), config())["gate_status"] == "not_applicable"


def test_review_can_resolve_unmapped_wording_as_neutral_without_bonus():
    resume = cv(["Python"])
    detail = resume["document"]["skills"][0]["details"][0]
    detail["raw_proficiency"] = "Proficient"
    resume["annotation"]["review_status"] = "pending"
    job = jd([("Python", "core")])
    pending = score_skills_v2(resume, job, config())
    assert pending["score"] == 50 and pending["gate_status"] == "unknown"
    detail["provenance"] = {"review_status": "approved"}
    reviewed = score_skills_v2(resume, job, config())
    assert reviewed["score"] == 100 and reviewed["bonus"] == 0
    assert reviewed["gate_status"] == "passed"


def relation(candidate, requirement, kind="near", status="approved", identifier="relation-1"):
    return {"id": identifier, "candidate": candidate, "requirement": requirement, "relation": kind, "review_status": status, "reason": "Reviewed transferable skill relationship", "source": "Test reviewer"}


@pytest.mark.parametrize("groups", [combination for count in range(1, 4) for combination in itertools.combinations(("core", "standard", "preferred"), count)])
def test_group_shares_normalize_every_present_group_combination(groups):
    requirements = []
    candidates = []
    for group in groups:
        requirements.extend([(f"{group}-matched", group), (f"{group}-missing", group)])
        candidates.append(f"{group}-matched")
    result = score_skills_v2(cv(candidates), jd(requirements), config(variant="group_weights"))
    original = {"core": 0.7, "standard": 0.2, "preferred": 0.1}
    denominator = sum(original[group] for group in groups)
    assert result["base_skills"] == pytest.approx(50)
    for group in original:
        assert result["group_fulfillment"][group] == (0.5 if group in groups else None)
        assert result["group_shares"][group] == (pytest.approx(original[group] / denominator) if group in groups else None)


def test_preferred_has_ten_percent_when_all_groups_exist_and_full_share_alone():
    result = score_skills_v2(cv(["Kubernetes"]), jd([("Python", "core"), ("SQL", "standard"), ("Kubernetes", "preferred")]), config(variant="group_weights"))
    assert result["base_skills"] == pytest.approx(10)
    assert result["preferred_skill_coverage"] == 100
    result = score_skills_v2(cv(["Kubernetes"]), jd([("Kubernetes", "preferred")]), config())
    assert result["score"] == 100
    assert result["group_shares"]["preferred"] == 1
    assert result["gate_status"] == "not_applicable"


def test_group_weights_are_not_applied_twice():
    result = score_skills_v2(cv(["Python", "SQL"]), jd([("Python", "core"), ("SQL", "standard"), ("Docker", "preferred")]), config())
    assert result["base_skills"] == pytest.approx(90)
    assert result["score"] == 90


@pytest.mark.parametrize("level,factor,bonus", [("beginner", 0.5, 0), ("normal", 1, 0), ("intermediate", 1, 0), ("advanced", 1, 2.5), ("expert", 1, 5)])
def test_explicit_proficiency_and_required_bonus(level, factor, bonus):
    result = score_skills_v2(cv([("Python", level)]), jd([("Python", "standard")]), config())
    assert result["base_skills"] == pytest.approx(factor * 100)
    assert result["bonus"] == pytest.approx(bonus)
    assert result["score"] <= 100


def test_policy_example_uses_weighted_required_bonus_denominator():
    result = score_skills_v2(cv([("Python", "expert"), "SQL"]), jd([("Python", "core"), ("SQL", "standard"), ("Docker", "standard"), ("Kubernetes", "preferred")]), config())
    assert result["group_fulfillment"] == {"core": 1, "standard": 0.5, "preferred": 0}
    assert result["base_skills"] == pytest.approx(80)
    assert result["bonus"] == pytest.approx(25 * 0.2 / 2.2)
    assert result["skills_before_gate"] == pytest.approx(82.2727272727)
    assert result["score"] == 82


def test_preferred_and_outside_skills_create_no_proficiency_bonus():
    result = score_skills_v2(cv([("Python", "expert"), ("Docker", "expert")]), jd([("Python", "preferred")]), config())
    assert result["base_skills"] == 100
    assert result["bonus"] == 0
    assert len(result["requirements"]) == 1


def test_legacy_group_level_does_not_award_per_skill_proficiency():
    result = score_skills_v2(cv(["Python"], group_level="Expert"), jd([("Python", "standard")]), config())
    assert result["bonus"] == 0
    assert result["requirements"][0]["proficiency"] == "normal"


def test_aliases_and_duplicate_requirements_preserve_full_equivalence():
    candidate = cv(["JS", "JavaScript"])
    job = jd([("JavaScript", "standard"), ("JS", "core")])
    result = score_skills_v2(candidate, job, config())
    assert result["score"] == 100
    assert len(result["requirements"]) == 1
    assert result["requirements"][0]["importance"] == "core"
    assert result["requirements"][0]["match_type"] == "alias"
    assert result["core_gate_fulfillment"] == 1
    assert {item["kind"] for item in result["merges"]} == {"candidate", "requirement"}


def test_or_is_one_requirement_and_assignment_preserves_scarce_exact_matches():
    result = score_skills_v2(cv(["Python", "SQL"]), jd([(["Python", "SQL"], "core"), ("Python", "standard")]), config())
    assert result["score"] == 100
    assert len(result["requirements"]) == 2
    assert [item["candidate_skill"] for item in result["requirements"]] == ["sql", "python"]
    only_one = score_skills_v2(cv(["Python"]), jd([(["Python", "SQL"], "core")]), config())
    assert only_one["score"] == 100


def test_one_skill_cannot_match_two_independent_requirements():
    result = score_skills_v2(cv(["Python"]), jd([("Python", "core"), ("Data science", "standard")]), config(relations=[relation("Python", "Data science")]))
    assert result["core_gate_fulfillment"] == 1
    assert result["group_fulfillment"]["standard"] == 0
    assert result["requirements"][1]["candidate_skill"] is None


def test_allocation_is_order_independent_including_ties():
    candidate = cv(["Skill B", "Skill A"])
    job = jd([("Requirement B", "standard"), ("Requirement A", "standard")])
    relationships = [relation(c, r, identifier=c + r) for c in ("Skill A", "Skill B") for r in ("Requirement A", "Requirement B")]
    original = score_skills_v2(candidate, job, config(relations=relationships))
    candidate["document"]["skills"][0]["details"].reverse()
    candidate["document"]["skills"][0]["keywords"].reverse()
    job["document"]["skills"]["requirements"].reverse()
    reordered = score_skills_v2(candidate, job, config(relations=list(reversed(relationships))))
    assert original["requirements"] == reordered["requirements"]
    assert original["score"] == reordered["score"]


def test_assignment_prioritizes_core_exact_then_required_exact_over_semantic_total():
    result = score_skills_v2(cv(["Python"]), jd([("Python", "core"), ("Data science", "standard"), ("Scripting", "preferred")]), config(relations=[relation("Python", "Data science"), relation("Python", "Scripting")]))
    assert result["requirements"][0]["exact"] is True
    assert result["requirements"][1]["match_type"] == "none"
    assert result["requirements"][2]["match_type"] == "none"


def test_assignment_matches_exhaustive_lexicographic_optimum_on_small_graphs():
    """Compare observable allocations with every legal assignment, not flow internals."""
    rng = random.Random(4129)
    weights = {"core": 100, "standard": 60, "preferred": 40}
    for _ in range(24):
        names = [f"Candidate {index}" for index in range(4)]
        levels = [rng.choice(("beginner", "normal", "expert")) for _ in names]
        groups = [rng.choice(tuple(weights)) for _ in names]
        options = [[f"Target {index}", *(name for name in names if rng.random() < 0.35)] for index in range(4)]
        relation_coefficients = {}
        relations = []
        for candidate_index, name in enumerate(names):
            for requirement_index in range(4):
                if rng.random() < 0.55:
                    coefficient = rng.choice((0.4, 0.7))
                    relation_coefficients[candidate_index, requirement_index] = coefficient
                    relations.append(relation(name, f"Target {requirement_index}", "near" if coefficient == 0.7 else "transfer", identifier=f"{candidate_index}:{requirement_index}"))

        def objective(assignment):
            totals = [0, 0, 0, 0]
            for requirement_index, candidate_index in enumerate(assignment):
                if candidate_index < 0:
                    continue
                exact = names[candidate_index] in options[requirement_index]
                coefficient = 1.0 if exact else relation_coefficients.get((candidate_index, requirement_index), 0)
                if not coefficient:
                    return None
                factor = 0.5 if levels[candidate_index] == "beginner" else 1
                weight = weights[groups[requirement_index]]
                exact_fulfillment = int(weight * factor * 10) if exact else 0
                fulfillment = int(round(weight * factor * coefficient * 10))
                if groups[requirement_index] == "core":
                    totals[0] += exact_fulfillment
                if groups[requirement_index] != "preferred":
                    totals[1] += exact_fulfillment
                    totals[2] += fulfillment
                else:
                    totals[3] += fulfillment
            return tuple(totals)

        legal_objectives = []
        for assignment in itertools.product(range(-1, 4), repeat=4):
            used = [index for index in assignment if index >= 0]
            if len(used) != len(set(used)):
                continue
            value = objective(assignment)
            if value is not None:
                legal_objectives.append(value)
        result = score_skills_v2(cv(list(zip(names, levels))), jd(list(zip(options, groups))), config(relations=relations))
        assert result["supported"] is True, result["warnings"]
        observed_assignment = [-1 if row["candidate_skill"] is None else names.index(row["candidate_skill"].title()) for row in result["requirements"]]
        assert objective(observed_assignment) == max(legal_objectives)


@pytest.mark.parametrize("kind,expected", [("near", 70), ("transfer", 40)])
def test_semantic_is_directed_approved_and_not_transitive(kind, expected):
    rel = relation("TensorFlow", "PyTorch", kind)
    result = score_skills_v2(cv(["TensorFlow"]), jd([("PyTorch", "standard")]), config(relations=[rel]))
    assert result["score"] == expected
    reverse = score_skills_v2(cv(["PyTorch"]), jd([("TensorFlow", "standard")]), config(relations=[rel]))
    assert reverse["score"] == 0
    transitive = score_skills_v2(cv(["TensorFlow"]), jd([("JAX", "standard")]), config(relations=[rel, relation("PyTorch", "JAX")]))
    assert transitive["score"] == 0


def test_pending_relations_require_explicit_provisional_opt_in_and_rejected_never_score():
    pending = relation("TensorFlow", "PyTorch", status="pending")
    rejected = relation("TensorFlow", "PyTorch", status="rejected")
    assert score_skills_v2(cv(["TensorFlow"]), jd([("PyTorch", "standard")]), config(relations=[pending]))["score"] == 0
    allowed = score_skills_v2(cv(["TensorFlow"]), jd([("PyTorch", "standard")]), config(relations=[pending], allow_provisional_relations=True))
    assert allowed["score"] == 70
    assert allowed["identity"]["provisional"] is True
    assert allowed["requirements"][0]["relation_provisional"] is True
    assert score_skills_v2(cv(["TensorFlow"]), jd([("PyTorch", "standard")]), config(relations=[rejected], allow_provisional_relations=True))["score"] == 0


def test_semantic_does_not_satisfy_core_gate_or_award_bonus():
    result = score_skills_v2(cv([("TensorFlow", "expert")]), jd([("PyTorch", "core")]), config(relations=[relation("TensorFlow", "PyTorch")]))
    assert result["base_skills"] == 70
    assert result["core_gate_fulfillment"] == 0
    assert result["bonus"] == 0
    assert result["score"] == 49
    assert result["force_mismatch"] is True


@pytest.mark.parametrize("matches,gate,blocked,mismatch,score", [(7, 0.7, False, False, 70), (6, 0.6, True, False, 49), (5, 0.5, True, False, 49), (4, 0.4, True, True, 40)])
def test_core_gate_boundaries_use_unrounded_fulfillment(matches, gate, blocked, mismatch, score):
    result = score_skills_v2(cv([f"Skill {index}" for index in range(matches)]), jd([(f"Skill {index}", "core") for index in range(10)]), config())
    assert result["core_gate_fulfillment"] == pytest.approx(gate)
    assert result["strong_blocked"] is blocked
    assert result["force_mismatch"] is mismatch
    assert result["score"] == score


def test_beginner_affects_gate_and_bonus_cannot_rescue_failure():
    result = score_skills_v2(cv([("Python", "beginner")]), jd([("Python", "core")]), config())
    assert result["core_gate_fulfillment"] == 0.5
    assert result["score"] == 49
    assert result["force_mismatch"] is False
    result = score_skills_v2(cv([("Python", "expert"), ("SQL", "expert")]), jd([("Python", "core"), ("Rust", "core"), ("SQL", "standard")]), config())
    assert result["bonus"] > 0
    assert result["skills_before_gate"] > 49
    assert result["score"] == 49
    assert result["strong_blocked"] is True


def test_missing_data_is_neutral_while_supported_no_match_scores_zero():
    no_match = score_skills_v2(cv(["Python"]), jd([("Rust", "core")]), config())
    assert no_match["supported"] is True
    assert no_match["score"] == 0
    assert no_match["force_mismatch"] is True
    missing = score_skills_v2(cv(["Python"], complete=False), jd([("Rust", "core")]), config())
    assert missing["supported"] is False
    assert missing["score"] == 50
    assert missing["gate_status"] == "unknown"
    assert missing["force_mismatch"] is False
    empty_supported = score_skills_v2(cv(), jd([("Rust", "core")]), config())
    assert empty_supported["supported"] is True
    assert empty_supported["score"] == 0


def test_no_requirement_or_missing_annotation_is_neutral():
    result = score_skills_v2(cv(["Python"]), jd(), config())
    assert result["score"] == 50
    assert result["supported"] is False
    assert result["gate_status"] == "not_applicable"
    candidate = cv(["Python"])
    del candidate["annotation"]
    result = score_skills_v2(candidate, jd([("Python", "core")]), config())
    assert result["score"] == 50
    assert result["gate_status"] == "unknown"


@pytest.mark.parametrize("mutation", ["proficiency_quote", "proficiency_label", "proficiency_raw", "proficiency_raw_missing", "proficiency_group", "importance_quote", "skill_quote", "keywords", "overlap", "compound", "conflicting_proficiency"])
def test_incomplete_or_ambiguous_evidence_does_not_become_core_failure(mutation):
    candidate = cv([("Python", "expert")])
    job = jd([("Python", "core")])
    if mutation == "proficiency_quote":
        candidate["document"]["skills"][0]["details"][0]["proficiency_quote"] = "Expert"
    elif mutation == "proficiency_label":
        candidate["document"]["skills"][0]["details"][0]["proficiency_quote"] = "Proficient in Python"
    elif mutation == "proficiency_raw":
        candidate["document"]["skills"][0]["details"][0]["raw_proficiency"] = "Master"
    elif mutation == "proficiency_raw_missing":
        candidate["document"]["skills"][0]["details"][0]["raw_proficiency"] = None
    elif mutation == "proficiency_group":
        candidate = cv([("Python", "expert"), "SQL"])
        candidate["document"]["skills"][0]["details"][0]["proficiency_quote"] = "Expert Python and SQL"
    elif mutation == "importance_quote":
        job["document"]["skills"]["requirements"][0]["importance_quote"] = None
    elif mutation == "skill_quote":
        candidate["document"]["skills"][0]["details"][0]["skill_quote"] = "SQL"
    elif mutation == "keywords":
        candidate["document"]["skills"][0]["keywords"] = ["SQL"]
    elif mutation == "overlap":
        job = jd([("Python", "core"), (["Python", "SQL"], "standard")])
        job["document"]["skills"]["requirements"][1]["review_status"] = "pending"
    elif mutation == "compound":
        job = jd([("Python OR SQL", "core")])
    else:
        candidate = cv([("Python", "expert"), ("Python", "advanced")])
    result = score_skills_v2(candidate, job, config())
    assert result["supported"] is False
    assert result["score"] == 50
    assert result["gate_status"] == "unknown"
    assert result["force_mismatch"] is False


def test_complete_pending_annotations_are_explicitly_provisional_and_scoreable():
    candidate = cv(["Python"])
    job = jd([("Python", "core")])
    candidate["annotation"]["review_status"] = "pending"
    job["annotation"]["review_status"] = "pending"
    job["document"]["skills"]["requirements"][0]["review_status"] = "pending"
    result = score_skills_v2(candidate, job, config())
    assert result["supported"] is True
    assert result["score"] == 100
    assert result["identity"]["provisional"] is True


def test_incremental_variants_control_groups_semantic_proficiency_bonus_and_gate():
    candidate = cv([("Python", "beginner"), ("TensorFlow", "expert"), ("Kubernetes", "expert")])
    job = jd([("Python", "core"), ("PyTorch", "standard"), ("Kubernetes", "preferred")])
    settings = config(relations=[relation("TensorFlow", "PyTorch")])
    results = {variant: score_skills_v2(candidate, job, {**settings, "variant": variant}) for variant in ("equal_groups", "group_weights", "semantic", "proficiency", "full")}
    assert results["equal_groups"]["base_skills"] == pytest.approx(100 * 2 / 3)
    assert results["group_weights"]["base_skills"] == pytest.approx(80)
    assert results["semantic"]["base_skills"] == pytest.approx(94)
    assert results["proficiency"]["base_skills"] == pytest.approx(59)
    assert results["proficiency"]["gate_status"] == "not_applicable"
    assert results["full"]["score"] == 49
    assert results["full"]["gate_status"] == "failed"
    assert all(result["bonus"] == 0 for result in results.values())


def test_output_is_json_serializable_and_identity_tracks_evidence_and_relations():
    candidate, job = cv(["Python"]), jd([("Python", "core")])
    first = score_skills_v2(candidate, job, config())
    assert json.loads(json.dumps(first)) == first
    modified = deepcopy(candidate)
    modified["annotation"]["source_sha256"] = "different-source"
    second = score_skills_v2(modified, job, config())
    assert first["identity"]["annotation_sha256"] != second["identity"]["annotation_sha256"]
    relation_result = score_skills_v2(candidate, job, config(relations=[relation("Python", "Rust")]))
    assert first["identity"]["relations_sha256"] != relation_result["identity"]["relations_sha256"]


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown skills v2 variant"):
        score_skills_v2(cv(["Python"]), jd([("Python", "standard")]), config(variant="undefined"))
