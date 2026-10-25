"""D3-30 regressions: v3 ignores literal Beginner depth; v2 retains its limit."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.parser.skill_v3 import CVSkillDetailV3, RequirementOption
from src.scoring import ResumeInput, ScorerConfig
from src.scoring.extraction import ExtractionCache, _prompts
from src.scoring.skills_v2 import score_skills_v2
from src.scoring.skills_v3 import BONUS, FULFILLMENT, score_skills_v3
from src.scoring.v3_corpus import refresh, remove_beginner_level, validate_identity
from src.scoring.v3_validation import POLICY_VERSION, RANK, direct_level, validate_skill_evidence
from tests.unit.test_skills_v3 import pair, config


@pytest.mark.parametrize("policy", ["off", "score_only", "full"])
def test_beginner_equals_silence_but_retains_raw_evidence(policy):
    beginner, jd = pair("beginner", "expert")
    silent, _ = pair("normal", "expert")
    result = score_skills_v3(beginner, jd, config(policy))
    reference = score_skills_v3(silent, jd, config(policy))
    for key in ("score", "base_skills", "bonus", "core_gate_fulfillment", "gate_status", "force_mismatch"):
        assert result[key] == reference[key]
    row = result["requirements"][0]
    assert row["proficiency"] == "normal" and row["raw_proficiency"] == "beginner"
    assert row["evidence"]["proficiency_quote"] == "beginner Python"
    assert row["depth_status"] == "unknown_candidate_depth"
    assert row["depth_credit"] == row["effective_fulfillment"] == 1
    assert row["depth_gap"] is None and not row["bonus_eligible"]
    assert result["identity"]["depth_policy_version"] == POLICY_VERSION


def test_beginner_excluded_from_v3_tables_and_schema():
    assert "beginner" not in RANK and "beginner" not in FULFILLMENT and "beginner" not in BONUS
    for cls, field in ((CVSkillDetailV3, "proficiency"), (RequirementOption, "minimum_proficiency")):
        assert cls.model_json_schema()["properties"][field]["enum"] == ["normal", "intermediate", "advanced", "expert"]


def test_beginner_jd_minimum_is_unspecified_and_raw_is_preserved():
    cv, jd = pair("beginner", "beginner")
    minimum = jd.document.skills.requirements[0].option_details[0]
    assert minimum.minimum_proficiency == "normal" and minimum.raw_proficiency == "beginner"
    assert minimum.proficiency_quote == "beginner Python"
    row = score_skills_v3(cv, jd, config())["requirements"][0]
    assert row["required_level"] == "normal" and row["depth_status"] == "unspecified_requirement"
    assert row["fulfillment"] == 1


def test_v2_still_scores_beginner_at_half_credit():
    cv, jd = pair("beginner", "expert")
    legacy = cv.model_dump(mode="json")
    legacy["document"]["schema_version"] = "resume-v2"
    legacy["document"]["skills"][0]["details"][0]["proficiency"] = "beginner"
    legacy = ResumeInput.model_validate(legacy)
    result = score_skills_v2(legacy, jd, ScorerConfig(version="ats-en-v2"))
    assert legacy.document.skills[0].details[0].proficiency == "beginner"
    assert result["core_gate_fulfillment"] == .5 and result["score"] == 49
    assert score_skills_v3(cv, jd, config("off"))["score"] == 100


def test_basic_does_not_become_supported_beginner():
    cv, jd = pair("beginner", "expert")
    detail = cv.document.skills[0].details[0]
    detail.raw_proficiency, detail.proficiency_quote = "Basic", "Basic Python"
    result = score_skills_v3(cv, jd, config())
    assert not result["supported"] and result["score"] == 50
    assert direct_level("Python", "Basic Python") is None


@pytest.mark.parametrize("text", ["Not beginner Python", "Trained others in beginner Python", "Java only"])
def test_ignored_beginner_wording_still_requires_real_positive_skill_evidence(text):
    cv, _ = pair("beginner", "expert")
    assert any(issue["severity"] == "error" for issue in validate_skill_evidence(cv, text, strict=False))


def test_local_annotation_mapping_retains_beginner_wording():
    from src.scoring.v3_corpus import _local_claim
    assert _local_claim("Java", "Languages: Java (Beginner), Python") == ("normal", "Beginner", "Java (Beginner)")
    cv, _ = pair("beginner", "expert")
    assert not validate_skill_evidence(cv, "Python. beginner Python", strict=False)


def test_v3_prompt_and_cache_versions_remove_beginner_enum(tmp_path):
    prompt, _ = _prompts("resume", "Beginner Python", "v3")
    assert "Explicit Beginner wording maps to normal" in prompt
    schema = json.loads(prompt.split("document follows this schema: ")[1].split(" facts follows this schema: ")[0])
    assert "beginner" not in schema["$defs"]["CVSkillDetail"]["properties"]["proficiency"]["enum"]
    old = ExtractionCache(tmp_path, None, "offline", "test", ruleset="v2")
    new = ExtractionCache(tmp_path, None, "offline", "test", ruleset="v3")
    assert old.identity["prompt"] == "ats-facts-en-v6-skills-v2" and old.identity["schema"] == "ats-envelope-v2"
    assert new.identity["validation_policy"] == POLICY_VERSION


def migration_corpus(tmp_path):
    cv, _ = pair("beginner", "expert")
    envelope = cv.model_dump(mode="json")
    envelope["document"]["skills"][0]["details"][0]["proficiency"] = "beginner"
    text = "Python. beginner Python"
    (tmp_path / "source.txt").write_bytes(text.encode())
    sha = hashlib.sha256(text.encode()).hexdigest()
    corpus = {"version": "ats-source-corpus-v3", "documents": [{"id": "cv-test", "kind": "resume",
        "source": {"source_path": "source.txt", "text_path": "source.txt", "source_sha256": sha, "text_sha256": sha},
        "envelope": envelope}], "relations": {"relations": []}, "rules": {}, "changes": [], "review_history": []}
    refresh(corpus)
    return corpus


def test_beginner_migration_is_checksum_bound_idempotent_and_not_envelope_approval(tmp_path):
    corpus = migration_corpus(tmp_path)
    records = remove_beginner_level(corpus, tmp_path)
    assert len(records) == 1 and records[0]["review_kind"] == "policy_derived"
    assert records[0]["before"]["proficiency"] == "beginner" and records[0]["after"]["proficiency"] == "normal"
    assert records[0]["after"]["raw_proficiency"] == "beginner"
    assert corpus["documents"][0]["envelope"]["annotation"]["review_status"] == "pending"
    first = deepcopy(corpus)
    assert remove_beginner_level(corpus, tmp_path) == [] and corpus == first
    validate_identity(corpus)


def test_beginner_migration_rejects_changed_source_without_mutating_corpus(tmp_path):
    corpus = migration_corpus(tmp_path)
    before = deepcopy(corpus)
    (tmp_path / "source.txt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="source checksum mismatch"):
        remove_beginner_level(corpus, tmp_path)
    assert corpus == before


def test_beginner_migration_uses_existing_canonical_text_hash_for_windows_newlines(tmp_path):
    corpus = migration_corpus(tmp_path)
    text = "Python.\nbeginner Python"
    (tmp_path / "source.txt").write_bytes(text.replace("\n", "\r\n").encode())
    corpus["documents"][0]["source"]["source_sha256"] = hashlib.sha256((tmp_path / "source.txt").read_bytes()).hexdigest()
    corpus["documents"][0]["source"]["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    refresh(corpus)
    assert len(remove_beginner_level(corpus, tmp_path)) == 1


def test_v3_cv_prompt_carries_label_rules_and_nullable_employer():
    prompt, _ = _prompts("resume", "Analyst 2019", "v3")
    for rule in ("a past date alone suffices", "Juris Doctor maps to master", "main function of the candidate's real jobs",
                 "pivot tables, macros", "skill group named Languages", "no fixed word limit"):
        assert rule in prompt
    assert "Copy SHORT exact quotes" not in prompt
    defs = json.loads(prompt.split("document follows this schema: ")[1].split(" facts follows this schema: ")[0])["$defs"]
    assert {"type": "null"} in defs["WorkExperience"]["properties"]["name"]["anyOf"]
    assert {"type": "null"} in defs["Education"]["properties"]["institution"]["anyOf"]
    job, _ = _prompts("job", "Python required", "v3")
    assert "Copy SHORT exact quotes" in job  # JD prompt unchanged


def test_cv_v3_prompt_carries_the_as_of_date_for_past_dated_degrees():
    import json
    prompt, user = _prompts("resume", "BSc 2026", "v3", "2026-10-25")
    assert json.loads(user)["as_of_date"] == "2026-10-25" and "as_of_date" in prompt
    assert "as_of_date" not in json.loads(_prompts("resume", "BSc 2026", "v3")[1])
    assert "as_of_date" not in json.loads(_prompts("job", "BSc 2026", "v3", "2026-10-25")[1])


def test_cv_v3_cache_requires_a_valid_as_of_and_keys_it_for_resumes_only(tmp_path):
    import pytest
    from src.scoring.extraction import ExtractionError
    class Client:
        def generate(self, *args, **kwargs):
            raise AssertionError("not reached")
    with pytest.raises(ExtractionError, match="requires as_of"):
        ExtractionCache(tmp_path, Client(), "offline", "test", ruleset="v3").get("BSc 2026", "resume")
    with pytest.raises(ValueError):
        ExtractionCache(tmp_path, None, "offline", "test", ruleset="v3", as_of="20261025")
    with pytest.raises(ExtractionError, match="Cache miss"):  # a job extraction needs no date
        ExtractionCache(tmp_path, None, "offline", "test", ruleset="v3").get("Python required", "job")
