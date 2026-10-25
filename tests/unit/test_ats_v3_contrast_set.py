"""Executable SPEC oracles for behavioural verification, not effectiveness.

SPEC drafted by Gemini (agy) with NotebookLM, verified by Codex against the
code, and reviewed by Claude Code. All fixtures are synthetic and in memory.
Findings assert current behaviour: documented finding, not endorsed policy.
Beginner oracles were superseded by direct human decision D3-30 (2026-10-12).
Golang->Go and question 5 were superseded by direct human decision D3-31.
Other findings remain documented finding, not endorsed policy.
All design questions are resolved by D3-30 through D3-36; v3 retains full precision.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import pytest

from src.parser.skill_v2 import SkillRelation
from src.scoring import JobInput, ResumeInput, ScorerConfig, score_pair
from src.scoring.skills_v3 import score_skills_v3
from src.scoring.models_v3 import display_score


SPEC_SHA256 = "3e4ddd2575680f8923415f79fb2468e5895bdf0306c876666c9260e94e804d13"
AS_OF = date(2026, 9, 30)
WORDS = {"normal": "", "beginner": "Beginner", "intermediate": "Intermediate",
         "advanced": "Advanced", "expert": "Expert"}
UR, UCD, MM, EG = ("unspecified_requirement", "unknown_candidate_depth", "meets_minimum", "explicit_gap")
UNMATCHED, RSU, UAD = "unmatched", "related_skill_depth_unknown", "unsupported_annotation_depth"


@dataclass(frozen=True)
class Requirement:
    options: tuple[tuple[str, str], ...]
    importance: str = "core"
    quote: str | None = None
    importance_quote: str | None = None


@dataclass(frozen=True)
class Candidate:
    name: str
    level: str = "normal"
    wording: str | None = None


@dataclass(frozen=True)
class Fixture:
    requirements: tuple[Requirement, ...]
    candidates: tuple[Candidate, ...]
    expected: dict
    depth_policy: str = "full"
    relations: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    fixtures: tuple[Fixture, ...]
    finding: str | None = None


def requirement(name="Python", minimum="normal", importance="core", **kwargs):
    return Requirement(((name, minimum),), importance, **kwargs)


def depth(status, credit=1.0, effective=1.0, gap=None, selected=None, exact=None):
    result = {"depth_status": status, "depth_credit": credit,
              "effective_fulfillment": effective, "depth_gap": gap}
    if selected is not None:
        result["selected_option"] = selected
    if exact is not None:
        result["exact"] = exact
    return result


def expected(rows, raw=100.0, score=100, bonus=0.0, gate="passed", gate_f=1.0,
             force=False, supported=True, warning=None, raw_places=None):
    return {"requirements": rows, "skills_after_gate": raw, "score": score,
            "bonus": bonus, "gate_status": gate, "core_gate_fulfillment": gate_f,
            "force_mismatch": force, "supported": supported,
            "warning_contains": warning, "raw_display_places": raw_places}


def single(identifier, minimum, level, oracle, *, policy="full", finding=None, wording=None):
    return Case(identifier, "MFT", (Fixture((requirement(minimum=minimum),),
        (Candidate("Python", level, wording),), oracle, policy),), finding)


MFT_CASES = [
    single("MFT-01", "normal", "normal", expected([depth(UR)])),
    single("MFT-02", "intermediate", "normal", expected([depth(UCD)])),
    single("MFT-03", "intermediate", "intermediate", expected([depth(MM, gap=0)])),
    single("MFT-04", "advanced", "intermediate", expected([depth(EG, .75, .75, 1)], 75, 75, gate_f=.75)),
    single("MFT-05", "expert", "intermediate", expected([depth(EG, .5, .5, 2)], 49, 49, gate="failed", gate_f=.5)),
    single("MFT-06", "expert", "beginner", expected([depth(UCD)])),
    single("MFT-06b", "expert", "beginner", expected([depth(UAD, None, None)], 50, 50,
        gate="unknown", gate_f=None, supported=False, warning="unmapped wording"),
        wording="Basic", finding="Basic encoded as Beginner is unsupported; neutral fallback"),
    single("MFT-07", "intermediate", "advanced", expected([depth(MM, gap=0)], bonus=2.5)),
    single("MFT-08", "intermediate", "expert", expected([depth(MM, gap=0)], bonus=5)),
    single("MFT-09", "expert", "advanced", expected([depth(EG, .75, .75, 1)], 75, 75, gate_f=.75)),
    single("MFT-10", "expert", "advanced", expected([depth(EG, .75, .75, 1)], 77.5, 77.5, 2.5),
        policy="score_only", finding="D3-36: full precision retains 77.5"),
    single("MFT-11", "expert", "advanced", expected([depth(EG, .75, 1, 1)], bonus=2.5), policy="off"),
]

for identifier, level, credit, gap, raw, score, gate, gate_f in (
    ("MFT-12", "advanced", .75, 1, 90, 90, "passed", .875),
    ("MFT-13", "intermediate", .5, 2, 77.5, 77.5, "passed", .75),
    ("MFT-14", "beginner", 1, None, 100, 100, "passed", 1),
    ("MFT-15", None, 1, None, 49, 49, "failed", .5),
):
    rows = [depth(UCD) if level == "beginner" else depth(EG, credit, credit, gap) if level else depth(UNMATCHED, 1, 0), depth(MM, gap=0)]
    candidates = ((Candidate("Python", level),) if level else ()) + (Candidate("SQL", "expert"),)
    MFT_CASES.append(Case(identifier, "MFT", (Fixture(
        (requirement(minimum="expert"), requirement("SQL", "expert")), candidates,
        expected(rows, raw, score, 2.5, gate, gate_f)),),
        "D3-36: retains 77.5 and 76.5 without integer rounding" if identifier == "MFT-13" else None))

MFT_CASES.extend([
    Case("MFT-16", "MFT", (Fixture((requirement(minimum="expert"), requirement("SQL", "expert")),
        (Candidate("Python", "beginner"), Candidate("SQL", "beginner")),
        expected([depth(UCD), depth(UCD)])),)),
    Case("MFT-17", "MFT", (Fixture((requirement("SQL"), requirement(minimum="expert", importance="preferred")),
        (Candidate("SQL"), Candidate("Python", "expert")), expected([depth(UR), depth(MM, gap=0)])),)),
    Case("MFT-18", "MFT", (Fixture((requirement(minimum="expert", importance="preferred"),),
        (Candidate("Python", "expert"),), expected([depth(MM, gap=0)], gate="not_applicable", gate_f=None)),)),
    Case("MFT-19", "MFT", (Fixture((requirement(minimum="expert"),), (Candidate("Django"),),
        expected([depth(RSU, 1, .7, exact=False)], 49, 49, gate="failed", gate_f=0, force=True),
        relations=(("Django", "Python", "near"),)),)),
    Case("MFT-20", "MFT", (Fixture((Requirement((("Python", "advanced"), ("R", "intermediate"))),),
        (Candidate("Python", "intermediate"), Candidate("R", "intermediate")),
        expected([depth(MM, gap=0, selected="r", exact=True)])),)),
    single("MFT-21", "beginner", "beginner", expected([depth(UR)])),
])

for identifier, total, met, raw, score, gate, gate_f, force, places in (
    ("MFT-22", 10, 7, 70, 70, "passed", .7, False, None),
    ("MFT-23", 9, 4, 100 * 4 / 9, 100 * 4 / 9, "failed", 4 / 9, True, None),
):
    MFT_CASES.append(Case(identifier, "MFT", (Fixture(
        tuple(requirement(f"Capability{i}", "intermediate") for i in range(total)),
        tuple(Candidate(f"Capability{i}", "intermediate") for i in range(met)),
        expected([depth(MM, gap=0)] * met + [depth(UNMATCHED, 1, 0)] * (total - met),
                 raw, score, gate=gate, gate_f=gate_f, force=force, raw_places=places)),)))

MFT_CASES.extend([
    Case("MFT-24", "MFT", (Fixture((requirement(minimum="advanced"), requirement("SQL", "intermediate", "standard")),
        (Candidate("Python", "advanced"),), expected([depth(MM, gap=0), depth(UNMATCHED, 1, 0)],
            100 * .7 / (.7 + .2) + 1.5625, 100 * .7 / (.7 + .2) + 1.5625, 1.5625)),)),
    Case("MFT-25", "MFT", (Fixture((requirement(),), (Candidate("Flask"),),
        expected([depth(UR, 1, .4, exact=False)], 40, 40, gate="failed", gate_f=0, force=True),
        relations=(("Flask", "Python", "transfer"),)),)),
    Case("MFT-26", "MFT", (Fixture((requirement(),), (Candidate("Django", "beginner"),),
        expected([depth(UR, 1, .7, exact=False)], 49, 49, gate="failed", gate_f=0, force=True),
        relations=(("Django", "Python", "near"),)),)),
    Case("MFT-27", "MFT", (Fixture((Requirement((("Python", "normal"), ("Django", "normal"))),),
        (Candidate("Python"), Candidate("Flask")), expected([depth(UR, selected="python", exact=True)]),
        relations=(("Flask", "Django", "near"),)),)),
])

DIR_CASES = [
    Case("DIR-01", "DIR", (
        Fixture((requirement(minimum="intermediate"),), (Candidate("Python"),), expected([depth(UCD)])),
        Fixture((requirement(minimum="intermediate"),), (Candidate("Python", "beginner"),),
            expected([depth(UCD)])))),
    Case("DIR-02", "DIR", tuple(Fixture((requirement(minimum="expert"),), (Candidate("Python", level),), oracle)
        for level, oracle in (
            ("beginner", expected([depth(UCD)])),
            ("intermediate", expected([depth(EG, .5, .5, 2)], 49, 49, gate="failed", gate_f=.5)),
            ("advanced", expected([depth(EG, .75, .75, 1)], 75, 75, gate_f=.75)),
            ("expert", expected([depth(MM, gap=0)], bonus=5))))),
    Case("DIR-03", "DIR", (
        Fixture((requirement(minimum="advanced"),), (Candidate("Python", "expert"),), expected([depth(MM, gap=0)], bonus=5)),
        Fixture((requirement(minimum="advanced"),), (Candidate("Python", "beginner"),),
            expected([depth(UCD)])))),
]

INV_CASES = [
    Case("INV-01", "INV", tuple(Fixture(reqs, cvs, expected([depth(UR), depth(UR)]))
        for reqs in ((requirement(), requirement("SQL", importance="standard")),
                     (requirement("SQL", importance="standard"), requirement()))
        for cvs in ((Candidate("SQL"), Candidate("Python")), (Candidate("Python"), Candidate("SQL"))))),
    Case("INV-02", "INV", tuple(Fixture((requirement(minimum="intermediate"),),
        (Candidate(word),), expected([depth(UCD)])) for word in ("PYTHON", " python ")),
        "Case/whitespace invariant; punctuation Python! remains unmatched"),
    Case("INV-03", "INV", (Fixture((requirement(minimum="intermediate"),),
        (Candidate("Py", "intermediate"),), expected([depth(MM, gap=0, exact=True)])),)),
    Case("INV-04", "INV", tuple(Fixture((Requirement((("Python", "intermediate"), ("SQL", "intermediate"))),),
        (Candidate(name, "intermediate"),), expected([depth(MM, gap=0)])) for name in ("Python", "SQL"))),
    Case("INV-05", "INV", tuple(Fixture((requirement("Go", "intermediate"),),
        (Candidate(name, "intermediate"),), expected([depth(MM, gap=0, selected="go", exact=True)]))
        for name in ("Golang", "Go"))),
]

GAP_CASES = [Case("GAP-01", "GAP", (Fixture((requirement("Go", "intermediate"),),
    (Candidate("Golang", "intermediate"),), expected([depth(MM, gap=0, selected="go", exact=True)])),),
    "Resolved D3-31: human-approved Golang->Go alias; former score-0 finding retained in history")]

# The SPEC's cited lines 210/201 validate JD evidence, despite its CV-column label.
ERR_CASES = [
    Case("ERR-01", "ERR", (Fixture((requirement(minimum="advanced", quote="Title: Lead Python Eng"),),
        (Candidate("Python", "advanced"),), expected([depth(UAD, None, None)], 50, 50,
            gate="unknown", gate_f=None, supported=False, warning="JD title requirement")),)),
    Case("ERR-02", "ERR", (Fixture((requirement(minimum="advanced", quote="Not required: Python"),),
        (Candidate("Python", "advanced"),), expected([depth(UAD, None, None)], 50, 50,
            gate="unknown", gate_f=None, supported=False, warning="negated importance scope")),)),
]
CASES = tuple(MFT_CASES + DIR_CASES + INV_CASES + GAP_CASES + ERR_CASES)


def build_fixture(fixture):
    details = []
    for candidate in fixture.candidates:
        word = candidate.wording if candidate.wording is not None else WORDS[candidate.level]
        details.append({"name": candidate.name, "skill_quote": candidate.name,
            "proficiency": candidate.level, "raw_proficiency": word or None,
            "proficiency_quote": f"{word} {candidate.name}" if word else None})
    cv = ResumeInput.model_validate({"document": {"schema_version": "resume-v3", "skills": [{
        "name": "Synthetic skills", "keywords": [c.name for c in fixture.candidates], "details": details}]},
        "facts": {"evidence": {"skills": [{"quote": "; ".join(d["proficiency_quote"] or d["name"] for d in details)}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    units = []
    for unit in fixture.requirements:
        quote = unit.quote or " or ".join(f"{WORDS[level]} {name}".strip() for name, level in unit.options) + (
            " required" if unit.importance == "core" else " experience")
        units.append({"id": "req:" + unit.options[0][0], "options": [name for name, _ in unit.options],
            "importance": unit.importance, "quote": quote,
            "importance_quote": (unit.importance_quote or quote) if unit.importance == "core" else None,
            "review_status": "approved", "option_details": [{"name": name,
                "minimum_proficiency": level, "raw_proficiency": WORDS[level] or None,
                "proficiency_quote": f"{WORDS[level]} {name}" if level != "normal" else None}
                for name, level in unit.options]})
    jd = JobInput.model_validate({"document": {"schema_version": "job-v3", "id": "synthetic-jd",
        "url": "urn:contrast-set", "scraped_at": "none", "title": "Engineer",
        "title_normalized": "engineer", "category": "engineer", "skills": {"requirements": units}},
        "facts": {"evidence": {"skills": [{"quote": "\n".join(u["quote"] for u in units)}]}},
        "annotation": {"schema_version": "ats-skills-v3", "skills_status": "complete"}})
    relations = [SkillRelation(id=f"synthetic-{i}", candidate=a, requirement=b, relation=kind,
        reason="Controlled behavioural fixture, not a corpus relation approval", source="SPEC fixture",
        review_status="approved") for i, (a, b, kind) in enumerate(fixture.relations)]
    config = ScorerConfig(version="ats-en-v3", variant="full", depth_policy=fixture.depth_policy,
        preferred_policy="separate", relations=relations)
    return cv, jd, config


def execute_fixture(fixture):
    cv, jd, config = build_fixture(fixture)
    result = score_skills_v3(cv, jd, config)
    runtime = score_pair(cv, jd, config=config, as_of_date=AS_OF)
    assert runtime.unrounded_scores.skills == result["score"]
    assert runtime.scores.skills == display_score(result["score"])
    if result["force_mismatch"]:
        assert runtime.grade == 0
    return result


def assert_expected(result, oracle, requirements):
    assert len(result["requirements"]) == len(oracle["requirements"])
    assert len(requirements) == len(oracle["requirements"])
    # Requirement IDs remain stable when the input lists are permuted.
    actual_rows = {row["id"]: row for row in result["requirements"]}
    for unit, item in zip(requirements, oracle["requirements"]):
        row = actual_rows["req:" + unit.options[0][0]]
        for key, value in item.items():
            assert row[key] == pytest.approx(value, abs=1e-12) if isinstance(value, float) else row[key] == value
    for key in ("score", "bonus", "gate_status", "core_gate_fulfillment", "force_mismatch", "supported"):
        value = oracle[key]
        assert result[key] == pytest.approx(value, abs=1e-12) if isinstance(value, float) else result[key] == value
    places = oracle["raw_display_places"]
    raw = result["skills_after_gate"]
    assert (round(raw, places) == oracle["skills_after_gate"] if places is not None
            else raw == pytest.approx(oracle["skills_after_gate"], abs=1e-12))
    if oracle["warning_contains"]:
        assert any(oracle["warning_contains"] in warning for warning in result["warnings"])
    else:
        assert not result["warnings"]


def snapshot(result):
    return {key: result[key] for key in ("skills_after_gate", "score", "bonus", "gate_status",
        "core_gate_fulfillment", "force_mismatch", "supported", "warnings")} | {"requirements": [
        {key: row.get(key) for key in ("id", "options", "depth_status", "depth_credit", "effective_fulfillment",
                "depth_gap", "selected_option", "exact", "required_level", "candidate_level",
                "raw_proficiency", "candidate_depth_evidence_status", "bonus_eligible", "bonus_factor", "evidence")} for row in result["requirements"]]}


def verify_case(case):
    observations = []
    for fixture in case.fixtures:
        result = execute_fixture(fixture)
        assert_expected(result, fixture.expected, fixture.requirements)
        labelled = fixture.expected | {"requirements": [item | {"id": "req:" + unit.options[0][0]}
            for unit, item in zip(fixture.requirements, fixture.expected["requirements"])]}
        observations.append({"expected": labelled, "actual": snapshot(result)})
    extras = {}
    if case.id == "DIR-01":
        assert observations[0]["actual"]["score"] == observations[1]["actual"]["score"]
    elif case.id == "DIR-02":
        # Beginner is unranked; monotonicity applies only to the remaining ladder.
        assert [r["actual"]["score"] for r in observations] == [100, 49, 75, 100]
    elif case.id == "DIR-03":
        assert observations[0]["actual"]["score"] - observations[1]["actual"]["score"] == 0
    elif case.id.startswith("INV-"):
        for key in ("score", "skills_after_gate", "bonus", "gate_status", "core_gate_fulfillment"):
            assert all(r["actual"][key] == observations[0]["actual"][key] for r in observations)
    if case.id == "INV-02":
        # documented finding, not endorsed policy: punctuation is preserved.
        probe = replace(case.fixtures[0], candidates=(Candidate("Python!"),), expected=expected(
            [depth(UNMATCHED, 1, 0)], 0, 0, gate="failed", gate_f=0, force=True))
        result = execute_fixture(probe)
        assert_expected(result, probe.expected, probe.requirements)
        extras["punctuation_finding"] = {"expected": probe.expected, "actual": snapshot(result)}
    if case.id == "MFT-13":
        # Twenty Core items, fifteen full matches, six Experts: 75 + 1.5 = 76.5 exactly.
        # D3-36 resolves early integer rounding; both .5 values remain intact.
        probe = Fixture(tuple(requirement(f"Capability{i}") for i in range(20)),
            tuple(Candidate(f"Capability{i}", "expert" if i < 6 else "normal") for i in range(15)),
            expected([depth(UR)] * 15 + [depth(UNMATCHED, 1, 0)] * 5,
                     76.5, 76.5, 1.5, gate_f=.75))
        result = execute_fixture(probe)
        assert result["skills_after_gate"] == 76.5
        assert_expected(result, probe.expected, probe.requirements)
        extras["precision_half_retained"] = {"expected": probe.expected, "actual": snapshot(result)}
    return {"id": case.id, "type": case.kind, "finding": case.finding,
            "observations": observations, "extra_checks": extras, "pass": True}


@pytest.mark.parametrize("case", MFT_CASES, ids=lambda c: c.id)
def test_mft(case):
    """D3-30/36 resolve Beginner/rounding; Basic remains a documented finding, not endorsed policy."""
    verify_case(case)


@pytest.mark.parametrize("case", DIR_CASES, ids=lambda c: c.id)
def test_dir(case):
    """D3-30 resolves Beginner-versus-silence; the ranked ladder excludes Beginner."""
    verify_case(case)


@pytest.mark.parametrize("case", INV_CASES, ids=lambda c: c.id)
def test_inv(case):
    """SPEC INV oracles; punctuation sensitivity is documented finding, not endorsed policy."""
    verify_case(case)


@pytest.mark.parametrize("case", GAP_CASES, ids=lambda c: c.id)
def test_gap(case):
    """D3-31 resolves the gap: approved Golang->Go is an exact depth match."""
    verify_case(case)


@pytest.mark.parametrize("case", ERR_CASES, ids=lambda c: c.id)
def test_err(case):
    """SPEC ERR JD-evidence guards; warnings must produce neutral credit and an unknown gate."""
    verify_case(case)
