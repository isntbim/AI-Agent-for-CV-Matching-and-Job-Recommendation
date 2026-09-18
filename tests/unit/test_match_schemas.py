"""
Unit tests for ATS Scoring Matrix Match Schemas (src/parser/match_schemas.py)
Verifies ATSDimensionalScores, compute_overall(), compute_grade(),
LLMJudgeResponse, and SilverMatchPair data models.
"""

import pytest
from pydantic import ValidationError

from src.parser.match_schemas import (
    ATS_WEIGHTS,
    ATSDimensionalScores,
    LLMJudgeResponse,
    SilverMatchPair,
    compute_grade,
)


class TestATSDimensionalScores:
    """Test the 7-dimensional score container and formula calculation."""

    def test_weights_sum_to_one(self):
        """Verify the ATS weights sum exactly to 1.0 (100%)."""
        total = sum(ATS_WEIGHTS.values())
        assert round(total, 4) == 1.0000

    def test_valid_scores(self):
        scores = ATSDimensionalScores(
            skills=90,
            title=85,
            experience=80,
            education=75,
            industry=80,
            location=100,
            salary=70,
        )
        assert scores.skills == 90
        assert scores.title == 85
        assert scores.experience == 80
        assert scores.education == 75
        assert scores.industry == 80
        assert scores.location == 100
        assert scores.salary == 70

    def test_compute_overall_formula(self):
        """
        Total Score = 0.30*skills + 0.20*title + 0.15*exp + 0.10*edu + 0.10*industry + 0.08*loc + 0.07*sal
        90*0.30 = 27.0
        85*0.20 = 17.0
        80*0.15 = 12.0
        75*0.10 = 7.5
        80*0.10 = 8.0
        100*0.08 = 8.0
        70*0.07 = 4.9
        Total = 27 + 17 + 12 + 7.5 + 8 + 8 + 4.9 = 84.4
        """
        scores = ATSDimensionalScores(
            skills=90,
            title=85,
            experience=80,
            education=75,
            industry=80,
            location=100,
            salary=70,
        )
        overall = scores.compute_overall()
        assert overall == 84.4

    def test_compute_overall_all_100(self):
        scores = ATSDimensionalScores(
            skills=100,
            title=100,
            experience=100,
            education=100,
            industry=100,
            location=100,
            salary=100,
        )
        assert scores.compute_overall() == 100.0

    def test_compute_overall_all_zero(self):
        scores = ATSDimensionalScores(
            skills=0,
            title=0,
            experience=0,
            education=0,
            industry=0,
            location=0,
            salary=0,
        )
        assert scores.compute_overall() == 0.0

    def test_bounds_validation_underflow(self):
        with pytest.raises(ValidationError):
            ATSDimensionalScores(
                skills=-1,
                title=80,
                experience=80,
                education=80,
                industry=80,
                location=80,
                salary=80,
            )

    def test_bounds_validation_overflow(self):
        with pytest.raises(ValidationError):
            ATSDimensionalScores(
                skills=101,
                title=80,
                experience=80,
                education=80,
                industry=80,
                location=80,
                salary=80,
            )


class TestComputeGrade:
    """Test deterministic grade assignment logic."""

    def test_grade_2_strong_match(self):
        # overall >= 75 and skills >= 70 and title >= 70 -> Grade 2
        assert compute_grade(overall_score=75.0, skills=70, title=70) == 2
        assert compute_grade(overall_score=85.0, skills=90, title=85) == 2

    def test_grade_1_when_overall_high_but_skills_low(self):
        # overall >= 75 but skills < 70 -> Grade 1
        assert compute_grade(overall_score=78.0, skills=65, title=85) == 1

    def test_grade_1_when_overall_high_but_title_low(self):
        # overall >= 75 but title < 70 -> Grade 1
        assert compute_grade(overall_score=79.0, skills=75, title=50) == 1

    def test_grade_1_partial_match(self):
        # 50 <= overall < 75 -> Grade 1
        assert compute_grade(overall_score=50.0, skills=50, title=50) == 1
        assert compute_grade(overall_score=68.5, skills=70, title=65) == 1

    def test_grade_0_mismatch(self):
        # overall < 50 -> Grade 0
        assert compute_grade(overall_score=49.9, skills=80, title=80) == 0
        assert compute_grade(overall_score=20.0, skills=10, title=10) == 0
        assert compute_grade(overall_score=0.0, skills=0, title=0) == 0


class TestLLMJudgeResponse:
    """Test the LLM response container schema."""

    def test_valid_llm_response(self):
        raw = {
            "scores": {
                "skills": 85,
                "title": 80,
                "experience": 75,
                "education": 70,
                "industry": 80,
                "location": 90,
                "salary": 75,
            },
            "rationale": "Strong candidate with solid Python background.",
            "missing_skills": ["Kubernetes"],
            "matched_skills": ["Python", "FastAPI", "PostgreSQL"],
        }
        res = LLMJudgeResponse.model_validate(raw)
        assert res.scores.skills == 85
        assert res.rationale == "Strong candidate with solid Python background."
        assert res.missing_skills == ["Kubernetes"]
        assert res.matched_skills == ["Python", "FastAPI", "PostgreSQL"]

    def test_defaults_for_skill_lists(self):
        raw = {
            "scores": {
                "skills": 50,
                "title": 50,
                "experience": 50,
                "education": 50,
                "industry": 50,
                "location": 50,
                "salary": 50,
            },
            "rationale": "Average match.",
        }
        res = LLMJudgeResponse.model_validate(raw)
        assert res.missing_skills == []
        assert res.matched_skills == []


class TestSilverMatchPair:
    """Test the full enriched match pair record."""

    def test_valid_silver_match_pair(self):
        pair = SilverMatchPair(
            cv_id="cv_001",
            jd_id="jd_015",
            cv_domain="software_engineer",
            jd_domain="software_engineer",
            pair_type="in_domain",
            grade=2,
            overall_score=84.4,
            scores=ATSDimensionalScores(
                skills=90,
                title=85,
                experience=80,
                education=75,
                industry=80,
                location=100,
                salary=70,
            ),
            rationale="Excellent fit.",
            missing_skills=["Kubernetes"],
            matched_skills=["Python", "FastAPI"],
        )
        assert pair.cv_id == "cv_001"
        assert pair.grade == 2
        assert pair.pair_type == "in_domain"
        assert pair.overall_score == 84.4
