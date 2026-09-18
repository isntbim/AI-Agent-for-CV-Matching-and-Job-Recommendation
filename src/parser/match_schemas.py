"""ATS Scoring Matrix Pydantic schemas for CV-JD matching evaluation.

Implements the 7-dimensional weighted scoring rubric used by the
LLM-as-a-Judge silver annotation pipeline (see implementation plan).

Separation of concerns: the LLM judge outputs only the 7 dimensional
subscores (0-100 each) plus rationale and skill overlap lists;
`overall_score` and `grade` are computed deterministically by the script
via `ATS_WEIGHTS` and `compute_grade()` so results are fully reproducible.
"""

from typing import List

from pydantic import BaseModel, Field

ATS_WEIGHTS = {
    "skills": 0.30,
    "title": 0.20,
    "experience": 0.15,
    "education": 0.10,
    "industry": 0.10,
    "location": 0.08,
    "salary": 0.07,
}


class ATSDimensionalScores(BaseModel):
    """The 7 ATS dimensions evaluated by the LLM judge (0-100 each)."""

    skills: int = Field(ge=0, le=100)
    title: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)
    industry: int = Field(ge=0, le=100)
    location: int = Field(ge=0, le=100)
    salary: int = Field(ge=0, le=100)

    def compute_overall(self) -> float:
        """Deterministic weighted total score (ATS formula)."""
        return round(sum(
            ATS_WEIGHTS[k] * getattr(self, k) for k in ATS_WEIGHTS
        ), 1)


class LLMJudgeResponse(BaseModel):
    """Schema for what the LLM returns (7 scores + rationale only)."""

    scores: ATSDimensionalScores
    rationale: str
    missing_skills: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)


class SilverMatchPair(BaseModel):
    """Full annotated pair record (LLM output + script-computed fields)."""

    cv_id: str
    jd_id: str
    cv_domain: str
    jd_domain: str
    pair_type: str  # "in_domain", "cross_domain", "heuristic_negative"
    grade: int = Field(ge=0, le=2)
    overall_score: float = Field(ge=0, le=100)
    scores: ATSDimensionalScores
    rationale: str
    missing_skills: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)


def compute_grade(overall_score: float, skills: int, title: int) -> int:
    """Deterministic grade assignment.

    Returns:
        2 (Strong Match): overall >= 75 and skills >= 70 and title >= 70
        1 (Partial Match): overall >= 50
        0 (Mismatch): otherwise
    """
    if overall_score >= 75 and skills >= 70 and title >= 70:
        return 2
    elif overall_score >= 50:
        return 1
    else:
        return 0