"""Runtime models kept separate from the historical silver annotation schema."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.parser.job_schemas import JobDescriptionSchema
from src.parser.match_schemas import ATS_WEIGHTS, ATSDimensionalScores
from src.parser.schemas import ResumeSchema

from .normalization import ROLE_FAMILIES

RULE_VERSION = "ats-en-v1"
PROMPT_VERSION = "ats-facts-en-v5"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(StrictModel):
    """A literal source quote, including explicit negative statements."""

    quote: str = Field(min_length=1)


class Facts(StrictModel):
    evidence: dict[str, list[Evidence]] = Field(default_factory=dict, json_schema_extra={
        "properties": {key: {"type": "array", "items": {"type": "object", "additionalProperties": False,
                       "properties": {"quote": {"type": "string", "minLength": 1}}, "required": ["quote"]}}
                       for key in ("skills", "title", "experience", "education", "industry", "location", "salary")},
        "additionalProperties": False,
    })
    domains: list[str] = Field(default_factory=list, json_schema_extra={
        "items": {"type": "string", "enum": list(ROLE_FAMILIES)},
    })
    completed_degrees: list[Literal["high_school", "associate", "bachelor", "master", "doctorate"]] = Field(default_factory=list)
    minimum_degree: Literal["high_school", "associate", "bachelor", "master", "doctorate"] | None = None
    no_experience_required: bool = False
    no_education_required: bool = False
    no_skills: bool = False
    remote: bool = False
    eligible_countries: list[str] = Field(default_factory=list)
    worldwide_remote: bool = False
    # Prevent parser defaults (VN/VND) from supplying unspecified facts.
    country: str | None = None
    currency: str | None = None

    def has(self, dimension: str) -> bool:
        return bool(self.evidence.get(dimension))


class ResumeInput(StrictModel):
    document: ResumeSchema
    facts: Facts = Field(default_factory=Facts)


class JobInput(StrictModel):
    document: JobDescriptionSchema
    facts: Facts = Field(default_factory=Facts)


class SalaryPreference(StrictModel):
    mode: Literal["negotiable", "explicit"] = "negotiable"
    minimum: float | None = Field(default=None, gt=0)
    currency: str | None = None
    period: Literal["monthly", "annual"] = "monthly"

    @model_validator(mode="after")
    def explicit_amount(self):
        if self.mode == "explicit" and (self.minimum is None or not self.currency):
            raise ValueError("Explicit salary requires minimum and currency")
        return self


class ScorerConfig(StrictModel):
    version: str = RULE_VERSION
    lower_threshold: int = Field(default=50, ge=0, le=100)
    upper_threshold: int = Field(default=75, ge=0, le=100)
    neutral: int = Field(default=50, ge=0, le=100)
    salary_neutral: int = Field(default=75, ge=0, le=100)
    strong_skill_min: int = 70
    strong_title_min: int = 70

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if self.lower_threshold >= self.upper_threshold:
            raise ValueError("lower_threshold must be less than upper_threshold")
        return self


class DimensionResult(StrictModel):
    score: int = Field(ge=0, le=100)
    supported: bool
    basis: str
    rule_id: str
    evidence: list[str] = Field(default_factory=list)


class ScoreResult(StrictModel):
    scores: ATSDimensionalScores
    overall_score: float
    grade: int = Field(ge=0, le=2)
    coverage: float = Field(ge=0, le=1)
    dimensions: dict[str, DimensionResult]
    matched_skills: list[str]
    missing_skills: list[str]
    preferred_skill_coverage: float | None = None
    config: ScorerConfig
    as_of_date: str


def runtime_grade(overall: float, skills: int, title: int, config: ScorerConfig) -> int:
    if overall >= config.upper_threshold and skills >= config.strong_skill_min and title >= config.strong_title_min:
        return 2
    return 1 if overall >= config.lower_threshold else 0


def supported_coverage(dimensions: dict[str, DimensionResult]) -> float:
    return round(sum(ATS_WEIGHTS[k] for k, d in dimensions.items() if d.supported), 4)
