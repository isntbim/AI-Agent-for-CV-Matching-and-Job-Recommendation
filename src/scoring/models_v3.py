"""V3-only display models; legacy integer models and source hashes stay intact."""
from decimal import Decimal, ROUND_HALF_UP
from pydantic import Field

from .models import DimensionResult, ScoreResult, StrictModel

PRECISION_POLICY_VERSION = "unrounded-components-half-up-display-v3-v1"


def display_score(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


class DimensionResultV3(DimensionResult):
    score: float = Field(ge=0, le=100)
    unrounded_score: float = Field(ge=0, le=100)


class ATSDimensionalScoresV3(StrictModel):
    skills: float = Field(ge=0, le=100)
    title: float = Field(ge=0, le=100)
    experience: float = Field(ge=0, le=100)
    education: float = Field(ge=0, le=100)
    industry: float = Field(ge=0, le=100)
    location: float = Field(ge=0, le=100)
    salary: float = Field(ge=0, le=100)


class ScoreResultV3(ScoreResult):
    scores: ATSDimensionalScoresV3
    dimensions: dict[str, DimensionResultV3]
    unrounded_scores: ATSDimensionalScoresV3
    unrounded_overall_score: float = Field(ge=0, le=100)
    precision_policy_version: str = PRECISION_POLICY_VERSION

