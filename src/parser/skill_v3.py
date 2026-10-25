"""Skill-local JD depth and specialization evidence; importance stays separate."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from .skill_v2 import CVSkillDetail, RequirementUnit


ProficiencyV3 = Literal["normal", "intermediate", "advanced", "expert"]


def normalize_beginner(value, field="proficiency"):
    """D3-30: retain literal wording while removing Beginner from v3 scoring."""
    if isinstance(value, dict) and str(value.get(field, "")).strip().lower() == "beginner":
        value = dict(value)
        value["raw_proficiency"] = value.get("raw_proficiency") or value[field]
        value[field] = "normal"
    return value


class CVSkillDetailV3(CVSkillDetail):
    proficiency: ProficiencyV3 = "normal"

    @model_validator(mode="before")
    @classmethod
    def beginner_is_unspecified(cls, value):
        return normalize_beginner(value)


class SkillExperienceEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    min_years: float | None = Field(default=None, ge=0)
    max_years: float | None = Field(default=None, ge=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_years(self):
        if self.min_years is not None and self.max_years is not None and self.min_years > self.max_years:
            raise ValueError("Skill experience years must be ordered")
        return self


class RequirementOption(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    minimum_proficiency: ProficiencyV3 = "normal"
    raw_proficiency: str | None = None
    proficiency_quote: str | None = None
    experience: list[SkillExperienceEvidence] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def beginner_is_unspecified(cls, value):
        return normalize_beginner(value, "minimum_proficiency")

    @model_validator(mode="after")
    def explicit_minimum(self):
        if self.minimum_proficiency != "normal" and not (self.raw_proficiency and self.proficiency_quote):
            raise ValueError("Explicit JD minimum requires raw wording and a skill-local quote")
        return self


class RequirementUnitV3(RequirementUnit):
    # Keep string options for compatibility with v1/v2. Details identify each
    # option explicitly, so different alternatives can have different minima.
    option_details: list[RequirementOption] = Field(default_factory=list)

    @model_validator(mode="after")
    def option_depth_identity(self):
        from src.scoring.v3_aliases import skill_name
        options = {skill_name(name) for name in self.options}
        names = [skill_name(detail.name) for detail in self.option_details]
        if len(names) != len(set(names)) or any(name not in options for name in names):
            raise ValueError("JD option details must uniquely identify an existing alternative")
        return self
