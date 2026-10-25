"""Source-backed skill annotations, independent of runtime scoring and embeddings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewStatus = Literal["pending", "approved", "rejected"]
Proficiency = Literal["beginner", "normal", "intermediate", "advanced", "expert"]
Importance = Literal["core", "standard", "preferred"]


class SkillProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    source_id: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None
    text_sha256: str | None = None
    review_status: ReviewStatus = "pending"


class SkillAnnotationMetadata(SkillProvenance):
    schema_version: Literal["ats-skills-v2", "ats-skills-v3"] = "ats-skills-v2"
    skills_status: Literal["complete", "partial", "missing"] = "partial"
    # Approval of skill fields does not approve every field in the envelope.
    envelope_review_status: ReviewStatus = "pending"
    reviewer: str | None = None


class CVSkillDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    proficiency: Proficiency = "normal"
    raw_proficiency: str | None = None
    skill_quote: str | None = None
    proficiency_quote: str | None = None
    provenance: SkillProvenance | None = None

    @model_validator(mode="after")
    def explicit_proficiency(self):
        if self.proficiency != "normal" and not (self.raw_proficiency and self.proficiency_quote):
            raise ValueError("Non-normal proficiency requires raw wording and a skill-local quote")
        return self


class RequirementUnit(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    options: list[str] = Field(min_length=1)
    importance: Importance = "standard"
    quote: str | None = None
    importance_quote: str | None = None
    review_status: ReviewStatus = "pending"
    issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_options(self):
        if any(not option.strip() for option in self.options):
            raise ValueError("Requirement options must be nonempty skill names")
        if self.importance == "core" and not self.importance_quote:
            raise ValueError("Core requires source evidence of mandatory importance")
        return self


class SkillRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    relation: Literal["near", "transfer"]
    reason: str = Field(min_length=1)
    source: str = Field(min_length=1)
    review_status: ReviewStatus = "pending"


class SkillRelationTable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str = Field(min_length=1)
    relations: list[SkillRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_directions(self):
        pairs = [(r.candidate.casefold(), r.requirement.casefold()) for r in self.relations]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Duplicate directed semantic relationship")
        if len({r.id for r in self.relations}) != len(self.relations):
            raise ValueError("Duplicate relationship ID")
        return self
