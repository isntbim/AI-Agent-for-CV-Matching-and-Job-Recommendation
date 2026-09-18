"""
Job Description Schema Models (Pydantic v2)
Optimized for AI Agent CV Matching & Job Recommendation System.
Provides strict validation, serialization, and embedding text generation for BGE-M3 dense retrieval.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class BaseJobModel(BaseModel):
    """Base configuration for all job models allowing alias and field name access."""
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True
    )


class Company(BaseJobModel):
    name: str = Field(default="Confidential", description="Company or employer name")
    size: Optional[str] = Field(default=None, description="Company size, e.g. '50-200'")
    industry: Optional[str] = Field(default=None, description="Industry sector")


class JobLocation(BaseJobModel):
    city: Optional[str] = Field(default=None, description="City name, e.g. 'Ho Chi Minh City'")
    region: Optional[str] = Field(default=None, description="Region, e.g. 'South', 'North'")
    country: str = Field(default="Vietnam", description="Country name")
    country_code: str = Field(default="VN", alias="countryCode", description="ISO Alpha-2 country code")
    remote_policy: Optional[str] = Field(default=None, description="Normalized policy: 'Onsite', 'Hybrid', 'Remote'")
    remote_policy_raw: Optional[str] = Field(default=None, description="Raw working mode from job portal")


class Seniority(BaseJobModel):
    level: Optional[str] = Field(default=None, description="Normalized level: 'Intern', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Manager', 'Director'")
    level_raw: Optional[str] = Field(default=None, description="Raw seniority text from job title/description")
    min_years: Optional[int] = Field(default=None, description="Minimum years of required experience")
    max_years: Optional[int] = Field(default=None, description="Maximum years of required experience")


class SkillItem(BaseJobModel):
    name: str = Field(..., description="Skill name")
    proficiency: str = Field(default="required", description="'required' or 'preferred'")


class SkillsSection(BaseJobModel):
    required: List[SkillItem] = Field(default_factory=list, description="Mandatory required skills")
    preferred: List[SkillItem] = Field(default_factory=list, description="Nice-to-have / preferred skills")
    raw_text: Optional[str] = Field(default=None, description="Raw skills summary string")


class DescriptionSection(BaseJobModel):
    summary: Optional[str] = Field(default=None, description="Job summary or key highlight points")
    responsibilities: List[str] = Field(default_factory=list, description="List of key duties and responsibilities")
    requirements: List[str] = Field(default_factory=list, description="List of candidate requirements/qualifications")
    raw_text: Optional[str] = Field(default=None, description="Full raw description and requirement text")


class Compensation(BaseJobModel):
    currency: Optional[str] = Field(default="VND", description="Currency symbol/code: 'VND', 'USD'")
    min_monthly: Optional[int] = Field(default=None, description="Minimum monthly salary")
    max_monthly: Optional[int] = Field(default=None, description="Maximum monthly salary")
    min_annual: Optional[int] = Field(default=None, description="Minimum annual salary")
    max_annual: Optional[int] = Field(default=None, description="Maximum annual salary")
    is_negotiable: bool = Field(default=True, description="Whether salary is negotiable / undisclosed")
    raw_text: Optional[str] = Field(default=None, description="Raw salary text from job portal")


class Benefits(BaseJobModel):
    raw_text: Optional[str] = Field(default=None, description="Benefits, perks, and compensation summary")


class JobMetadata(BaseJobModel):
    posted_date: Optional[str] = Field(default=None, description="Date or relative time job was posted")
    deadline: Optional[str] = Field(default=None, description="Application deadline if applicable")
    job_type: str = Field(default="Full-time", description="Job type: 'Full-time', 'Part-time', 'Contract', 'Internship'")
    work_arrangement: Optional[str] = Field(default=None, description="Work arrangement: 'Hybrid', 'Onsite', 'Remote'")
    language: str = Field(default="en", description="Primary language: 'en', 'vi', 'mixed'")


class JobDescriptionSchema(BaseJobModel):
    """
    Standardized Job Description Schema for the AI Agent Matching System.
    Guarantees structural parity with CV representations and powers dense vector embeddings.
    """
    id: str = Field(..., description="Unique job identifier, e.g. 'jd_001'")
    source: str = Field(default="itviec", description="Origin source portal")
    url: str = Field(..., description="Direct URL link to job posting")
    scraped_at: str = Field(..., description="ISO 8601 scraping timestamp")

    title: str = Field(..., description="Advertised job title")
    title_normalized: str = Field(..., description="Standardized snake_case job role")
    category: str = Field(..., description="Domain category, e.g. 'software_engineer', 'data_scientist_analyst'")

    company: Company = Field(default_factory=Company, description="Company details")
    location: JobLocation = Field(default_factory=JobLocation, description="Location and workplace policy")
    seniority: Seniority = Field(default_factory=Seniority, description="Seniority and experience requirements")
    skills: SkillsSection = Field(default_factory=SkillsSection, description="Skills breakdown")
    description: DescriptionSection = Field(default_factory=DescriptionSection, description="Description sections")
    compensation: Compensation = Field(default_factory=Compensation, description="Compensation information")
    benefits: Benefits = Field(default_factory=Benefits, description="Perks and benefits")
    metadata: JobMetadata = Field(default_factory=JobMetadata, description="Posting metadata")

    def to_embedding_text(self) -> str:
        """
        Convert structured Job Description into dense semantic text for embedding (BGE-M3 / OpenAI).
        Mirrors ResumeSchema.to_embedding_text() for optimal bi-encoder cosine similarity.
        """
        parts = [
            f"Job Title: {self.title}",
            f"Category: {self.category}",
            f"Seniority: {self.seniority.level or 'Not specified'}" + (f" ({self.seniority.min_years}+ years)" if self.seniority.min_years else ""),
            f"Company: {self.company.name} ({self.company.industry or 'Technology'})",
            f"Location: {self.location.city or 'Vietnam'} ({self.location.remote_policy or 'Onsite'})",
            f"Required Skills: {', '.join(s.name for s in self.skills.required)}",
        ]
        if self.skills.preferred:
            parts.append(f"Preferred Skills: {', '.join(s.name for s in self.skills.preferred)}")
        if self.description.summary:
            parts.append(f"Summary: {self.description.summary}")
        if self.description.responsibilities:
            parts.append("Responsibilities:\n" + "\n".join(f"- {r}" for r in self.description.responsibilities[:6]))
        if self.description.requirements:
            parts.append("Requirements:\n" + "\n".join(f"- {r}" for r in self.description.requirements[:6]))
        return "\n".join(parts).strip()
