"""
JSON Resume Schema Models (Pydantic v2)
Optimized for AI Agent CV Matching & Job Recommendation System.
Adheres to JSON Resume standard (jsonresume.org) with streamlined fields for RAG & Matching.
Robust against null/empty LLM-generated values.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseSchemaModel(BaseModel):
    """Base configuration for all parser models allowing alias and field name access."""
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        str_strip_whitespace=True
    )


class Location(BaseSchemaModel):
    address: Optional[str] = Field(default=None, description="Street address")
    postal_code: Optional[str] = Field(default=None, alias="postalCode", description="Postal / Zip code")
    city: Optional[str] = Field(default=None, description="City name, e.g. Ho Chi Minh City, Ha Noi")
    country_code: Optional[str] = Field(default=None, alias="countryCode", description="Country code, e.g. VN")
    region: Optional[str] = Field(default=None, description="Region or Province")


class Profile(BaseSchemaModel):
    network: Optional[str] = Field(default=None, description="Platform name, e.g. GitHub, LinkedIn")
    username: Optional[str] = Field(default=None, description="Username or handle")
    url: Optional[str] = Field(default=None, description="Direct URL link to profile")


class Basics(BaseSchemaModel):
    name: str = Field(default="Candidate", description="Candidate full name")
    label: Optional[str] = Field(default=None, description="Target job title or profession, e.g. AI Engineer, Fullstack Developer")
    email: Optional[str] = Field(default=None, description="Contact email address")
    phone: Optional[str] = Field(default=None, description="Contact phone number")
    url: Optional[str] = Field(default=None, description="Personal website or portfolio URL")
    summary: Optional[str] = Field(default=None, description="Short professional summary / objective")
    location: Optional[Location] = Field(default=None, description="Candidate location details")
    profiles: List[Profile] = Field(default_factory=list, description="List of social / professional profiles")

    @field_validator("name", mode="before")
    @classmethod
    def coerce_name(cls, v):
        return v or "Candidate"

    @field_validator("profiles", mode="before")
    @classmethod
    def coerce_profiles(cls, v):
        return v or []


class WorkExperience(BaseSchemaModel):
    name: str = Field(default="Organization", description="Company or employer organization name")
    position: Optional[str] = Field(default=None, description="Job title / Role held")
    url: Optional[str] = Field(default=None, description="Company website URL")
    start_date: Optional[str] = Field(default=None, alias="startDate", description="Start date (YYYY-MM or YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, alias="endDate", description="End date or 'Present'")
    summary: Optional[str] = Field(default=None, description="Overview of duties and responsibilities")
    highlights: List[str] = Field(default_factory=list, description="Key achievements, bullets, or metrics delivered")

    @field_validator("name", mode="before")
    @classmethod
    def coerce_work_name(cls, v):
        return v or "Organization"

    @field_validator("highlights", mode="before")
    @classmethod
    def coerce_highlights(cls, v):
        return v or []


class Education(BaseSchemaModel):
    institution: str = Field(default="Educational Institution", description="University, college, or educational institution name")
    url: Optional[str] = Field(default=None, description="Institution website")
    area: Optional[str] = Field(default=None, description="Field of study or major, e.g. Computer Science, Information Technology")
    study_type: Optional[str] = Field(default=None, alias="studyType", description="Degree level: Bachelor, Master, Engineer, High School")
    start_date: Optional[str] = Field(default=None, alias="startDate", description="Start date")
    end_date: Optional[str] = Field(default=None, alias="endDate", description="Graduation / Completion date")
    score: Optional[str] = Field(default=None, description="GPA, grade or honors, e.g. 3.6/4.0, Good")
    courses: List[str] = Field(default_factory=list, description="Notable completed courses")

    @field_validator("institution", mode="before")
    @classmethod
    def coerce_institution(cls, v):
        return v or "Educational Institution"

    @field_validator("courses", mode="before")
    @classmethod
    def coerce_courses(cls, v):
        return v or []


class Skill(BaseSchemaModel):
    name: str = Field(default="General", description="Skill group or domain name, e.g. Backend Development, Machine Learning, Cloud")
    level: Optional[str] = Field(default=None, description="Proficiency level: Beginner, Intermediate, Advanced, Master")
    keywords: List[str] = Field(default_factory=list, description="Specific tech keywords / tools, e.g. ['Python', 'FastAPI', 'Docker']")

    @field_validator("name", mode="before")
    @classmethod
    def coerce_skill_name(cls, v):
        return v or "General"

    @field_validator("keywords", mode="before")
    @classmethod
    def coerce_keywords(cls, v):
        return v or []


class Project(BaseSchemaModel):
    name: str = Field(default="Project", description="Project name or publication title")
    start_date: Optional[str] = Field(default=None, alias="startDate", description="Start date")
    end_date: Optional[str] = Field(default=None, alias="endDate", description="End date")
    description: Optional[str] = Field(default=None, description="Overview description of the project")
    highlights: List[str] = Field(default_factory=list, description="Key contributions, metrics, or technical achievements")
    keywords: List[str] = Field(default_factory=list, description="Technologies / stack utilized")
    url: Optional[str] = Field(default=None, description="Live URL or GitHub repository")
    project_type: Optional[str] = Field(default="project", alias="type", description="Classification: 'project' or 'publication'")

    @field_validator("name", mode="before")
    @classmethod
    def coerce_project_name(cls, v):
        return v or "Project"

    @field_validator("highlights", "keywords", mode="before")
    @classmethod
    def coerce_project_lists(cls, v):
        return v or []


class Certificate(BaseSchemaModel):
    name: str = Field(default="Certificate", description="Certificate name or award title")
    date: Optional[str] = Field(default=None, description="Date achieved or issued")
    issuer: Optional[str] = Field(default=None, description="Issuing body, platform, or contest, e.g. AWS, Coursera, Hackathon")
    url: Optional[str] = Field(default=None, description="Verification URL")
    certificate_type: Optional[str] = Field(default="certificate", alias="type", description="Classification: 'certificate' or 'award'")

    @field_validator("name", mode="before")
    @classmethod
    def coerce_cert_name(cls, v):
        return v or "Certificate"


class Language(BaseSchemaModel):
    language: str = Field(default="English", description="Language name, e.g. Vietnamese, English, Japanese")
    fluency: Optional[str] = Field(default=None, description="Fluency descriptor, e.g. Native, Fluent, IELTS 7.5, Professional")

    @field_validator("language", mode="before")
    @classmethod
    def coerce_language(cls, v):
        return v or "Language"


class ResumeSchema(BaseSchemaModel):
    """
    Root candidate resume schema.
    Provides utility methods to produce optimized text for BGE-M3 embeddings
    and metadata payloads for Qdrant vector database.
    """
    basics: Basics = Field(default_factory=Basics)
    work: List[WorkExperience] = Field(default_factory=list, description="Work and employment history")
    education: List[Education] = Field(default_factory=list, description="Educational background")
    skills: List[Skill] = Field(default_factory=list, description="Categorized technical & professional skills")
    projects: List[Project] = Field(default_factory=list, description="Notable academic or personal projects")
    certificates: List[Certificate] = Field(default_factory=list, description="Certificates and competitive awards")
    languages: List[Language] = Field(default_factory=list, description="Spoken / written languages")
    summary_text: Optional[str] = Field(default=None, description="Pre-computed text summary for dense embedding")

    @field_validator("work", "education", "skills", "projects", "certificates", "languages", mode="before")
    @classmethod
    def coerce_schema_lists(cls, v):
        return v or []

    def get_flat_skills(self) -> List[str]:
        """Returns a flat, deduplicated list of all technical skill keywords."""
        flat = []
        for s in self.skills:
            for kw in s.keywords:
                clean_kw = kw.strip()
                if clean_kw and clean_kw.lower() not in [x.lower() for x in flat]:
                    flat.append(clean_kw)
        return flat

    def to_embedding_text(self) -> str:
        """
        Synthesizes core candidate strengths into an information-dense paragraph
        optimized for dense embedding models (e.g. BGE-M3 / Qwen Embedding).
        """
        parts = []

        # 1. Candidate target role & summary
        title = self.basics.label or "Professional"
        parts.append(f"Candidate Target Role: {title}.")
        if self.basics.summary:
            parts.append(f"Summary: {self.basics.summary}")

        # 2. Key Skills
        flat_skills = self.get_flat_skills()
        if flat_skills:
            parts.append(f"Core Skills & Technologies: {', '.join(flat_skills)}.")

        # 3. Work Experience Highlights
        if self.work:
            work_summaries = []
            for w in self.work[:3]:  # prioritize 3 most recent
                role = w.position or "Team Member"
                comp = w.name
                desc = f"{role} at {comp}"
                if w.highlights:
                    desc += f" ({'; '.join(w.highlights[:2])})"
                work_summaries.append(desc)
            parts.append(f"Experience: {'. '.join(work_summaries)}.")

        # 4. Projects
        if self.projects:
            proj_summaries = []
            for p in self.projects[:2]:  # prioritize 2 key projects
                proj_desc = p.name
                if p.keywords:
                    proj_desc += f" using {', '.join(p.keywords)}"
                if p.highlights:
                    proj_desc += f": {'; '.join(p.highlights[:1])}"
                proj_summaries.append(proj_desc)
            parts.append(f"Key Projects: {'. '.join(proj_summaries)}.")

        # 5. Education
        if self.education:
            edu_summaries = []
            for e in self.education[:1]:
                deg = e.study_type or "Degree"
                major = e.area or "Technology"
                edu_summaries.append(f"{deg} in {major} from {e.institution}")
            parts.append(f"Education: {', '.join(edu_summaries)}.")

        return "\n".join(parts)
