"""
conftest.py — Shared pytest fixtures for the test suite

Provides reusable fixtures for unit and integration tests across
the entire src/parser module.
"""

import json
from pathlib import Path
import pytest

from src.parser.schemas import ResumeSchema
from src.parser.regex_utils import ContactInfo

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_CVS = PROJECT_ROOT / "data" / "raw" / "cvs"


# ---------------------------------------------------------------------------
# Sample CV Text Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cv_text_vn() -> str:
    """Typical Vietnamese CV header with contact info and sections."""
    return (
        "LÊ TRÍ DŨNG\n"
        "AI Engineer | Ho Chi Minh City, Vietnam\n"
        "Email: tridungit2005@gmail.com | Điện thoại: 0941 423 545\n"
        "GitHub: github.com/isntbim | LinkedIn: linkedin.com/in/tri-dung\n"
        "Portfolio: https://tridung.dev\n\n"
        "TÓM TẮT\n"
        "AI Engineer với 2 năm kinh nghiệm trong lĩnh vực NLP, LLM, và RAG systems.\n\n"
        "KINH NGHIỆM LÀM VIỆC\n"
        "FPT Software — AI Engineer (01/2025 - Hiện tại)\n"
        "- Xây dựng pipeline RAG phục vụ 100k users/ngày\n"
        "- Triển khai vLLM serving với throughput 50 req/s\n\n"
        "HỌC VẤN\n"
        "Đại học FPT — Kỹ thuật Phần mềm (2021 - 2025) | GPA: 3.7/4.0\n\n"
        "KỸ NĂNG\n"
        "Python, PyTorch, Transformers, vLLM, LangChain, FastAPI, Docker, Qdrant\n\n"
        "DỰ ÁN\n"
        "CV Matching AI Agent — NLP, RAG, Qdrant (2026)\n"
        "- NDCG@5 = 0.85 trên tập benchmark tuyển dụng\n\n"
        "CHỨNG CHỈ\n"
        "Deep Learning Specialization — Coursera (2024)\n\n"
        "NGOẠI NGỮ\n"
        "Tiếng Anh: IELTS 7.0 | Tiếng Việt: Bản ngữ\n"
    )


@pytest.fixture
def sample_cv_text_en() -> str:
    """Typical English CV (international format)."""
    return (
        "John Doe\n"
        "Senior Software Engineer | San Francisco, CA\n"
        "john.doe@gmail.com | https://johndoe.dev | github.com/johndoe\n\n"
        "SUMMARY\n"
        "Full-stack engineer with 5+ years building scalable web systems.\n\n"
        "EXPERIENCE\n"
        "Google LLC — Senior SWE (2022 - Present)\n"
        "- Led team of 8 engineers on Ads infrastructure\n"
        "- Reduced query latency by 40%\n\n"
        "EDUCATION\n"
        "Stanford University — M.S. Computer Science (2018 - 2020)\n\n"
        "SKILLS\n"
        "Python, Go, Java, Kubernetes, GCP, PostgreSQL, Redis\n"
    )


# ---------------------------------------------------------------------------
# ContactInfo Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_contacts_vn() -> ContactInfo:
    """Pre-extracted ContactInfo for the Vietnamese CV fixture."""
    return ContactInfo(
        emails=["tridungit2005@gmail.com"],
        phones=["+84941423545"],
        urls=["https://tridung.dev"],
        linkedin=["linkedin.com/in/tri-dung"],
        github=["github.com/isntbim"],
    )


@pytest.fixture
def sample_contacts_en() -> ContactInfo:
    """Pre-extracted ContactInfo for the English CV fixture."""
    return ContactInfo(
        emails=["john.doe@gmail.com"],
        phones=[],
        urls=["https://johndoe.dev"],
        linkedin=[],
        github=["github.com/johndoe"],
    )


@pytest.fixture
def empty_contacts() -> ContactInfo:
    return ContactInfo(emails=[], phones=[], urls=[], linkedin=[], github=[])


# ---------------------------------------------------------------------------
# ResumeSchema Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_resume_dict() -> dict:
    """Minimal valid dict that passes ResumeSchema validation."""
    return {
        "basics": {
            "name": "Test User",
            "label": "Software Engineer",
            "email": "test@example.com",
            "phone": "+84900000000",
            "summary": "Experienced SE.",
            "location": {"city": "Ho Chi Minh City", "countryCode": "VN"},
            "profiles": [
                {"network": "GitHub", "username": "testuser", "url": "https://github.com/testuser"}
            ],
        },
        "work": [
            {
                "name": "TestCorp",
                "position": "Software Engineer",
                "startDate": "2023-01-01",
                "endDate": "Present",
                "highlights": ["Built feature X", "Improved performance by 30%"],
            }
        ],
        "education": [
            {
                "institution": "FPT University",
                "area": "Computer Science",
                "studyType": "Bachelor",
                "startDate": "2019-09-01",
                "endDate": "2023-09-01",
                "score": "3.5/4.0",
            }
        ],
        "skills": [
            {
                "name": "Backend",
                "level": "Advanced",
                "keywords": ["Python", "FastAPI", "Docker", "Redis"],
            }
        ],
        "projects": [
            {
                "name": "Test Project",
                "description": "A test project.",
                "keywords": ["Python", "FastAPI"],
                "highlights": ["Won hackathon"],
                "type": "project",
            }
        ],
        "certificates": [
            {
                "name": "AWS Solutions Architect",
                "date": "2024-01-01",
                "issuer": "Amazon",
                "type": "certificate",
            }
        ],
        "languages": [
            {"language": "Vietnamese", "fluency": "Native"},
            {"language": "English", "fluency": "Professional"},
        ],
    }


@pytest.fixture
def sample_resume(sample_resume_dict) -> ResumeSchema:
    """Validated ResumeSchema object from sample dict."""
    return ResumeSchema.model_validate(sample_resume_dict)


# ---------------------------------------------------------------------------
# Dataset File Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def first_pdf_in_dataset() -> Path | None:
    """Returns the first available PDF from the dataset, or None if empty."""
    files = list(DATA_RAW_CVS.rglob("*.pdf"))
    return files[0] if files else None


@pytest.fixture
def pdf_files_sample() -> list[Path]:
    """Returns up to 5 PDF files from the dataset for smoke tests."""
    return list(DATA_RAW_CVS.rglob("*.pdf"))[:5]
