"""
src/parser — CV Parsing Module

Public API:
    ResumeSchema             — Pydantic CV data model (JSON Resume standard)
    extract_text_from_file() — Extract raw text from PDF/DOCX (handles 2-column layouts)
    extract_all_contacts()   — Regex-based contact info extraction
    parse_cv()               — Full pipeline: file → ResumeSchema
    parse_cv_tier1()         — Tier 1 only: file → (raw_text, contacts)
    batch_parse_cvs()        — Batch process a directory of CV files

LLM Backends (for parse_cv's llm_client parameter):
    MockLLMClient            — Test/development backend (no server needed)
    VLLMClient               — Production backend (vLLM server)
    OllamaClient             — Local fallback backend (Ollama server)
"""

from src.parser.schemas import ResumeSchema
from src.parser.extractors import extract_text_from_file
from src.parser.regex_utils import extract_all_contacts, ContactInfo
from src.parser.llm_extractor import (
    LLMClient,
    MockLLMClient,
    VLLMClient,
    OllamaClient,
    ExtractionError,
    LLMConnectionError,
)
from src.parser.pipeline import parse_cv, parse_cv_tier1, batch_parse_cvs

__all__ = [
    # Schema
    "ResumeSchema",
    # Extractors
    "extract_text_from_file",
    # Regex utils
    "extract_all_contacts",
    "ContactInfo",
    # LLM extractor
    "LLMClient",
    "MockLLMClient",
    "VLLMClient",
    "OllamaClient",
    "ExtractionError",
    "LLMConnectionError",
    # Pipeline
    "parse_cv",
    "parse_cv_tier1",
    "batch_parse_cvs",
]
