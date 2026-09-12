"""
pipeline.py — CV Parsing Pipeline Orchestrator

Combines all Tier 1 and Tier 2 components into a single, clean public API:
    Tier 1 (CPU-fast): File reading → text extraction → regex contact extraction
    Tier 2 (LLM):      Structured JSON extraction → Pydantic ResumeSchema validation

Usage:
    # Full pipeline with MockLLMClient (default for development/testing)
    from src.parser.pipeline import parse_cv
    resume = parse_cv("data/raw/cvs/resume.pdf")

    # Full pipeline with production vLLM backend
    from src.parser.llm_extractor import VLLMClient
    client = VLLMClient(base_url="http://localhost:8000")
    resume = parse_cv("data/raw/cvs/resume.pdf", llm_client=client)

    # Tier 1 only (no LLM) — returns raw text + contacts only
    raw_text, contacts = parse_cv_tier1("data/raw/cvs/resume.pdf")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from src.parser.extractors import PDFExtractor, DOCXExtractor, extract_text_from_file
from src.parser.regex_utils import ContactInfo, extract_all_contacts
from src.parser.llm_extractor import LLMClient, extract_resume_from_text
from src.parser.schemas import ResumeSchema


# ---------------------------------------------------------------------------
# Tier 1 Only (CPU, no LLM)
# ---------------------------------------------------------------------------

def parse_cv_tier1(file_path: str | Path) -> tuple[str, ContactInfo]:
    """
    Run only the CPU-fast Tier 1 pipeline on a CV file.

    Extracts raw text (with correct reading order for 2-column layouts)
    and pre-extracts contact information using regex patterns.

    Args:
        file_path: Path to a .pdf or .docx CV file.

    Returns:
        Tuple of (raw_text: str, contacts: ContactInfo).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is not supported.

    Examples:
        >>> raw_text, contacts = parse_cv_tier1("data/raw/cvs/resume.pdf")
        >>> print(contacts["emails"])
    """
    path = Path(file_path)
    logger.info(f"[Tier 1] Starting extraction: {path.name}")

    # Stage 1: Extract text from file
    raw_text = extract_text_from_file(path)
    logger.info(f"[Tier 1] Text extracted: {len(raw_text)} chars")

    # Stage 2: Extract contact info via regex
    contacts = extract_all_contacts(raw_text)
    logger.info(
        f"[Tier 1] Contacts found — "
        f"emails: {len(contacts['emails'])}, "
        f"phones: {len(contacts['phones'])}, "
        f"linkedin: {len(contacts['linkedin'])}, "
        f"github: {len(contacts['github'])}"
    )

    return raw_text, contacts


# ---------------------------------------------------------------------------
# Full Pipeline (Tier 1 + Tier 2 LLM)
# ---------------------------------------------------------------------------

def parse_cv(
    file_path: str | Path,
    llm_client: Optional[LLMClient] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ResumeSchema:
    """
    Full CV parsing pipeline: file → ResumeSchema.

    Executes in 3 stages:
        1. [Tier 1] Extract raw text from PDF/DOCX (handles 2-column layouts).
        2. [Tier 1] Extract contact info via regex (email, phone, URLs).
        3. [Tier 2] LLM extraction → validated ResumeSchema.

    Args:
        file_path:   Path to a .pdf or .docx CV file.
        llm_client:  LLMClient backend instance.
                     Defaults to MockLLMClient() for testing/development.
                     Use VLLMClient or OllamaClient for production.
        temperature: LLM sampling temperature (default 0.0 = deterministic).
        max_tokens:  Maximum tokens for LLM generation (default 4096).

    Returns:
        Validated ResumeSchema Pydantic object.

    Raises:
        FileNotFoundError: If the CV file does not exist.
        ValueError: If the file type is not supported (.pdf or .docx only).
        ExtractionError: If the LLM fails to produce a valid ResumeSchema.
        LLMConnectionError: If the LLM server is unreachable.

    Examples:
        # Development (MockLLMClient):
        resume = parse_cv("data/raw/cvs/Nguyen_Bao_resume.pdf")
        print(resume.basics.name)
        print(resume.get_flat_skills())

        # Production (vLLM):
        from src.parser.llm_extractor import VLLMClient
        client = VLLMClient("http://localhost:8000")
        resume = parse_cv("data/raw/cvs/resume.pdf", llm_client=client)
    """
    path = Path(file_path)
    logger.info(f"[Pipeline] parse_cv started: {path.name}")

    # --- Stage 1 & 2: Tier 1 ---
    raw_text, contacts = parse_cv_tier1(path)

    if not raw_text.strip():
        logger.warning(f"[Pipeline] No text extracted from '{path.name}'. Possible scan/image PDF.")

    # --- Stage 3: Tier 2 LLM ---
    logger.info(f"[Pipeline] Starting Tier 2 LLM extraction...")
    resume = extract_resume_from_text(
        raw_text=raw_text,
        contacts=contacts,
        client=llm_client,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.info(
        f"[Pipeline] Done: '{resume.basics.name}' | "
        f"skills: {len(resume.get_flat_skills())} | "
        f"work: {len(resume.work)} | "
        f"education: {len(resume.education)}"
    )
    return resume


# ---------------------------------------------------------------------------
# Batch Processing Utility
# ---------------------------------------------------------------------------

def batch_parse_cvs(
    cv_dir: str | Path,
    llm_client: Optional[LLMClient] = None,
    extensions: tuple[str, ...] = (".pdf", ".docx"),
    max_files: Optional[int] = None,
) -> list[tuple[str, ResumeSchema | Exception]]:
    """
    Parse multiple CV files from a directory.

    Args:
        cv_dir:     Directory containing CV files.
        llm_client: LLMClient backend. Defaults to MockLLMClient().
        extensions: File extensions to process (default: .pdf and .docx).
        max_files:  Optional cap on number of files to process.

    Returns:
        List of (filename, ResumeSchema | Exception) tuples.
        On failure for a specific file, the Exception is captured and returned
        rather than stopping the entire batch.

    Examples:
        results = batch_parse_cvs("data/raw/cvs/Software Engineer/", max_files=5)
        for filename, result in results:
            if isinstance(result, Exception):
                print(f"FAILED {filename}: {result}")
            else:
                print(f"OK {filename}: {result.basics.name}")
    """
    dir_path = Path(cv_dir)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    files = [
        f for f in sorted(dir_path.iterdir())
        if f.is_file() and f.suffix.lower() in extensions
    ]

    if max_files is not None:
        files = files[:max_files]

    logger.info(f"[Batch] Processing {len(files)} CV files from '{dir_path.name}'...")

    results: list[tuple[str, ResumeSchema | Exception]] = []
    for i, file_path in enumerate(files, start=1):
        logger.info(f"[Batch] [{i}/{len(files)}] {file_path.name}")
        try:
            resume = parse_cv(file_path, llm_client=llm_client)
            results.append((file_path.name, resume))
        except Exception as exc:
            logger.error(f"[Batch] FAILED '{file_path.name}': {exc}")
            results.append((file_path.name, exc))

    success = sum(1 for _, r in results if isinstance(r, ResumeSchema))
    logger.info(f"[Batch] Complete: {success}/{len(files)} succeeded.")
    return results
