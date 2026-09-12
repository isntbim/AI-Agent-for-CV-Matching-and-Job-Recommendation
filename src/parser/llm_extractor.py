"""
llm_extractor.py — Tier 2 LLM-Based CV Structured Extraction

Sends cleaned CV text to a Large Language Model (Qwen2.5-7B-Instruct) and
parses the JSON response into a validated ResumeSchema Pydantic object.

Architecture:
    LLMClient (Protocol) — Abstract interface for any LLM backend.
    VLLMClient           — Calls a vLLM server (/v1/chat/completions, OpenAI-compatible).
    OllamaClient         — Calls an Ollama server (/api/generate).
    MockLLMClient        — Returns fixture JSON for testing without a real LLM.

    CVExtractionPrompt   — Builds the system/user prompt sent to the LLM.
    extract_resume_from_text() — Public API: text + contacts → ResumeSchema.

Usage:
    # Production (vLLM)
    client = VLLMClient(base_url="http://localhost:8000", model="Qwen2.5-7B-Instruct")
    resume = extract_resume_from_text(raw_text, contacts, client)

    # Testing (Mock)
    client = MockLLMClient()
    resume = extract_resume_from_text(raw_text, contacts, client)
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

import httpx
from loguru import logger

from src.parser.schemas import (
    Basics,
    Certificate,
    Education,
    Language,
    Location,
    Profile,
    Project,
    ResumeSchema,
    Skill,
    WorkExperience,
)
from src.parser.regex_utils import ContactInfo

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ExtractionError(Exception):
    """Raised when the LLM response cannot be parsed into a ResumeSchema."""


class LLMConnectionError(ExtractionError):
    """Raised when the LLM endpoint is unreachable."""


# ---------------------------------------------------------------------------
# LLMClient Protocol (Plan B: swappable backend interface)
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """
    Abstract interface for LLM backends.

    Any class implementing generate() can be used as a drop-in backend.
    This allows switching between vLLM, Ollama, and Mock without changing
    the calling code.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            system_prompt: System instruction for the LLM role/behavior.
            user_prompt:   User message containing CV text.
            temperature:   Sampling temperature (0.0 = deterministic).
            max_tokens:    Maximum number of tokens to generate.

        Returns:
            Raw string response from the LLM (expected to be valid JSON).

        Raises:
            LLMConnectionError: If the server is unreachable.
            ExtractionError: If the LLM returns an unexpected error.
        """
        ...


# ---------------------------------------------------------------------------
# VLLMClient — Production Backend
# ---------------------------------------------------------------------------

class VLLMClient:
    """
    LLM client for a vLLM server using OpenAI-compatible /v1/chat/completions API.

    Forces JSON output via response_format={"type": "json_object"} (Guided Decoding).

    Args:
        base_url: vLLM server base URL, e.g. "http://localhost:8000".
        model:    Model name as registered in vLLM, e.g. "Qwen2.5-7B-Instruct".
        timeout:  HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "Qwen2.5-7B-Instruct",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/v1/chat/completions"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = httpx.post(
                self._endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
            raise LLMConnectionError(
                f"Cannot reach vLLM at {self._endpoint}. Is the server running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ExtractionError(
                f"vLLM returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        data = response.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# OllamaClient — Local Fallback Backend
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    LLM client for an Ollama server using /api/generate API.

    Args:
        base_url: Ollama server base URL, e.g. "http://localhost:11434".
        model:    Ollama model tag, e.g. "qwen2.5:7b-instruct".
        timeout:  HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._endpoint = f"{self.base_url}/api/generate"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = httpx.post(
                self._endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
            raise LLMConnectionError(
                f"Cannot reach Ollama at {self._endpoint}. Is Ollama running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ExtractionError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        data = response.json()
        return data.get("response", "")


# ---------------------------------------------------------------------------
# MockLLMClient — Testing Backend (Plan B)
# ---------------------------------------------------------------------------

_MOCK_RESUME_JSON: dict[str, Any] = {
    "basics": {
        "name": "Nguyễn Văn Mock",
        "label": "AI Engineer",
        "email": "mock@example.com",
        "phone": "+84900000000",
        "url": "https://mock.dev",
        "summary": "Experienced AI Engineer with focus on LLMs and RAG systems.",
        "location": {
            "city": "Ho Chi Minh City",
            "countryCode": "VN",
        },
        "profiles": [
            {"network": "GitHub", "username": "mock-user", "url": "https://github.com/mock-user"},
        ],
    },
    "work": [
        {
            "name": "TechCorp VN",
            "position": "Senior AI Engineer",
            "startDate": "2023-01-01",
            "endDate": "Present",
            "summary": "Developed RAG pipelines and deployed LLM microservices.",
            "highlights": ["Reduced inference latency by 60%", "Built Qdrant vector index for 1M documents"],
        }
    ],
    "education": [
        {
            "institution": "FPT University",
            "area": "Artificial Intelligence",
            "studyType": "Bachelor",
            "startDate": "2021-09-01",
            "endDate": "2025-09-01",
            "score": "3.5/4.0",
        }
    ],
    "skills": [
        {
            "name": "Machine Learning & AI",
            "level": "Advanced",
            "keywords": ["PyTorch", "Transformers", "vLLM", "LangChain", "Qdrant"],
        },
        {
            "name": "Backend",
            "level": "Intermediate",
            "keywords": ["Python", "FastAPI", "Docker", "Redis"],
        },
    ],
    "projects": [
        {
            "name": "CV Matching AI Agent",
            "description": "End-to-end AI agent for CV matching and job recommendation.",
            "keywords": ["Qwen2.5", "BGE-M3", "FastAPI", "Qdrant"],
            "highlights": ["NDCG@5 = 0.85 on benchmark dataset"],
            "type": "project",
        }
    ],
    "certificates": [
        {
            "name": "Deep Learning Specialization",
            "date": "2024-06-01",
            "issuer": "Coursera / DeepLearning.AI",
            "type": "certificate",
        }
    ],
    "languages": [
        {"language": "Vietnamese", "fluency": "Native"},
        {"language": "English",    "fluency": "Professional (IELTS 7.0)"},
    ],
}


class MockLLMClient:
    """
    Mock LLM client for unit testing and pipeline development (Plan B).

    Returns a pre-defined valid ResumeSchema JSON fixture without
    making any network calls. Optionally configurable to simulate errors.

    Args:
        mock_json:       Custom JSON dict to return (defaults to built-in fixture).
        raise_error:     If True, raises ExtractionError to test error handling.
        error_message:   Custom error message when raise_error=True.
    """

    def __init__(
        self,
        mock_json: dict[str, Any] | None = None,
        raise_error: bool = False,
        error_message: str = "Mock LLM error",
    ) -> None:
        self._mock_json = mock_json if mock_json is not None else _MOCK_RESUME_JSON
        self._raise_error = raise_error
        self._error_message = error_message

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        if self._raise_error:
            raise ExtractionError(self._error_message)
        logger.debug("MockLLMClient: returning fixture JSON response.")
        return json.dumps(self._mock_json, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior HR Resume Parser AI. Your task is to extract structured information from a CV/Resume text and return it as a single, valid JSON object.

OUTPUT RULES (STRICT):
1. Return ONLY a JSON object. No explanations, no markdown, no code blocks.
2. Follow the schema precisely. Do not add extra fields.
3. If a field is not present in the CV, use null for optional strings or [] for lists.
4. Date rules & Strict Anti-Hallucination:
   - ONLY extract dates that explicitly appear in the CV text. NEVER invent, infer, or guess start dates or durations.
   - If a single date is listed (e.g. graduation date "May 2026"), put it in `endDate` ("2026-05") and set `startDate` to null.
   - Use YYYY-MM or YYYY-MM-DD format when possible; use "Present" for current ongoing positions.
5. Education rules:
   - Put the degree type (e.g. "Bachelor of Science", "Master", "Ph.D.") in `studyType`.
   - Put the major AND any minors, concentrations, or double majors in `area` (e.g. "Computer Science with Math Minor").
   - If only graduation or expected graduation date is written, set `endDate` and keep `startDate: null`. Do NOT guess when the student enrolled.
6. Basics & URL rules:
   - Extract the candidate's personal portfolio, academic homepage, or website into `basics.url` (e.g. "people.tamu.edu/~bnguyen4656").
7. Awards, Hackathons & Competitions:
   - Always extract awards, hackathons, datathons, coding contests, and honors into `certificates` with `type="award"`.
   - Put competition or organizer in `issuer`, award title in `name`, and year/date in `date`.
8. Projects:
   - Extract academic, personal, or open-source projects into `projects` with `type="project"`. Merge research papers/publications with `type="publication"`.
9. Skills:
   - Group related technologies under a descriptive domain name (e.g. "Languages", "Frontend", "Backend", "Tools/Practices", "Soft Skills").
   - Keep each individual technology or tool granular.

JSON SCHEMA:
{
  "basics": {
    "name": "string (required)",
    "label": "string | null",
    "email": "string | null",
    "phone": "string | null",
    "url": "string | null",
    "summary": "string | null",
    "location": { "address": "string|null", "city": "string|null", "countryCode": "string|null", "region": "string|null" } | null,
    "profiles": [{ "network": "string", "username": "string|null", "url": "string|null" }]
  },
  "work": [{ "name": "string", "position": "string|null", "startDate": "string|null", "endDate": "string|null", "summary": "string|null", "highlights": ["string"] }],
  "education": [{ "institution": "string", "area": "string|null", "studyType": "string|null", "startDate": "string|null", "endDate": "string|null", "score": "string|null", "courses": ["string"] }],
  "skills": [{ "name": "string", "level": "string|null", "keywords": ["string"] }],
  "projects": [{ "name": "string", "description": "string|null", "startDate": "string|null", "endDate": "string|null", "highlights": ["string"], "keywords": ["string"], "url": "string|null", "type": "project|publication" }],
  "certificates": [{ "name": "string", "date": "string|null", "issuer": "string|null", "url": "string|null", "type": "certificate|award" }],
  "languages": [{ "language": "string", "fluency": "string|null" }]
}"""


class CVExtractionPrompt:
    """
    Builds the system and user prompts for the CV extraction LLM call.

    The user prompt injects:
    1. Pre-extracted contact info (from regex_utils) as hints for the basics section.
    2. The full cleaned CV text.
    """

    @staticmethod
    def build_user_prompt(raw_text: str, contacts: ContactInfo | None = None) -> str:
        """
        Build the user prompt combining contact hints and CV text.

        Args:
            raw_text: Cleaned CV text from the extractor.
            contacts: Pre-extracted ContactInfo from regex_utils (optional).

        Returns:
            Formatted user prompt string.
        """
        contact_hint = ""
        if contacts:
            hints = []
            if contacts.get("emails"):
                hints.append(f"Emails found: {', '.join(contacts['emails'])}")
            if contacts.get("phones"):
                hints.append(f"Phones found (E.164): {', '.join(contacts['phones'])}")
            if contacts.get("linkedin"):
                hints.append(f"LinkedIn: {', '.join(contacts['linkedin'])}")
            if contacts.get("github"):
                hints.append(f"GitHub: {', '.join(contacts['github'])}")
            if contacts.get("urls"):
                other_urls = [
                    u for u in contacts["urls"]
                    if "linkedin" not in u and "github" not in u
                ]
                if other_urls:
                    hints.append(f"Other URLs: {', '.join(other_urls[:3])}")
            if hints:
                contact_hint = (
                    "CONTACT HINTS (pre-extracted, use these to fill basics fields accurately):\n"
                    + "\n".join(f"  - {h}" for h in hints)
                    + "\n\n"
                )

        return (
            f"{contact_hint}"
            f"CV TEXT TO PARSE:\n"
            f"{'=' * 60}\n"
            f"{raw_text[:12000]}\n"  # Truncate to avoid context overflow
            f"{'=' * 60}\n\n"
            f"Extract the structured JSON from the CV text above."
        )


# ---------------------------------------------------------------------------
# JSON Parsing Helper
# ---------------------------------------------------------------------------

def _extract_json_from_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response, handling common formatting quirks.

    Tries:
    1. Direct json.loads() on the full response.
    2. Extract JSON from ```json ... ``` markdown code blocks.
    3. Find the first {...} block in the response.

    Raises:
        ExtractionError: If no valid JSON object can be found.
    """
    text = response.strip()

    # Attempt 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown code block
    md_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Attempt 3: Find first { ... } block
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ExtractionError(
        f"LLM response does not contain valid JSON. "
        f"First 200 chars: {response[:200]!r}"
    )


def _sanitize_llm_json(data: Any) -> Any:
    """
    Recursively sanitize LLM JSON output before Pydantic validation:
    - Replace None with [] for fields expected to be lists
    - Replace None with fallback strings for entity names
    """
    if isinstance(data, dict):
        cleaned = {}
        for k, v in data.items():
            if k in ("courses", "highlights", "keywords", "profiles", "work", "education", "skills", "projects", "certificates", "languages"):
                cleaned[k] = [] if v is None else _sanitize_llm_json(v)
            elif k in ("institution", "name") and v is None:
                cleaned[k] = "Unknown"
            else:
                cleaned[k] = _sanitize_llm_json(v)
        return cleaned
    elif isinstance(data, list):
        return [_sanitize_llm_json(item) for item in data if item is not None]
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_resume_from_text(
    raw_text: str,
    contacts: ContactInfo | None = None,
    client: LLMClient | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ResumeSchema:
    """
    Extract a structured ResumeSchema from raw CV text using an LLM.

    This is the primary Tier 2 extraction function. It:
    1. Builds system + user prompts (injecting pre-extracted contact hints).
    2. Calls the LLM backend (vLLM / Ollama / Mock).
    3. Parses the JSON response.
    4. Validates the parsed dict through Pydantic ResumeSchema.

    Args:
        raw_text:    Cleaned CV text (output from extractors.py).
        contacts:    Pre-extracted ContactInfo from regex_utils (optional but recommended).
        client:      LLMClient instance. Defaults to MockLLMClient() if None.
        temperature: LLM sampling temperature (0.0 = deterministic, recommended).
        max_tokens:  Max tokens for LLM generation.

    Returns:
        Validated ResumeSchema Pydantic object.

    Raises:
        ExtractionError: If LLM response cannot be parsed or fails schema validation.
        LLMConnectionError: If the LLM server is unreachable.

    Examples:
        # With Mock (testing)
        resume = extract_resume_from_text(raw_text, contacts)

        # With vLLM (production)
        client = VLLMClient(base_url="http://localhost:8000")
        resume = extract_resume_from_text(raw_text, contacts, client)
    """
    if client is None:
        logger.debug("No LLM client provided — using MockLLMClient.")
        client = MockLLMClient()

    if not raw_text.strip():
        raise ExtractionError("Empty CV text provided — cannot extract resume.")

    # Build prompts
    user_prompt = CVExtractionPrompt.build_user_prompt(raw_text, contacts)

    logger.info(
        f"Calling LLM ({type(client).__name__}) for CV extraction "
        f"[{len(raw_text)} chars]..."
    )

    # Call LLM
    response = client.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    logger.debug(f"LLM response received ({len(response)} chars).")

    # Parse JSON from response
    parsed_dict = _extract_json_from_response(response)
    sanitized_dict = _sanitize_llm_json(parsed_dict)

    # Validate through Pydantic
    try:
        resume = ResumeSchema.model_validate(sanitized_dict)
        logger.info(
            f"Extraction successful: '{resume.basics.name}' | "
            f"{len(resume.work)} work | {len(resume.skills)} skill groups | "
            f"{len(resume.education)} education"
        )
        return resume
    except Exception as exc:
        raise ExtractionError(
            f"Pydantic validation failed for extracted JSON: {exc}\n"
            f"Raw dict keys: {list(parsed_dict.keys())}"
        ) from exc
