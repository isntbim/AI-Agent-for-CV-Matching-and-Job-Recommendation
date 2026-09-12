"""
test_llm_extractor.py — Unit tests for src/parser/llm_extractor.py

Tests all LLM client implementations (MockLLMClient, protocol compliance),
prompt builder, JSON parsing, and end-to-end extraction with schema validation.
Run with: .venv/Scripts/python.exe -m pytest tests/unit/test_llm_extractor.py -v
"""

import json
import pytest

from src.parser.llm_extractor import (
    CVExtractionPrompt,
    ExtractionError,
    LLMConnectionError,
    LLMClient,
    MockLLMClient,
    VLLMClient,
    OllamaClient,
    _MOCK_RESUME_JSON,
    _SYSTEM_PROMPT,
    _extract_json_from_response,
    extract_resume_from_text,
)
from src.parser.regex_utils import ContactInfo
from src.parser.schemas import ResumeSchema


# ---------------------------------------------------------------------------
# MockLLMClient Tests
# ---------------------------------------------------------------------------

class TestMockLLMClient:
    def test_returns_valid_json_string(self):
        client = MockLLMClient()
        result = client.generate(system_prompt="sys", user_prompt="user")
        data = json.loads(result)
        assert isinstance(data, dict)
        assert "basics" in data

    def test_returns_default_fixture(self):
        client = MockLLMClient()
        result = client.generate(system_prompt="sys", user_prompt="user")
        data = json.loads(result)
        assert data["basics"]["name"] == _MOCK_RESUME_JSON["basics"]["name"]

    def test_returns_custom_json(self):
        custom = {"basics": {"name": "Custom User"}, "work": [], "education": [],
                  "skills": [], "projects": [], "certificates": [], "languages": []}
        client = MockLLMClient(mock_json=custom)
        result = client.generate(system_prompt="sys", user_prompt="user")
        data = json.loads(result)
        assert data["basics"]["name"] == "Custom User"

    def test_raise_error_mode(self):
        client = MockLLMClient(raise_error=True, error_message="Simulated failure")
        with pytest.raises(ExtractionError, match="Simulated failure"):
            client.generate(system_prompt="sys", user_prompt="user")

    def test_implements_llm_client_protocol(self):
        """MockLLMClient must satisfy the LLMClient Protocol."""
        client = MockLLMClient()
        assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# VLLMClient & OllamaClient — Interface Tests (no real server needed)
# ---------------------------------------------------------------------------

class TestVLLMClientInterface:
    def test_initializes_correctly(self):
        client = VLLMClient(base_url="http://localhost:8000", model="Qwen2.5-7B-Instruct")
        assert client.model == "Qwen2.5-7B-Instruct"
        assert "localhost:8000" in client._endpoint

    def test_trailing_slash_stripped(self):
        client = VLLMClient(base_url="http://localhost:8000/")
        assert not client._endpoint.startswith("http://localhost:8000//")

    def test_raises_connection_error_when_no_server(self):
        """With no server running, should raise LLMConnectionError."""
        client = VLLMClient(base_url="http://localhost:19999", timeout=2.0)
        with pytest.raises(LLMConnectionError):
            client.generate(system_prompt="sys", user_prompt="user")

    def test_implements_llm_client_protocol(self):
        client = VLLMClient()
        assert isinstance(client, LLMClient)


class TestOllamaClientInterface:
    def test_initializes_correctly(self):
        client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:7b-instruct")
        assert "11434" in client._endpoint

    def test_raises_connection_error_when_no_server(self):
        client = OllamaClient(base_url="http://localhost:19998", timeout=2.0)
        with pytest.raises(LLMConnectionError):
            client.generate(system_prompt="sys", user_prompt="user")

    def test_implements_llm_client_protocol(self):
        client = OllamaClient()
        assert isinstance(client, LLMClient)


# ---------------------------------------------------------------------------
# CVExtractionPrompt Tests
# ---------------------------------------------------------------------------

class TestCVExtractionPrompt:
    def test_builds_user_prompt_with_raw_text(self):
        prompt = CVExtractionPrompt.build_user_prompt("John Doe AI Engineer")
        assert "John Doe AI Engineer" in prompt
        assert "CV TEXT TO PARSE" in prompt

    def test_builds_user_prompt_with_contacts(self):
        contacts: ContactInfo = {
            "emails": ["john@gmail.com"],
            "phones": ["+84941423545"],
            "urls": ["https://johndoe.dev"],
            "linkedin": ["linkedin.com/in/john"],
            "github": ["github.com/johndoe"],
        }
        prompt = CVExtractionPrompt.build_user_prompt("Some CV text", contacts)
        assert "john@gmail.com" in prompt
        assert "+84941423545" in prompt
        assert "linkedin.com/in/john" in prompt
        assert "github.com/johndoe" in prompt
        assert "CONTACT HINTS" in prompt

    def test_builds_user_prompt_without_contacts(self):
        prompt = CVExtractionPrompt.build_user_prompt("Some CV text", contacts=None)
        assert "CONTACT HINTS" not in prompt
        assert "Some CV text" in prompt

    def test_truncates_long_cv_text(self):
        """CV text should be truncated to avoid LLM context overflow."""
        long_text = "A" * 20000
        prompt = CVExtractionPrompt.build_user_prompt(long_text)
        # The raw text portion should be capped to 12000 chars in the source,
        # but the prompt template adds a few chars of its own — allow up to 12100
        assert prompt.count("A") <= 12100

    def test_system_prompt_contains_schema(self):
        assert "basics" in _SYSTEM_PROMPT
        assert "work" in _SYSTEM_PROMPT
        assert "skills" in _SYSTEM_PROMPT
        assert "education" in _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# JSON Parsing Helper Tests
# ---------------------------------------------------------------------------

class TestExtractJsonFromResponse:
    def test_parses_clean_json(self):
        data = json.dumps({"basics": {"name": "Test"}})
        result = _extract_json_from_response(data)
        assert result["basics"]["name"] == "Test"

    def test_parses_json_in_markdown_block(self):
        response = '```json\n{"basics": {"name": "Jane"}}\n```'
        result = _extract_json_from_response(response)
        assert result["basics"]["name"] == "Jane"

    def test_parses_json_in_plain_code_block(self):
        response = '```\n{"basics": {"name": "Jane"}}\n```'
        result = _extract_json_from_response(response)
        assert result["basics"]["name"] == "Jane"

    def test_extracts_json_from_prose(self):
        response = 'Here is the result: {"basics": {"name": "Bob"}} - done.'
        result = _extract_json_from_response(response)
        assert result["basics"]["name"] == "Bob"

    def test_raises_error_on_no_json(self):
        with pytest.raises(ExtractionError, match="does not contain valid JSON"):
            _extract_json_from_response("Sorry, I cannot parse this CV.")

    def test_handles_whitespace_around_json(self):
        result = _extract_json_from_response('  \n{"basics": {"name": "Alice"}}\n  ')
        assert result["basics"]["name"] == "Alice"


# ---------------------------------------------------------------------------
# extract_resume_from_text Integration Tests
# ---------------------------------------------------------------------------

class TestExtractResumeFromText:
    @pytest.fixture
    def sample_cv_text(self) -> str:
        return (
            "Nguyễn Văn A\n"
            "AI Engineer | Ho Chi Minh City\n"
            "Email: mock@example.com | Phone: +84900000000\n\n"
            "WORK EXPERIENCE\n"
            "TechCorp VN — Senior AI Engineer (2023 - Present)\n"
            "- Developed RAG pipelines\n"
            "- Deployed LLM microservices\n\n"
            "EDUCATION\n"
            "FPT University — Bachelor in AI (2021 - 2025)\n\n"
            "SKILLS\n"
            "Python, PyTorch, FastAPI, Docker, Qdrant"
        )

    @pytest.fixture
    def sample_contacts(self) -> ContactInfo:
        return ContactInfo(
            emails=["mock@example.com"],
            phones=["+84900000000"],
            urls=["https://mock.dev"],
            linkedin=[],
            github=["github.com/mock-user"],
        )

    def test_returns_resume_schema_with_default_mock(self, sample_cv_text):
        """With no client provided, should use MockLLMClient."""
        resume = extract_resume_from_text(sample_cv_text)
        assert isinstance(resume, ResumeSchema)
        assert resume.basics.name

    def test_returns_resume_schema_with_explicit_mock(self, sample_cv_text, sample_contacts):
        client = MockLLMClient()
        resume = extract_resume_from_text(sample_cv_text, sample_contacts, client)
        assert isinstance(resume, ResumeSchema)

    def test_resume_has_expected_structure(self, sample_cv_text):
        resume = extract_resume_from_text(sample_cv_text)
        # Structure checks
        assert resume.basics is not None
        assert isinstance(resume.work, list)
        assert isinstance(resume.education, list)
        assert isinstance(resume.skills, list)
        assert isinstance(resume.projects, list)
        assert isinstance(resume.certificates, list)
        assert isinstance(resume.languages, list)

    def test_mock_data_passes_pydantic_validation(self, sample_cv_text):
        """The mock fixture must produce a fully valid ResumeSchema."""
        resume = extract_resume_from_text(sample_cv_text)
        # Should have work experience and skills from fixture
        assert len(resume.work) > 0
        assert len(resume.skills) > 0
        assert len(resume.education) > 0

    def test_get_flat_skills_from_mock_output(self, sample_cv_text):
        resume = extract_resume_from_text(sample_cv_text)
        flat_skills = resume.get_flat_skills()
        assert isinstance(flat_skills, list)
        assert len(flat_skills) > 0

    def test_embedding_text_from_mock_output(self, sample_cv_text):
        resume = extract_resume_from_text(sample_cv_text)
        emb_text = resume.to_embedding_text()
        assert isinstance(emb_text, str)
        assert len(emb_text) > 50

    def test_raises_extraction_error_on_empty_text(self):
        with pytest.raises(ExtractionError, match="Empty CV text"):
            extract_resume_from_text("")

    def test_raises_extraction_error_on_whitespace_only(self):
        with pytest.raises(ExtractionError, match="Empty CV text"):
            extract_resume_from_text("   \n\n  ")

    def test_raises_extraction_error_when_llm_fails(self, sample_cv_text):
        client = MockLLMClient(raise_error=True, error_message="LLM timeout")
        with pytest.raises(ExtractionError, match="LLM timeout"):
            extract_resume_from_text(sample_cv_text, client=client)

    def test_raises_extraction_error_on_invalid_json_response(self, sample_cv_text):
        """When LLM returns non-JSON, ExtractionError should be raised."""
        class BadResponseClient:
            def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=4096):
                return "Sorry, I cannot process this resume."

        with pytest.raises(ExtractionError):
            extract_resume_from_text(sample_cv_text, client=BadResponseClient())

    def test_handles_json_with_markdown_wrapper(self, sample_cv_text):
        """LLM sometimes wraps output in ```json ... ``` — should handle gracefully."""
        json_str = json.dumps(_MOCK_RESUME_JSON, ensure_ascii=False)

        class MarkdownWrappedClient:
            def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=4096):
                return f"```json\n{json_str}\n```"

        resume = extract_resume_from_text(sample_cv_text, client=MarkdownWrappedClient())
        assert isinstance(resume, ResumeSchema)
