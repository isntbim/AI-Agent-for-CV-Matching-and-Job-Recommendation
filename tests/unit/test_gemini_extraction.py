"""Gemini transport, schema and credential handling without real API calls."""

import json

import httpx
import pytest

from src.parser.llm_extractor import ExtractionError
from src.scoring.gemini import GeminiClient


def test_gemini_schema_and_non_thought_text(monkeypatch):
    captured = {}

    def request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return httpx.Response(200, json={"modelVersion": "gemini-3.7-flash-test", "usageMetadata": {"totalTokenCount": 100},
                             "candidates": [{"finishReason": "STOP", "content": {"parts": [
                                 {"text": "internal reasoning", "thought": True}, {"text": "{}"}]}}]})

    monkeypatch.setattr("src.scoring.gemini.httpx.request", request)
    client = GeminiClient(api_key="test-only-key")
    assert client.generate("extract facts", json.dumps({"document_kind": "resume"})) == "{}"
    assert captured["headers"] == {"x-goog-api-key": "test-only-key"}
    assert "test-only-key" not in captured["url"]
    assert "test-only-key" not in json.dumps(captured["json"])
    config = captured["json"]["generationConfig"]
    assert config["responseJsonSchema"]["required"] == ["document", "facts"]
    assert config["responseMimeType"] == "application/json"
    assert client.last_metadata["model_version"] == "gemini-3.7-flash-test"


def test_provider_error_redacts_key(monkeypatch):
    monkeypatch.setattr("src.scoring.gemini.httpx.request", lambda *a, **k: httpx.Response(
        403, json={"error": {"message": "Rejected credential test-only-key"}}))
    with pytest.raises(ExtractionError) as error:
        GeminiClient(api_key="test-only-key").describe()
    assert "test-only-key" not in str(error.value)
    assert "[REDACTED]" in str(error.value)


def test_gemini_requires_official_endpoint_and_credential(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="GEMINI_API_KEY"):
        GeminiClient()
    with pytest.raises(ValueError, match="official HTTPS"):
        GeminiClient(base_url="https://example.com", api_key="test-only-key")
    with pytest.raises(ValueError, match="model name"):
        GeminiClient(model="invalid/path", api_key="test-only-key")


def test_incomplete_response_is_rejected(monkeypatch):
    monkeypatch.setattr("src.scoring.gemini.httpx.request", lambda *a, **k: httpx.Response(
        200, json={"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]}))
    with pytest.raises(ExtractionError, match="incomplete"):
        GeminiClient(api_key="test-only-key").generate("extract", '{"document_kind":"job"}')


def test_five_rpm_pacing_includes_retries_and_model_requests(monkeypatch):
    clock, times, replies = {"now": 0.0}, [], [503, 200, 200]
    monkeypatch.setattr("src.scoring.gemini.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("src.scoring.gemini.time.sleep", lambda seconds: clock.update(now=clock["now"] + seconds))

    def request(*args, **kwargs):
        times.append(clock["now"])
        return httpx.Response(replies.pop(0), json={})

    monkeypatch.setattr("src.scoring.gemini.httpx.request", request)
    client = GeminiClient(api_key="test-only-key")
    client.describe()
    client.describe()
    assert times == [0, 13, 26]


def test_daily_quota_stops_retries_and_following_requests(monkeypatch):
    calls = []

    def request(*args, **kwargs):
        calls.append(True)
        return httpx.Response(429, json={"error": {"message": "Daily quota exhausted", "details": [
            {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
            {"retryDelay": "18s"}]}})

    monkeypatch.setattr("src.scoring.gemini.httpx.request", request)
    client = GeminiClient(api_key="test-only-key")
    with pytest.raises(ExtractionError, match="429"):
        client.describe()
    with pytest.raises(ExtractionError, match="not sent"):
        client.describe()
    assert len(calls) == 1
