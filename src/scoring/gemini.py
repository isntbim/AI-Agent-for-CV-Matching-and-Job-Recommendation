"""Gemini extraction backend. Credentials remain in process memory only.

REST reference: https://ai.google.dev/api/generate-content
"""

import json
import os
import re
import time

import httpx

from src.parser.llm_extractor import ExtractionError

from .models import JobInput, ResumeInput


class GeminiClient:
    def __init__(self, base_url: str = "https://generativelanguage.googleapis.com/v1beta",
                 model: str = "gemini-3.7-flash", api_key: str | None = None, timeout: float = 120,
                 request_interval: float = 13.0, max_attempts: int = 3):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout
        if self.base_url != "https://generativelanguage.googleapis.com/v1beta":
            raise ValueError("Gemini credentials may only be sent to the official HTTPS API")
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
            raise ValueError("Invalid Gemini model name")
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ExtractionError("Set GEMINI_API_KEY in the process environment")
        self.last_response = None
        self.last_metadata = None
        self.last_api_error = None
        self._daily_quota_error = None
        if max_attempts not in {1, 2, 3}:
            raise ValueError("Gemini max_attempts must be 1, 2, or 3")
        self.max_attempts = max_attempts
        if request_interval < 13:
            raise ValueError("Gemini request interval must be at least 13 seconds for this project's 5 RPM quota")
        self.request_interval = request_interval
        self._next_request = 0.0

    def _request(self, method: str, path: str, payload=None) -> dict:
        if self._daily_quota_error is not None:
            self.last_api_error = self._daily_quota_error
            raise ExtractionError("Gemini daily quota is exhausted; further requests were not sent")
        for attempt in range(self.max_attempts):
            remaining = self._next_request - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            self._next_request = time.monotonic() + self.request_interval
            try:
                response = httpx.request(method, self.base_url + path,
                                        headers={"x-goog-api-key": self._api_key},
                                        json=payload, timeout=self.timeout)
            except httpx.HTTPError as exc:
                raise ExtractionError(f"Gemini transport error: {type(exc).__name__}") from None
            if response.is_error:
                try:
                    error = response.json().get("error", {})
                    self.last_api_error = json.loads(json.dumps(error).replace(self._api_key, "[REDACTED]"))
                    message = str(error.get("message", "Request rejected"))
                except ValueError:
                    message = "Non-JSON API error"
                details = self.last_api_error.get("details", []) if self.last_api_error else []
                daily_limit = any("PerDay" in str(item) for item in details)
                if daily_limit:
                    self._daily_quota_error = self.last_api_error
                if response.status_code in {429, 503} and not daily_limit and attempt < self.max_attempts - 1:
                    delay = 2 * (attempt + 1)
                    for item in details:
                        retry = item.get("retryDelay", "")
                        if isinstance(retry, str) and retry.endswith("s"):
                            try:
                                delay = max(delay, float(retry[:-1]) + 1)
                            except ValueError:
                                pass
                    if delay <= 60:
                        time.sleep(delay)
                        continue
                message = message.replace(self._api_key, "[REDACTED]")
                raise ExtractionError(f"Gemini HTTP {response.status_code}: {message[:700]}")
            return response.json()
        raise ExtractionError("Gemini unavailable after retries")

    def describe(self) -> dict:
        return self._request("GET", "/models/" + self.model)

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=4096):
        self.last_response = None
        self.last_metadata = None
        self.last_api_error = None
        kind = json.loads(user_prompt)["document_kind"]
        schema = ResumeInput.model_json_schema() if kind == "resume" else JobInput.model_json_schema()
        schema["required"] = ["document", "facts"]
        payload = self._request("POST", "/models/" + self.model + ":generateContent", {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json", "responseJsonSchema": schema,
                                 "thinkingConfig": {"thinkingLevel": "LOW"}},
        })
        candidates = payload.get("candidates", [])
        if not candidates:
            raise ExtractionError("Gemini returned no candidate (blocked or empty response)")
        candidate = candidates[0]
        self.last_metadata = {"model_version": payload.get("modelVersion"),
                              "usage": payload.get("usageMetadata"), "finish_reason": candidate.get("finishReason")}
        self.last_response = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
                                     if not part.get("thought"))
        if candidate.get("finishReason") != "STOP" or not self.last_response:
            raise ExtractionError(f"Gemini extraction incomplete: {candidate.get('finishReason', 'unknown')}")
        return self.last_response
