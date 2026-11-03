import json

import pytest

from src.parser.llm_extractor import ExtractionError
from src.scoring import v4_client
from src.scoring.v4_client import V4Client


class FakeLLM:
    """Replays answers in order and records the repetition penalty of each call."""

    def __init__(self, answers):
        self.answers, self.penalties = list(answers), []

    def complete(self, system, user, repetition_penalty):
        self.penalties.append(repetition_penalty)
        return self.answers.pop(0), "stop"


def call(client, kind="job", text="Python is required."):
    return client.generate("system", json.dumps({"document_kind": kind, "source_text": text}))


def test_retries_at_rising_penalty_until_valid(monkeypatch):
    statuses = iter(["policy_fail", "pass"])
    monkeypatch.setattr("scripts.qwen_repair.repair", lambda label, text, task: (label, next(statuses), ["fixed"]))
    llm = FakeLLM(["{not json", '{"document": {}, "facts": {}}', '{"document": {"a": 1}, "facts": {}}'])
    client = V4Client(llm)
    assert json.loads(call(client)) == {"document": {"a": 1}, "facts": {}}
    assert llm.penalties == list(v4_client.PENALTIES[:3])
    assert client.last_metadata["attempts"] == 3 and client.last_metadata["status"] == "pass"


def test_gives_up_after_the_last_penalty():
    llm = FakeLLM(["{not json"] * len(v4_client.PENALTIES))
    with pytest.raises(ExtractionError):
        call(V4Client(llm))
    assert llm.penalties == list(v4_client.PENALTIES)
