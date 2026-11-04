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


def test_token_cap_scales_with_the_source_and_keeps_the_hard_limit():
    assert v4_client.token_cap(1000) == 7000  # a median CV is ~1k source tokens; its answer is ~2k
    assert v4_client.token_cap(10_000) == v4_client.MAX_NEW_TOKENS


def test_a_cut_off_answer_is_never_accepted(monkeypatch):
    monkeypatch.setattr("scripts.qwen_repair.repair", lambda label, text, task: (label, "pass", []))

    class Capped(FakeLLM):
        def complete(self, system, user, repetition_penalty):
            return super().complete(system, user, repetition_penalty)[0], "length"

    llm = Capped(['{"document": {}, "facts": {}}'] * len(v4_client.PENALTIES))
    with pytest.raises(ExtractionError, match="truncated"):
        call(V4Client(llm))
    assert llm.penalties == list(v4_client.PENALTIES)
