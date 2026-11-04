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


def test_loop_cut_removes_a_repeating_run_and_closes_the_list():
    head = '{"document":{"skills":[{"name":"Dev","keywords":["Java","JSP","JDBC"'
    looped = head + ',"JSP","JDBC"' * 5 + ',"JS'
    assert v4_client.loop_cut(looped) == head + "]"
    objs = '{"facts":{"evidence":{"title":[{"quote":"Accountant"}' + ',{"quote":"Accountant"}' * 9 + ',{"quo'
    assert v4_client.loop_cut(objs) == '{"facts":{"evidence":{"title":[{"quote":"Accountant"}]'
    assert v4_client.loop_cut(head + ',"Go","Rust","C","C++","SQL","R","Bash","Perl","Lua",') is None  # no repeats


def test_loop_cut_keeps_an_item_that_leaves_the_loop():
    head = '{"facts":{"evidence":{"q":[' + ",".join(['{"quote":"A"}'] * 10)
    assert v4_client.loop_cut(head + ',{"quote":"B') is None  # B is new: the model is escaping, keep it
    assert v4_client.loop_cut(head + ',{"quote":"A') == '{"facts":{"evidence":{"q":[{"quote":"A"}]'


class FakeStream:
    def __init__(self, lines): self.lines = lines
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def raise_for_status(self): pass
    def iter_lines(self): return iter(self.lines)


def fake_v4(monkeypatch, lines):
    import httpx
    llm = v4_client.QwenV4.__new__(v4_client.QwenV4)
    llm.model, llm.url, llm.timeout = "m", "u", 1
    llm.tok = type("T", (), {"apply_chat_template": lambda self, *a, **k: "P", "__call__": lambda self, t: {"input_ids": [0] * 10}})()
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: FakeStream([json.dumps(x) for x in lines]))
    return llm


def test_stream_uses_eval_count_and_fails_on_errors(monkeypatch):
    user = json.dumps({"source_text": "x"})
    ok = [{"response": "{}"}] * 3 + [{"done": True, "done_reason": "stop", "eval_count": 3, "response": ""}]
    assert fake_v4(monkeypatch, ok).complete("s", user, 1.05) == ("{}{}{}", "stop")
    with pytest.raises(ExtractionError):
        fake_v4(monkeypatch, [{"response": "{}"}, {"error": "backend failed"}]).complete("s", user, 1.05)
    with pytest.raises(ExtractionError):
        fake_v4(monkeypatch, [{"response": "{}"}]).complete("s", user, 1.05)  # no completion message
