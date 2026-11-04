"""Unit test for scripts/o4_report.py replay logic."""

from __future__ import annotations

from scripts import o4_report


def test_replay_document_attempt1_and_attempt2(monkeypatch) -> None:
    def fake_parse_balanced(raw: str, schema=None):
        return None if raw == "UNPARSED" else {"raw": raw}

    def fake_validate(label: dict, text: str, task: str) -> tuple[str, str | None]:
        return ("pass", None) if label["raw"] == "RAW_OK" else ("fail", "needs repair or retry")

    def fake_repair(label: dict, text: str, task: str) -> tuple[dict, str, list[str]]:
        if label["raw"] in ("RAW_OK", "REPAIR_OK", "RETRY_OK"):
            return label, "pass", []
        return label, "fail", ["still bad"]

    monkeypatch.setattr(o4_report, "parse_balanced", fake_parse_balanced)
    monkeypatch.setattr(o4_report, "validate", fake_validate)
    monkeypatch.setattr(o4_report, "repair", fake_repair)

    rows_by_key = {
        ("cv_1", 1.05): {
            "id": "cv_1",
            "task": "cv_extraction",
            "prediction": "REPAIR_OK",
            "finish_reason": "stop",
            "repetition_penalty": 1.05,
            "prompt_tokens": 500,
            "output_tokens": 300,
        },
        ("jd_2", 1.05): {
            "id": "jd_2",
            "task": "jd_extraction",
            "prediction": "UNPARSED",
            "finish_reason": "length",
            "repetition_penalty": 1.05,
            "prompt_tokens": 400,
            "output_tokens": 16384,
        },
        ("jd_2", 1.1): {
            "id": "jd_2",
            "task": "jd_extraction",
            "prediction": "RETRY_OK",
            "finish_reason": "stop",
            "repetition_penalty": 1.1,
            "prompt_tokens": 400,
            "output_tokens": 420,
        },
    }

    r1 = o4_report.replay_document("cv_1", "resume", "cv text", rows_by_key, {})
    assert not r1["raw_valid"]
    assert r1["repair_valid"]
    assert r1["retry_valid"]
    assert r1["final_valid"]
    assert r1["attempts_count"] == 1
    assert not r1["retried"]
    assert r1["attempt1_tokens"] == 300
    assert r1["total_tokens"] == 300
    assert r1["added_tokens"] == 0

    r2 = o4_report.replay_document("jd_2", "job", "jd text", rows_by_key, {})
    assert not r2["raw_valid"]
    assert not r2["repair_valid"]
    assert r2["retry_valid"]
    assert r2["final_valid"]
    assert r2["attempts_count"] == 2
    assert r2["retried"]
    assert r2["attempt1_tokens"] == 16384
    assert r2["total_tokens"] == 16804
    assert r2["added_tokens"] == 420
    assert r2["length_finish_count"] == 1
