"""Unit tests for the Qwen SFT exporter (scripts/qwen_export_sft.py)."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import qwen_export_sft
from src.scoring import extraction
from src.scoring.extraction import write_json


def _fake_label(skill_name: str) -> dict:
    return {
        "document": {
            "schema_version": "resume-v3",
            "basics": {"name": "Synthetic Candidate", "label": "Engineer"},
            "work": [],
            "education": [],
            "skills": [
                {
                    "name": "Core",
                    "level": None,
                    "keywords": [skill_name],
                    "details": [
                        {
                            "name": skill_name,
                            "proficiency": "normal",
                            "raw_proficiency": None,
                            "skill_quote": skill_name,
                            "proficiency_quote": None,
                        }
                    ],
                }
            ],
            "projects": [],
            "certificates": [],
            "languages": [],
        },
        "facts": {
            "domains": ["software"],
            "completed_degrees": [],
            "country": None,
            "evidence": {"skills": [{"quote": skill_name}], "industry": [{"quote": "Engineer"}]},
        },
    }


def _fake_jd_label() -> dict:
    return {
        "document": {
            "schema_version": "job-v3",
            "id": "jd_syn",
            "url": "https://example.com/jd",
            "scraped_at": "2026-10-26T00:00:00Z",
            "title": "Backend Developer",
            "title_normalized": "backend_developer",
            "category": "software_engineer",
            "location": {"city": "Hanoi", "country": "Vietnam", "countryCode": "VN"},
            "seniority": {"min_years": None, "max_years": None},
            "skills": {
                "required": [{"name": "Python", "proficiency": "required"}],
                "preferred": [],
                "raw_text": None,
                "requirements": [
                    {
                        "id": "req_01",
                        "options": ["Python"],
                        "option_details": [{"name": "Python", "minimum_proficiency": "normal"}],
                        "importance": "core",
                        "quote": "Must have Python.",
                        "importance_quote": "Must have Python.",
                        "review_status": "pending",
                        "issues": [],
                    }
                ],
            },
            "description": {"raw_text": None},
            "compensation": {"currency": None, "min_monthly": None, "max_monthly": None, "raw_text": None},
            "benefits": {"raw_text": None},
            "metadata": {"language": "en"},
        },
        "facts": {
            "domains": ["software"],
            "completed_degrees": [],
            "minimum_degree": None,
            "country": "VN",
            "currency": None,
            "evidence": {
                "skills": [{"quote": "Python"}],
                "industry": [{"quote": "Backend Developer"}],
                "location": [{"quote": "Hanoi, Vietnam"}],
                "salary": [],
            },
        },
    }


def test_export_sft_v3_splits_roundtrip_and_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_calls: list[tuple[str, str, str]] = []
    real_prompts = extraction._prompts

    def spy_prompts(kind: str, text: str, ruleset: str = "v1", as_of: str | None = None) -> tuple[str, str]:
        recorded_calls.append((kind, text, ruleset))
        return real_prompts(kind, text, ruleset, as_of)

    monkeypatch.setattr(extraction, "_prompts", spy_prompts)

    label_train = _fake_label("Python")
    label_val = _fake_label("SQL")
    text_train = "Synthetic Engineer CV with Python experience."
    text_val = "Synthetic Engineer CV with SQL experience."
    text_long = "A" * 40001

    write_json(tmp_path / "labels/cv_train.json", label_train)
    write_json(tmp_path / "labels/cv_val.json", label_val)
    (tmp_path / "texts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "texts/cv_train.txt").write_text(text_train, encoding="utf-8")
    (tmp_path / "texts/cv_val.txt").write_text(text_val, encoding="utf-8")
    (tmp_path / "texts/cv_long.txt").write_text(text_long, encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {
            "tasks": {
                "cv_extraction": [
                    {
                        "id": "cv_val_01",
                        "split": "validation",
                        "source": "gold",
                        "text": "texts/cv_val.txt",
                        "label": "labels/cv_val.json",
                        "status": "partial",
                        "as_of": "2026-10-25",
                    },
                    {
                        "id": "cv_train_01",
                        "split": "train",
                        "source": "pool/reference_b",
                        "text": "texts/cv_train.txt",
                        "label": "labels/cv_train.json",
                        "status": "pass",
                        "as_of": "2026-10-25",
                    },
                    {
                        "id": "cv_bad_depth",
                        "split": "train",
                        "source": "pool/reference_b",
                        "text": "texts/cv_train.txt",
                        "label": "labels/cv_train.json",
                        "status": "depth_error",
                        "as_of": "2026-10-25",
                    },
                    {
                        "id": "cv_too_long",
                        "split": "test",
                        "source": "gold",
                        "text": "texts/cv_long.txt",
                        "label": "labels/cv_train.json",
                        "status": "pass",
                        "as_of": "2026-10-25",
                    },
                ]
            }
        },
    )

    out_dir = tmp_path / "out"
    stats = qwen_export_sft.export_sft(manifest_path=manifest_path, out_dir=out_dir, root=tmp_path)

    assert recorded_calls and all(ruleset == "v3" and kind == "resume" for kind, _, ruleset in recorded_calls)

    train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (out_dir / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    test_lines = (out_dir / "test.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(train_lines) == 1
    assert len(val_lines) == 1
    assert len(test_lines) == 0

    train_obj = json.loads(train_lines[0])
    val_obj = json.loads(val_lines[0])

    assert train_obj["id"] == "cv_train_01"
    assert train_obj["split"] == "train"
    assert train_obj["source"] == "pool/reference_b"
    assert train_obj["text_sha256"] == hashlib.sha256(text_train.encode("utf-8")).hexdigest()
    assert [m["role"] for m in train_obj["messages"]] == ["system", "user", "assistant"]
    assert json.loads(train_obj["messages"][2]["content"]) == label_train
    assert json.loads(train_obj["messages"][1]["content"])["as_of_date"] == "2026-10-25" and stats["as_of_dates"] == ["2026-10-25"]

    assert val_obj["id"] == "cv_val_01"
    assert val_obj["split"] == "validation"
    assert json.loads(val_obj["messages"][2]["content"]) == label_val

    saved_stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    assert saved_stats == stats
    assert stats["counts"] == {"train": 1, "validation": 1, "test": 0}
    skipped_by_id = {item["id"]: item for item in stats["skipped"]}
    assert set(skipped_by_id) == {"cv_bad_depth", "cv_too_long"}
    assert skipped_by_id["cv_bad_depth"]["status"] == "depth_error"
    assert "40001" in skipped_by_id["cv_too_long"]["reason"]
    assert stats["system_prompt_sha256"] == hashlib.sha256(train_obj["messages"][0]["content"].encode("utf-8")).hexdigest()


def test_export_sft_jd_roundtrip_and_skips(tmp_path: Path) -> None:
    text_jd = "Title: Backend Developer\nLocation: Hanoi, Vietnam\nRequirements: Must have Python."
    good_jd = _fake_jd_label()
    bad_jd = _fake_jd_label()
    bad_jd["facts"]["completed_degrees"] = ["bachelor"]

    (tmp_path / "texts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "texts/jd_01.txt").write_text(text_jd, encoding="utf-8")
    write_json(tmp_path / "labels/jd_good.json", good_jd)
    write_json(tmp_path / "labels/jd_bad.json", bad_jd)

    manifest_path = tmp_path / "manifest.json"
    write_json(
        manifest_path,
        {
            "tasks": {
                "jd_extraction": [
                    {
                        "id": "jd_train_01",
                        "split": "train",
                        "source": "teacher/gemini",
                        "text": "texts/jd_01.txt",
                        "label": "labels/jd_good.json",
                        "status": "pass",
                    },
                    {
                        "id": "jd_bad_01",
                        "split": "train",
                        "source": "teacher/gemini",
                        "text": "texts/jd_01.txt",
                        "label": "labels/jd_bad.json",
                        "status": "pass",
                    },
                ]
            }
        },
    )

    out_dir = tmp_path / "out_jd"
    stats = qwen_export_sft.export_sft(manifest_path=manifest_path, out_dir=out_dir, root=tmp_path, task="jd_extraction")

    assert stats["task"] == "jd_extraction"
    assert stats["as_of_dates"] == []
    assert stats["counts"] == {"train": 1, "validation": 0, "test": 0}
    assert [s["id"] for s in stats["skipped"]] == ["jd_bad_01"]

    train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 1
    train_obj = json.loads(train_lines[0])
    expected_system, expected_user = extraction._prompts("job", text_jd, "v3")
    assert [m["role"] for m in train_obj["messages"]] == ["system", "user", "assistant"]
    assert train_obj["messages"][0]["content"] == expected_system
    assert train_obj["messages"][1]["content"] == expected_user
    assert json.loads(train_obj["messages"][2]["content"]) == good_jd


def test_export_sft_rejects_duplicate_id_across_splits_and_extra_label_keys(tmp_path: Path) -> None:
    manifest_dup = tmp_path / "manifest_dup.json"
    write_json(
        manifest_dup,
        {
            "tasks": {
                "cv_extraction": [
                    {"id": "dup_1", "split": "train", "source": "gold", "text": "t.txt", "label": "l.json", "status": "pass"},
                    {"id": "dup_1", "split": "test", "source": "gold", "text": "t.txt", "label": "l.json", "status": "pass"},
                ]
            }
        },
    )
    with pytest.raises(AssertionError):
        qwen_export_sft.export_sft(manifest_path=manifest_dup, out_dir=tmp_path / "out_dup", root=tmp_path)

    (tmp_path / "t.txt").write_text("Synthetic text", encoding="utf-8")
    write_json(tmp_path / "bad_label.json", {"document": {}, "facts": {}, "extra": 1})
    manifest_extra = tmp_path / "manifest_extra.json"
    write_json(
        manifest_extra,
        {
            "tasks": {
                "cv_extraction": [
                    {
                        "id": "extra_1",
                        "split": "train",
                        "source": "gold",
                        "text": "t.txt",
                        "label": "bad_label.json",
                        "status": "pass",
                        "as_of": "2026-10-25",
                    }
                ]
            }
        },
    )
    with pytest.raises(ValueError, match="document"):
        qwen_export_sft.export_sft(manifest_path=manifest_extra, out_dir=tmp_path / "out_extra", root=tmp_path)


def test_export_sft_rechecks_labels_and_blocks_same_text_across_splits(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Synthetic Engineer CV with Python experience.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Synthetic Engineer  CV with Python\nexperience.", encoding="utf-8")
    write_json(tmp_path / "good.json", _fake_label("Python"))
    stale = _fake_label("Python")
    stale["document"]["schema_version"] = "resume-v1"
    write_json(tmp_path / "stale.json", stale)
    row = {"source": "gold", "text": "a.txt", "status": "pass", "as_of": "2026-10-25"}
    write_json(tmp_path / "m1.json", {"tasks": {"cv_extraction": [dict(row, id="ok", split="train", label="good.json"),
                                                                   dict(row, id="stale", split="train", label="stale.json")]}})
    stats = qwen_export_sft.export_sft(manifest_path=tmp_path / "m1.json", out_dir=tmp_path / "o1", root=tmp_path)
    assert stats["counts"]["train"] == 1 and [s["id"] for s in stats["skipped"]] == ["stale"]

    write_json(tmp_path / "m2.json", {"tasks": {"cv_extraction": [dict(row, id="x", split="train", label="good.json"),
                                                                   dict(row, id="y", split="test", text="b.txt", label="good.json")]}})
    with pytest.raises(ValueError, match="Same text"):
        qwen_export_sft.export_sft(manifest_path=tmp_path / "m2.json", out_dir=tmp_path / "o2", root=tmp_path)
