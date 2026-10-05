import hashlib
import json
from pathlib import Path

import pytest

from src.scoring.extraction import write_json
from src.scoring.pilot import freeze_splits, REVIEW_STATUS, similarity, shingles
from src.retrieval.azure import table_name


def test_duplicate_groups_and_public_overlap_are_isolated_and_reproducible(tmp_path):
    output = tmp_path / "output"
    rows = []
    for i in range(20):
        text = "Shared duplicate source a b c d e" if i in (0, 1) else f"Original source {i} unique content for document with distinct tokens {i}"
        path = output / "sources" / f"cv_{i}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        rows.append({"id": f"cv_{i}", "kind": "resume", "domain": "software_engineer", "text_path": path.relative_to(tmp_path).as_posix(),
                     "normalized_sha256": hashlib.sha256(text.encode()).hexdigest()})
    write_json(output / "source_manifest.json", {"documents": rows})
    write_json(output / "near_duplicate_nominations.json", [])
    write_json(output / "near_duplicate_review.json", [])
    public = tmp_path / "public"
    write_json(public / "manifest.json", {"seed": 123, "split_policy": "preserved"})
    write_json(public / "pairs.json", [{"resume_text": "Shared duplicate source a b c d e", "job_description_text": "Public job text"}])
    first = freeze_splits(tmp_path, output, output / "near_duplicate_review.json", public)
    before = (output / "split_manifest.json").read_bytes()
    second = freeze_splits(tmp_path, output, output / "near_duplicate_review.json", public)
    assert first == second
    assert before == (output / "split_manifest.json").read_bytes()
    assert first["documents"][0]["split"] == first["documents"][1]["split"] == "excluded_public_overlap"
    groups = {}
    for row in first["documents"]:
        groups.setdefault(row["duplicate_group"], set()).add(row["split"])
    assert all(len(s) == 1 for s in groups.values())
    assert first["public_manifest"]["seed"] == 123


def test_near_duplicate_jaccard_whitespace_case():
    assert similarity(shingles("A b c d e f"), shingles("a  B c d e f")) == 1
    assert similarity(set(), set()) == 0


def test_atomic_json_write_retries_windows_sharing_error(tmp_path, monkeypatch):
    import src.scoring.extraction as extraction
    replace = extraction.os.replace
    attempts = []
    def transient(source, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError("Transient file lock")
        return replace(source, target)
    monkeypatch.setattr(extraction.os, "replace", transient)
    monkeypatch.setattr(extraction.time, "sleep", lambda _: None)
    path = tmp_path / "result.json"
    write_json(path, {"value": 123})
    assert json.loads(path.read_text()) == {"value": 123}
    assert len(attempts) == 3
    assert not list(tmp_path.glob("*.tmp"))


def test_azure_table_is_isolated_and_only_accepts_manifest_hash():
    assert table_name("a" * 64) == "retrieval_" + "a" * 20
    with pytest.raises(ValueError):
        table_name("existing_table; DROP TABLE jobs")


def test_azure_audit_accepts_pgvector_wrapper_and_numpy_results():
    import numpy as np
    from pgvector import Vector
    from src.retrieval.azure import vector_array
    expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert np.array_equal(vector_array(Vector(expected), dimension=3), expected)
    assert np.array_equal(vector_array(expected, dimension=3), expected)
    with pytest.raises(ValueError, match="dimensions"):
        vector_array(Vector(expected), dimension=1024)
    with pytest.raises(ValueError, match="non-finite"):
        vector_array(np.array([np.nan, 0, 1]), dimension=3)


def test_disagreement_validation_rejects_stale_split_and_changed_originals(tmp_path):
    from copy import deepcopy
    from src.scoring.disagreements import validate_review
    cases = []
    for domain in ("accountant", "data_scientist_analyst", "frontend_web_developer", "hr", "marketing_executive", "software_engineer"):
        for i in range(4):
            cases.append({"original_comparison": {"cv_domain": domain, "silver_grade": i % 3},
                          "sources": {"resume": {"split": "train"}, "job": {"split": "train"}},
                          "selection_reason": "large_score_gap", "category_fallback": False,
                          "source_excerpts": {"resume": "Original", "job": "Original"},
                          "review_status": REVIEW_STATUS, "classification": "scoring_rule_behavior", "findings": ["Reviewed"]})
    report = {"cases": cases, "comparison_sha256": "comparison", "split_manifest_sha256": "split"}
    write_json(tmp_path / "disagreement_nominations.json", report)
    write_json(tmp_path / "disagreement_review.json", report)
    assert validate_review(tmp_path)["validation"]["count"] == 24
    stale = deepcopy(report)
    stale["split_manifest_sha256"] = "changed"
    write_json(tmp_path / "disagreement_review.json", stale)
    with pytest.raises(ValueError, match="stale"):
        validate_review(tmp_path)
    altered = deepcopy(report)
    altered["cases"][0]["original_comparison"]["silver_grade"] = 2
    write_json(tmp_path / "disagreement_review.json", altered)
    with pytest.raises(ValueError, match="original labels"):
        validate_review(tmp_path)
