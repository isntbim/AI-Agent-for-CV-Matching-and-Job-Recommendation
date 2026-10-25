"""Ablation identity, failure accounting and unchanged-label diagnostics."""

import json
from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest

from src.parser.skill_v2 import SkillRelationTable
from src.scoring.models import ScorerConfig
from src.scoring.v2_benchmark import (
    VARIANTS, annotation_summary, benchmark_identity, canonical_hash, grade_changes,
    load_corpus, require_same_identity, run_benchmark, score_stored_pair, summarize,
    variant_config,
)


AS_OF = date(2026, 9, 30)


def _write(path, value):
    if isinstance(value, dict) and value.get("version") == "ats-source-corpus-v2":
        for document in value["documents"]:
            document["checksum_sha256"] = canonical_hash(document["envelope"])
        value["identity_sha256"] = canonical_hash({key: current for key, current in value.items() if key not in {"identity_sha256", "review_history"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixtures(tmp_path):
    cv = {"cv_id": "cv_001", "basics": {"label": "Software Engineer"}, "skills": [{"keywords": ["Python"]}]}
    jd = {"id": "jd_001", "url": "https://example.test/job", "scraped_at": "2026-09-01", "title": "Software Engineer",
          "title_normalized": "software_engineer", "category": "software_engineer", "skills": {"required": [{"name": "Python"}]}}
    pair = {"cv_id": "cv_001", "jd_id": "jd_001", "cv_domain": "software_engineer", "jd_domain": "software_engineer",
            "pair_type": "llm_positive", "grade": 2, "overall_score": 90, "scores": {}}
    docs = []
    for identifier, kind, document in (("cv_001", "resume", cv), ("jd_001", "job", jd)):
        envelope = {"document": document, "facts": {}}
        docs.append({"id": identifier, "kind": kind, "source": {"source_sha256": "abc"}, "before": deepcopy(envelope),
                     "envelope": envelope, "checksum_sha256": "abc", "review_status": "pending", "skills_status": "partial"})
    corpus = {"version": "ats-source-corpus-v2", "as_of_date": AS_OF.isoformat(), "documents": docs,
              "relations": {"version": "empty-v1", "relations": []}}
    _write(tmp_path / "data/datasets/matching_pairs/benchmark_matching_pairs.json", [pair])
    _write(tmp_path / "data/processed/structured_cvs/all_cvs.json", [cv])
    _write(tmp_path / "data/processed/structured_jobs/all_jobs_en.json", [jd])
    corpus_path = tmp_path / "corpus.json"
    _write(corpus_path, corpus)
    return pair, corpus, corpus_path


def _row(identifier, grade, *, status="ok", force=False):
    row = {"pair_id": identifier, "status": status, "latency_ms": 1}
    if status == "failed":
        return {**row, "error_type": "ValueError", "error": "bad data"}
    return {**row, "silver_grade": 2, "result": {"grade": grade, "overall_score": 90,
            "coverage": .4, "scores": {"skills": 80, "title": 90},
            "dimensions": {"skills": {"supported": True}, "title": {"supported": False}},
            "skills_breakdown": {"force_mismatch": force, "gate_status": "failed" if force else "passed"}}}


def test_identity_refuses_config_date_source_code_and_selection_mixing(tmp_path):
    source = tmp_path / "corpus.json"
    _write(source, {"annotation": "pending"})
    code = tmp_path / "src/scoring/rules.py"
    code.parent.mkdir(parents=True)
    code.write_text("VERSION = 1\n")
    table = SkillRelationTable(version="empty-v1")
    identity = benchmark_identity(tmp_path, {"corpus": source}, AS_OF, table, False, [{"cv_id": "a", "jd_id": "b"}])
    assert require_same_identity(identity, {"identity": deepcopy(identity)}) == identity["family_sha256"]
    for key in ("as_of_date", "code", "configs", "inputs", "relations", "selected_pairs_sha256"):
        changed = deepcopy(identity)
        changed[key] = "different"
        changed["family_sha256"] = canonical_hash({k: v for k, v in changed.items() if k != "family_sha256"})
        with pytest.raises(ValueError, match="different benchmark identities"):
            require_same_identity(identity, changed)
    forged = deepcopy(identity)
    forged["as_of_date"] = "2027-01-01"
    with pytest.raises(ValueError, match="checksum"):
        require_same_identity(forged)
    code.write_text("VERSION = 2\n")
    assert benchmark_identity(tmp_path, {"corpus": source}, AS_OF, table, False)["family_sha256"] != identity["family_sha256"]


def test_corpus_validates_date_duplicate_identity_population_and_provenance(tmp_path):
    _, corpus, path = _fixtures(tmp_path)
    assert len(load_corpus(path, AS_OF, require_population=False)[1]) == 2
    with pytest.raises(ValueError, match="same fixed as-of"):
        load_corpus(path, date(2026, 10, 1), require_population=False)
    with pytest.raises(ValueError, match="88 CV"):
        load_corpus(path, AS_OF)
    corpus["documents"].append(deepcopy(corpus["documents"][0]))
    _write(path, corpus)
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(path, AS_OF, require_population=False)
    corpus["documents"].pop()
    corpus["documents"][0]["source"].pop("source_sha256")
    _write(path, corpus)
    with pytest.raises(ValueError, match="source checksum"):
        load_corpus(path, AS_OF, require_population=False)


def test_all_failed_and_empty_subsets_never_fabricate_metrics():
    failed = summarize([_row("a", 0, status="failed")])
    assert failed["status"] == "all_failed" and failed["metrics"] is None
    assert failed["expected"] == failed["failed"] == 1
    assert summarize([])["status"] == "empty"
    assert summarize([])["metrics"] is None


def test_metrics_use_runtime_grade_and_retain_failures():
    report = summarize([_row("a", 0, force=True), _row("b", 0, status="failed")])
    assert report["status"] == "partial_failure"
    assert report["metrics"]["confusion_matrix"][2][0] == 1
    assert report["forced_mismatch"] == 1
    assert report["failed"] == 1 and report["expected"] == 2


def test_changes_match_ids_preserve_failures_and_ignore_input_order():
    old = [_row("b", 1), _row("a", 2), _row("c", 0, status="failed")]
    new = [_row("a", 0), _row("b", 1), _row("c", 2)]
    result = grade_changes(old, new)
    assert result == grade_changes(list(reversed(old)), list(reversed(new)))
    assert result["changed"] == 1 and result["pairs"][0]["pair_id"] == "a"
    assert result["uncomparable_pairs"] == ["c"]


def test_seven_variants_have_explicit_v1_and_v2_configuration():
    table = SkillRelationTable(version="reviewed-v1")
    assert len(VARIANTS) == 7
    for name in VARIANTS:
        config = variant_config(name, table)
        assert config.version == ("ats-en-v1" if name.startswith("v1_") else "ats-en-v2")
        if not name.startswith("v1_"):
            assert config.variant == name and config.relation_version == "reviewed-v1"
    assert ScorerConfig().version == "ats-en-v1"


def test_skill_only_review_never_reports_full_envelope_approval():
    corpus = {"documents": [{"kind": "resume", "skills_review_status": "approved", "skills_status": "complete"}]}
    report = annotation_summary(corpus)
    assert report["skills_reviewed_fraction"] == 1
    assert report["full_envelope_reviewed_fraction"] == 0
    assert report["provisional"] is True


def test_pair_scoring_keeps_original_labels_and_failure_detail(tmp_path):
    pair, corpus, _ = _fixtures(tmp_path)
    annotations = {item["id"]: item for item in corpus["documents"]}
    result = _row("pair", 0)["result"]
    def fake_scorer(cv, jd, **kwargs):
        assert not hasattr(cv.facts, "silver_grade")
        return SimpleNamespace(model_dump=lambda **kw: result)
    row = score_stored_pair(pair, "full", {}, annotations, ScorerConfig(), AS_OF, fake_scorer)
    assert row["silver_grade"] == 2 and row["result"]["grade"] == 0
    def broken(*args, **kwargs):
        raise RuntimeError("GPU must not be needed")
    failed = score_stored_pair(pair, "full", {}, annotations, ScorerConfig(), AS_OF, broken)
    assert failed["status"] == "failed" and failed["error_type"] == "RuntimeError"


def test_benchmark_writes_failed_pair_and_rejects_stale_artifact(tmp_path):
    _, _, corpus_path = _fixtures(tmp_path)
    def broken(*args, **kwargs):
        raise RuntimeError("retained failure")
    output = tmp_path / "benchmark"
    report = run_benchmark(tmp_path, as_of=AS_OF, corpus_path=corpus_path, output=output,
                           variants=("full",), require_population=False, scorer=broken)
    assert report["status"] == "failed"
    assert report["variants"]["full"]["gemini_primary"]["status"] == "all_failed"
    artifact = json.loads((output / "full.json").read_text())
    assert artifact["rows"][0]["error"] == "retained failure"
    corpus = json.loads(corpus_path.read_text())
    corpus["documents"][0]["skills_status"] = "complete"
    _write(corpus_path, corpus)
    with pytest.raises(ValueError, match="different benchmark identities"):
        run_benchmark(tmp_path, as_of=AS_OF, corpus_path=corpus_path, output=output,
                      variants=("full",), require_population=False, scorer=broken)
