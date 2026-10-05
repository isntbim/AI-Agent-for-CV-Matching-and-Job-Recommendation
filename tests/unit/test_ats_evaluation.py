"""Data split, calibration, cache and pipeline contract tests."""

import csv
import json

import pytest

from scripts.ats_eval import main
from src.parser.llm_extractor import ExtractionError, MockLLMClient
from src.scoring.dataset import LABELS, load_pairs, prepare_dataset
from src.scoring.evaluation import calibrate, classification_metrics, evaluate
from src.scoring.extraction import (
    ExtractionCache,
    ScoringOllamaClient,
    extract_document,
    validate_evidence,
)
from src.scoring.models import Evidence, Facts


def csv_file(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["resume_text", "job_description_text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def population():
    return [{"resume_text": f"Candidate {i} software engineer Python bachelor 2020-01 to Present US",
                 "job_description_text": f"Job {j} software engineer Python bachelor five years US",
                 "label": list(LABELS)[(i + j) % 3]} for i in range(24) for j in range(24)]


def test_disjoint_split_duplicates_conflicts_and_reproducibility(tmp_path):
    rows = population()
    conflict = dict(rows[0], label="Potential Fit")
    csv_file(tmp_path / "data.csv", rows + [rows[1], conflict])
    first = prepare_dataset([tmp_path / "data.csv"], tmp_path / "a")
    second = prepare_dataset([tmp_path / "data.csv"], tmp_path / "b")
    assert first == second
    assert first["duplicates"] == 2
    assert first["excluded_by_reason"]["conflicting_labels"] == 1
    assert first["ready_for_calibration"]
    _, pairs = load_pairs(tmp_path / "a")
    for kind in ["cv_id", "jd_id"]:
        assert not ({p[kind] for p in pairs if p["split"] == "development"}
                    & {p[kind] for p in pairs if p["split"] == "test"})
    assert {p["label"] for p in pairs} == set(LABELS)
    assert json.loads((tmp_path / "a" / "excluded.json").read_text())


def test_missing_class_and_bad_labels(tmp_path):
    csv_file(tmp_path / "data.csv", [{"resume_text": "CV", "job_description_text": "JD", "label": "No Fit"}])
    assert not prepare_dataset([tmp_path / "data.csv"], tmp_path / "prepared")["ready_for_calibration"]
    csv_file(tmp_path / "data.csv", [{"resume_text": "CV", "job_description_text": "JD", "label": "Hired"}])
    with pytest.raises(ValueError, match="Unknown label"):
        prepare_dataset([tmp_path / "data.csv"], tmp_path / "prepared")


def rows_for_metrics(split="development"):
    return [{"split": split, "label_id": label, "pair_id": str(label), "cv_id": str(label), "baseline_skill_score": score,
                 "result": {"overall_score": score, "scores": {"skills": score, "title": score}, "coverage": .93,
                             "dimensions": {"skills": {"supported": True}}}} for label, score in [(0, 10), (1, 60), (2, 90)]]


def test_calibration_and_metrics_known_answers():
    config = calibrate(rows_for_metrics())
    assert (config.lower_threshold, config.upper_threshold) == (50, 75)
    report = evaluate(rows_for_metrics("test"), config, calibrate(rows_for_metrics(), baseline=True), 0, bootstrap=20)
    assert report["scorer"]["macro_f1"] == 1
    assert report["majority_baseline"]["accuracy"] == pytest.approx(1 / 3)
    assert report["score_overlap"]["0-1"] == 0
    assert report["scores_by_label"]["1"]["mean"] == 60
    assert report["bootstrap"]["iterations"] == 20
    with pytest.raises(ValueError, match="development"):
        calibrate(rows_for_metrics("test"))
    with pytest.raises(ValueError, match="held-out"):
        evaluate(rows_for_metrics(), config, config, 0)
    with pytest.raises(ValueError):
        classification_metrics([], [])


class EnvelopeClient:
    """Test-only response client. Production CLI exposes only real backends."""

    def __init__(self, bad_quote=False):
        self.calls = []
        self.bad_quote = bad_quote

    def generate(self, system_prompt, user_prompt, **kwargs):
        self.calls.append((system_prompt, user_prompt))
        source = json.loads(user_prompt)
        quote = "invented quote" if self.bad_quote else "software engineer"
        facts = {"evidence": {key: [{"quote": quote}] for key in ["skills", "title", "experience", "education", "industry", "location"]},
                     "domains": ["software"], "country": "US"}
        if source["document_kind"] == "resume":
            document = {"basics": {"label": "Software Engineer", "location": {"city": "Boston", "countryCode": "US"}},
                            "skills": [{"name": "Technical", "keywords": ["Python"]}],
                            "work": [{"position": "Software Engineer", "startDate": "2020-01", "endDate": "Present"}]}
            facts["completed_degrees"] = ["bachelor"]
        else:
            document = {"title": "Software Engineer", "title_normalized": "software_engineer", "category": "software",
                            "skills": {"required": [{"name": "Python"}]}, "seniority": {"min_years": 5},
                            "location": {"city": "Boston", "country": "US", "countryCode": "US"}}
            facts["minimum_degree"] = "bachelor"
        return json.dumps({"document": document, "facts": facts})


def test_envelope_reuses_cv_extraction_and_cache(tmp_path):
    text = "software engineer Python bachelor US " + "long text " * 1300
    client = EnvelopeClient()
    cache = ExtractionCache(tmp_path, client, "fixture", "test-model")
    resume = cache.get(text, "resume")
    assert resume.document.basics.label == "Software Engineer"
    assert len(client.calls) == 1
    assert json.loads(client.calls[0][1])["source_text"] == text  # No old 12k truncation.
    assert cache.get(text, "resume") == resume
    assert len(client.calls) == 1
    ExtractionCache(tmp_path, client, "fixture", "different-model").get(text, "resume")
    assert len(client.calls) == 2
    assert "label" not in json.loads(client.calls[0][1])


def test_bad_quotes_fail_and_retry_instead_of_caching_fake_success(tmp_path):
    client = EnvelopeClient(bad_quote=True)
    cache = ExtractionCache(tmp_path, client, "fixture", "model")
    with pytest.raises(ExtractionError, match="literal"):
        cache.get("software engineer", "job")
    assert len(list(tmp_path.glob("*.error.json"))) == 1
    assert len(list((tmp_path / "failure_archive").glob("*.json"))) == 1
    client.bad_quote = False
    assert cache.get("software engineer", "job").document.title == "Software Engineer"
    assert not list(tmp_path.glob("*.error.json"))
    assert len(list((tmp_path / "failure_archive").glob("*.json"))) == 1


def test_mock_and_oversized_inputs_rejected():
    with pytest.raises(ExtractionError, match="real"):
        extract_document("text", "resume", MockLLMClient())
    with pytest.raises(ExtractionError, match="truncation"):
        extract_document("x" * 10, "resume", EnvelopeClient(), max_chars=5)


def test_layout_whitespace_restores_original_quote_but_paraphrases_fail():
    facts = Facts(evidence={"skills": [Evidence(quote="Python SQL")]})
    validate_evidence(facts, "Skills: Python\n    SQL")
    assert facts.evidence["skills"][0].quote == "Python\n    SQL"
    with pytest.raises(ExtractionError, match="literal source"):
        validate_evidence(Facts(evidence={"skills": [Evidence(quote="Python and SQL")]}), "Python SQL")


def test_ollama_context_budget_and_generation_payload(monkeypatch):
    client = ScoringOllamaClient("http://fixture", "model", context_window=2048)
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "{}", "done_reason": "stop"}

    def post(url, **kwargs):
        captured.update(kwargs["json"])
        return Response()

    monkeypatch.setattr("src.scoring.extraction.httpx.post", post)
    assert client.generate("system", '{"document_kind":"resume"}', max_tokens=128) == "{}"
    assert captured["options"]["num_ctx"] == 2048
    assert captured["format"]["type"] == "object"
    assert set(captured["format"]["required"]) == {"document", "facts"}
    evidence_schema = captured["format"]["$defs"]["Facts"]["properties"]["evidence"]
    assert evidence_schema["additionalProperties"] is False
    assert set(evidence_schema["properties"]) == {"skills", "title", "experience", "education", "industry", "location", "salary"}
    with pytest.raises(ExtractionError, match="context budget"):
        client.generate("system", "x" * 2048, max_tokens=128)


def test_default_country_currency_not_evidence():
    job = extract_document("software engineer", "job", EnvelopeClient())
    assert job.document.compensation.currency == "VND"
    assert job.facts.currency is None
    assert not job.facts.has("salary")


def test_cli_cached_pipeline_and_identity_guards(tmp_path, capsys):
    prepared, cache_path, results = tmp_path / "prepared", tmp_path / "cache", tmp_path / "results"
    source = tmp_path / "data.csv"
    csv_file(source, population())
    prepare_dataset([source], prepared)
    _, pairs = load_pairs(prepared)
    cache = ExtractionCache(cache_path, EnvelopeClient(), "vllm:http://fixture", "test-model")
    for pair in pairs:
        cache.get(pair["resume_text"], "resume")
        cache.get(pair["job_description_text"], "job")
    common = ["--prepared", str(prepared), "--cache", str(cache_path), "--output", str(results),
              "--backend", "vllm", "--endpoint", "http://fixture", "--model", "test-model", "--as-of", "2026-09-30"]
    main(["score", "--split", "development", *common])
    calibration = tmp_path / "calibration.json"
    main(["calibrate", "--prepared", str(prepared), "--scores", str(results / "development_scores.json"), "--output", str(calibration)])
    main(["score", "--split", "test", "--calibration", str(calibration), *common])
    main(["report", "--scores", str(results / "test_scores.json"), "--calibration", str(calibration), "--output", str(results), "--bootstrap", "5"])
    assert (results / "report.md").exists()
    assert json.loads((results / "report.json").read_text())["failure_rate"] == 0
    changed = common[:-1] + ["2026-08-30"]
    with pytest.raises(ValueError, match="identity differs"):
        main(["score", "--split", "test", "--calibration", str(calibration), *changed])
    with pytest.raises(SystemExit):
        main(["score", "--split", "test", *common])
