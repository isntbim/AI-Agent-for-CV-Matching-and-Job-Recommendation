import json

from scripts.qwen_eval import parse, prf, quotes, score, skills


def test_parse_strips_fences_and_rejects_non_objects():
    assert parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse("[1, 2]") is None
    assert parse("not json") is None


def test_skills_and_quotes_shapes():
    cv = {"document": {"skills": [{"details": [{"name": "Excel "}, {"name": "SQL"}]}]}}
    jd = {"document": {"skills": {"required": [{"name": "SQL"}], "preferred": [{"name": "Redshift"}]}}}
    assert skills(cv, "cv_extraction") == {"excel", "sql"}
    assert skills(jd, "jd_extraction") == {"sql", "redshift"}
    assert quotes({"facts": {"evidence": {"skills": [{"quote": "a"}, {"quote": "b"}]}}}) == ["a", "b"]


def test_wrong_typed_parts_count_as_empty():
    assert skills({"document": "a string"}, "cv_extraction") == set()
    assert skills({"document": {"skills": ["Excel", {"details": "x"}]}}, "cv_extraction") == set()
    assert skills({"document": {"skills": [{"name": "SQL"}]}}, "jd_extraction") == set()
    line = (__import__("scripts.qwen_eval", fromlist=["DATA"]).DATA / "cv_extraction/test.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    bad = json.dumps({"document": "x", "facts": {"domains": "finance", "completed_degrees": "bachelor"}})
    r = score([{"id": row["id"], "task": "cv_extraction", "split": "test", "prediction": bad, "finish_reason": "stop"}], boot=20)
    assert r["cv_extraction"]["json_parse_rate"]["value"] == 1.0 and r["cv_extraction"]["skills_recall"]["value"] == 0.0
    assert r["cv_extraction"]["domains_f1"]["value"] == 0.0


def test_relaxed_matches_is_one_to_one():
    from scripts.qwen_eval import relaxed_matches
    assert relaxed_matches({"us gaap"}, {"us gaap principles"}) == 1
    assert relaxed_matches({"reconciliations"}, {"reconciliation"}) == 1
    assert relaxed_matches({"excel"}, {"excel", "advanced excel"}) == 1  # one prediction can't match two gold skills
    assert relaxed_matches({"java"}, {"javascript"}) == 0  # word-level, not substring
    assert relaxed_matches(set(), {"sql"}) == 0


def test_prf_handles_empty():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert prf(1, 2, 1)["f1"] == 0.667


def test_gold_prediction_scores_perfectly():
    line = (__import__("scripts.qwen_eval", fromlist=["DATA"]).DATA / "cv_extraction/test.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    pred = {"id": row["id"], "task": "cv_extraction", "split": "test", "prediction": row["messages"][2]["content"], "finish_reason": "stop"}
    r = score([pred], boot=20)["cv_extraction"]
    for key in ("json_parse_rate", "schema_valid_rate", "skills_f1", "completed_degrees_accuracy", "quote_faithfulness"):
        assert r[key] == {"value": 1.0, "ci95": [1.0, 1.0]}, key


def test_unparseable_prediction_scores_zero():
    line = (__import__("scripts.qwen_eval", fromlist=["DATA"]).DATA / "cv_extraction/test.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    r = score([{"id": row["id"], "task": "cv_extraction", "split": "test", "prediction": "oops", "finish_reason": "length"}], boot=20)
    assert r["cv_extraction"]["json_parse_rate"]["value"] == 0.0 and r["cv_extraction"]["truncated"] == 1
    assert r["cv_extraction"]["skills_recall"]["value"] == 0.0
