import json

from scripts.qwen_eval import DATA
from scripts.qwen_failure_analysis import analyse


def first_row() -> dict:
    return json.loads((DATA / "cv_extraction/test.jsonl").read_text(encoding="utf-8").splitlines()[0])


def pred(row: dict, prediction: str, finish: str = "stop") -> dict:
    return {"id": row["id"], "task": "cv_extraction", "split": "test", "prediction": prediction, "finish_reason": finish}


def test_gold_prediction_has_no_failures():
    row = first_row()
    o = analyse([pred(row, row["messages"][2]["content"])])["cv_extraction"]
    assert o["n"] == 1 and sum(o["counts"].values()) == 0


def test_truncated_and_unparseable_buckets():
    row = first_row()
    o = analyse([pred(row, '{"document": ', "length"), pred(row, "oops")])["cv_extraction"]
    assert o["counts"]["truncated"] == 1 and o["counts"]["unparseable"] == 1


def test_invented_quote_counts():
    row = first_row()
    label = json.loads(row["messages"][2]["content"])
    label["facts"]["evidence"].setdefault("skills", []).append({"quote": "zzz not in the source qqq"})
    o = analyse([pred(row, json.dumps(label))])["cv_extraction"]
    assert o["counts"]["invented_quotes"] == 1 and o["quotes_invented"] == 1


def test_task_mismatch_flags_other_task_format():
    from scripts.qwen_failure_analysis import task_mismatch
    assert task_mismatch({"document": {"schema_version": "resume-v3", "work": []}, "facts": {}}, "cv_extraction") is None
    assert "schema_version" in task_mismatch({"document": {"schema_version": "resume-v3"}, "facts": {}}, "jd_extraction")
    assert task_mismatch({"document": {}, "facts": {}}, "jd_extraction") is None  # missing version: format error only
    assert "other-task keys work" in task_mismatch({"document": {"schema_version": "job-v3", "work": []}, "facts": {}}, "jd_extraction")
    assert "completed_degrees" in task_mismatch({"document": {"schema_version": "job-v3"}, "facts": {"completed_degrees": ["bachelor"]}}, "jd_extraction")
    assert "company" in task_mismatch({"document": {"schema_version": "resume-v3", "company": {}}, "facts": {}}, "cv_extraction")
