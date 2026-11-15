"""Web service contract, with the models stubbed out (no Ollama, no bge-m3)."""
import pytest
from fastapi.testclient import TestClient

from scripts import ats_service as svc

CV = "Software engineer with five years of Python, Django and PostgreSQL experience building web services. " * 4


@pytest.fixture
def client(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("HISTORY_DB", str(tmp_path / "history.sqlite"))
    (tmp_path / "jd_001.txt").write_text("Backend Developer. Python, Django, PostgreSQL.", encoding="utf-8")
    monkeypatch.setattr(svc, "ROOT", tmp_path)
    monkeypatch.setattr(svc, "corpus_ranker", lambda: "RANKER")
    monkeypatch.setattr(svc, "_retrieval_index", lambda: None)  # start-up loads: the real index is git-ignored data
    monkeypatch.setattr(svc, "_llm", lambda: None)
    monkeypatch.setattr(svc, "_jd_paths", lambda: {"jd_001": "jd_001.txt"})

    def fake_match(text, as_of, top, pool, ranker, on_step=None, on_token=None):
        calls.append((text, as_of, top, ranker))
        for t in '{"skills": [' * 4 if on_token else ():
            on_token(t)
        for node in ("extract", "retrieve", "score", "rank") if on_step else ():
            on_step(node)
        return {"as_of": as_of, "retrieved": True, "pool": 20, "skipped": {}, "ranking_model": "hybrid_qwen_v1",
                "matches": [{"jd": "jd_001", "title": "Backend Developer", "fit": 81.0, "grade": "Good Fit"}]}
    monkeypatch.setattr(svc, "match", fake_match)
    from src.api.app import app
    with TestClient(app) as c:
        c.calls = calls
        yield c


def test_pasted_text_is_matched_with_the_corpus_ranker(client):
    r = client.post("/api/match", data={"text": CV, "top": "3", "as_of": "2026-11-04"})
    assert r.status_code == 200 and r.json()["matches"][0]["jd"] == "jd_001" and "job_text" in r.json()["matches"][0]
    assert client.calls == [(CV, "2026-11-04", 3, "RANKER")]


def test_bad_inputs_are_refused_with_a_reason(client):
    assert client.post("/api/match", data={"text": "too short"}).status_code == 400
    assert client.post("/api/match", files={"file": ("cv.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert client.post("/api/match", files={"file": ("cv.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain")}).status_code == 413
    assert client.post("/api/match", files={"file": ("cv.txt", b"too short", "text/plain")}).status_code == 422
    assert client.post("/api/match", data={"text": CV, "top": "99"}).status_code == 400
    assert client.post("/api/match", data={"text": CV, "as_of": "tomorrow"}).status_code == 400
    assert client.calls == []


def test_stream_sends_tokens_steps_then_the_saved_result(client):
    import json
    r = client.post("/api/match/stream", data={"text": CV, "top": "3"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [(e.split("\n")[0].removeprefix("event: "), json.loads(e.split("\n")[1].removeprefix("data: ")))
              for e in r.text.strip().split("\n\n")]
    kinds = [k for k, _ in events]
    assert kinds[0] == "token" and kinds[-1] == "result"
    assert [d["step"] for k, d in events if k == "step"] == ["extract", "retrieve", "score", "rank"]
    result = events[-1][1]
    assert result["run_id"] and "job_text" in result["matches"][0]
    assert client.get(f"/api/history/{result['run_id']}").json()["matches"][0]["jd"] == "jd_001"


def test_stream_refuses_bad_input_before_starting(client):
    assert client.post("/api/match/stream", data={"text": "too short"}).status_code == 400 and client.calls == []


def test_history_and_feedback(client):
    run = client.post("/api/match", data={"text": CV}).json()["run_id"]
    assert client.post("/api/feedback", json={"run_id": run, "jd": "jd_001", "vote": -1}).status_code == 200
    assert client.post("/api/feedback", json={"run_id": run, "jd": "jd_001", "vote": 1, "grade": "Potential Fit",
                                              "comment": "close, but no SQL"}).status_code == 200  # replaces the first vote
    assert client.get(f"/api/history/{run}").json()["feedback"] == {
        "jd_001": {"vote": 1, "grade": "Potential Fit", "comment": "close, but no SQL"}}
    assert client.get("/api/history").json()[0] | {"created": None, "seconds": None} == {
        "run_id": run, "created": None, "seconds": None, "feedback": 1,
        "top": [{"jd": "jd_001", "title": "Backend Developer", "grade": "Good Fit", "fit": 81.0}]}
    assert client.post("/api/feedback", json={"run_id": run, "jd": "jd_999", "vote": 1}).status_code == 404
    assert client.post("/api/feedback", json={"run_id": run, "jd": "jd_001", "vote": 5}).status_code == 422
    assert client.get("/api/history/999").status_code == 404


def test_metrics_count_matches_by_route_template(client):
    run = client.post("/api/match", data={"text": CV}).json()["run_id"]
    client.get(f"/api/history/{run}")
    client.post("/api/match", data={"text": "too short"})
    text = client.get("/metrics").text
    assert 'api_response_seconds_count{route="/api/match",status="200"}' in text
    assert 'api_response_seconds_count{route="/api/match",status="400"}' in text
    assert 'route="/api/history/{run_id}"' in text and f"/api/history/{run}" not in text  # one series per route
    assert "match_seconds_count" in text and "host_cpu_percent" in text and "matches_waiting 0.0" in text


def test_txt_upload_and_index_page(client):
    r = client.post("/api/match", files={"file": ("cv.txt", CV.encode(), "text/plain")})
    assert r.status_code == 200 and client.calls[0][0] == CV
    assert "CV Match" in client.get("/").text
