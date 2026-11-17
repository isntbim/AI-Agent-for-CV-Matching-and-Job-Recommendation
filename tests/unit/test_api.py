"""Web service contract, with the models stubbed out (no Ollama, no bge-m3). Every test runs on the SQLite history and,
when PG_TEST_DSN points at a PostgreSQL server (as in CI), again on a fresh PostgreSQL history database."""
import os
import uuid

import pytest
from fastapi.testclient import TestClient

from scripts import ats_service as svc

CV = "Software engineer with five years of Python, Django and PostgreSQL experience building web services. " * 4


@pytest.fixture(params=["sqlite", "postgres"])
def history(request, monkeypatch, tmp_path):
    monkeypatch.delenv("HISTORY_DSN", raising=False)
    monkeypatch.setenv("HISTORY_DB", str(tmp_path / "history.sqlite"))
    if request.param == "sqlite":
        yield
        return
    base = os.environ.get("PG_TEST_DSN") or pytest.skip("PG_TEST_DSN not set: PostgreSQL history not tested")
    import psycopg
    from psycopg.conninfo import make_conninfo
    name = f"history_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f"CREATE DATABASE {name}")
    monkeypatch.setenv("HISTORY_DSN", make_conninfo(base, dbname=name))
    yield
    with psycopg.connect(base, autocommit=True) as admin:
        admin.execute(f"DROP DATABASE {name} WITH (FORCE)")


@pytest.fixture
def client(monkeypatch, tmp_path, history):
    calls = []
    (tmp_path / "jd_001.txt").write_text("Backend Developer. Python, Django, PostgreSQL.", encoding="utf-8")
    monkeypatch.setattr(svc, "ROOT", tmp_path)
    monkeypatch.setattr(svc, "corpus_ranker", lambda: "RANKER")
    monkeypatch.setattr(svc, "_retrieval_index", lambda: None)  # start-up loads: the real index is git-ignored data
    monkeypatch.setattr(svc, "_llm", lambda: None)
    monkeypatch.setattr(svc, "_jd_paths", lambda: {"jd_001": "jd_001.txt"})

    def fake_match(text, as_of, top, pool, ranker, on_step=None, on_token=None, jd_text=None):
        calls.append((text, as_of, top, ranker) if jd_text is None else (text, as_of, top, ranker, jd_text))
        for t in '{"skills": [' * 4 if on_token else ():
            on_token(t)
        nodes = ("extract", "score", "rank") if jd_text is not None else ("extract", "retrieve", "score", "rank")
        for node in nodes if on_step else ():
            on_step(node)
        if jd_text is not None:
            return {"as_of": as_of, "retrieved": False, "pool": 1, "skipped": {}, "ranking_model": "hybrid_qwen_v1",
                    "matches": [{"jd": "pasted", "title": "Pasted Backend Role", "fit": 88.5, "grade": "Good Fit",
                                 "job_text": jd_text}]}
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


def test_remote_ollama_is_never_polled(client, monkeypatch):
    """With a serverless Ollama (OLLAMA_API_KEY set), health checks and metric scrapes must not wake a billed GPU worker."""
    import httpx

    from src.scoring.v4_client import ollama_headers
    monkeypatch.setenv("OLLAMA_API_KEY", "rpa_test")
    monkeypatch.setattr(svc, "corpus_ids", lambda: ["jd_001"])
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError(f"polled Ollama: {a}")))
    assert client.get("/api/health").json()["extractor"].startswith("remote")
    assert client.get("/metrics").status_code == 200
    assert ollama_headers() == {"Authorization": "Bearer rpa_test"}
    monkeypatch.delenv("OLLAMA_API_KEY")
    assert ollama_headers() == {}


def test_pasted_jd_validation_with_jobs_stream_and_history_reopen(client, monkeypatch):
    import json
    from fastapi import HTTPException
    from src.api import app as app_mod
    from src.parser.llm_extractor import ExtractionError

    jd = ("We are hiring a Backend Python Engineer to design APIs with FastAPI, PostgreSQL, and Docker. " * 3).strip()

    # Direct helper checks: inputs() validation (200..12000) and with_jobs() preservation / no KeyError
    assert app_mod.inputs(None, CV, 5, "2026-11-04", "   \r\n\t  ")[2] is None
    assert app_mod.inputs(None, CV, 5, "2026-11-04", f"  {jd}  ")[2] == jd
    assert len(app_mod.inputs(None, CV, 5, "2026-11-04", "x" * 12000)[2]) == 12000
    with pytest.raises(HTTPException) as short_exc:
        app_mod.inputs(None, CV, 5, "2026-11-04", "x" * 199)
    assert short_exc.value.status_code == 400
    with pytest.raises(HTTPException) as long_exc:
        app_mod.inputs(None, CV, 5, "2026-11-04", "x" * 12001)
    assert long_exc.value.status_code == 400

    preserved = app_mod.with_jobs({"matches": [
        {"jd": "pasted", "job_text": "custom pasted posting"},
        {"jd": "jd_001", "job_text": "already present"},
        {"jd": "pasted"},
    ]})
    assert preserved["matches"][0]["job_text"] == "custom pasted posting"
    assert preserved["matches"][1]["job_text"] == "already present"
    assert "job_text" not in preserved["matches"][2]

    # Route validation: too short / >12000 -> 400; blank -> corpus mode
    assert client.post("/api/match", data={"text": CV, "jd_text": "too short"}).status_code == 400
    assert client.post("/api/match/stream", data={"text": CV, "jd_text": "too short"}).status_code == 400
    assert client.post("/api/match", data={"text": CV, "jd_text": "x" * 12001}).status_code == 400
    assert client.calls == []

    blank = client.post("/api/match", data={"text": CV, "as_of": "2026-11-04", "jd_text": "   "})
    assert blank.status_code == 200 and blank.json()["matches"][0]["jd"] == "jd_001"

    # Streamed pasted-JD match -> skips retrieve, saves job_text, reopens from /api/history/{run_id}, accepts feedback
    r = client.post("/api/match/stream", data={"text": CV, "as_of": "2026-11-04", "jd_text": f"\r\n{jd}\r\n"})
    assert r.status_code == 200
    events = [(e.split("\n")[0].removeprefix("event: "), json.loads(e.split("\n")[1].removeprefix("data: ")))
              for e in r.text.strip().split("\n\n")]
    assert [d["step"] for k, d in events if k == "step"] == ["extract", "score", "rank"]
    result = events[-1][1]
    assert result["retrieved"] is False and result["matches"][0]["jd"] == "pasted"
    assert result["matches"][0]["job_text"] == jd

    run_id = result["run_id"]
    reopened = client.get(f"/api/history/{run_id}").json()
    assert reopened["matches"][0]["jd"] == "pasted" and reopened["matches"][0]["job_text"] == jd
    assert client.get("/api/history").json()[0]["top"][0]["jd"] == "pasted"
    assert client.post("/api/feedback", json={"run_id": run_id, "jd": "pasted", "vote": 1}).status_code == 200

    # Addendum B: pasted-JD ExtractionError -> 422 JD-specific message on /api/match and SSE error event on /api/match/stream
    def fail_jd_match(*args, **kwargs):
        err = ExtractionError("JD context budget exceeded")
        err.source = "jd"
        raise err

    monkeypatch.setattr(svc, "match", fail_jd_match)
    sync_err = client.post("/api/match", data={"text": CV, "jd_text": jd})
    assert sync_err.status_code == 422 and "job description could not be read reliably" in sync_err.json()["detail"].lower()

    sse_err = client.post("/api/match/stream", data={"text": CV, "jd_text": jd})
    assert sse_err.status_code == 200
    err_events = [(e.split("\n")[0].removeprefix("event: "), json.loads(e.split("\n")[1].removeprefix("data: ")))
                  for e in sse_err.text.strip().split("\n\n")]
    assert err_events[-1][0] == "error" and err_events[-1][1]["status"] == 422
    assert "job description could not be read reliably" in err_events[-1][1]["detail"].lower()

    # CV ExtractionError during a pasted-JD request still reports the CV message
    def fail_cv_match(*args, **kwargs):
        err = ExtractionError("CV unreadable")
        err.source = "cv"
        raise err

    monkeypatch.setattr(svc, "match", fail_cv_match)
    cv_err = client.post("/api/match", data={"text": CV, "jd_text": jd})
    assert cv_err.status_code == 422 and "cv could not be read reliably" in cv_err.json()["detail"].lower()
