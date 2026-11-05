"""Web service contract, with the models stubbed out (no Ollama, no bge-m3)."""
import pytest
from fastapi.testclient import TestClient

from scripts import ats_service as svc

CV = "Software engineer with five years of Python, Django and PostgreSQL experience building web services. " * 4


@pytest.fixture
def client(monkeypatch):
    calls = []
    monkeypatch.setattr(svc, "corpus_ranker", lambda: "RANKER")
    monkeypatch.setattr(svc, "_jd_paths", lambda: {"jd_001": "output/retrieval/corpus_v2/manifest.json"})

    def fake_match(text, as_of, top, pool, ranker):
        calls.append((text, as_of, top, ranker))
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
    assert client.post("/api/match", data={"text": CV, "top": "99"}).status_code == 400
    assert client.post("/api/match", data={"text": CV, "as_of": "tomorrow"}).status_code == 400
    assert client.calls == []


def test_txt_upload_and_index_page(client):
    r = client.post("/api/match", files={"file": ("cv.txt", CV.encode(), "text/plain")})
    assert r.status_code == 200 and client.calls[0][0] == CV
    assert "CV Match" in client.get("/").text
