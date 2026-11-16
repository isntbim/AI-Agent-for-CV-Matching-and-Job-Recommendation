"""The serverless Ollama worker's /ping: 204 while the models load, 200 once ready (docker/ollama-serverless/health.py)."""
import importlib.util
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

spec = importlib.util.spec_from_file_location("health", Path(__file__).parents[2] / "docker/ollama-serverless/health.py")
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


def test_ping_is_204_until_ready_then_200(monkeypatch):
    ready = []
    monkeypatch.setattr(health.os.path, "exists", lambda path: path == "/tmp/ready" and bool(ready))
    server = ThreadingHTTPServer(("127.0.0.1", 0), health.Ping)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert urllib.request.urlopen(url + "/ping").status == 204
        ready.append(True)
        assert urllib.request.urlopen(url + "/ping").status == 200
        try:
            urllib.request.urlopen(url + "/api/tags")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()
