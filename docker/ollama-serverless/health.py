"""Runpod's load-balancer health probe: GET /ping is 204 while the worker starts, 200 once /tmp/ready exists."""
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Ping(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response((200 if os.path.exists("/tmp/ready") else 204) if self.path == "/ping" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT_HEALTH", "8080"))), Ping).serve_forever()
