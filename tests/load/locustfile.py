"""Load test for the matching API (checklist: concurrent users, requests per second, response time, failure rate).

  LOAD_CV_DIR=<folder of .txt CVs> uvx locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 \
      --headless -u 20 -r 2 -t 3m --csv output/review3/load/u20

Users mostly re-match CVs the service has seen (cached extraction, about 1 s each), and read history and health,
which is how a recruiter re-uses the tool. A brand-new CV takes 17-53 s of GPU time and the API runs one match at a
time (one GPU), so new CVs are measured separately with -u 1. CV texts stay outside the repository (personal data).
"""
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

CVS = [p.read_text(encoding="utf-8") for p in sorted(Path(os.environ.get("LOAD_CV_DIR", "output/review3/load/cvs")).glob("*.txt"))]


class Recruiter(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def match_known_cv(self):
        with self.client.post("/api/match", data={"text": random.choice(CVS), "top": "5"}, name="/api/match",
                              timeout=600, catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"{r.status_code}: {r.text[:120]}")

    @task(2)
    def history(self):
        self.client.get("/api/history?limit=8", name="/api/history")

    @task(1)
    def health(self):
        self.client.get("/api/health", name="/api/health")
