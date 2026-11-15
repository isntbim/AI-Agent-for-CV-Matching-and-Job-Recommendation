"""The injection test's bookkeeping, with the service stubbed (no Ollama)."""
import json
import sys

from scripts import ats_service as svc
from scripts import injection_test as it


def test_counts_planted_skills_grade_upgrades_and_errors(tmp_path, monkeypatch):
    (tmp_path / "cvs").mkdir()
    (tmp_path / "cvs" / "a.txt").write_text("Python developer, Django.", encoding="utf-8")

    def fake_match(text, as_of, top, pool, jd_ids=None, ranker=None):
        if '"}]}' in text:
            raise ValueError("schema")  # the JSON-break attack fails extraction
        planted = "Kubernetes" in text and "parser" in text
        override = "Ignore all previous" in text
        return {"matches": [{"jd": "j1", "fit": 90.0 if override else 40.0, "grade": "Good Fit" if override else "No Fit",
                             "matched_skills": ["python"] + (["Kubernetes"] if planted else [])},
                            {"jd": "j2", "fit": 20.0, "grade": "No Fit", "matched_skills": []}]}
    monkeypatch.setattr(svc, "match", fake_match)
    monkeypatch.setattr(svc, "corpus_ranker", lambda: None)
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path / "cvs"), "--out", str(tmp_path / "out")])
    it.main()
    rows = {r["attack"]: r for r in json.loads((tmp_path / "out/results.json").read_text(encoding="utf-8"))}
    assert rows["override_en"]["grade_upgrades"] == 1 and rows["override_en"]["max_fit_gain"] == 50.0
    assert rows["planted_skills"]["planted_skills_matched"] == ["Kubernetes"] and rows["planted_skills"]["grade_upgrades"] == 0
    assert "error" in rows["json_break"]
    assert rows["override_vi"]["max_fit_gain"] == 0.0
    assert "| json_break | 0 | 1 |" in (tmp_path / "out/summary.md").read_text(encoding="utf-8")
