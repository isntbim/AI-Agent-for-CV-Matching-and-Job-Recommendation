import json
import sys

from scripts import jd_judge_apply


def test_later_row_file_indexes_the_label_left_by_earlier_files(tmp_path, monkeypatch):
    (tmp_path / "gemini").mkdir()
    label = {"skills": {"requirements": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}}
    (tmp_path / "gemini" / "jd_001.json").write_text(json.dumps(label), encoding="utf-8")
    # round 1 drops a; round 2 then sees [b, c] and edits index 1 (= c)
    (tmp_path / "r1.json").write_text(json.dumps([
        {"id": "jd_001", "path": "skills.requirements[0]", "from": {"id": "a"}, "to": "DROP"}]), encoding="utf-8")
    (tmp_path / "r2.json").write_text(json.dumps([
        {"id": "jd_001", "path": "skills.requirements[1]", "from": {"id": "c"}, "to": {"id": "c2"}}]), encoding="utf-8")
    monkeypatch.setattr(jd_judge_apply, "JD", tmp_path)
    monkeypatch.setattr(sys, "argv", ["jd_judge_apply", "--rows", "r1.json", "r2.json"])
    jd_judge_apply.main()
    out = json.loads((tmp_path / "reference" / "jd_001.json").read_text(encoding="utf-8"))
    assert out["skills"]["requirements"] == [{"id": "b"}, {"id": "c2"}]
