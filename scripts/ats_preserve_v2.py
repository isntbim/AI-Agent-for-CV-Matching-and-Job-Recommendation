"""Preserve the pre-v3 working tree and local v2 research artifacts, once."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "output/scorer_v3/frozen_v2"
    if target.exists():
        raise SystemExit("Frozen baseline already exists; refusing to overwrite it")
    target.mkdir(parents=True)
    selected = set()
    # Include ignored source files and relevant untracked implementation files.
    for folder in ("src/scoring", "src/parser", "tests/unit", "scripts"):
        selected.update(p for p in (root / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".mjs"})
    for name in (".gitignore", ".miruignore", "requirements.txt", "pyproject.toml",
                 "docs/guides/ats_scorer.md", "docs/explanations/scorer_explanation.md",
                 "docs/reference/scorer_references.md", "knowledge-base/sessions/ats-scorer/scorer_v2_handoff.md"):
        path = root / name
        if path.is_file():
            selected.add(path)
    for folder in ("output/scorer_v2", "human-verify/scorer-v2", "output/ats/silver_full", "data/silver"):
        selected.update(p for p in (root / folder).rglob("*") if p.is_file())
    manifest = []
    with zipfile.ZipFile(target / "snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative = path.relative_to(root).as_posix()
            archive.write(path, relative)
            manifest.append({"path": relative, "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {"version": "ats-frozen-v2-v1", "preserved_on": "2026-10-08",
              "python": platform.python_version(), "platform": platform.platform(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
              "files": manifest,
              "snapshot_sha256": hashlib.sha256((target / "snapshot.zip").read_bytes()).hexdigest()}
    (target / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    receipt = root / "output/scorer_v2/verification.json"
    if receipt.is_file():
        shutil.copy2(receipt, target / "verification.json")
    print(json.dumps({"files": len(manifest), "snapshot_sha256": result["snapshot_sha256"], "target": str(target)}))


if __name__ == "__main__":
    main()
