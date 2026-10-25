"""Add exact source inputs and an executable source tree to the frozen v2 record."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "output/scorer_v3/frozen_v2"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if (target / "inputs_manifest.json").exists():
        raise SystemExit("Supplement already preserved")
    corpus = json.loads((root / "output/scorer_v2/corpus.json").read_text(encoding="utf-8"))
    paths = {root / d["source"][key] for d in corpus["documents"] for key in ("source_path", "text_path") if d["source"].get(key)}
    paths.update(root / p for p in ("data/datasets/matching_pairs/benchmark_matching_pairs.json", "data/processed/structured_cvs/all_cvs.json",
                                   "data/processed/structured_jobs/all_jobs_en.json", "output/extraction_pilot/disagreement_nominations.json",
                                   "output/extraction_pilot/disagreement_review.json"))
    # Add runtime dependencies outside the original scorer/parser snapshot.
    paths.update(p for p in (root / "src").rglob("*.py") if not p.is_relative_to(root / "src/scoring") and not p.is_relative_to(root / "src/parser"))
    files = []
    with zipfile.ZipFile(target / "inputs.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if not path.is_file():
                continue
            name = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            archive.writestr(name, payload)
            files.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
    record = {"version": "ats-frozen-v2-inputs-v1", "files": files,
              "source_snapshot_sha256": manifest["snapshot_sha256"],
              "inputs_sha256": hashlib.sha256((target / "inputs.zip").read_bytes()).hexdigest()}
    (target / "inputs_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    # Only extract executable code and baseline data necessary to reproduce it.
    tree = target / "worktree"
    tree.mkdir()
    for archive_name in ("snapshot.zip", "inputs.zip"):
        with zipfile.ZipFile(target / archive_name) as archive:
            for name in archive.namelist():
                if name.startswith(("src/", "scripts/", "tests/", "data/processed/", "output/extraction_pilot/")) or name in {"output/scorer_v2/corpus.json", "output/scorer_v2/relations.json"}:
                    archive.extract(name, tree)
    print(json.dumps({"source_files": len(manifest["files"]), "input_files": len(files), "inputs_sha256": record["inputs_sha256"]}))


if __name__ == "__main__":
    main()
