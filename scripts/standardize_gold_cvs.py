"""
Standardize Gold CV Corpus (Priority 1 / Step 1 — Silver Matching Pairs Pipeline)

Reads all 88 human-audited gold CV JSONs from data/evaluation/{DOMAIN}/*.json and:
1. Assigns sequential IDs cv_001..cv_088 ordered by domain alphabetically.
2. Injects metadata: cv_id, domain (normalized snake_case matching JD categories),
   quality_tier ("gold"), original_file.
3. Validates 100% against ResumeSchema (Pydantic v2) and verifies to_embedding_text()
   is non-empty.
4. Writes individual files to data/processed/structured_cvs/cv_NNN.json.
5. Writes consolidated indexes: gold_cvs.json and all_cvs.json.

NOTE on metadata: ResumeSchema is configured with extra="ignore", so the injected
metadata keys (cv_id, domain, ...) are NOT schema fields: they are stored as sibling
keys in the JSON (surviving round-trips because re-validation ignores them) and are
read back from the raw dict by downstream scripts (e.g. generate_silver_matching_pairs.py).
Dump uses by_alias=True so stored keys keep the JSON Resume camelCase aliases
(e.g. startDate, studyType, postalCode) exactly as authored in data/evaluation/.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parser.schemas import ResumeSchema

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# Canonical domain normalization: evaluation folder name -> JD category
DOMAIN_MAP = {
    "ACCOUNTANT": "accountant",
    "Data Scientist Analyst": "data_scientist_analyst",
    "FrontendWebDeveloper": "frontend_web_developer",
    "HR": "hr",
    "Marketing Executive": "marketing_executive",
    "Software Engineer": "software_engineer",
}

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "structured_cvs"


def collect_gold_records(eval_root: Path):
    """Discover all gold CV JSONs grouped by domain (sorted for determinism).

    Returns (ordered_domains, records) where records is a list of
    (domain, folder_name, cv_index_within_domain, original_path).
    """
    records = []
    for folder in sorted(eval_root.iterdir()):
        if not folder.is_dir() or folder.name not in DOMAIN_MAP:
            continue
        jsons = sorted(folder.glob("*.json"))
        if not jsons:
            print(f"  WARNING: no JSON files in {folder.name} — skipped")
            continue
        records.append((folder.name, jsons))
    return records


def main():
    project_root = Path(__file__).resolve().parent.parent
    eval_root = project_root / "data" / "evaluation"

    print("=" * 80)
    print("STANDARDIZING GOLD CV CORPUS (88 CVs)")
    print("=" * 80)

    if not eval_root.is_dir():
        print(f"FATAL: evaluation directory not found: {eval_root}")
        sys.exit(1)

    groups = collect_gold_records(eval_root)
    if not groups:
        print("FATAL: no evaluation domains found")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    errors = []
    empty_embedding = []
    written = []
    next_cv_id = 1  # cv_001 across all domains, ordered alphabetically

    for folder_name, jsons in groups:
        domain = DOMAIN_MAP[folder_name]
        print(f"\n[{folder_name}] -> domain '{domain}' ({len(jsons)} CVs)")
        for jf in jsons:
            cv_id = f"cv_{next_cv_id:03d}"
            next_cv_id += 1
            try:
                with open(jf, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)

                resume = ResumeSchema.model_validate(raw)
                embedding_text = resume.to_embedding_text()
                if not embedding_text.strip():
                    empty_embedding.append((cv_id, jf.name))

                record = {
                    **resume.model_dump(by_alias=True, exclude_none=False),
                    "cv_id": cv_id,
                    "domain": domain,
                    "quality_tier": "gold",
                    "original_file": jf.name,
                }

                out_path = OUT_DIR / f"{cv_id}.json"
                with open(out_path, "w", encoding="utf-8") as fp:
                    json.dump(record, fp, ensure_ascii=False, indent=2)
                written.append(record)
                print(f"  {cv_id}  <- {jf.name}  "
                      f"(skills={len(record['skills'])}, work={len(record['work'])}, "
                      f"edu={len(record['education'])})")
            except Exception as exc:  # noqa: BLE001 — report all failures
                errors.append((cv_id, jf.name, str(exc)))
                print(f"  {cv_id}  !! FAILED: {jf.name}: {exc}")

    total = len(written)
    print("=" * 80)
    print(f"VALIDATION SUMMARY")
    print(f"  Files written      : {total}")
    print(f"  Validation errors  : {len(errors)}")
    print(f"  Empty embedding    : {len(empty_embedding)}")

    if errors:
        print("\n  ERRORS (must be 0):")
        for cv_id, name, msg in errors:
            print(f"    {cv_id} {name}: {msg}")
        sys.exit(2)

    if empty_embedding:
        print("\n  WARNINGS (embedding text empty — schema valid but semantically empty):")
        for cv_id, name in empty_embedding:
            print(f"    {cv_id} {name}")
        sys.exit(3)

    # Consolidated indexes
    gold_index = OUT_DIR / "gold_cvs.json"
    all_index = OUT_DIR / "all_cvs.json"
    # all_cvs.json is the superset index; today it equals the gold set (no silver tier yet).
    with open(gold_index, "w", encoding="utf-8") as fp:
        json.dump(written, fp, ensure_ascii=False, indent=2)
    with open(all_index, "w", encoding="utf-8") as fp:
        json.dump(written, fp, ensure_ascii=False, indent=2)

    print(f"\n  Indexes: {gold_index.name}, {all_index.name} ({total} records)")

    # Per-domain tally for the console
    from collections import Counter
    tally = Counter(r["domain"] for r in written)
    print("\n  Domain breakdown:")
    for d in sorted(tally):
        print(f"    {d}: {tally[d]}")

    print("\n  DONE — 100% schema validation passed, embedding text non-empty.")


if __name__ == "__main__":
    main()