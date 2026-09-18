"""
Comprehensive Corpus Verification Script
Audits:
1. Pydantic v2 validation for all 361 primary structured JDs.
2. Embedding text generation for dense retrieval.
3. Raw text file integrity (non-empty, properly formatted).
4. Secondary Vietnamese corpus validation (467 JDs).
5. Detailed distribution reports across all 6 domains.
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.parser.job_schemas import JobDescriptionSchema

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def verify_corpus():
    project_root = Path(__file__).resolve().parent.parent
    primary_json_dir = project_root / "data" / "processed" / "structured_jobs"
    primary_raw_dir = project_root / "data" / "raw" / "jobs"
    vi_json_dir = project_root / "data" / "processed" / "structured_jobs_vi"
    vi_raw_dir = project_root / "data" / "raw" / "jobs_vi"

    print("=" * 80)
    print("AUDITING PRIMARY CORPUS (EN + BILINGUAL/MIXED)")
    print("=" * 80)

    primary_jsons = sorted(primary_json_dir.glob("jd_*.json"))
    print(f"Found {len(primary_jsons)} structured JSON files in {primary_json_dir}")

    primary_errors = []
    primary_domain_counts = Counter()
    primary_seniority_counts = Counter()
    primary_salary_counts = Counter()
    primary_empty_skills = 0
    primary_raw_missing_or_short = 0
    primary_total_chars = 0

    for jf in primary_jsons:
        job_id = jf.stem
        try:
            with open(jf, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            job = JobDescriptionSchema.model_validate(data)

            emb_text = job.to_embedding_text()
            if not emb_text or len(emb_text.strip()) < 50:
                primary_errors.append(f"{job_id}: to_embedding_text() too short ({len(emb_text)} chars)")
            primary_total_chars += len(emb_text)

            primary_domain_counts[job.category] += 1
            primary_seniority_counts[job.seniority.level or "Unspecified"] += 1
            if job.compensation.is_negotiable:
                primary_salary_counts["Negotiable / Undisclosed"] += 1
            else:
                primary_salary_counts[f"Numeric ({job.compensation.currency})"] += 1

            if not job.skills.required:
                primary_empty_skills += 1

            # Check corresponding raw file
            raw_path = primary_raw_dir / f"{job_id}.txt"
            if not raw_path.exists() or raw_path.stat().st_size < 100:
                primary_raw_missing_or_short += 1

        except Exception as e:
            primary_errors.append(f"{job_id}: Validation error: {str(e)}")

    print(f"Validation Status: {'PASS' if not primary_errors else 'FAIL'}")
    if primary_errors:
        print(f"Found {len(primary_errors)} errors:")
        for err in primary_errors[:10]:
            print(f"  ❌ {err}")
    else:
        print(f"  ✅ 100% of {len(primary_jsons)} Primary JDs successfully validated against JobDescriptionSchema!")
        print(f"  ✅ All {len(primary_jsons)} generate valid embedding text (avg: {primary_total_chars // len(primary_jsons)} chars/job)")
        print(f"  ✅ Raw text files check: {primary_raw_missing_or_short} missing/short files (0 expected)")
        print(f"  ✅ Jobs with 0 skills: {primary_empty_skills} (0 expected)")

    print("\n📊 Primary Corpus Breakdown by Domain:")
    for dom, cnt in sorted(primary_domain_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / len(primary_jsons)) * 100
        print(f"  • {dom:<25}: {cnt:>3} jobs ({pct:>5.1f}%)")

    print("\n📊 Primary Corpus Breakdown by Seniority:")
    for sen, cnt in sorted(primary_seniority_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / len(primary_jsons)) * 100
        print(f"  • {sen:<25}: {cnt:>3} jobs ({pct:>5.1f}%)")

    print("\n📊 Primary Corpus Breakdown by Compensation:")
    for sal, cnt in sorted(primary_salary_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / len(primary_jsons)) * 100
        print(f"  • {sal:<25}: {cnt:>3} jobs ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("AUDITING SECONDARY CORPUS (VIETNAMESE JDs)")
    print("=" * 80)

    vi_jsons = sorted(vi_json_dir.glob("jd_vi_*.json"))
    print(f"Found {len(vi_jsons)} structured JSON files in {vi_json_dir}")

    vi_errors = []
    vi_domain_counts = Counter()
    vi_raw_missing_or_short = 0
    vi_empty_skills = 0
    vi_total_chars = 0

    for jf in vi_jsons:
        job_id = jf.stem
        try:
            with open(jf, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            job = JobDescriptionSchema.model_validate(data)

            emb_text = job.to_embedding_text()
            if not emb_text or len(emb_text.strip()) < 50:
                vi_errors.append(f"{job_id}: to_embedding_text() too short")
            vi_total_chars += len(emb_text)

            vi_domain_counts[job.category] += 1
            if not job.skills.required:
                vi_empty_skills += 1

            raw_path = vi_raw_dir / f"{job_id}.txt"
            if not raw_path.exists() or raw_path.stat().st_size < 100:
                vi_raw_missing_or_short += 1

        except Exception as e:
            vi_errors.append(f"{job_id}: Validation error: {str(e)}")

    print(f"Validation Status: {'PASS' if not vi_errors else 'FAIL'}")
    if vi_errors:
        print(f"Found {len(vi_errors)} errors:")
        for err in vi_errors[:10]:
            print(f"  ❌ {err}")
    else:
        print(f"  ✅ 100% of {len(vi_jsons)} Vietnamese JDs successfully validated against JobDescriptionSchema!")
        print(f"  ✅ All {len(vi_jsons)} generate valid embedding text (avg: {vi_total_chars // len(vi_jsons)} chars/job)")
        print(f"  ✅ Raw text files check: {vi_raw_missing_or_short} missing/short files (0 expected)")
        print(f"  ✅ Jobs with 0 skills: {vi_empty_skills} (0 expected)")

    print("\n📊 Vietnamese Corpus Breakdown by Domain:")
    for dom, cnt in sorted(vi_domain_counts.items(), key=lambda x: -x[1]):
        pct = (cnt / len(vi_jsons)) * 100
        print(f"  • {dom:<25}: {cnt:>3} jobs ({pct:>5.1f}%)")

    print("\n" + "=" * 80)
    print("CONSOLIDATED GRAND TOTALS")
    print("=" * 80)
    print(f"  • Primary Benchmark (EN + Mixed) : {len(primary_jsons):>4} JDs")
    print(f"  • Secondary Corpus (Vietnamese)  : {len(vi_jsons):>4} JDs")
    print(f"  • Total Curated & Validated JDs  : {len(primary_jsons) + len(vi_jsons):>4} JDs")
    print("=" * 80)


if __name__ == "__main__":
    verify_corpus()
