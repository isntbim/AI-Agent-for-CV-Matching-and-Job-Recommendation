"""
Repair and Audit Script for Job Descriptions (data/raw/jobs and data/processed/structured_jobs).
Extracts missing job descriptions from JSON-LD schema metadata, parses bullets,
updates structured JSONs, and generates full raw text files.
"""

import json
import logging
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.parser.job_schemas import JobDescriptionSchema
from src.scraper.import_itviec import (
    clean_bullet,
    detect_seniority_level,
    extract_years_experience,
    normalize_title_category,
    parse_bullets,
    parse_compensation,
    parse_location,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def fetch_jsonld_jobposting(url: str) -> Optional[Dict]:
    """Fetch and parse schema.org/JobPosting from URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        matches = re.findall(
            r'<script type=[\'"]application/ld\+json[\'"]>(.*?)</script>', html, re.DOTALL
        )
        for m in matches:
            if "JobPosting" in m:
                try:
                    return json.loads(m.strip())
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Failed to fetch JSON-LD for {url}: {e}")
    return None


def split_description_and_requirements(full_html: str) -> Tuple[str, str]:
    """Split raw HTML/text into description/responsibilities part and requirements part."""
    # Split text by common section breaks
    markers = [
        r"your skills and experience",
        r"yêu cầu bắt buộc",
        r"yêu cầu công việc",
        r"job requirements",
        r"what you need",
        r"qualifications",
    ]
    pattern = "|".join(markers)
    parts = re.split(pattern, full_html, maxsplit=1, flags=re.IGNORECASE)

    if len(parts) == 2:
        desc_part = parts[0]
        req_part = parts[1]
    else:
        desc_part = full_html
        req_part = full_html

    # Clean HTML tags
    desc_clean = re.sub(r"<[^>]+>", "\n", desc_part)
    desc_clean = re.sub(r"\n{3,}", "\n\n", desc_clean).strip()

    req_clean = re.sub(r"<[^>]+>", "\n", req_part)
    req_clean = re.sub(r"\n{3,}", "\n\n", req_clean).strip()

    return desc_clean, req_clean


def repair_all_jobs(
    structured_dir: Path = Path("data/processed/structured_jobs"),
    raw_dir: Path = Path("data/raw/jobs"),
):
    json_files = sorted(structured_dir.glob("jd_*.json"))
    logger.info(f"Auditing and repairing {len(json_files)} jobs...")

    repaired_count = 0
    all_jobs_data = []

    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as fp:
            job_dict = json.load(fp)

        desc = job_dict.get("description", {})
        raw_text = desc.get("raw_text", "")
        needs_repair = len(raw_text.strip()) < 300

        if needs_repair:
            url = job_dict.get("url")
            logger.info(f"Repairing {job_dict['id']} ({job_dict['title']}) from {url}...")
            ld_data = fetch_jsonld_jobposting(url)

            if ld_data and ld_data.get("description"):
                full_raw_html = ld_data.get("description", "")
                desc_text, req_text = split_description_and_requirements(full_raw_html)

                resp_bullets = parse_bullets(desc_text)
                req_bullets = parse_bullets(req_text)

                combined_raw = f"{desc_text}\n\n{req_text}".strip()
                job_dict["description"]["responsibilities"] = resp_bullets[:12]
                job_dict["description"]["requirements"] = req_bullets[:12]
                job_dict["description"]["raw_text"] = combined_raw

                # Check years of experience
                min_y, max_y = extract_years_experience(combined_raw)
                if min_y is not None and job_dict["seniority"]["min_years"] is None:
                    job_dict["seniority"]["min_years"] = min_y
                    job_dict["seniority"]["max_years"] = max_y

                # Check salary in JSON-LD if currently null
                comp = job_dict.get("compensation", {})
                if comp.get("min_monthly") is None and ld_data.get("baseSalary"):
                    bs = ld_data["baseSalary"]
                    val = bs.get("value", {})
                    curr = bs.get("currency", "USD")
                    if isinstance(val, dict):
                        if val.get("minValue") or val.get("maxValue"):
                            comp["currency"] = curr
                            comp["min_monthly"] = val.get("minValue")
                            comp["max_monthly"] = val.get("maxValue")
                            comp["is_negotiable"] = False
                            comp["raw_text"] = f"{val.get('minValue')} - {val.get('maxValue')} {curr}"

                repaired_count += 1
            else:
                logger.warning(f"Could not fetch JSON-LD for {job_dict['id']}")

        # Validate with Pydantic
        job_obj = JobDescriptionSchema.model_validate(job_dict)

        # Overwrite structured JSON
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(job_obj.model_dump_json(by_alias=True, indent=2))

        # Overwrite raw .txt file with complete plain text
        raw_txt_path = raw_dir / f"{job_obj.id}.txt"
        with open(raw_txt_path, "w", encoding="utf-8") as fp:
            raw_content = (
                f"Title: {job_obj.title}\n"
                f"Company: {job_obj.company.name}\n"
                f"Location: {job_obj.location.city}, {job_obj.location.country}\n"
                f"Category: {job_obj.category}\n"
                f"Seniority: {job_obj.seniority.level or 'Not specified'}\n"
                f"Compensation: {job_obj.compensation.raw_text or 'Negotiable'}\n"
                f"\n--- JOB DESCRIPTION & REQUIREMENTS ---\n"
                f"{job_obj.description.raw_text}\n"
            )
            if job_obj.benefits and job_obj.benefits.raw_text:
                raw_content += f"\n--- BENEFITS & PERKS ---\n{job_obj.benefits.raw_text}\n"
            fp.write(raw_content)

        all_jobs_data.append(job_obj.model_dump(by_alias=True))

    # Save consolidated dataset
    consolidated_path = structured_dir / "itviec_jobs_all.json"
    with open(consolidated_path, "w", encoding="utf-8") as fp:
        json.dump(all_jobs_data, fp, ensure_ascii=False, indent=2)

    logger.info(f"Repair complete! {repaired_count} incomplete jobs successfully restored.")


def run_full_audit(raw_dir: Path = Path("data/raw/jobs")):
    txt_files = sorted(raw_dir.glob("*.txt"))
    print("\n" + "=" * 60)
    print(f"📊 FULL AUDIT OF {raw_dir} ({len(txt_files)} files)")
    print("=" * 60)

    short_files = []
    total_bytes = 0

    for f in txt_files:
        size = f.stat().st_size
        total_bytes += size
        text = f.read_text(encoding="utf-8")
        lines = len(text.splitlines())
        chars = len(text)
        if size < 500:
            short_files.append((f.name, size, chars))
        status = "✅ PASS" if size >= 500 else "❌ SHORT"
        print(f"{f.name:<12} | {size:>6} bytes | {lines:>3} lines | {chars:>5} chars | {status}")

    print("-" * 60)
    print(f"Total size: {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"Short/corrupted files (< 500 bytes): {len(short_files)}")
    if short_files:
        print("Failed files:", short_files)
    else:
        print("🎉 100% of raw job files passed verification!")
    print("=" * 60)


if __name__ == "__main__":
    repair_all_jobs()
    run_full_audit()
