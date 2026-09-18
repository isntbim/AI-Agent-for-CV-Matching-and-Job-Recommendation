"""
ITviec Scraper Importer and Standardizer
Transforms raw scraped ITviec jobs from crawlers/itviec-scraper/itviec-jobs.json
into the standardized JobDescriptionSchema (Pydantic v2).
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.parser.job_schemas import JobDescriptionSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_bullet(line: str) -> str:
    """Clean bullet symbols and numbers while preserving text like '5+ years'."""
    line = line.strip()
    # Strip leading bullet icons, dashes, asterisks
    line = re.sub(r"^[\s*•\-–—]+\s*", "", line)
    # Strip leading numbered list markers e.g. "1. " or "2) "
    line = re.sub(r"^\d+[\.\)]\s+", "", line)
    return line.strip()


def parse_bullets(text: Optional[str]) -> List[str]:
    """Extract clean bullet points from description/requirements markdown."""
    if not text:
        return []
    lines = text.split("\n")
    bullets = []
    for line in lines:
        cleaned = clean_bullet(line)
        # Exclude tiny lines or generic section headers
        if len(cleaned) > 10 and not cleaned.lower().endswith(("responsibilities:", "qualifications:", "requirements:")):
            bullets.append(cleaned)
    return bullets


def extract_years_experience(text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Extract min and max years of experience from requirements text."""
    if not text:
        return None, None

    # Range format: "3 to 5 years", "3-5 years", "3 - 5 năm"
    range_match = re.search(r"(\d+)\s*(?:to|-)\s*(\d+)\s*(?:years?|năm)", text, re.IGNORECASE)
    if range_match:
        min_y, max_y = int(range_match.group(1)), int(range_match.group(2))
        if 0 <= min_y <= 25 and min_y <= max_y <= 30:
            return min_y, max_y

    # Plus format: "5+ years", "5+ năm", "tối thiểu 3 năm", "at least 3 years"
    min_match = re.search(r"(?:tối thiểu|at least|min(?:imum)?|\+)?\s*(\d+)\s*(?:\+|years?|năm)", text, re.IGNORECASE)
    if min_match:
        val = int(min_match.group(1))
        if 1 <= val <= 25:
            return val, None

    # Single format: "5 years experience", "3 năm kinh nghiệm"
    single_match = re.search(r"(\d+)\s*(?:years?|năm)\s*(?:of\s+)?experience", text, re.IGNORECASE)
    if single_match:
        val = int(single_match.group(1))
        if 1 <= val <= 25:
            return val, None

    return None, None


def detect_seniority_level(title: str, min_years: Optional[int] = None) -> Tuple[str, str]:
    """Determine normalized seniority level and raw level label."""
    title_lower = title.lower()
    first_token = title.split()[0] if title else "Middle"

    if any(w in title_lower for w in ["director", "head of", "vp", "chief"]):
        return "Director", "Director"
    if any(w in title_lower for w in ["principal", "expert", "chuyên gia"]):
        return "Lead", "Principal"
    if any(w in title_lower for w in ["lead", "trưởng nhóm"]):
        return "Lead", "Lead"
    if any(w in title_lower for w in ["manager", "deputy manager", "quản lý"]):
        return "Manager", "Manager"
    if any(w in title_lower for w in ["senior", "sr.", "sr ", "cvcc"]):
        return "Senior", "Senior"
    if any(w in title_lower for w in ["junior", "jr.", "fresher", "entry"]):
        return "Junior", "Junior"
    if any(w in title_lower for w in ["intern", "thực tập"]):
        return "Intern", "Intern"
    if any(w in title_lower for w in ["middle", "mid-level", "mid"]):
        return "Mid-level", "Middle"

    # Infer from years of experience if title doesn't specify
    if min_years is not None:
        if min_years >= 7:
            return "Lead", "Lead"
        elif min_years >= 4:
            return "Senior", "Senior"
        elif min_years >= 2:
            return "Mid-level", "Middle"
        elif min_years >= 1:
            return "Junior", "Junior"
        else:
            return "Junior", "Entry"

    return "Mid-level", first_token


def normalize_title_category(title: str) -> Tuple[str, str]:
    """Normalize job title to snake_case identifier and assign domain category."""
    t = title.lower()

    if any(w in t for w in ["data scientist", "data analyst", "bi analyst", "business intelligence", "analytics"]):
        return "data_analyst", "data_scientist_analyst"
    elif any(w in t for w in ["data engineer", "data warehouse", "data platform"]):
        return "data_engineer", "data_scientist_analyst"
    elif any(w in t for w in ["frontend", "front-end", "react", "vue", "angular", "ui/ux", "web developer"]):
        return "frontend_developer", "frontend_web_developer"
    elif any(w in t for w in ["security", "cybersecurity", "incident response", "appsec", "devsecops", "soc"]):
        return "security_engineer", "security_engineer"
    elif any(w in t for w in ["devops", "cloud", "platform", "sre", "infrastructure", "appops", "sysadmin"]):
        return "devops_engineer", "devops_cloud"
    elif any(w in t for w in ["solution architect", "enterprise architect", "architect"]):
        return "solution_architect", "solution_architecture"
    elif any(w in t for w in ["qa", "qc", "tester", "quality assurance", "test engineer"]):
        return "qa_engineer", "quality_assurance"
    elif any(w in t for w in ["product manager", "product owner", "po", "pm"]):
        return "product_manager", "product_management"
    elif any(w in t for w in ["accountant", "kế toán", "accounting"]):
        return "accountant", "accountant"
    elif any(w in t for w in ["hr", "human resources", "talent acquisition", "recruiter", "nhân sự"]):
        return "hr_specialist", "hr"
    elif any(w in t for w in ["marketing", "seo", "content", "growth"]):
        return "marketing_executive", "marketing_executive"
    elif any(w in t for w in ["backend", "back-end", "python", "golang", "java", "c#", ".net", "php", "nodejs", "software engineer", "developer"]):
        return "software_engineer", "software_engineer"
    else:
        clean_slug = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
        return clean_slug[:30], "software_engineer"


def parse_location(raw_loc: str) -> Tuple[str, str, str, str]:
    """Parse city, region, country, and countryCode."""
    raw = raw_loc or "Ho Chi Minh City"
    has_hcm = "ho chi minh" in raw.lower()
    has_hn = "ha noi" in raw.lower() or "hà nội" in raw.lower()
    has_dn = "da nang" in raw.lower() or "đà nẵng" in raw.lower()

    if has_hcm and has_hn:
        city = "Ho Chi Minh City, Ha Noi"
        region = "South / North"
    elif has_hcm:
        city = "Ho Chi Minh City"
        region = "South"
    elif has_hn:
        city = "Ha Noi"
        region = "North"
    elif has_dn:
        city = "Da Nang"
        region = "Central"
    else:
        city = raw
        region = "South"

    return city, region, "Vietnam", "VN"


def parse_compensation(salary_str: Optional[str]) -> Tuple[Optional[str], Optional[int], Optional[int], bool]:
    """Extract currency, min monthly, max monthly, and negotiability flag."""
    if not salary_str:
        return "VND", None, None, True

    curr = "USD" if ("$" in salary_str or "usd" in salary_str.lower()) else "VND"
    s_lower = salary_str.lower()
    is_neg = "view" in s_lower or "thoả thuận" in s_lower or "negotiable" in s_lower or "love it" in s_lower

    clean_str = salary_str.replace(",", "")
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", clean_str)]
    min_m, max_m = None, None

    if nums and not is_neg:
        multiplier = 1000000 if (curr == "VND" and max(nums) < 1000) else 1
        if len(nums) == 1:
            max_m = int(nums[0] * multiplier)
        elif len(nums) >= 2:
            min_m = int(nums[0] * multiplier)
            max_m = int(nums[1] * multiplier)

    return curr, min_m, max_m, is_neg


def transform_raw_job(raw: Dict, index: int) -> JobDescriptionSchema:
    """Transform a single raw scraped job dict from ITviec into JobDescriptionSchema."""
    title = raw.get("title", "").strip()
    req_text = raw.get("requirements", "") or ""
    desc_text = raw.get("jobDescription", "") or ""
    reasons = raw.get("reasons", "") or ""

    min_yrs, max_yrs = extract_years_experience(req_text)
    seniority_level, level_raw = detect_seniority_level(title, min_yrs)
    norm_title, category = normalize_title_category(title)

    city, region, country, country_code = parse_location(raw.get("location", ""))
    working_mode = raw.get("workingMode", "At office")
    wm_lower = working_mode.lower()
    if "hybrid" in wm_lower:
        remote_policy = "Hybrid"
    elif "remote" in wm_lower:
        remote_policy = "Remote"
    else:
        remote_policy = "Onsite"

    comp_info = raw.get("companyInfo") or {}
    company_name = raw.get("company") or comp_info.get("name") or "Confidential"
    size_raw = comp_info.get("Company size", "")
    company_size = re.sub(r"\s*employees?", "", size_raw).strip() if size_raw else None
    industry = comp_info.get("Company industry") or comp_info.get("Company type") or "Information Technology"

    tags = raw.get("tags") or []
    skills_raw = raw.get("skills") or []
    combined_skills = list(dict.fromkeys(tags + skills_raw))

    # Identify preferred vs required skills
    preferred_names = set()
    pref_match = re.search(r"(?:preferred|nice to have|ưu tiên)[\s\S]+", req_text, re.IGNORECASE)
    if pref_match:
        pref_block = pref_match.group(0).lower()
        for sk in combined_skills:
            if sk.lower() in pref_block and sk not in tags[:3]:
                preferred_names.add(sk)

    required_items = []
    preferred_items = []
    for sk in combined_skills:
        if sk in preferred_names:
            preferred_items.append({"name": sk, "proficiency": "preferred"})
        else:
            required_items.append({"name": sk, "proficiency": "required"})

    if not preferred_items and len(required_items) > 5:
        # If no explicit split found, prioritize top tags as required and tail as preferred
        preferred_items = [{"name": it["name"], "proficiency": "preferred"} for it in required_items[5:]]
        required_items = required_items[:5]

    resp_bullets = parse_bullets(desc_text)
    req_bullets = parse_bullets(req_text)
    summary = reasons.strip().replace("\n", "; ") if reasons else (desc_text[:200].strip() if desc_text else None)

    curr, min_m, max_m, is_neg = parse_compensation(raw.get("salary"))

    job_id = f"jd_{index+1:03d}"
    raw_full_text = f"{desc_text}\n\n{req_text}".strip()

    job_dict = {
        "id": job_id,
        "source": "itviec",
        "url": raw.get("url", ""),
        "scraped_at": raw.get("scrapedAt", datetime.now().isoformat()),
        "title": title,
        "title_normalized": norm_title,
        "category": category,
        "company": {
            "name": company_name,
            "size": company_size,
            "industry": industry,
        },
        "location": {
            "city": city,
            "region": region,
            "country": country,
            "countryCode": country_code,
            "remote_policy": remote_policy,
            "remote_policy_raw": working_mode,
        },
        "seniority": {
            "level": seniority_level,
            "level_raw": level_raw,
            "min_years": min_yrs,
            "max_years": max_yrs,
        },
        "skills": {
            "required": required_items,
            "preferred": preferred_items,
            "raw_text": ", ".join(combined_skills),
        },
        "description": {
            "summary": summary,
            "responsibilities": resp_bullets[:10],
            "requirements": req_bullets[:10],
            "raw_text": raw_full_text,
        },
        "compensation": {
            "currency": curr,
            "min_monthly": min_m,
            "max_monthly": max_m,
            "min_annual": None,
            "max_annual": None,
            "is_negotiable": is_neg,
            "raw_text": raw.get("salary") or "Negotiable",
        },
        "benefits": {
            "raw_text": raw.get("benefits", "").strip() or None,
        },
        "metadata": {
            "posted_date": raw.get("postedTime"),
            "deadline": None,
            "job_type": "Full-time",
            "work_arrangement": remote_policy,
        },
    }

    return JobDescriptionSchema.model_validate(job_dict)


def get_existing_jobs_index_and_urls(output_dir: Path) -> Tuple[int, Dict[str, str]]:
    """Find maximum existing job ID index and map existing URLs to their job IDs."""
    max_idx = 0
    url_to_id = {}
    for p in output_dir.glob("jd_*.json"):
        m = re.search(r"jd_(\d+)", p.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "url" in d and "id" in d:
                    url_to_id[d["url"]] = d["id"]
        except Exception:
            pass
    return max_idx, url_to_id


def import_itviec_jobs(
    input_file: Path = Path("crawlers/itviec-scraper/itviec-jobs.json"),
    output_dir: Path = Path("data/processed/structured_jobs"),
    raw_dir: Path = Path("data/raw/jobs"),
) -> List[JobDescriptionSchema]:
    """
    Read ITviec scraped jobs, transform them to JobDescriptionSchema, and persist them.
    Supports incremental additions without overwriting existing jobs.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"ITviec crawled jobs file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    max_idx, url_to_id = get_existing_jobs_index_and_urls(output_dir)

    with open(input_file, "r", encoding="utf-8") as f:
        raw_jobs = json.load(f)

    logger.info(f"Loaded {len(raw_jobs)} raw jobs from {input_file} (current max index: {max_idx})")
    processed_jobs: List[JobDescriptionSchema] = []

    for raw in raw_jobs:
        try:
            url = raw.get("url", "")
            if url in url_to_id:
                # Reuse existing ID for updates
                m = re.search(r"jd_(\d+)", url_to_id[url])
                target_idx = int(m.group(1)) - 1 if m else max_idx
            else:
                max_idx += 1
                target_idx = max_idx - 1

            job_obj = transform_raw_job(raw, target_idx)
            url_to_id[job_obj.url] = job_obj.id
            processed_jobs.append(job_obj)

            # Save individual structured JSON
            job_path = output_dir / f"{job_obj.id}.json"
            with open(job_path, "w", encoding="utf-8") as f:
                f.write(job_obj.model_dump_json(by_alias=True, indent=2))

            # Save raw description text for BM25 / NLP baseline
            raw_path = raw_dir / f"{job_obj.id}.txt"
            raw_content = f"Title: {job_obj.title}\nCompany: {job_obj.company.name}\n\n{job_obj.description.raw_text}"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(raw_content)

        except Exception as e:
            logger.error(f"Failed to process job ({raw.get('title')}): {e}")

    # Save consolidated dataset
    consolidated_path = output_dir / "itviec_jobs_all.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        data = [job.model_dump(by_alias=True) for job in processed_jobs]
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Successfully processed and saved {len(processed_jobs)} / {len(raw_jobs)} jobs to {output_dir}"
    )
    return processed_jobs


if __name__ == "__main__":
    import_itviec_jobs()
